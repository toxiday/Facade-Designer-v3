"""
optimize_facade_allsides_climate.py  (all-sides + climate + generalized room)
Gradient-based facade optimizer for the 4-facade all-sides climate transformer,
evaluated across the WHOLE TMYx year at a user-chosen hour, for a chosen city, a
chosen room W/D/H, and a user-chosen subset of walls (S/N/E/W) that may carry windows.

Fuses two optimizers:
  * optimize_facade_allsides.py (repo root) — the 4-facade logit parameterization:
        X_logits (4, R, C) -> sigmoid(X/tau) -> P (window prob) -> 1-P (model input,
        <0.5 = window) -> surrogate -> lux.
  * sDA_Generalized Model/.../optimize_facade_climate.py — the 8-vector climate + room
    conditioning (annual_conditions + reduce_conditions + append_room), the comfort-band
    zone loss, and chunked grad accumulation over the year.

The design is ONE set of 4 facades evaluated under every (sun, DNI, DHI) condition of the
year.  `facade_open` (bool per S/N/E/W) freezes unpicked walls to fully solid: their P is
forced to 0 each step and in the returned binary, so they never receive windows.

Facade convention (matches the dataset / model): index = row*FACADE_COLS+col, row 0 = top
(ceiling).

S/N/E/W are ROOM-LOCAL wall ids, NOT compass directions:
    S = wall at y=0    N = wall at y=D    E = wall at x=W    W = wall at x=0
    col 0 = the x=0 end for S/N;  col 0 = the y=0 end for E/W.
They coincide with true compass directions only when north_angle = 0.  A rotated building is
handled by rotating the SUN into the room frame (annual_conditions(..., north_angle=...)),
never by relabelling walls — the room geometry the surrogate sees is always this fixed frame.

Window encodings — the authoritative mapping for the whole project (see WIN_VAL/WALL_VAL below):
    dataset / model input   {side}_Pix_k  float, <0.5 = window
    optimizer internal      P             window probability, 1 = window
    both output CSVs        is_window     1 = window
"""

import math
import glob
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_allsides_climate import (
    AllSidesClimateTransformer, MODEL_CONFIG, BEST_MODEL_PATH,
    N_FACADES, FACADE_ROWS, FACADE_COLS, GRID_N,
    DNI_SCALE, DHI_SCALE, W_MIN, W_MAX, D_MIN, D_MAX, H_MIN, H_MAX)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parents[2]
CLIMATE_DIR = REPO_ROOT / "allside_Generalised model" / "climate data" / "library"

GLARE_LUX = 3000.0
MIN_ALT   = 5.0

DEFAULT_TARGET_LUX = 400.0
ZONE_MIN_RATIO     = 0.75
ZONE_MAX_RATIO     = 1.25
ZONE_GLARE_RATIO   = 2.0

DEFAULT_WIDTH  = 6.0
DEFAULT_DEPTH  = 8.0
DEFAULT_HEIGHT = 3.0

SIDES = ('S', 'N', 'E', 'W')     # facade index order (matches the dataset)

# The surrogate was trained on facade values window ~U[0,0.5), wall ~U[0.5,1] (class means
# 0.25 / 0.75) — it NEVER saw hard 0/1.  Feeding exact 0/1 is OUT-OF-DISTRIBUTION and makes
# the model hallucinate phantom sun beams (~40k lux) on adverse orientations.  So the model
# input is a FLOAT kept in the training band, not 0/1:  facade_in = WALL_VAL-(WALL_VAL-WIN_VAL)*p
# with p = window probability / binary (1 = window).  window(p=1)->0.25, wall(p=0)->0.75.
WIN_VAL, WALL_VAL = 0.25, 0.75


def facade_in(p):
    """Map window-prob / binary (1=window) -> in-distribution model facade input (float)."""
    return WALL_VAL - (WALL_VAL - WIN_VAL) * p


# ---------------------------------------------------------------------------
# Solar position  (verbatim from generate_dataset_climate_pit.py)
# ---------------------------------------------------------------------------

def solar_position(lat_deg, lon_deg, tz_h, month, day, hour_decimal):
    """(altitude_deg, azimuth_north_cw_deg, (sx, sy, sz)). sz<0 when sun is up."""
    lat = math.radians(lat_deg)
    doy = int(275 * month / 9 - 30 + day - 2)
    if month <= 2:
        doy += 2
    B    = math.radians(360 / 365 * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    ET   = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    solar_time = hour_decimal + (lon_deg - tz_h * 15) / 15 + ET / 60
    omega      = math.radians(15 * (solar_time - 12))
    sin_alt = max(-1.0, min(1.0,
        math.sin(lat) * math.sin(decl) +
        math.cos(lat) * math.cos(decl) * math.cos(omega)))
    altitude_deg = math.degrees(math.asin(sin_alt))
    az_south_rad = math.atan2(
        math.sin(omega),
        math.cos(omega) * math.sin(lat) - math.tan(decl) * math.cos(lat))
    az = (math.degrees(az_south_rad) + 180.0) % 360.0
    alt_r, az_r = math.radians(altitude_deg), math.radians(az)
    sx = -math.sin(az_r) * math.cos(alt_r)
    sy = -math.cos(az_r) * math.cos(alt_r)
    sz = -math.sin(alt_r)
    return altitude_deg, az, (sx, sy, sz)


# ---------------------------------------------------------------------------
# Climate library
# ---------------------------------------------------------------------------

def discover_cities():
    files = sorted(glob.glob(str(CLIMATE_DIR / "**" / "*.wea"), recursive=True))
    if not files:
        raise RuntimeError(f"No .wea files found under {CLIMATE_DIR}")
    return files


def city_label(path):
    return Path(path).stem.split("_TMYx")[0]


def resolve_city(spec, cities=None):
    """Resolve a city spec to a .wea path.  Accepts, in precedence order:
        * an int (or all-digit string) index into discover_cities()  -- back-compat with --city N
        * an exact filename stem     "USA_TX_Houston-....722430_TMYx.2011-2025"
        * an exact city_label()      "USA_TX_Houston-Bush.Intercontinental.AP.722430"
        * a UNIQUE case-insensitive substring of the label   "houston"

    The positional index is fragile — it shifts whenever a .wea is added to or removed from
    the library — so job files should carry a NAME.  An ambiguous substring raises and lists
    the candidates rather than silently picking one, because resolving to the wrong city
    produces a completely plausible design for the wrong place.
    """
    if cities is None:
        cities = discover_cities()

    idx = None
    if isinstance(spec, bool):
        raise TypeError("city spec must be an index or a name, not a bool")
    if isinstance(spec, int):
        idx = spec
    elif isinstance(spec, str) and spec.strip().lstrip("+-").isdigit():
        idx = int(spec.strip())
    if idx is not None:
        if not (0 <= idx < len(cities)):
            raise IndexError(f"city index {idx} out of range (0..{len(cities) - 1})")
        return cities[idx]

    name = str(spec).strip()
    if not name:
        raise ValueError("empty city spec")
    for c in cities:
        if Path(c).stem == name:
            return c
    for c in cities:
        if city_label(c) == name:
            return c
    low  = name.lower()
    hits = [c for c in cities if low in city_label(c).lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no city matching {name!r} under {CLIMATE_DIR}")
    raise ValueError(f"city spec {name!r} is ambiguous ({len(hits)} matches):\n  "
                     + "\n  ".join(city_label(c) for c in hits[:20])
                     + ("\n  ..." if len(hits) > 20 else ""))


def annual_conditions(wea_path, hour_slot, include_below_horizon=False, north_angle=0.0):
    """
    (label, latitude, cond, days) with cond (D,5) = [Sun_X,Sun_Y,Sun_Z,DNI/s,DHI/s]
    (climate only; append_room adds the 3 room channels -> (D,8)).

    north_angle -- rotation of the BUILDING, in Ladybug's convention: the clockwise angle
    from the room's local +Y axis to true North.  0 = +Y is true north (legacy behaviour,
    bit-identical), 90 = +Y faces West, 270 = +Y faces East.  Adopted verbatim from
    ladybug/compass.py ("90 is West and 270 is East"; angle_clockwise(north_vector)) because
    Ladybug/Honeybee are installed alongside this project and a privately inverted north
    would be a vicious bug the day the two are wired together.

    THE ROOM NEVER ROTATES -- THE SUN DOES.  The surrogate's room frame is fixed (S at y=0,
    N at y=D, E at x=W, W at x=0) and its only positional input is a sun vector in that
    frame.  Since the azimuth of +Y is -north_angle, the sun's azimuth as the room
    experiences it is az_true + north_angle, which is the rotation applied below.  Altitude
    is untouched.

        north_angle | +Y faces |  S(y=0)  N(y=D)  E(x=W)  W(x=0)
        ------------+----------+--------------------------------
              0     |  North   |  South   North    East    West
             90     |  West    |  East    West     North   South
            180     |  South   |  North   South    West    East
            270     |  East    |  West    East     South   North

    Worked example (north_angle=270, so E is the wall physically facing south): a noon sun
    at az=180, alt=40 has sx=0, sy=+0.766; rotating gives sx=-0.766, sy=0, i.e. light
    travelling in -X, entering through the x=W (E) wall.  Correct.

    IS A ROTATED ROOM IN-DISTRIBUTION?  Yes, and it was measured rather than assumed.  The
    model has no north input at all, so rotating only asks it for a different sun vector.
    Binning all 672,603 lit sun vectors of the 170-city library (alt>5, DNI+DHI>0) by
    altitude and azimuth gives 36/36 azimuth bins occupied in EVERY altitude band, 5-90
    degrees -- there is no azimuth hole to fall into, so any north_angle lands in a
    populated region.  The only thin band is 85-90 deg (1,791 samples), and that sparsity
    already affects unrotated tropical-noon use; rotation adds no new deficiency.  Re-run
    "rhino bridge/check_sun_coverage.py" if the climate library is ever expanded.
    """
    lat = lon_w = tz_deg = None
    rows = []
    with open(wea_path) as f:
        for line in f:
            p = line.split()
            if   len(p) == 2 and p[0] == "latitude":  lat    = float(p[1])
            elif len(p) == 2 and p[0] == "longitude": lon_w  = float(p[1])
            elif len(p) == 2 and p[0] == "time_zone": tz_deg = float(p[1])
            elif len(p) == 5:
                rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
    lon_east = -lon_w
    tz_h     = -tz_deg / 15.0

    # Rotate the horizontal sun vector into the room-local frame (see docstring).  Guarded on
    # a falsy north_angle so the legacy path does literally zero float ops and stays
    # bit-identical -- that exactness is what verification asserts.
    rotate = bool(north_angle)
    if rotate:
        th = math.radians(north_angle)
        ct, st = math.cos(th), math.sin(th)

    cond, days = [], []
    for month, day, hour, dni, dhi in rows:
        if int(hour) != hour_slot:
            continue
        alt, az, (sx, sy, sz) = solar_position(lat, lon_east, tz_h, month, day, hour)
        if alt <= MIN_ALT and not include_below_horizon:
            continue
        if rotate:
            sx, sy = sx * ct + sy * st, -sx * st + sy * ct
        cond.append([sx, sy, sz, dni / DNI_SCALE, dhi / DHI_SCALE])
        days.append((month, day, alt, dni, dhi))
    return city_label(wea_path), lat, np.array(cond, dtype=np.float32), days


def city_location(wea_path):
    lat = lon_w = tz_deg = None
    with open(wea_path) as f:
        for line in f:
            p = line.split()
            if   len(p) == 2 and p[0] == "latitude":  lat    = float(p[1])
            elif len(p) == 2 and p[0] == "longitude": lon_w  = float(p[1])
            elif len(p) == 2 and p[0] == "time_zone": tz_deg = float(p[1])
            if lat is not None and lon_w is not None and tz_deg is not None:
                break
    return lat, -lon_w, -tz_deg / 15.0


# ---------------------------------------------------------------------------
# Room geometry conditioning
# ---------------------------------------------------------------------------

def normalize_room(width, depth, height):
    wn = (np.clip(width,  W_MIN, W_MAX) - W_MIN) / (W_MAX - W_MIN)
    dn = (np.clip(depth,  D_MIN, D_MAX) - D_MIN) / (D_MAX - D_MIN)
    hn = (np.clip(height, H_MIN, H_MAX) - H_MIN) / (H_MAX - H_MIN)
    return np.array([wn, dn, hn], dtype=np.float32)


def append_room(cond5, width, depth, height):
    """(T,5) climate cond -> (T,8) by appending constant normalized W/D/H columns."""
    cond5 = np.asarray(cond5, dtype=np.float32)
    room  = np.broadcast_to(normalize_room(width, depth, height), (len(cond5), 3))
    return np.concatenate([cond5, room], axis=1).astype(np.float32)


N_REPR = 48


def reduce_conditions(cond, n_clusters=N_REPR, iters=30, seed=0):
    """Weighted k-means over the (D,5) climate conditions -> <=n_clusters weighted reps."""
    cond = np.asarray(cond, dtype=np.float32)
    D = cond.shape[0]
    if D <= n_clusters:
        return cond, np.ones(D, dtype=np.float32)
    rng       = np.random.default_rng(seed)
    centroids = cond[rng.choice(D, n_clusters, replace=False)].copy()
    assign    = np.full(D, -1, dtype=np.int64)
    for _ in range(iters):
        d = ((cond[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new_assign = d.argmin(1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for k in range(n_clusters):
            m = assign == k
            if m.any():
                centroids[k] = cond[m].mean(0)
    counts = np.bincount(assign, minlength=n_clusters).astype(np.float32)
    keep   = counts > 0
    return centroids[keep].astype(np.float32), counts[keep]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def zone_penalty_sum(E_pred, target_lux, w=None):
    """Asymmetric comfort-band penalty SUM over occupied cells (target_lux > 0)."""
    occupied  = target_lux > 0.0
    zeros     = torch.zeros_like(E_pred)
    tgt       = torch.where(occupied, target_lux, torch.ones_like(target_lux))
    band_min  = tgt * ZONE_MIN_RATIO
    band_max  = tgt * ZONE_MAX_RATIO
    hard_max  = tgt * ZONE_GLARE_RATIO
    too_dark  = torch.where(occupied, F.relu(band_min - E_pred) / band_min, zeros)
    mild_over = torch.where(occupied, F.relu(E_pred - band_max) / band_max, zeros)
    hard_over = torch.where(occupied, F.relu(E_pred - hard_max) / hard_max, zeros)
    glare_in  = torch.where(occupied, F.relu(E_pred - GLARE_LUX) / GLARE_LUX, zeros)
    pen = (2.0 * too_dark + 0.5 * mild_over + 2.0 * hard_over + 4.0 * glare_in)
    if w is not None:
        pen = pen * w.view(-1, 1, 1)
    return pen.sum()


DEFAULT_GLARE_WEIGHT = 0.0      # off by default — see outside_glare_sum()


def outside_glare_sum(E_pred, target_lux, glare_cap=GLARE_LUX, w=None):
    """Glare penalty SUM over UNOCCUPIED cells (target_lux == 0) — the exact complement of
    zone_penalty_sum, which masks every one of its terms (glare included) by `occupied`.

    WHY THIS EXISTS.  Without it the optimizer is completely indifferent to the unpainted
    floor, and worse, it exploits that: dumping surplus daylight outside the zone is the
    cheapest way to hit the in-zone target without overshooting it.  A measured example —
    a design scoring 0% in-zone glare simultaneously put 30,276 lux on unpainted floor,
    with the unoccupied mean (844 lux) at twice the occupied mean (415).  Glare is a
    field-of-view phenomenon, so a sun patch two metres from a desk is a real problem even
    though no scored sensor sees it.

    NORMALISATION.  Callers divide this by the UNOCCUPIED cell-day count, not by n_occ,
    so it becomes a per-cell mean like the zone term.  That keeps `glare_weight`
    scale-free: a small zone in a big room would otherwise carry ~5x the glare pressure of
    a large zone, purely from the cell-count ratio.

    Default weight is 0.0 (OFF) so the interactive designer and the legacy CLI keep their
    existing behaviour exactly.  The Rhino job runner opts in.
    """
    unocc = target_lux <= 0.0
    pen   = torch.where(unocc, F.relu(E_pred - glare_cap) / glare_cap,
                        torch.zeros_like(E_pred))
    if w is not None:
        pen = pen * w.view(-1, 1, 1)
    return pen.sum()


def tv_loss(P):
    """Total variation over the 4 facade window-prob grids (contiguity, not scatter)."""
    tv_h = torch.abs(P[:, 1:, :] - P[:, :-1, :]).sum()
    tv_w = torch.abs(P[:, :, 1:] - P[:, :, :-1]).sum()
    return (tv_h + tv_w) / P.numel()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class FacadeOptimizerAllSidesClimate:
    def __init__(self, weights_path=BEST_MODEL_PATH, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = AllSidesClimateTransformer(**MODEL_CONFIG).to(device)
        state = torch.load(str(weights_path), weights_only=True, map_location=device)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.use_amp   = (device.type == "cuda")
        self.amp_dtype = (torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported())
                          else torch.float16)
        print(f"All-sides climate surrogate loaded on {device.type.upper()}"
              f"  (AMP {self.amp_dtype if self.use_amp else 'off'})")

    def _open_mask(self, facade_open):
        """(4,1,1) tensor: 1 for walls that may hold windows, 0 for frozen-solid walls."""
        m = torch.tensor([1.0 if o else 0.0 for o in facade_open],
                         dtype=torch.float32, device=self.device)
        return m.view(N_FACADES, 1, 1)

    def _lux(self, cond_chunk, fac_in):
        """fac_in (1,4,R,C) window-prob complement -> lux (b,32,32) for cond_chunk (b,8)."""
        b   = cond_chunk.shape[0]
        fac = fac_in.expand(b, N_FACADES, FACADE_ROWS, FACADE_COLS).contiguous()
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            pred = self.model(cond_chunk, fac)
        return pred.float().squeeze(1) ** 2

    def optimize(self, cond, target_lux, weights=None, facade_open=(True, True, True, True), *,
                 tau=1.0, tau_binarize=0.05, lambda_tv=0.02,
                 n_steps=1000, lr=0.05, chunk=16,
                 glare_weight=DEFAULT_GLARE_WEIGHT, glare_cap=GLARE_LUX,
                 stable_patience=120, stable_tol=3, loss_patience=200,
                 progress_callback=None, log_every=25):
        # NOTE on `chunk`: backprop to the facade logits retains the full forward graph
        # for each chunk, so peak VRAM scales with chunk x (512 tokens).  chunk=16 keeps
        # one forward+backward well under 16 GB; the loss accumulates across chunks so the
        # result is identical to a large chunk — only memory (and thus speed, no spill) changes.
        """
        cond        : (T, 8)      float32 — climate + room conditioning per condition.
        target_lux  : (T, 32, 32) float32 — desired lux per floor cell (zone x target).
        weights     : (T,)        float32 or None — days-represented per condition.
        facade_open : (4,) bool   — which of S/N/E/W may hold windows (rest = solid).
        glare_weight: 0 = off (legacy).  Weight on mean UNOCCUPIED-cell glare above
                      glare_cap — see outside_glare_sum().  Deliberately subordinate: the
                      in-zone comfort band stays the primary objective.
        Returns (binary (4,R,C) int, P_final (4,R,C), history).
        """
        T       = cond.shape[0]
        cond_t  = torch.tensor(cond,       dtype=torch.float32, device=self.device)
        tgt_t   = torch.tensor(target_lux, dtype=torch.float32, device=self.device)
        w_t     = (torch.ones(T, device=self.device) if weights is None
                   else torch.as_tensor(weights, dtype=torch.float32, device=self.device))
        n_occ   = (w_t.view(-1, 1, 1) * (tgt_t > 0).float()).sum().clamp(min=1.0)
        n_unocc = (w_t.view(-1, 1, 1) * (tgt_t <= 0).float()).sum().clamp(min=1.0)
        open_m  = self._open_mask(facade_open)

        X   = (torch.randn(N_FACADES, FACADE_ROWS, FACADE_COLS, device=self.device) * 0.1
               ).requires_grad_(True)
        opt = torch.optim.Adam([X], lr=lr)
        best_L, best_X, loss_ctr = float("inf"), X.detach().clone(), 0
        bin_hist   = deque(maxlen=stable_patience)
        history    = []
        ramp_denom = max(1, n_steps - 1)
        t_start    = time.time()

        def masked_P(tau_):
            # window prob, with frozen walls forced to 0 (fully solid).
            return torch.sigmoid(X / tau_) * open_m

        for step in range(n_steps):
            opt.zero_grad()
            ramp = 0.2 + 0.8 * (step / ramp_denom)

            l_zone_val = 0.0
            for s in range(0, T, chunk):
                e      = min(s + chunk, T)
                P      = masked_P(tau)
                fac_in = facade_in(P).unsqueeze(0)             # (1,4,R,C), in-distribution float
                lux    = self._lux(cond_t[s:e], fac_in)
                l_zone = zone_penalty_sum(lux, tgt_t[s:e], w_t[s:e]) / n_occ
                if glare_weight:
                    # Normalised by the UNOCCUPIED count so the weight is scale-free.
                    l_zone = l_zone + glare_weight * (
                        outside_glare_sum(lux, tgt_t[s:e], glare_cap, w_t[s:e]) / n_unocc)
                l_zone.backward()
                l_zone_val += float(l_zone.item())

            P    = masked_P(tau)
            L_tv = tv_loss(P)
            (lambda_tv * ramp * L_tv).backward()
            opt.step()

            L_total = l_zone_val + lambda_tv * ramp * L_tv.item()
            if L_total < best_L:
                best_L, best_X, loss_ctr = L_total, X.detach().clone(), 0
            else:
                loss_ctr += 1
            if loss_ctr >= loss_patience:
                print(f"\n  Loss not improved for {loss_patience} steps — early stop at step {step+1}.")
                break

            with torch.no_grad():
                cur_bin = (masked_P(tau) > 0.5)
            drift = (int((cur_bin != bin_hist[0]).sum().item()) if bin_hist
                     else N_FACADES * FACADE_ROWS * FACADE_COLS)
            if len(bin_hist) == stable_patience and drift <= stable_tol:
                print(f"\n  Facade binary settled (<= {stable_tol} cells over {stable_patience} "
                      f"steps) — early stop at step {step+1}.")
                break
            bin_hist.append(cur_bin)

            with torch.no_grad():
                pct = (masked_P(tau) > 0.5).float().sum().item() / (N_FACADES*FACADE_ROWS*FACADE_COLS) * 100

            if progress_callback is not None:
                with torch.no_grad():
                    curr_np = masked_P(tau).detach().cpu().numpy()
                    best_np = ((torch.sigmoid(best_X / tau_binarize) * open_m) > 0.5).float().cpu().numpy()
                progress_callback(step, n_steps, l_zone_val, best_L, pct, curr_np, best_np)
            elif step % log_every == 0 or step == n_steps - 1:
                history.append(dict(step=step, L_zone=l_zone_val, L_tv=L_tv.item(),
                                    L_total=L_total, pct_window=pct, drift=drift))
                el   = time.time() - t_start
                frac = min((step + 1) / max(n_steps, 1), 1.0)
                print(f"  step {step+1:4d}/{n_steps} ({frac*100:4.1f}%)  L={L_total:.4f}  "
                      f"best={best_L:.4f}  windows={pct:4.1f}%  drift={drift}  "
                      f"{el:5.1f}s elapsed, ~{el/frac - el:4.0f}s left", flush=True)

        with torch.no_grad():
            P_final = (torch.sigmoid(best_X / tau_binarize) * open_m).cpu().numpy()
            binary  = (P_final > 0.5).astype(np.int32)
        print(f"  Best L {best_L:.5f}  |  glazing {binary.mean()*100:.1f}%  "
              f"|  sides open {[s for s, o in zip(SIDES, facade_open) if o]}")
        return binary, P_final, history

    @staticmethod
    def _upsample(X, out_rows, out_cols):
        """Bilinear-upsample the 4 facade logit grids (4, r, c) -> (4, out_rows, out_cols).
        Identity if already at that size.  Differentiable (grads flow to the coarse grid)."""
        if X.shape[-2] == out_rows and X.shape[-1] == out_cols:
            return X
        return F.interpolate(X.unsqueeze(0), size=(out_rows, out_cols),
                             mode="bilinear", align_corners=False).squeeze(0)

    def optimize_coarse_to_fine(self, cond, target_lux, weights=None,
                                facade_open=(True, True, True, True), *,
                                schedule=((2, 250), (4, 250), (8, 300), (16, 150)),
                                tau=1.0, tau_final=0.3, tau_binarize=0.05,
                                lambda_tv=0.02, lr=0.05, chunk=16,
                                glare_weight=DEFAULT_GLARE_WEIGHT, glare_cap=GLARE_LUX,
                                eval_every=10, bin_patience=12, callback_every=3,
                                log_every=25, progress_callback=None, verbose=True):
        """
        Coarse-to-fine (multigrid) 4-facade optimization, to escape the init-dependent
        local minima a direct optimization falls into.

        Each facade is parameterised by a LOW-resolution logit grid bilinearly upsampled
        to the surrogate's fixed 16x32 input.  `schedule` is a list of (facade_rows,
        n_steps) stages (cols = 2*rows, keeping the 16:32 facade aspect); each stage is
        warm-started from the previous stage's CONTINUOUS best grid.  Few DOF at low res
        => smooth loss landscape => the optimiser commits to the global "which walls,
        where" basin first, then adds detail.  Model forward cost per step is unchanged.

        The shipped design is BINARY, so the search is judged on the binarised facades:
          * every `eval_every` steps the HARD facade (X>0, masked to open walls) is scored
            with the exact deployed loss, and the globally best binary across ALL stages
            is kept.
          * the final stage anneals tau -> tau_final so the soft facade sharpens to binary.
          * ANY stage early-stops once its binary score plateaus for `bin_patience` evals.
        `facade_open` freezes unpicked walls to solid (their P forced 0) at every stage.
        Returns (binary (4,R,C), P_final (4,R,C), history) — same contract as optimize().
        """
        T      = cond.shape[0]
        cond_t = torch.tensor(cond,       dtype=torch.float32, device=self.device)
        tgt_t  = torch.tensor(target_lux, dtype=torch.float32, device=self.device)
        w_t    = (torch.ones(T, device=self.device) if weights is None
                  else torch.as_tensor(weights, dtype=torch.float32, device=self.device))
        n_occ   = (w_t.view(-1, 1, 1) * (tgt_t > 0).float()).sum().clamp(min=1.0)
        n_unocc = (w_t.view(-1, 1, 1) * (tgt_t <= 0).float()).sum().clamp(min=1.0)
        open_m  = self._open_mask(facade_open)

        def binarized_loss(Xfull):
            """Exact deployed loss of the HARD facades (Xfull>0, masked to open walls).

            This MUST include the same glare term as the training loss — the search keeps
            the globally best binarised design by this score, so scoring on a different
            objective than it optimises would quietly discard the designs the glare term
            was steering towards.
            """
            with torch.no_grad():
                hard   = (Xfull > 0).float() * open_m
                fac_in = facade_in(hard).unsqueeze(0)          # in-distribution float, not 0/1
                tot = glare = 0.0
                for s in range(0, T, chunk):
                    e = min(s + chunk, T)
                    lux = self._lux(cond_t[s:e], fac_in)
                    tot += float(zone_penalty_sum(lux, tgt_t[s:e], w_t[s:e]).item())
                    if glare_weight:
                        glare += float(outside_glare_sum(
                            lux, tgt_t[s:e], glare_cap, w_t[s:e]).item())
            return (tot / float(n_occ.item())
                    + glare_weight * glare / float(n_unocc.item()))

        history, global_step = [], 0
        total_steps = sum(n for _, n in schedule)
        # Progress logging is gated on ELAPSED STEPS SINCE THE LAST LINE, not on
        # `global_step % log_every`.  The old form was nested inside the `eval_every` check
        # below, so it needed step%eval_every and global_step%log_every to coincide — and a
        # stage ending early shifts the phase between those two counters, after which they
        # essentially never align again.  The symptom was one line at step 1 and silence for
        # the rest of the run.
        t_start, last_log = time.time(), -10 ** 9
        best_bin_L, best_bin_Xfull = float("inf"), None
        r0 = schedule[0][0]
        X = torch.randn(N_FACADES, r0, 2 * r0, device=self.device) * 0.1
        for si, (res_rows, n_steps) in enumerate(schedule):
            is_final = (si == len(schedule) - 1)
            X = self._upsample(X.detach(), res_rows, 2 * res_rows).clone().requires_grad_(True)
            opt = torch.optim.Adam([X], lr=lr)
            best_L_cont, best_X_cont = float("inf"), X.detach().clone()
            ramp_denom = max(1, n_steps - 1)
            stage_best_bin, stall = float("inf"), 0
            stage_early = False          # did this stage plateau, or did it spend its budget?
            step = -1
            for step in range(n_steps):
                opt.zero_grad()
                ramp    = 0.2 + 0.8 * (step / ramp_denom)
                cur_tau = (tau + (tau_final - tau) * (step / ramp_denom)) if is_final else tau

                l_zone_val = 0.0
                for s in range(0, T, chunk):
                    e      = min(s + chunk, T)
                    Xfull  = self._upsample(X, FACADE_ROWS, FACADE_COLS)
                    P      = torch.sigmoid(Xfull / cur_tau) * open_m
                    fac_in = facade_in(P).unsqueeze(0)         # in-distribution float, not 0/1
                    lux    = self._lux(cond_t[s:e], fac_in)
                    l_zone = zone_penalty_sum(lux, tgt_t[s:e], w_t[s:e]) / n_occ
                    if glare_weight:
                        l_zone = l_zone + glare_weight * (
                            outside_glare_sum(lux, tgt_t[s:e], glare_cap, w_t[s:e]) / n_unocc)
                    l_zone.backward()
                    l_zone_val += float(l_zone.item())
                Xfull = self._upsample(X, FACADE_ROWS, FACADE_COLS)
                L_tv  = tv_loss(torch.sigmoid(Xfull / cur_tau) * open_m)
                (lambda_tv * ramp * L_tv).backward()
                opt.step()

                L_total = l_zone_val + lambda_tv * ramp * L_tv.item()
                if L_total < best_L_cont:
                    best_L_cont, best_X_cont = L_total, X.detach().clone()

                if progress_callback is not None and global_step % callback_every == 0:
                    with torch.no_grad():
                        Xf_d    = self._upsample(X, FACADE_ROWS, FACADE_COLS)
                        curr_np = (torch.sigmoid(Xf_d / cur_tau) * open_m).cpu().numpy()
                        pct     = (((Xf_d > 0).float() * open_m).mean().item()) * 100
                        bb      = best_bin_Xfull if best_bin_Xfull is not None else Xf_d
                        best_np = ((bb > 0).float() * open_m).cpu().numpy()
                    progress_callback(global_step, total_steps, l_zone_val,
                                      best_bin_L, pct, curr_np, best_np)

                if step % eval_every == 0 or step == n_steps - 1:
                    with torch.no_grad():
                        Xf = self._upsample(X, FACADE_ROWS, FACADE_COLS)
                    bl = binarized_loss(Xf)
                    if bl < best_bin_L:
                        best_bin_L, best_bin_Xfull = bl, Xf.detach().clone()
                    if bl < stage_best_bin - 1e-4:
                        stage_best_bin, stall = bl, 0
                    else:
                        stall += 1
                    if verbose and (global_step - last_log) >= log_every:
                        last_log = global_step
                        pctw = (((Xf > 0).float() * open_m).mean().item()) * 100
                        el   = time.time() - t_start
                        frac = min((global_step + 1) / max(total_steps, 1), 1.0)
                        eta  = el / frac - el if frac > 0 else 0.0
                        print(f"  [c2f {res_rows:2d}x{2*res_rows:2d}] "
                              f"{global_step+1:4d}/{total_steps} ({frac*100:4.1f}%)  "
                              f"bin={bl:.4f}  best={best_bin_L:.4f}  windows={pctw:4.1f}%  "
                              f"{el:5.1f}s elapsed, ~{eta:4.0f}s left", flush=True)
                    if stall >= bin_patience:
                        if verbose:
                            print(f"  [c2f {res_rows:2d}x{2*res_rows:2d}] binary plateaued — "
                                  f"early stop at step {step+1} "
                                  f"({time.time() - t_start:.1f}s elapsed)", flush=True)
                        stage_early = True
                        global_step += 1
                        break
                global_step += 1
            X = best_X_cont
            # `early_stop` distinguishes "this stage converged" from "this stage ran out of
            # budget while still improving" — the same fact the verbose line prints, but
            # available to callers.  Purely additive: every caller in the repo discards
            # `history`, and the live demo reads it to tell the user whether the design on
            # screen is settled or merely unfinished.
            history.append(dict(res=res_rows, stage_best_bin=stage_best_bin,
                                early_stop=stage_early, steps=step + 1))

        with torch.no_grad():
            P_final = (torch.sigmoid(best_bin_Xfull / tau_binarize) * open_m).cpu().numpy()
            binary  = ((best_bin_Xfull > 0).float() * open_m).cpu().numpy().astype(np.int32)
        print(f"  [c2f] best BINARISED L {best_bin_L:.5f}  |  glazing {binary.mean()*100:.1f}%  "
              f"|  sides open {[s for s, o in zip(SIDES, facade_open) if o]}  "
              f"|  {time.time() - t_start:.1f}s total", flush=True)
        return binary, P_final, history

    def predict_lux(self, binary_facades, cond, chunk=64):
        """Plan-view predicted lux (T,32,32) for a fixed 4-facade design over all conditions.
        binary_facades: (4,R,C) with 1=window."""
        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
        fac_in = torch.tensor(facade_in(binary_facades.astype(np.float32)),
                              device=self.device).unsqueeze(0)     # (1,4,R,C) in-distribution float
        outs = []
        with torch.no_grad():
            for s in range(0, cond.shape[0], chunk):
                outs.append(self._lux(cond_t[s:s+chunk], fac_in).cpu().numpy())
        return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Outputs / geometry  (shared by the interactive designer and the headless job runner)
# ---------------------------------------------------------------------------

def save_facades_csv(binary_facades, path):
    """All 4 facades in ONE wide CSV (training-dataset column layout): a header
    {S,N,E,W}_Pix_0..511 plus a single data row.  Each facade is flattened row-major
    (k = row*FACADE_COLS + col, row 0 = ceiling), facade order S,N,E,W — identical to how
    generate_dataset_allsides_climate_pit.py writes {side}_Pix_k.  Values are binary,
    1 = WINDOW (the optimizer/design convention; inverse of the model-input <0.5 = window —
    see the encoding table in this module's docstring).

    NOTE the side ids are ROOM-LOCAL: N_Pix_* is always the y=D wall regardless of
    north_angle.  A reader needs the job's sides_compass mapping to know which way that
    wall actually faces.
    """
    cols, vals = [], []
    for i, side in enumerate(SIDES):
        cols += [f"{side}_Pix_{k}" for k in range(binary_facades[i].size)]
        vals += [str(int(v)) for v in binary_facades[i].flatten()]
    Path(path).write_text(",".join(cols) + "\n" + ",".join(vals) + "\n")


def wall_azimuths(north_angle=0.0):
    """{side: true compass azimuth (deg clockwise from true north) the wall's OUTWARD face
    points at}, for a building rotated by north_angle.

    The room's +Y axis has azimuth -north_angle, and +X is 90 deg clockwise from +Y.
    Outward normals are N->+Y, E->+X, S->-Y, W->-X.  At north_angle=0 this returns the
    identity mapping S->180, N->0, E->90, W->270.
    """
    base = -float(north_angle)
    return {'N': base           % 360.0,
            'E': (base +  90.0) % 360.0,
            'S': (base + 180.0) % 360.0,
            'W': (base + 270.0) % 360.0}


def facade_panel_boxes(width, depth, height):
    """Yield (side, row, col, pix_index, (x0,y0,z0), (x1,y1,z1), (nx,ny,nz)) for every one of
    the N_FACADES*FACADE_ROWS*FACADE_COLS panels, in ROOM-LOCAL metres, grouped side-outer in
    SIDES order so the output lines up with save_facades_csv's column order.

    Every panel is a planar axis-aligned rectangle, so the min/max corner pair fully
    determines it.  Normals are OUTWARD (the Radiance builder lists inward ones).

    PORTED VERBATIM from the Radiance scene builder in
    generate_dataset_allsides_climate_pit.py (the S/N/E/W polygon block) — that is the ground
    truth for where a facade cell physically sits, and the two must stay in sync:
        row 0 = top (high Z), step H/FACADE_ROWS downward
        S (y=0) / N (y=D): col -> X, step W/FACADE_COLS, col 0 at x=0
        E (x=W) / W (x=0): col -> Y, step D/FACADE_COLS, col 0 at y=0
    """
    W, D, H = float(width), float(depth), float(height)
    cw_sn = W / FACADE_COLS      # S/N column width along X
    cw_ew = D / FACADE_COLS      # E/W column width along Y
    ch    = H / FACADE_ROWS      # row height along Z
    normals = {'S': (0.0, -1.0, 0.0), 'N': (0.0, 1.0, 0.0),
               'E': (1.0, 0.0, 0.0),  'W': (-1.0, 0.0, 0.0)}
    for side in SIDES:
        for row in range(FACADE_ROWS):
            z0 = (FACADE_ROWS - 1 - row) * ch
            z1 = z0 + ch
            for col in range(FACADE_COLS):
                idx = row * FACADE_COLS + col
                if side in ('S', 'N'):
                    a0, a1 = col * cw_sn, (col + 1) * cw_sn
                    y      = 0.0 if side == 'S' else D
                    p0, p1 = (a0, y, z0), (a1, y, z1)
                else:
                    b0, b1 = col * cw_ew, (col + 1) * cw_ew
                    x      = W if side == 'E' else 0.0
                    p0, p1 = (x, b0, z0), (x, b1, z1)
                yield side, row, col, idx, p0, p1, normals[side]


# ---------------------------------------------------------------------------
# Headless CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="All-sides annual climate + geometry facade optimizer (headless).")
    ap.add_argument("--city", type=int, default=None, help="index into the city list")
    ap.add_argument("--hour", type=int, default=12, help="hour slot 0-23 (default 12)")
    ap.add_argument("--width",  type=float, default=DEFAULT_WIDTH,  help=f"room width  X, m ({W_MIN}-{W_MAX})")
    ap.add_argument("--depth",  type=float, default=DEFAULT_DEPTH,  help=f"room depth  Y, m ({D_MIN}-{D_MAX})")
    ap.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help=f"room height Z, m ({H_MIN}-{H_MAX})")
    ap.add_argument("--sides", type=str, default="SNEW",
                    help="which walls may hold windows, e.g. SNEW, SE, S (rest are solid)")
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET_LUX)
    ap.add_argument("--repr", type=int, default=N_REPR, help="representative conditions (0 = all days)")
    ap.add_argument("--steps", type=int, default=800, help="direct-mode steps (ignored in coarse-to-fine)")
    ap.add_argument("--direct", action="store_true",
                    help="use the single-stage direct optimizer instead of coarse-to-fine (default)")
    args = ap.parse_args()

    cities = discover_cities()
    if args.city is None:
        print(f"{len(cities)} cities available. Pass --city N. Showing first 10:")
        for i, c in enumerate(cities[:10]):
            print(f"  [{i:3d}] {city_label(c)}")
        raise SystemExit(0)

    W, D, H = args.width, args.depth, args.height
    facade_open = tuple(s in args.sides.upper() for s in SIDES)
    if not any(facade_open):
        raise SystemExit("At least one wall must be open (--sides). e.g. --sides SNEW")

    label, lat, cond, days = annual_conditions(cities[args.city], args.hour)
    print(f"City: {label}  (lat {lat:.1f})  |  hour {args.hour}:00  |  "
          f"room {W:.1f}x{D:.1f}x{H:.1f} m  |  open {[s for s,o in zip(SIDES,facade_open) if o]}")
    print(f"Days above horizon at this hour: {len(days)} / 365")
    if len(days) == 0:
        raise SystemExit("Sun never above horizon at that hour for this city.")

    if args.repr and len(cond) > args.repr:
        cond_opt, weights = reduce_conditions(cond, n_clusters=args.repr)
        print(f"Reduced {len(cond)} days -> {len(cond_opt)} weighted conditions")
    else:
        cond_opt, weights = cond, None
    cond_opt  = append_room(cond_opt, W, D, H)
    cond_full = append_room(cond,     W, D, H)

    zone = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    zone[12:20, 10:22] = args.target
    target = np.broadcast_to(zone, (len(cond_opt), GRID_N, GRID_N)).copy()

    opt = FacadeOptimizerAllSidesClimate()
    if args.direct:
        binary, _, _ = opt.optimize(cond_opt, target, weights, facade_open, n_steps=args.steps)
    else:
        binary, _, _ = opt.optimize_coarse_to_fine(cond_opt, target, weights, facade_open)

    E = opt.predict_lux(binary, cond_full)
    zn = E[:, 12:20, 10:22]
    lo = args.target * ZONE_MIN_RATIO
    print(f"\n  Annual zone mean lux : {zn.mean():.0f}")
    print(f"  Underlit (<{lo:.0f}) : {(zn < lo).mean()*100:.1f}% of zone-days")
    print(f"  Glare (>3000)        : {(E > GLARE_LUX).mean()*100:.1f}% of all cell-days")

    out_dir = REPO_ROOT / "allside_Generalised model" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, side in enumerate(SIDES):
        out = out_dir / f"facade_allsides_climate_{label[:20]}_{side}.csv"
        np.savetxt(out, binary[i], delimiter=",", fmt="%d")
    print(f"  Facades saved: {out_dir}\\facade_allsides_climate_{label[:20]}_{{S,N,E,W}}.csv")
