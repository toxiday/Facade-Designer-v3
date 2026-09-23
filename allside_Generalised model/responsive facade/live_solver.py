"""
live_solver.py — the engine behind the responsive-facade demo.

WHAT THIS IS FOR.  The rest of this project designs a facade ONCE, for a whole year, from a
brief of scheduled occupancy zones (see `model trainer/run_facade_job.py`).  This module
inverts that: the occupancy is LIVE and the facade is re-solved as it changes.  The premise
is a building whose facade is a grid of operable panels, and a sensor (a camera, in the real
system) that reports where people are and what they are doing.  Activity implies a required
illuminance; the facade opens exactly the panels that deliver it, and closes them again when
the person moves on.

Only the sensing is faked here.  The occupants are dragged around by hand, but everything
downstream — the sun vector, the surrogate, the optimizer, the lux prediction — is the real
pipeline the annual designer uses.

WHY IT IS FAST ENOUGH TO BE LIVE.  The annual job solves against ~48 representative climate
conditions clustered out of 365 days, which is what makes it a multi-minute job.  A live
facade only ever has to answer for the sun that is in the sky RIGHT NOW: T = 1.  Measured on
an RTX 5070 Ti:

    forward pass (T=1)                        8.4 ms
    cold solve, SHORT_SCHEDULE                 2.6 s
    warm refine after an occupant moves        1.2 s

Multiple occupants are free.  Everyone is painted into ONE 32x32 target grid, so five people
with five different target levels cost exactly what one person costs — the batch dimension is
the number of sun conditions, not the number of people.

COLD vs WARM.  Anything that changes the room or the sun (W/D/H, city, date, hour,
north_angle, which walls may open) invalidates the whole search and forces a cold
coarse-to-fine solve.  Anything that only moves people or retargets them is a warm refine
from the previous logits — which is both faster AND gives the facade visual continuity: it
morphs by ~30 cells instead of jumping to an unrelated design.

NO GUI HERE.  This module never imports matplotlib, so it can be tested headlessly and so the
app can choose its own backend.  See `responsive_facade.py` for the interface.

CONVENTIONS (identical to the rest of the project — see optimize_facade_allsides_climate.py):
    room-local wall ids   S = y=0   N = y=D   E = x=W   W = x=0
    floor grid            row 0 = the +Y (N) edge, col 0 = the x=0 (W) edge
    facade grid           row 0 = ceiling, 16 rows x 32 cols, 1 = window
    north_angle           Ladybug: clockwise angle from room +Y to true North
"""

import sys
import math
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, replace

import numpy as np
import torch

_TRAINER = Path(__file__).resolve().parents[1] / "model trainer"
if str(_TRAINER) not in sys.path:
    sys.path.insert(0, str(_TRAINER))

from optimize_facade_allsides_climate import (            # noqa: E402
    FacadeOptimizerAllSidesClimate, facade_in, solar_position,
    zone_penalty_sum, outside_glare_sum, tv_loss,
    discover_cities, city_label, resolve_city, city_location,
    normalize_room, append_room,
    SIDES, GRID_N, MIN_ALT, GLARE_LUX,
    DNI_SCALE, DHI_SCALE, ZONE_MIN_RATIO, ZONE_MAX_RATIO,
    W_MIN, W_MAX, D_MIN, D_MAX, H_MIN, H_MAX,
    N_FACADES, FACADE_ROWS, FACADE_COLS)

__all__ = [
    "Occupant", "Scene", "SolveResult", "FacadePreview", "LiveSolver", "SolverWorker",
    "single_condition", "occupant_owner_grid", "target_grid", "zone_stats",
    "ACTIVITIES", "SHORT_SCHEDULE", "REFINE_STEPS", "PREVIEW_COOLDOWN",
]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Cold-solve schedule.  The annual runner's default is ((2,250),(4,250),(8,300),(16,150)) =
# ~15 s at T=1.  Measured against it on a 3-occupant scene, this 130-step schedule costs 2.6 s
# and still lands every occupant inside its comfort band; the extra 12 s buys a lower loss
# (0.0004 vs 0.0025) that is invisible on the floor.  Interactivity is worth more here.
SHORT_SCHEDULE = ((2, 30), (4, 30), (8, 40), (16, 30))

REFINE_STEPS   = 60        # warm-start refine at full 16x32 resolution
GLARE_WEIGHT   = 0.25      # same as the Rhino job runner

# Minimum seconds between two intermediate facades being pushed to the screen mid-solve.
# The optimizer improves on its best binary dozens of times per solve; painting every one makes
# the facade flicker and buries the fact that it is converging.  Held on the LiveSolver, not
# per-call, so a run of fast warm refines cannot exceed this rate either.  FINAL results ignore
# the cooldown entirely — they are the answer, not a progress report.
PREVIEW_COOLDOWN = 0.5
# Steps between optimizer callbacks.  The optimizer builds the preview arrays (two GPU->CPU
# copies, a sync point) BEFORE handing them over, so the cost is paid whether or not the
# cooldown lets the push through.  10 gives ~13 chances across a 130-step cold solve — plenty to
# feed a 2 Hz cooldown — at a third of the default's sync overhead.
PREVIEW_EVERY = 10

# When the warm refine counts as having settled.  It evaluates the binarised design every 10
# steps out of 60, so 2 consecutive flat evaluations means 20 steps produced nothing — a real
# plateau at this budget, not a single unlucky evaluation.  The tolerance matches the
# `bin_patience` bar the coarse-to-fine optimizer uses for the same judgement.
REFINE_STALL_PATIENCE = 2
REFINE_STALL_TOL      = 1e-4
# Matches the bar optimize_coarse_to_fine uses internally (`bl < stage_best_bin - 1e-4`), so
# "improved" means the same thing on both sides of the call.
STAGE_IMPROVE_TOL     = 1e-4

# Largest |logit| a warm start is allowed to carry in.  A warm start must be CONFIDENT enough
# to reproduce the previous facade exactly and UNCERTAIN enough to still have a gradient.
# Seeding at +-4 against tau=0.6 puts sigmoid at 0.9987, where its derivative is ~0.0013 —
# the descent stalls and the facade simply refuses to react to the occupant moving (measured:
# 0 of 2048 panels changed, the occupant left in 377 lux against a 500 ask).  Capping at 2.5
# leaves sigmoid(2.5/0.6) = 0.985, a ~12x live gradient, and since rescaling is positive it
# cannot flip a single cell — the warm start still reproduces the previous design exactly.
WARM_LOGIT_CAP = 2.5
DEFAULT_RADIUS = 1.0       # m — the floor patch one person's activity occupies

# Activity -> required illuminance.  Rounded from EN 12464-1 / IESNA task levels; these are
# the labels the "camera" would be emitting in the real system.
ACTIVITIES = [
    ("Sleeping",      100.0),
    ("Relaxing / TV", 150.0),
    ("Dining",        300.0),
    ("Cooking",       500.0),
    ("Reading",       500.0),
    ("Desk work",     750.0),
    ("Drafting",     1000.0),
]

TARGET_LUX_MIN, TARGET_LUX_MAX = 50.0, 2000.0
RADIUS_MIN,     RADIUS_MAX     = 0.3, 3.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass
class Occupant:
    """One person on the floor.  `x`/`y` are metres in the ROOM-LOCAL frame: x from the W
    wall (x=0) toward E (x=W), y from the S wall (y=0) toward N (y=D)."""
    oid:        int
    x:          float
    y:          float
    target_lux: float = 500.0
    radius_m:   float = DEFAULT_RADIUS
    activity:   str   = "Reading"

    def clamped(self, width, depth):
        """Keep the occupant inside the room, allowing for its own footprint."""
        r = min(self.radius_m, width / 2.0, depth / 2.0)
        return replace(self,
                       x=float(np.clip(self.x, r, max(r, width  - r))),
                       y=float(np.clip(self.y, r, max(r, depth  - r))))


@dataclass(frozen=True)
class Scene:
    """An immutable snapshot of everything the solver needs.  The app hands one of these to
    the worker; because it is frozen, the UI may keep mutating its own copy while a solve for
    an older snapshot is still in flight."""
    city_path:   str
    month:       int
    day:         int
    hour:        int
    north_angle: float
    width:       float
    depth:       float
    height:      float
    sides_open:  tuple           # 4 bools in SIDES order (S, N, E, W)
    occupants:   tuple           # tuple of Occupant
    seq:         int = 0         # bumped by the app; echoed back so stale results are dropped

    def structural_key(self):
        """Everything that changes the SEARCH PROBLEM rather than just its target.

        A change here means the sun vector or the room geometry the surrogate sees is
        different, so the previous logits describe a facade for a different building and are
        worthless as a warm start.  Moving a person does NOT appear here — the physics is
        unchanged, only where the light has to land, which is exactly the case a warm refine
        handles well.
        """
        return (self.city_path, self.month, self.day, self.hour, self.north_angle,
                round(self.width, 3), round(self.depth, 3), round(self.height, 3),
                tuple(self.sides_open))


@dataclass
class FacadePreview:
    """An intermediate facade, pushed mid-solve when the optimizer improves on its best.

    Deliberately NOT a SolveResult: it carries no lux and no per-occupant stats, because
    computing them would cost a forward pass on the solver thread for a design that is about to
    be superseded.  The app re-predicts the map from `binary` on its own thread instead.
    """
    seq:    int
    scene:  "Scene"               # what it is being designed for — Scene is frozen, so sharing
                                  # the reference across threads is safe
    binary: np.ndarray            # (4, 16, 32) int, 1 = window
    best_L: float                 # binarised loss of this design — monotonically falling
    mode:   str = "cold"


@dataclass
class SolveResult:
    seq:        int
    scene:      Scene
    binary:     np.ndarray = None       # (4, 16, 32) int, 1 = window
    lux:        np.ndarray = None       # (32, 32) float, predicted floor illuminance
    owner:      np.ndarray = None       # (32, 32) int, occupant index per cell, -1 = empty
    stats:      list  = field(default_factory=list)   # per-occupant dicts, see zone_stats()
    sun:        dict  = field(default_factory=dict)   # alt / az / dni / dhi / ok
    mode:       str   = "cold"          # "cold" | "warm" | "dark"
    seconds:    float = 0.0
    glazing:    float = 0.0
    message:    str   = ""
    # True when the optimizer stopped because its best design plateaued rather than because it
    # ran out of steps — i.e. this is a local minimum and more of the same search will not
    # improve it.  None for the dark paths, where nothing was optimized.
    converged:  bool  = None
    stop_reason: str  = ""


# ---------------------------------------------------------------------------
# One sun, one moment
# ---------------------------------------------------------------------------

_WEA_CACHE = {}


def _wea_rows(wea_path):
    """(lat, lon_east, tz_h, {(month, day, int(hour)): (file_hour, dni, dhi)}) — parsed once.

    annual_conditions() walks the whole 8760-row file for every call, which is the right
    trade when you want all 365 days of one hour.  Here the app re-asks for a single
    (month, day, hour) on every slider drag, so the file is indexed once and kept.

    THE FILE'S OWN HOUR IS KEPT, not just the integer it is filed under.  TMYx .wea rows are
    stamped at the MIDDLE of the hour (12.5, not 12.0) and annual_conditions feeds that value
    straight to solar_position while bucketing on int(hour).  Half an hour is 7.5 degrees of
    azimuth, which showed up as a 0.13 discrepancy in the sun vector when this index rounded
    it away — a real and entirely invisible error, since the facade it produces looks
    perfectly plausible, just designed for the wrong half-hour.
    """
    key = str(wea_path)
    hit = _WEA_CACHE.get(key)
    if hit is not None:
        return hit
    lat, lon_w, tz_deg, rows = None, None, None, {}
    with open(wea_path) as f:
        for line in f:
            p = line.split()
            if   len(p) == 2 and p[0] == "latitude":  lat    = float(p[1])
            elif len(p) == 2 and p[0] == "longitude": lon_w  = float(p[1])
            elif len(p) == 2 and p[0] == "time_zone": tz_deg = float(p[1])
            elif len(p) == 5:
                h = float(p[2])
                rows[(int(p[0]), int(p[1]), int(h))] = (h, float(p[3]), float(p[4]))
    out = (lat, -lon_w, -tz_deg / 15.0, rows)
    _WEA_CACHE[key] = out
    return out


def single_condition(wea_path, month, day, hour, north_angle=0.0):
    """(cond5 (1,5) float32 or None, info dict) for ONE moment in the TMYx year.

    Built to be bit-comparable with annual_conditions(): the same solar_position(), the same
    DNI/DHI scaling, the same sun-into-room-frame rotation.  The difference is only that it
    picks one row instead of 365.

    `info["ok"]` is False when the sun is at or below MIN_ALT (5 deg).  The surrogate was
    trained only on hours with alt > 5, in BOTH the dataset generator and the optimizer, so
    below that there is no honest prediction to make — the caller should say so rather than
    feed an out-of-distribution sun vector and draw whatever comes back.
    """
    lat, lon_east, tz_h, rows = _wea_rows(wea_path)
    key = (int(month), int(day), int(hour))
    # Fall back to the integer hour only when the file has no such row at all (a clipped
    # .wea); otherwise always use the file's own mid-hour stamp — see _wea_rows().
    file_hour, dni, dhi = rows.get(key, (float(hour), 0.0, 0.0))
    alt, az, (sx, sy, sz) = solar_position(lat, lon_east, tz_h, int(month), int(day), file_hour)
    info = dict(alt=alt, az=az, dni=dni, dhi=dhi, lat=lat, lon=lon_east,
                hour=file_hour, ok=(alt > MIN_ALT), has_row=(key in rows))
    if not info["ok"]:
        return None, info
    if north_angle:
        th = math.radians(north_angle)
        ct, st = math.cos(th), math.sin(th)
        sx, sy = sx * ct + sy * st, -sx * st + sy * ct
    cond5 = np.array([[sx, sy, sz, dni / DNI_SCALE, dhi / DHI_SCALE]], dtype=np.float32)
    return cond5, info


def days_in_month(month):
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][int(month) - 1]


# ---------------------------------------------------------------------------
# Occupants -> floor target grid
# ---------------------------------------------------------------------------

def cell_centres(width, depth):
    """(X, Y) each (32,32) — the room-local metre coordinates of every floor cell centre,
    under the project convention row 0 = +Y edge, col 0 = x=0 edge."""
    cols = (np.arange(GRID_N) + 0.5) * (width / GRID_N)
    rows = (GRID_N - 0.5 - np.arange(GRID_N)) * (depth / GRID_N)
    return np.meshgrid(cols, rows)


def occupant_owner_grid(occupants, width, depth):
    """(32,32) int — which occupant each floor cell belongs to, -1 for unoccupied floor.

    A cell belongs to an occupant when its CENTRE falls inside that occupant's radius, which
    is the same binary centre-sampling rule the Rhino zone rasterizer uses, so a zone painted
    here and a zone painted in Grasshopper cover the same cells.

    WHERE TWO PEOPLE OVERLAP the NEAREST one wins, rather than the brighter target winning by
    np.maximum the way scheduled zones merge in build_slots().  Merging is right for one
    person doing two things at different hours; it is wrong for two people standing next to
    each other at the same instant, where taking the max would silently hold someone who
    asked for 150 lux at the 1000 lux their neighbour needs.
    """
    X, Y = cell_centres(width, depth)
    owner = np.full((GRID_N, GRID_N), -1, dtype=np.int32)
    best  = np.full((GRID_N, GRID_N), np.inf, dtype=np.float32)
    for i, occ in enumerate(occupants):
        d = np.hypot(X - occ.x, Y - occ.y)
        take = (d <= occ.radius_m) & (d < best)
        owner[take] = i
        best[take]  = d[take]
    return owner


def target_grid(occupants, width, depth, owner=None):
    """(1,32,32) float32 target illuminance per floor cell; 0 = unoccupied."""
    if owner is None:
        owner = occupant_owner_grid(occupants, width, depth)
    tgt = np.zeros((1, GRID_N, GRID_N), dtype=np.float32)
    for i, occ in enumerate(occupants):
        tgt[0][owner == i] = float(occ.target_lux)
    return tgt


def zone_stats(occupants, owner, lux):
    """Per-occupant achieved-vs-asked report.

    `in_band` uses the optimizer's own comfort band (ZONE_MIN_RATIO..ZONE_MAX_RATIO of the
    target), so the number shown in the app is the objective the optimizer was actually
    scored on, not a prettier one invented for the display.
    """
    out = []
    for i, occ in enumerate(occupants):
        m = (owner == i)
        if not m.any() or lux is None:
            out.append(dict(oid=occ.oid, cells=int(m.sum()), mean=float("nan"),
                            lo=float("nan"), hi=float("nan"), in_band=float("nan"),
                            target=occ.target_lux))
            continue
        v = lux[m]
        out.append(dict(
            oid=occ.oid, cells=int(m.sum()), mean=float(v.mean()),
            lo=float(v.min()), hi=float(v.max()),
            in_band=float(np.mean((v >= occ.target_lux * ZONE_MIN_RATIO) &
                                  (v <= occ.target_lux * ZONE_MAX_RATIO)) * 100.0),
            target=float(occ.target_lux)))
    return out


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class LiveSolver:
    """A thin live-use layer over FacadeOptimizerAllSidesClimate.

    Holds the previous solution's LOGITS (not just its binary) so a warm refine can continue
    the same descent.  Reconstructing logits from a binary would throw away the confidence
    the optimizer had in each cell and restart it on a plateau.
    """

    def __init__(self, weights_path=None, device=None, verbose=True):
        kw = {} if weights_path is None else dict(weights_path=weights_path)
        self.opt   = FacadeOptimizerAllSidesClimate(device=device, **kw)
        self._X    = None            # (4,16,32) logits from the last solve
        self._key  = None            # structural_key() those logits belong to
        self.verbose = verbose
        # The rate limit is GLOBAL (a burst of fast warm refines must not exceed it either),
        # but the improvement bar is PER SOLVE: losses from different occupant layouts are not
        # comparable, so carrying the bar across solves would mute a new scene whose honest
        # best simply starts higher.
        self._last_push = 0.0        # wall clock of the last preview let through
        self._push_best = float("inf")

    # ---- internals --------------------------------------------------------
    def _open_mask(self, sides_open):
        return self.opt._open_mask(tuple(bool(s) for s in sides_open))

    @staticmethod
    def _c2f_converged(hist):
        """Did the coarse-to-fine search run out of improvements, or out of budget?

        `bin_patience` is NOT the signal here.  It counts 12 flat EVALUATIONS, and at
        eval_every=10 against SHORT_SCHEDULE's 30-40 step stages there are only 3-4 evaluations
        per stage — so a stage early-stop is unreachable and the flag reads False every time
        (measured: 0 of 12 solves).  Lowering the patience to make it fire costs quality
        (measured: 93.4% in-band at patience 2 against 95.2% at the default), which is a bad
        trade for a nicer-looking indicator.

        The honest signal is already in the history: whether the FINEST stage beat the best any
        earlier stage had found.  If 16x32 cannot improve on 8x16, more of this search is not
        going to help.  Measured across 16 solves it fires 44% of the time and it is
        predictive — settled solves averaged 90.7% in-band against 95.3% for still-improving
        ones, i.e. it flags exactly the runs where starting over is worth it.
        """
        if not hist:
            return None
        if hist[-1].get("early_stop"):
            return True                       # a genuine plateau, if the budget ever allows one
        bests = [h["stage_best_bin"] for h in hist]
        if len(bests) < 2:
            return False
        return bool(bests[-1] >= min(bests[:-1]) - STAGE_IMPROVE_TOL)

    def _push_gate(self, best_L, binary):
        """True if this improved binary should go to the screen now.

        Two conditions, and both matter.  IMPROVED: the optimizer's callback fires on a step
        schedule, not on progress, so most calls carry the same best design as the last one and
        would be a repaint that shows nothing.  COOLED DOWN: early in a solve the best improves
        many times per second, and painting each one reads as flicker rather than convergence.
        """
        if best_L is None or not np.isfinite(best_L) or binary is None:
            return False                      # before the first eval there is no best yet
        if best_L >= self._push_best:
            return False
        now = time.time()
        if now - self._last_push < PREVIEW_COOLDOWN:
            return False                      # keep _push_best: re-offer it after the cooldown
        self._last_push, self._push_best = now, best_L
        return True

    def _refine(self, X0, cond, tgt, sides_open, n_steps=REFINE_STEPS,
                lr=0.05, tau=0.6, tau_final=0.3, lambda_tv=0.02,
                glare_weight=GLARE_WEIGHT, on_improve=None):
        """Single-stage full-resolution descent warm-started from `X0`.

        No coarse stages: the warm start has already committed to a basin ("windows high on
        the east wall"), and re-running the 2x4 stage would discard that and re-decide it from
        scratch — which is the whole cost the warm path exists to avoid.

        tau anneals 0.6 -> 0.3 rather than the cold path's 1.0 -> 0.3, because the incoming
        logits are already near-binary and re-softening them would undo the previous solve
        before improving on it.
        """
        dev     = self.opt.device
        cond_t  = torch.tensor(cond, dtype=torch.float32, device=dev)
        tgt_t   = torch.tensor(tgt,  dtype=torch.float32, device=dev)
        open_m  = self._open_mask(sides_open)
        n_occ   = (tgt_t > 0).float().sum().clamp(min=1.0)
        n_unocc = (tgt_t <= 0).float().sum().clamp(min=1.0)

        # Rescale into the live-gradient band (see WARM_LOGIT_CAP).  Division by a positive
        # scalar preserves every sign, so the incoming facade is bit-identical; what it
        # preserves beyond that is the RELATIVE confidence per cell, which is the whole reason
        # the logits are kept between solves instead of just the binary.  Without this the
        # magnitudes also ratchet upward over a chain of refines and the facade slowly freezes.
        X0 = X0.detach().to(dev)
        peak = float(X0.abs().max())
        if peak > WARM_LOGIT_CAP:
            X0 = X0 * (WARM_LOGIT_CAP / peak)

        X = X0.clone().requires_grad_(True)
        o = torch.optim.Adam([X], lr=lr)
        best_L, best_X = float("inf"), X.detach().clone()
        stall = 0            # consecutive evaluations with no improvement

        def binarised_loss(Xd):
            with torch.no_grad():
                hard = (Xd > 0).float() * open_m
                lx   = self.opt._lux(cond_t, facade_in(hard).unsqueeze(0))
                return float((zone_penalty_sum(lx, tgt_t) / n_occ
                              + glare_weight * outside_glare_sum(
                                  lx, tgt_t, GLARE_LUX) / n_unocc).item())

        for s in range(n_steps):
            o.zero_grad()
            cur_tau = tau + (tau_final - tau) * (s / max(1, n_steps - 1))
            P   = torch.sigmoid(X / cur_tau) * open_m
            lux = self.opt._lux(cond_t, facade_in(P).unsqueeze(0))
            L = (zone_penalty_sum(lux, tgt_t) / n_occ
                 + glare_weight * outside_glare_sum(lux, tgt_t, GLARE_LUX) / n_unocc
                 + lambda_tv * tv_loss(P))
            L.backward()
            o.step()
            # Judge on the BINARISED design, like optimize_coarse_to_fine does: the shipped
            # facade is binary, so a continuous loss that improves while the binary gets worse
            # is not an improvement.
            if s % 10 == 0 or s == n_steps - 1:
                bl = binarised_loss(X.detach())
                if bl >= best_L - REFINE_STALL_TOL:
                    stall += 1
                else:
                    stall = 0
                if bl < best_L:
                    best_L, best_X = bl, X.detach().clone()
                    # The warm path needs its own progress hook — optimize_coarse_to_fine has
                    # progress_callback but _refine is hand-rolled, and without this a cold
                    # solve would animate while every warm refine snapped, which reads as a bug
                    # rather than as two different code paths.
                    if on_improve is not None and s < n_steps - 1:
                        b = ((best_X > 0).float() * open_m).cpu().numpy().astype(np.int32)
                        if self._push_gate(best_L, b):
                            on_improve(b, best_L)

        binary = ((best_X > 0).float() * open_m).cpu().numpy().astype(np.int32)
        # DETECTION ONLY — the refine still runs its full budget.  Exiting here would save
        # ~0.5 s of a 1.1 s refine, but it would also change which design gets shipped, and the
        # ask was for an indicator, not a speed-up.  Reported so the app can say whether the
        # facade on screen is settled.
        return binary, best_X, best_L, stall >= REFINE_STALL_PATIENCE

    # ---- public -----------------------------------------------------------
    def invalidate(self):
        """Force the next solve to be cold."""
        self._X, self._key = None, None

    # ---- shared by solve() and predict() ----------------------------------
    @staticmethod
    def no_daylight(scene):
        """(cond5, sun, note) — the two conditions under which there is no daylight to speak
        of, resolved once for BOTH the solve and the predict path.

        `note` is None when daylight is possible; otherwise it says why not, and the caller
        should show a zero lux field and a closed facade rather than asking the surrogate.
        Factored out because the live map and the optimizer must never disagree about whether
        it is dark — if they did, the floor plan would show light for an hour the facade
        refuses to design for.
        """
        cond5, sun = single_condition(scene.city_path, scene.month, scene.day,
                                      scene.hour, scene.north_angle)
        if cond5 is None:
            # Below MIN_ALT.  Not an error and not something to paper over: the surrogate has
            # never seen a sun this low.  Same limitation the annual runner records in its
            # result.json warnings.
            return None, sun, (f"sun {sun['alt']:.1f}deg — below the {MIN_ALT:.0f}deg training "
                               f"floor, no daylight design possible at this hour")
        if not any(scene.sides_open):
            return None, sun, "every wall is frozen solid — open at least one"
        return cond5, sun, None

    def predict(self, scene, binary):
        """(lux (32,32) float32, sun dict, note str|None) for a KNOWN facade.

        One forward pass — ~8 ms idle, ~26 ms while a solve is saturating the GPU — versus the
        1.1-2.6 s it takes to design a new facade.  That gap is why this exists: the app can
        answer "what light is on the floor right now" continuously while the optimizer is still
        deciding what the facade should become.

        Deliberately does NOT touch self._X / self._key.  Those are the optimizer's warm-start
        state; a predict is a read of the world, and letting it perturb the search would make
        moving a slider change the facade the next refine starts from.
        """
        cond5, sun, note = self.no_daylight(scene)
        if note is not None:
            return np.zeros((GRID_N, GRID_N), dtype=np.float32), sun, note
        cond = append_room(cond5, scene.width, scene.depth, scene.height)
        return self.opt.predict_lux(np.asarray(binary), cond)[0], sun, None

    def solve(self, scene, force_cold=False, on_preview=None):
        """Solve `scene` and return a SolveResult.  Blocking; run it off the UI thread.

        `on_preview(FacadePreview)` is called from THIS thread each time the optimizer improves
        on its best binary and the cooldown allows — see PREVIEW_COOLDOWN.  It must be cheap and
        must not block: anything slow here stalls the optimizer itself.
        """
        t0    = time.time()
        occs  = list(scene.occupants)
        owner = occupant_owner_grid(occs, scene.width, scene.depth)
        res   = SolveResult(seq=scene.seq, scene=scene, owner=owner)

        cond5, sun, note = self.no_daylight(scene)
        res.sun = sun
        if note is not None:
            res.mode    = "dark"
            res.binary  = np.zeros((N_FACADES, FACADE_ROWS, FACADE_COLS), dtype=np.int32)
            res.lux     = np.zeros((GRID_N, GRID_N), dtype=np.float32)
            res.stats   = zone_stats(occs, owner, res.lux)
            res.seconds = time.time() - t0
            res.message = note
            return res

        cond = append_room(cond5, scene.width, scene.depth, scene.height)
        tgt  = target_grid(occs, scene.width, scene.depth, owner)

        if not occs or not (tgt > 0).any():
            # Nobody to light for.  A solve here would be scored on an empty occupied set,
            # which is meaningless; close the facade instead and show the daylight that
            # remains (none).
            res.mode    = "dark"
            res.binary  = np.zeros((N_FACADES, FACADE_ROWS, FACADE_COLS), dtype=np.int32)
            res.lux     = self.opt.predict_lux(res.binary, cond)[0]
            res.stats   = []
            res.seconds = time.time() - t0
            res.message = "no occupants — facade closed"
            return res

        # Intermediate facades go to the screen as the optimizer improves on them, so the user
        # watches the design converge instead of staring at the old one until the solve ends.
        # Rate-limited by _push_gate; see PREVIEW_COOLDOWN.
        def _emit(binary, best_L):
            if on_preview is not None:
                on_preview(FacadePreview(seq=scene.seq, scene=scene, binary=binary,
                                         best_L=float(best_L),
                                         mode="warm" if warm else "cold"))

        def _c2f_cb(step, total, l_zone, best_L, pct, curr_np, best_np):
            if self._push_gate(best_L, best_np):
                _emit((best_np > 0.5).astype(np.int32), best_L)

        self._push_best = float("inf")        # new scene, new loss scale — see __init__
        key  = scene.structural_key()
        warm = (not force_cold) and self._X is not None and self._key == key
        if warm:
            binary, X, _, res.converged = self._refine(
                self._X, cond, tgt, scene.sides_open,
                on_improve=_emit if on_preview else None)
            res.mode = "warm"
        else:
            binary, _, hist = self.opt.optimize_coarse_to_fine(
                cond, tgt, facade_open=tuple(bool(s) for s in scene.sides_open),
                schedule=SHORT_SCHEDULE, eval_every=10,
                glare_weight=GLARE_WEIGHT, verbose=self.verbose,
                progress_callback=_c2f_cb if on_preview else None,
                callback_every=PREVIEW_EVERY)
            res.converged = self._c2f_converged(hist)
            res.mode = "cold"
            # optimize_coarse_to_fine hands back its binary but not the logits behind it, so
            # seed the warm path with flat logits that reproduce that binary exactly.  The
            # magnitude is WARM_LOGIT_CAP, not something larger: see the note there.
            X = torch.tensor(np.where(binary > 0, WARM_LOGIT_CAP, -WARM_LOGIT_CAP),
                             dtype=torch.float32, device=self.opt.device)
            res.mode = "cold"

        self._X, self._key = X.detach(), key
        res.binary  = binary
        res.lux     = self.opt.predict_lux(binary, cond)[0]
        res.stats   = zone_stats(occs, owner, res.lux)
        res.glazing = float(binary.mean() * 100.0)
        res.seconds = time.time() - t0
        ok = [s["in_band"] for s in res.stats if s["in_band"] == s["in_band"]]
        # Two quite different things get called "settled", and conflating them would mislead:
        # the search can stop improving because it has met the brief, or because it is stuck in
        # a mediocre basin.  Only the second is worth re-rolling, so they are named separately.
        if not res.converged:
            res.stop_reason = "still improving when its step budget ran out"
        elif ok and np.mean(ok) >= 90.0:
            res.stop_reason = "settled — everyone is in band, nothing left to improve"
        else:
            res.stop_reason = "settled in a local minimum — a fresh start may do better"
        res.message = (f"{res.mode} solve {res.seconds:.1f}s | glazing {res.glazing:.1f}% | "
                       f"in-band {np.mean(ok):.0f}%" if ok else f"{res.mode} solve")
        return res


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class SolverWorker:
    """Runs a LiveSolver on its own thread, keeping only the NEWEST request.

    Coalescing is the whole point.  Holding down `W` emits a keypress every ~30 ms while a
    solve takes ~1.2 s, so a queue would build a backlog of positions the occupant has
    already walked past and the facade would lag further behind the longer you moved.  A
    single-slot mailbox that overwrites means the worker always picks up where the person
    actually IS when it next becomes free — the facade chases the occupant instead of
    replaying their path.
    """

    def __init__(self, weights_path=None, device=None, verbose=False):
        self._solver_kw = dict(weights_path=weights_path, device=device, verbose=verbose)
        self.solver  = None
        self._cv     = threading.Condition()
        self._next   = None          # single-slot mailbox (Scene, force_cold)
        self._out    = []            # newest-only result mailbox
        self._prev   = None          # newest-only mid-solve preview
        self._stop   = False
        self._busy   = False
        self.ready   = threading.Event()
        self.error   = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="facade-solver")

    def start(self):
        self._thread.start()
        return self

    def submit(self, scene, force_cold=False):
        with self._cv:
            self._next = (scene, force_cold)
            self._cv.notify()

    def poll(self):
        """Newest finished result, or None.  Older results are discarded unread — the app
        only ever wants the current facade.

        Taking a result also DROPS any pending preview: a preview is a worse version of the
        design the result just finalised, so applying it afterwards would step the facade
        backwards on screen.
        """
        with self._cv:
            if not self._out:
                return None
            r, self._out = self._out[-1], []
            self._prev = None
            return r

    def poll_preview(self):
        """Newest mid-solve facade, or None.  Single-slot for the same reason the request
        mailbox is: an older intermediate design is of no interest once a better one exists."""
        with self._cv:
            p, self._prev = self._prev, None
            return p

    @property
    def busy(self):
        with self._cv:
            return self._busy or self._next is not None

    def invalidate(self):
        if self.solver is not None:
            self.solver.invalidate()

    def predict(self, scene, binary):
        """Run a predict on the CALLER'S thread — it does not go through the mailbox.

        That is the point: the mailbox exists to make slow solves coalesce, and putting an 8 ms
        predict behind a 2.6 s solve would reintroduce exactly the lag this path was added to
        remove.  Safe to call while the solver thread is mid-solve: the model is in eval(), its
        parameters are requires_grad_(False), predict_lux runs under no_grad, and autocast state
        is thread-local, so a concurrent forward mutates nothing the solver owns.  Measured
        contention is ~26 ms p95 during an active cold solve.

        Returns None if the model has not finished loading yet (`ready` has not fired).
        """
        if self.solver is None:
            return None
        return self.solver.predict(scene, binary)

    def wait_idle(self, timeout=300.0):
        """Block until nothing is queued or running.  For headless callers (tests, the app's
        --screenshot path) that need one finished solve rather than a live stream."""
        end = time.time() + timeout
        with self._cv:
            while (self._busy or self._next is not None) and time.time() < end:
                self._cv.wait(0.05)
            return not (self._busy or self._next is not None)

    def stop(self, join_timeout=5.0):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def _run(self):
        try:
            self.solver = LiveSolver(**self._solver_kw)
        except Exception as exc:                      # noqa: BLE001 — surfaced to the UI
            self.error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            with self._cv:
                while self._next is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                scene, force_cold = self._next
                self._next = None
                self._busy = True
            def _preview(p):
                # Runs on this thread, inside the optimizer's inner loop — so it does the least
                # possible work: take the lock, drop the slot, release.
                with self._cv:
                    self._prev = p

            try:
                r = self.solver.solve(scene, force_cold=force_cold, on_preview=_preview)
            except Exception as exc:                  # noqa: BLE001
                r = SolveResult(seq=scene.seq, scene=scene, mode="error",
                                message=f"{type(exc).__name__}: {exc}")
            with self._cv:
                self._out.append(r)
                self._busy = False
                self._cv.notify_all()      # wakes wait_idle()
