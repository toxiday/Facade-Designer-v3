"""
run_facade_job.py  —  headless JSON-driven facade optimization job runner.

The file-based interface between Rhino/Grasshopper and the all-sides climate optimizer.
Grasshopper authors a `facade_job/1` JSON (see "rhino bridge/FACADE_JOB_SCHEMA.md"), this
script consumes it and writes back a facade design carrying EXPLICIT PANEL GEOMETRY, so the
Grasshopper reader can build Rhino geometry without re-deriving any index convention.

    python run_facade_job.py job.json [--out DIR]
    python run_facade_job.py --list-cities

It adds no optimization logic — it composes existing pieces of
optimize_facade_allsides_climate.py (resolve_city, annual_conditions, reduce_conditions,
append_room, optimize_coarse_to_fine, predict_lux, save_facades_csv, facade_panel_boxes,
wall_azimuths) and mirrors how the interactive designer collects multi-slot inputs.

FRAMES — the one thing to keep straight
    Everything in the job, input and output, is ROOM-LOCAL: S = the y=0 wall, N = y=D,
    E = x=W, W = x=0; grid row 0 = the +Y ("N") edge; panel coordinates are metres in the
    room's own frame.  None of that changes with rotation.  `north_angle` rotates the SUN
    into this frame, and `placement` says where the frame sits in Rhino world space.  The
    only compass-referenced value in the outputs is `faces_az` / `sides_compass`.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_facade_allsides_climate import (      # noqa: E402
    FacadeOptimizerAllSidesClimate, resolve_city, city_label, city_location,
    annual_conditions, reduce_conditions, append_room,
    save_facades_csv, facade_panel_boxes, wall_azimuths, discover_cities,
    GRID_N, GLARE_LUX, ZONE_MIN_RATIO, N_REPR, SIDES, DEFAULT_GLARE_WEIGHT,
    FACADE_ROWS, FACADE_COLS, N_FACADES,
    W_MIN, W_MAX, D_MIN, D_MAX, H_MIN, H_MAX)

SCHEMA      = "facade_job/1"
N_GRID_VALS = GRID_N * GRID_N          # 1024
OUT_ROOT    = Path(__file__).resolve().parents[1] / "outputs"

# Rhino jobs opt in to the unoccupied-floor glare penalty; the interactive designer and the
# legacy CLI keep DEFAULT_GLARE_WEIGHT = 0.0 so their behaviour is untouched.
DEFAULT_JOB_GLARE_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class JobError(Exception):
    """A job file that cannot be run as written."""


def _require(cond, msg):
    if not cond:
        raise JobError(msg)


def load_job(path):
    """Parse + validate a job file into a normalized dict.  Raises JobError with an
    actionable message rather than letting a bad job produce a plausible wrong design."""
    path = Path(path)
    _require(path.is_file(), f"job file not found: {path}")
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise JobError(f"{path.name} is not valid JSON: {e}") from None
    _require(isinstance(job, dict), "job file must be a JSON object")

    schema = job.get("schema", SCHEMA)
    _require(schema == SCHEMA, f"unsupported schema {schema!r} (expected {SCHEMA!r})")

    units = job.get("units", "m")
    _require(units == "m", f"units must be 'm' (got {units!r}) — convert in Grasshopper, "
                           "the model's room ranges are all in metres")

    # ---- room ----
    room = job.get("room")
    _require(isinstance(room, dict), "job.room is required")
    try:
        W = float(room["width"]); D = float(room["depth"]); H = float(room["height"])
    except (KeyError, TypeError, ValueError):
        raise JobError("job.room needs numeric width, depth and height (metres)") from None
    north_angle = float(room.get("north_angle", 0.0))

    # normalize_room CLAMPS silently, so out-of-range dims would otherwise produce a
    # confident design for a room that is not the one drawn.  Warn loudly.
    warnings = []
    for name, val, lo, hi in (("width", W, W_MIN, W_MAX),
                              ("depth", D, D_MIN, D_MAX),
                              ("height", H, H_MIN, H_MAX)):
        _require(val > 0, f"room.{name} must be positive (got {val})")
        if not (lo <= val <= hi):
            warnings.append(f"room.{name}={val:g} m is outside the model's trained range "
                            f"{lo}-{hi} m and WILL BE CLAMPED — the design will be for a "
                            f"{min(max(val, lo), hi):g} m {name}, not {val:g} m")

    # ---- sides ----
    sides = str(job.get("sides", "SNEW")).upper().strip()
    _require(sides, "job.sides is empty — at least one wall must be able to hold windows")
    bad = sorted(set(sides) - set("SNEW"))
    _require(not bad, f"job.sides contains invalid wall id(s) {bad} — allowed characters are "
                      f"S, N, E, W (room-local wall ids, not compass directions)")
    facade_open = tuple(s in sides for s in SIDES)

    # ---- climate ----
    climate = job.get("climate") or {}
    _require(isinstance(climate, dict) and "city" in climate,
             "job.climate.city is required (a name, e.g. 'Houston', or an index)")

    # ---- slots ----
    slots_in = job.get("slots")
    _require(isinstance(slots_in, list) and slots_in, "job.slots must be a non-empty list")
    slots = []
    for i, s in enumerate(slots_in):
        _require(isinstance(s, dict), f"slots[{i}] must be an object")
        try:
            hour = int(s["hour"])
        except (KeyError, TypeError, ValueError):
            raise JobError(f"slots[{i}].hour is required and must be an integer") from None
        _require(0 <= hour <= 23, f"slots[{i}].hour={hour} out of range 0-23")

        grid = s.get("grid")
        _require(isinstance(grid, list),
                 f"slots[{i}].grid must be a list of {N_GRID_VALS} numbers")
        _require(len(grid) == N_GRID_VALS,
                 f"slots[{i}].grid has {len(grid)} values, expected {N_GRID_VALS} "
                 f"({GRID_N}x{GRID_N} row-major)")
        try:
            g = np.asarray(grid, dtype=np.float32)
        except (TypeError, ValueError):
            raise JobError(f"slots[{i}].grid contains non-numeric values") from None
        _require(np.all(np.isfinite(g)), f"slots[{i}].grid contains NaN/inf")
        _require(np.all(g >= 0), f"slots[{i}].grid has negative lux values")
        _require(g.any(), f"slots[{i}].grid is all zeros — no occupied cells means zero "
                          f"gradient, so the optimizer would return noise. Paint a zone.")
        slots.append(dict(hour=hour, grid=g.reshape(GRID_N, GRID_N)))

    # ---- optimizer ----
    opt = job.get("optimizer") or {}
    mode = str(opt.get("mode", "coarse_to_fine"))
    _require(mode in ("coarse_to_fine", "direct"),
             f"optimizer.mode must be 'coarse_to_fine' or 'direct' (got {mode!r})")
    repr_conditions = int(opt.get("repr_conditions", N_REPR))
    _require(repr_conditions >= 0, "optimizer.repr_conditions must be >= 0 (0 = use all days)")
    seed  = opt.get("seed", None)
    steps = int(opt.get("steps", 800))

    # Unoccupied-floor glare. ON by default for Rhino jobs (the GUI and legacy CLI stay at
    # 0.0), because without it the optimizer treats unpainted floor as free disposal for
    # surplus daylight. 0.25 was sized from measurement, not taste: at cap 3000 the glare
    # term's initial magnitude is ~0.087 against a ~0.102 zone loss, so this puts it near
    # 18% of the objective — enough to steer, never enough to outvote the zone.
    def _num(key, default):
        """A present-but-null or non-numeric field should be a job rejection, not a
        TypeError traceback out of float()."""
        v = opt.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            raise JobError(f"optimizer.{key} must be a number (got {v!r})") from None

    # Steps between progress lines.  The optimizer treats this as a minimum GAP rather than
    # a modulus, so early-stopping stages cannot silence it.
    progress_every = int(_num("progress_every", 25))
    _require(progress_every >= 1, "optimizer.progress_every must be >= 1")

    glare_weight = _num("glare_weight", DEFAULT_JOB_GLARE_WEIGHT)
    glare_cap    = _num("glare_cap", GLARE_LUX)
    _require(glare_weight >= 0, "optimizer.glare_weight must be >= 0 (0 disables it)")
    _require(glare_cap > 0, "optimizer.glare_cap must be > 0 lux")

    out = job.get("output") or {}
    return dict(raw=job, path=path,
                width=W, depth=D, height=H, north_angle=north_angle,
                sides=sides, facade_open=facade_open,
                city=climate["city"], slots=slots,
                mode=mode, repr_conditions=repr_conditions, seed=seed, steps=steps,
                glare_weight=glare_weight, glare_cap=glare_cap,
                progress_every=progress_every,
                out_dir=out.get("dir"), out_name=out.get("name") or path.stem,
                placement=job.get("placement"), warnings=warnings)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def build_conditions(job, wea_path):
    """Mirror the designer's _collect_inputs: per slot build weighted representative
    conditions for the OPTIMIZER and the full annual set for METRICS, then concatenate on
    the T axis so one facade design is scored against every slot at once."""
    W, D, H = job["width"], job["depth"], job["height"]
    na      = job["north_angle"]

    conds, targets, weights, per_slot = [], [], [], []
    for s in job["slots"]:
        _, lat, cond, days = annual_conditions(wea_path, s["hour"], north_angle=na)
        if len(cond) == 0:
            # Recorded in job["warnings"], not just printed: save_result_json writes that
            # list into result.json, so a dropped slot stays visible long after the console
            # output has scrolled away.  A painted zone the design never saw is exactly the
            # kind of thing that should not be discoverable only by counting rows in a PNG.
            msg = (f"{s['hour']:02d}:00 slot SKIPPED — the sun is never above 5 deg at that "
                   f"hour in this climate, so it contributes no daylight to design for. "
                   f"Its painted zone was ignored.")
            print(f"  [!] {msg}")
            job["warnings"].append(msg)
            continue
        if job["repr_conditions"] and len(cond) > job["repr_conditions"]:
            cond_opt, w = reduce_conditions(cond, n_clusters=job["repr_conditions"])
        else:
            cond_opt, w = cond, np.ones(len(cond), dtype=np.float32)
        cond_opt = append_room(cond_opt, W, D, H)
        tgt      = np.broadcast_to(s["grid"], (len(cond_opt), GRID_N, GRID_N))

        conds.append(cond_opt)
        weights.append(w)
        targets.append(tgt)
        per_slot.append(dict(hour=s["hour"], grid=s["grid"], n_days=len(days),
                             n_repr=len(cond_opt), days=days,
                             cond_full=append_room(cond, W, D, H)))

    if not conds:
        raise JobError("no usable time slots — the sun is never above the horizon at any "
                       "requested hour for this city. Pick different hours.")
    return (np.concatenate(conds, 0),
            np.concatenate(targets, 0).copy(),
            np.concatenate(weights, 0),
            per_slot)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(opt, binary, per_slot):
    """Per-slot and overall daylight metrics, evaluated over the FULL year at each slot's
    hour (not the reduced conditions the optimizer saw).

    Glare is reported BOTH zone-only and whole-floor: zone_penalty_sum masks every term by
    `target_lux > 0`, so unoccupied cells contribute exactly zero gradient — including the
    glare term.  The whole-floor number therefore describes something the optimizer was
    never asked to control, and the two can disagree wildly.  Reporting only one of them
    is how that gap turns into a surprise.
    """
    slot_metrics = []
    zone_vals, zone_under, floor_glare_hits, floor_cells = [], [], 0, 0

    for ps in per_slot:
        E   = opt.predict_lux(binary, ps["cond_full"])          # (T,32,32) lux
        occ = ps["grid"] > 0.0
        tgt = ps["grid"][occ]                                   # per-cell target lux
        zv  = E[:, occ]                                         # (T, n_occ)
        under = zv < (tgt * ZONE_MIN_RATIO)

        slot_metrics.append(dict(
            hour=ps["hour"], days_above_horizon=ps["n_days"],
            repr_conditions=ps["n_repr"], occupied_cells=int(occ.sum()),
            target_lux_mean=float(tgt.mean()),
            zone_mean_lux=float(zv.mean()),
            zone_underlit_pct=float(under.mean() * 100.0),
            zone_glare_pct=float((zv > GLARE_LUX).mean() * 100.0),
            floor_glare_pct=float((E > GLARE_LUX).mean() * 100.0)))

        zone_vals.append(zv.ravel())
        zone_under.append(under.ravel())
        floor_glare_hits += int((E > GLARE_LUX).sum())
        floor_cells      += int(E.size)

    zv = np.concatenate(zone_vals)
    zu = np.concatenate(zone_under)
    return dict(
        slots=slot_metrics,
        zone_mean_lux=float(zv.mean()),
        zone_underlit_pct=float(zu.mean() * 100.0),
        zone_glare_pct=float((zv > GLARE_LUX).mean() * 100.0),
        floor_glare_pct=float(floor_glare_hits / max(floor_cells, 1) * 100.0),
        glazing_pct=float(binary.mean() * 100.0))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_panels_csv(binary, width, depth, height, north_angle, path):
    """One row per panel (all 2048, not only windows, so Grasshopper can build solid wall
    too), with explicit min/max corners in ROOM-LOCAL metres.

    is_window is 1 = WINDOW — the design convention, and the inverse of the model input
    where <0.5 = window.  That inversion is this project's most persistent trap.

    faces_az is the only compass-referenced column: the true azimuth the wall's outward
    face points at, given north_angle.  `side` is a room-local id, so `N` means "the y=D
    wall", never "north".  An azimuth also stays exact at non-cardinal rotations where an
    S/N/E/W letter would be meaningless.
    """
    az = wall_azimuths(north_angle)
    idx_of = {s: i for i, s in enumerate(SIDES)}
    lines = ["side,row,col,pix_index,is_window,x0,y0,z0,x1,y1,z1,nx,ny,nz,faces_az"]
    for side, row, col, pix, p0, p1, n in facade_panel_boxes(width, depth, height):
        w = int(binary[idx_of[side], row, col])
        lines.append(
            f"{side},{row},{col},{pix},{w},"
            f"{p0[0]:.6f},{p0[1]:.6f},{p0[2]:.6f},"
            f"{p1[0]:.6f},{p1[1]:.6f},{p1[2]:.6f},"
            f"{n[0]:g},{n[1]:g},{n[2]:g},{az[side]:.4f}")
    Path(path).write_text("\n".join(lines) + "\n")


def render_result_png(binary, opt, wea_path, job, per_slot, path):
    """The same unfolded-cross figure the interactive designer produces, drawn headlessly.

    matplotlib is imported HERE, not at module scope, and the Agg backend is selected before
    facade_viz is imported: this process has no display, and importing the designer module
    (which sets TkAgg at import time) would try to open a window and hang the subprocess
    Grasshopper is waiting on.  A missing matplotlib must not fail the job either — the CSVs
    are the deliverable, the picture is a convenience.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from facade_viz import visualize_result_cross
    except ImportError as e:
        print(f"  [!] skipping result.png — matplotlib unavailable ({e})", flush=True)
        return None

    # facade_viz wants painted-marker dicts; a slot's per-cell lux grid is the same thing
    # with the scalar target recovered from its occupied cells.
    markers = [dict(hour=ps["hour"], zone=ps["grid"],
                    target=float(ps["grid"][ps["grid"] > 0].mean()))
               for ps in per_slot]
    print("  Rendering result.png...", flush=True)
    return visualize_result_cross(
        binary, opt, wea_path, markers, job["width"], job["depth"], job["height"],
        job["facade_open"], path, north_angle=job["north_angle"], interactive=False)


def save_result_json(job, wea_path, metrics, out_dir, path):
    lat, lon_e, tz_h = city_location(wea_path)
    result = dict(
        schema="facade_result/1",
        job=job["raw"],                       # verbatim echo, including placement
        resolved=dict(
            city_label=city_label(wea_path),
            wea_file=Path(wea_path).name,
            latitude=lat, longitude_east=lon_e, timezone_hours=tz_h,
            sides_open=[s for s, o in zip(SIDES, job["facade_open"]) if o],
            north_angle=job["north_angle"],
            # Everything else in the outputs is room-local and rotation-invariant; this is
            # the decoder ring that makes facade_wide.csv unambiguous on its own.
            sides_compass=wall_azimuths(job["north_angle"]),
            room=dict(width=job["width"], depth=job["depth"], height=job["height"]),
            glare_weight=job["glare_weight"], glare_cap=job["glare_cap"],
            facade_grid=dict(facades=N_FACADES, rows=FACADE_ROWS, cols=FACADE_COLS),
            floor_grid=dict(rows=GRID_N, cols=GRID_N)),
        metrics=metrics,
        warnings=job["warnings"],
        outputs=dict(facade_wide="facade_wide.csv",
                     facade_panels="facade_panels.csv",
                     directory=str(out_dir)))
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(job, out_dir=None, no_png=False,
        verify=False, verify_conditions=32, verify_workers=None):
    wea_path = resolve_city(job["city"])
    label    = city_label(wea_path)

    for w in job["warnings"]:
        print(f"  [!] {w}")

    az = wall_azimuths(job["north_angle"])
    print(f"City : {label}")
    print(f"Room : {job['width']:.2f} x {job['depth']:.2f} x {job['height']:.2f} m  "
          f"| north_angle {job['north_angle']:g} deg")
    print(f"Sides: {job['sides']}  ->  " +
          "  ".join(f"{s}:{az[s]:.0f}deg" for s in SIDES if s in job["sides"]))

    cond, target, weights, per_slot = build_conditions(job, wea_path)
    for ps in per_slot:
        print(f"  {ps['hour']:02d}:00  {int((ps['grid'] > 0).sum()):4d} cells  "
              f"@ {ps['grid'][ps['grid'] > 0].mean():.0f} lux  "
              f"over {ps['n_days']} days -> {ps['n_repr']} repr conditions")

    if job["seed"] is not None:
        torch.manual_seed(int(job["seed"]))
        print(f"  seed {int(job['seed'])} (reproducible)")

    gkw = dict(glare_weight=job["glare_weight"], glare_cap=job["glare_cap"])
    if job["glare_weight"]:
        print(f"  unoccupied-floor glare penalty: weight {job['glare_weight']:g} "
              f"above {job['glare_cap']:.0f} lux")

    print(f"\n  Optimizing over {len(cond)} weighted conditions "
          f"({job['mode'].replace('_', '-')})...", flush=True)
    opt = FacadeOptimizerAllSidesClimate()
    if job["mode"] == "direct":
        binary, _, _ = opt.optimize(cond, target, weights, job["facade_open"],
                                    n_steps=job["steps"], log_every=job["progress_every"],
                                    **gkw)
    else:
        binary, _, _ = opt.optimize_coarse_to_fine(cond, target, weights,
                                                   job["facade_open"],
                                                   log_every=job["progress_every"], **gkw)

    print("\n  Evaluating the design over the full year...", flush=True)
    metrics = compute_metrics(opt, binary, per_slot)

    out_dir = Path(out_dir or job["out_dir"] or (OUT_ROOT / job["out_name"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_facades_csv(binary, out_dir / "facade_wide.csv")
    save_panels_csv(binary, job["width"], job["depth"], job["height"],
                    job["north_angle"], out_dir / "facade_panels.csv")

    png = None
    if not no_png:
        png = render_result_png(binary, opt, wea_path, job, per_slot,
                                out_dir / "result.png")
    save_result_json(job, wea_path, metrics, out_dir, out_dir / "result.json")

    print(f"\n  Glazing              : {metrics['glazing_pct']:.1f}%")
    print(f"  Zone mean lux        : {metrics['zone_mean_lux']:.0f}")
    print(f"  Zone underlit        : {metrics['zone_underlit_pct']:.1f}% of zone cell-days")
    print(f"  Zone glare (>{GLARE_LUX:.0f})   : {metrics['zone_glare_pct']:.1f}%   <- optimized")
    if job["glare_weight"]:
        print(f"  Floor glare (>{job['glare_cap']:.0f})  : {metrics['floor_glare_pct']:.1f}%   "
              f"<- optimized at weight {job['glare_weight']:g} (subordinate to the zone)")
    else:
        print(f"  Floor glare (>{GLARE_LUX:.0f})  : {metrics['floor_glare_pct']:.1f}%   "
              f"<- NOT optimized (set optimizer.glare_weight to constrain it)")
    print(f"\n  Written to {out_dir}")
    for f in ("facade_wide.csv", "facade_panels.csv", "result.json"):
        print(f"    {f}")
    if png:
        print(f"    result.png   <- unfolded facade cross + predicted-lux gallery")

    # Everything above this line is the SURROGATE's account of its own output.  --verify
    # re-simulates the finished design in Radiance so the design claim stops being
    # self-certified.  Imported here, not at module scope: Radiance is not needed to run a
    # job, and find_radiance_bin() raises at import time when it is absent.
    if verify:
        try:
            from verify_facade_radiance import verify as verify_radiance
        except Exception as e:
            print(f"\n  [!] verification skipped — {e}")
            return out_dir
        verify_radiance(binary, per_slot,
                        dict(width=job["width"], depth=job["depth"], height=job["height"],
                             north_angle=job["north_angle"]),
                        label, out_dir, opt=opt,
                        n_conditions=verify_conditions, workers=verify_workers)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run a facade_job/1 JSON through the all-sides climate optimizer.")
    ap.add_argument("job", nargs="?", help="path to the job JSON")
    ap.add_argument("--out", default=None, help="override the output directory")
    ap.add_argument("--list-cities", action="store_true",
                    help="print the climate library (name + index) and exit")
    ap.add_argument("--no-png", action="store_true",
                    help="skip the result.png figure (saves a few seconds)")
    ap.add_argument("--verify", action="store_true",
                    help="after optimizing, re-simulate the design in Radiance and report "
                         "radiance-vs-target, radiance-vs-surrogate and surrogate-vs-target")
    ap.add_argument("--verify-conditions", type=int, default=32,
                    help="Radiance runs per time slot, spread evenly through the year "
                         "(default 32)")
    ap.add_argument("--verify-workers", type=int, default=None,
                    help="worker processes for verification (default: cpu_count - 2)")
    args = ap.parse_args(argv)

    if args.list_cities:
        for i, c in enumerate(discover_cities()):
            print(f"  [{i:3d}] {city_label(c)}")
        return 0
    if not args.job:
        ap.error("a job JSON is required (or use --list-cities)")

    try:
        run(load_job(args.job), out_dir=args.out, no_png=args.no_png,
            verify=args.verify, verify_conditions=args.verify_conditions,
            verify_workers=args.verify_workers)
    except JobError as e:
        print(f"\nJob rejected: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
