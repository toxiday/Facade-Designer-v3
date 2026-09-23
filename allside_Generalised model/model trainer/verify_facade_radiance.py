"""
verify_facade_radiance.py — close the loop on an optimized facade with real Radiance.

The optimizer reports what the SURROGATE believes about its own output.  That is exactly
the claim that cannot be self-certified: gradient descent drives toward the extremes of the
surrogate's prediction surface, which is where a learned model is least reliable, so the
optimizer is adversarial to its own network.  Accuracy measured on RANDOM facades (which is
what the training/eval sweeps measure) does not transfer to OPTIMIZER-SELECTED facades.

So this module re-simulates the finished design in Radiance and reports three comparisons:

    RADIANCE vs TARGET      does the design actually work?          (the design claim)
    RADIANCE vs SURROGATE   is the model right where it landed?     (the method claim)
    SURROGATE vs TARGET     what the optimizer believed             (context for the above)

The middle one is the scientifically load-bearing row and it is nearly free — the Radiance
run is the cost, and the prediction is already in hand.

SCOPE — read this before interpreting a disagreement
    The surrogate was trained on an ISOLATED room: floor, ceiling, four walls of facade
    cells, sky and ground glow, and nothing else.  No neighbours, no slab overhangs, no
    balconies, no self-shading.  This verification rebuilds that SAME isolated room, so a
    disagreement here is a genuine model error.  Putting the room in real building context
    and comparing against this is a different experiment (out-of-scope context sensitivity),
    and conflating the two would misreport a scope mismatch as a model failure.

Radiance settings, materials, sensor grid and sensor height are taken from the dataset
generator itself, so ground truth here is produced the same way the training data was.

    python verify_facade_radiance.py ../outputs/job_001
    python verify_facade_radiance.py ../outputs/job_001 --conditions 48 --workers 22

or from run_facade_job.py via --verify.
"""

import argparse
import importlib.util
import json
import math
import multiprocessing
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_facade_allsides_climate import (      # noqa: E402
    FacadeOptimizerAllSidesClimate, resolve_city, city_label,
    annual_conditions, append_room,
    GRID_N, GLARE_LUX, ZONE_MIN_RATIO, SIDES,
    FACADE_ROWS, FACADE_COLS, N_FACADES)

GEN_PY = (Path(__file__).resolve().parents[1] / "data generator"
          / "generate_dataset_allsides_climate_pit.py")

N_CELLS = FACADE_ROWS * FACADE_COLS


def _load_generator():
    """Import the dataset generator by path — 'data generator' is not an importable package
    name, and we want the REAL scene builder rather than a second copy of it.  Importing it
    runs find_radiance_bin() at module scope, so a missing Radiance fails loudly here."""
    spec = importlib.util.spec_from_file_location("gen_pit_verify", str(GEN_PY))
    m = importlib.util.module_from_spec(spec)
    sys.modules["gen_pit_verify"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def cond_to_sun(cond_row):
    """(alt_deg, az_north_cw_deg, dni, dhi) from a model conditioning row.

    Inverts the generator's own construction
        sx = -sin(az)cos(alt);  sy = -cos(az)cos(alt);  sz = -sin(alt)
    The vector is already expressed in the ROOM frame (annual_conditions rotated it by
    north_angle), and the scene we rebuild is also room-local with +Y = the N wall, so
    feeding this straight back to gendaylit is consistent — the room never rotates.
    """
    sx, sy, sz = float(cond_row[0]), float(cond_row[1]), float(cond_row[2])
    alt = math.degrees(math.asin(max(-1.0, min(1.0, -sz))))
    az = math.degrees(math.atan2(-sx, -sy)) % 360.0
    return alt, az, float(cond_row[3]) * 1000.0, float(cond_row[4]) * 1000.0


def binary_to_scene_facades(binary):
    """(4,16,32) with 1=WINDOW  ->  {'S','N','E','W'}: flat 512-lists with <0.5 = window.

    This is the project's most persistent trap, inverted in exactly one place: the design
    convention is 1 = window, the scene builder (and the model input) treat <0.5 = window.
    """
    out = {}
    for i, side in enumerate(SIDES):
        flat = binary[i].reshape(-1)
        out[side] = [0.0 if v >= 0.5 else 1.0 for v in flat]
    return out


def sensors_to_plan(lux_flat):
    """Radiance sensor order -> the plan orientation the model and targets use.

    Identical to the trainer's target transform (rot90 then flip columns); without it the
    comparison would silently be against a rotated field.
    """
    a = np.asarray(lux_flat, dtype=np.float32).reshape(GRID_N, GRID_N)
    return np.fliplr(np.rot90(a, k=1)).copy()


# ---------------------------------------------------------------------------
# Radiance workers
# ---------------------------------------------------------------------------

_GEN = None
_ENV = None
_FAC = None
_ROOM = None


def _init_worker(gen_py, facades, room):
    global _GEN, _ENV, _FAC, _ROOM
    spec = importlib.util.spec_from_file_location("gen_pit_verify", gen_py)
    _GEN = importlib.util.module_from_spec(spec)
    sys.modules["gen_pit_verify"] = _GEN
    spec.loader.exec_module(_GEN)
    _ENV = _GEN.make_env()
    _FAC = facades
    _ROOM = room


def _simulate(task):
    """(key, alt, az, dni, dhi) -> (key, 1024 lux) or (key, None) on failure."""
    key, alt, az, dni, dhi = task
    work = Path(tempfile.mkdtemp(prefix="verify_"))
    try:
        W, D, H = _ROOM
        oct_path, pts_path = _GEN.write_scene(work, _FAC, W, D, H, alt, az, dni, dhi, _ENV)
        lux = _GEN.run_simulation(oct_path, pts_path, _ENV)
        if len(lux) != GRID_N * GRID_N:
            return key, None
        return key, lux
    except Exception:
        return key, None
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Metrics — mirror run_facade_job.compute_metrics so both sides are scored identically
# ---------------------------------------------------------------------------

def zone_metrics(E, grid):
    """Same definitions the optimizer reports, applied to ANY (T,32,32) lux stack.

    Using one function for both the Radiance and the surrogate stack is the point: a
    difference in the numbers is then a difference in the lux, never in the bookkeeping.
    """
    occ = grid > 0.0
    tgt = grid[occ]
    zv = E[:, occ]
    under = zv < (tgt * ZONE_MIN_RATIO)
    return dict(
        occupied_cells=int(occ.sum()),
        target_lux_mean=float(tgt.mean()),
        zone_mean_lux=float(zv.mean()),
        zone_underlit_pct=float(under.mean() * 100.0),
        zone_glare_pct=float((zv > GLARE_LUX).mean() * 100.0),
        floor_glare_pct=float((E > GLARE_LUX).mean() * 100.0),
        floor_mean_lux=float(E.mean()))


def agreement(a, b, grid=None):
    """Radiance-vs-surrogate agreement over two aligned (T,32,32) stacks.

    Reports the tail explicitly.  A surrogate can be excellent on the typical pixel and
    still be badly wrong on a handful of direct-sun cells; a single RMSE hides which of
    those two situations you are in, and they mean very different things here.
    """
    err = (b - a).ravel()                      # b = surrogate, a = radiance
    abserr = np.abs(err)
    denom = float(((a - a.mean()) ** 2).sum())
    out = dict(
        n_pixels=int(err.size),
        rmse=float(np.sqrt((err ** 2).mean())),
        mae=float(abserr.mean()),
        bias=float(err.mean()),
        r2=float(1.0 - (err ** 2).sum() / denom) if denom > 0 else float("nan"),
        p50=float(np.percentile(abserr, 50)),
        p90=float(np.percentile(abserr, 90)),
        p99=float(np.percentile(abserr, 99)),
        p999=float(np.percentile(abserr, 99.9)),
        max=float(abserr.max()))
    # per-condition room-mean lux is the number a designer actually reads
    am, bm = a.reshape(len(a), -1).mean(1), b.reshape(len(b), -1).mean(1)
    out["room_mean_mape_pct"] = float(np.mean(np.abs(bm - am) / np.maximum(am, 1.0)) * 100.0)
    if len(am) > 1:
        out["room_mean_corr"] = float(np.corrcoef(am, bm)[0, 1])
    if grid is not None and (grid > 0).any():
        occ = grid > 0
        ze = (b[:, occ] - a[:, occ]).ravel()
        out["zone_rmse"] = float(np.sqrt((ze ** 2).mean()))
        out["zone_mae"] = float(np.abs(ze).mean())
        out["zone_bias"] = float(ze.mean())
    return out


# ---------------------------------------------------------------------------
# Figures — two structurally identical galleries, so they can be flipped between
# ---------------------------------------------------------------------------

def _gallery(per_slot_E, slots, title, out_path, day_labels, vmax=GLARE_LUX, dpi=150):
    """rows = time slots, cols = [mean over sampled conditions] + sampled days.

    Deliberately the SAME layout, colour map and scale for both the Radiance and the
    surrogate figure — an A/B pair only reads if nothing but the data changes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from facade_viz import LUX_CMAP
    except ImportError as e:
        print(f"  [!] skipping {Path(out_path).name} — matplotlib unavailable ({e})")
        return None

    n_rows = len(slots)
    n_show = min(5, per_slot_E[0].shape[0])
    step = max(1, per_slot_E[0].shape[0] // n_show)
    picks = list(range(0, per_slot_E[0].shape[0], step))[:n_show]
    n_cols = 1 + len(picks)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.85 * n_cols + 1.0, 2.05 * n_rows + 0.9),
                             squeeze=False, facecolor="white")
    fig.suptitle(title, fontsize=11, fontweight="bold")

    im_last = None
    for r, (E, s) in enumerate(zip(per_slot_E, slots)):
        zmask = s["grid"] > 0
        arrs = [E.mean(0)] + [E[i] for i in picks]
        heads = ["mean (sampled)"] + [day_labels[r][i] for i in picks]
        for c, (arr, head) in enumerate(zip(arrs, heads)):
            ax = axes[r][c]
            im = ax.imshow(arr, cmap=LUX_CMAP, vmin=0, vmax=vmax,
                           interpolation="nearest", aspect="auto")
            if zmask.any():
                ax.contour(np.arange(GRID_N), np.arange(GRID_N), zmask.astype(float),
                           levels=[0.5], colors=["white"], linewidths=1.0)
            ax.set_xticks([0, GRID_N - 1]); ax.set_xticklabels(["W", "E"], fontsize=5)
            ax.set_yticks([0, GRID_N - 1]); ax.set_yticklabels(["N", "S"], fontsize=5)
            ax.tick_params(length=1.5)
            ax.set_title(head, fontsize=7, pad=2)
            zval = arr[zmask].mean() if zmask.any() else arr.mean()
            ax.set_xlabel(f"zone {zval:.0f} lux", fontsize=6)
            if c == 0:
                ax.set_ylabel(f"{s['hour']:02d}:00\n@ {s['grid'][zmask].mean():.0f} lux",
                              fontsize=8, rotation=0, ha="right", va="center", labelpad=20)
            im_last = im

    fig.tight_layout(rect=[0.02, 0.02, 0.92, 0.96])
    cax = fig.add_axes([0.94, 0.12, 0.012, 0.76])
    fig.colorbar(im_last, cax=cax, label=f"lux (max {vmax:.0f})")
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_metric_table(rad, sur, tgt_mean):
    rows = [
        ("target mean lux",   f"{tgt_mean:10.0f}", ""),
        ("zone mean lux",     f"{rad['zone_mean_lux']:10.0f}", f"{sur['zone_mean_lux']:10.0f}"),
        ("zone underlit %",   f"{rad['zone_underlit_pct']:10.2f}", f"{sur['zone_underlit_pct']:10.2f}"),
        ("zone glare %",      f"{rad['zone_glare_pct']:10.2f}", f"{sur['zone_glare_pct']:10.2f}"),
        ("floor glare %",     f"{rad['floor_glare_pct']:10.2f}", f"{sur['floor_glare_pct']:10.2f}"),
        ("floor mean lux",    f"{rad['floor_mean_lux']:10.0f}", f"{sur['floor_mean_lux']:10.0f}"),
    ]
    lines = [f"    {'':<20}{'RADIANCE':>12}{'SURROGATE':>12}{'delta':>12}"]
    for name, a, b in rows:
        if b == "":
            lines.append(f"    {name:<20}{a:>12}{'':>12}{'':>12}")
        else:
            d = float(b) - float(a)
            lines.append(f"    {name:<20}{a:>12}{b:>12}{d:>+12.2f}")
    return "\n".join(lines)


def write_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_report(report):
    r = report
    print("\n" + "=" * 78)
    print("RADIANCE VERIFICATION")
    print("=" * 78)
    print(f"  City            : {r['city']}")
    print(f"  Room            : {r['room']['width']:.2f} x {r['room']['depth']:.2f} "
          f"x {r['room']['height']:.2f} m   north_angle {r['room']['north_angle']:g} deg")
    print(f"  Glazing         : {r['glazing_pct']:.1f}%")
    print(f"  Conditions      : {r['n_conditions']} simulated "
          f"({r['n_failed']} failed)  |  Radiance {r['rad_params']}")
    print(f"  Scene           : ISOLATED room (no context) — same assumptions as training")

    for s in r["slots"]:
        print(f"\n  ---- {s['hour']:02d}:00  ({s['n_conditions']} conditions, "
              f"{s['occupied_cells']} occupied cells) ----")
        print(_fmt_metric_table(s["radiance_vs_target"], s["surrogate_vs_target"],
                                s["target_lux_mean"]))
        a = s["radiance_vs_surrogate"]
        print(f"\n    RADIANCE vs SURROGATE   RMSE {a['rmse']:8.1f}   MAE {a['mae']:7.1f}   "
              f"R2 {a['r2']:7.4f}   bias {a['bias']:+8.1f} lux")
        print(f"    abs err percentiles     p50 {a['p50']:7.1f}  p90 {a['p90']:7.1f}  "
              f"p99 {a['p99']:8.1f}  p99.9 {a['p999']:9.1f}  max {a['max']:9.1f}")
        if "zone_mae" in a:
            print(f"    in-zone only            MAE {a['zone_mae']:7.1f}   "
                  f"RMSE {a['zone_rmse']:8.1f}   bias {a['zone_bias']:+8.1f} lux")
        print(f"    room-mean lux           MAPE {a['room_mean_mape_pct']:6.2f}%"
              + (f"   corr {a['room_mean_corr']:.5f}" if "room_mean_corr" in a else ""))

    o = r["overall"]
    print("\n" + "-" * 78)
    print("  OVERALL (all slots pooled)")
    print("-" * 78)
    print(_fmt_metric_table(o["radiance_vs_target"], o["surrogate_vs_target"],
                            o["target_lux_mean"]))
    a = o["radiance_vs_surrogate"]
    print(f"\n    RADIANCE vs SURROGATE   RMSE {a['rmse']:8.1f}   MAE {a['mae']:7.1f}   "
          f"R2 {a['r2']:7.4f}   bias {a['bias']:+8.1f} lux")
    print(f"    abs err percentiles     p50 {a['p50']:7.1f}  p90 {a['p90']:7.1f}  "
          f"p99 {a['p99']:8.1f}  p99.9 {a['p999']:9.1f}  max {a['max']:9.1f}")
    if "zone_mae" in a:
        print(f"    in-zone only            MAE {a['zone_mae']:7.1f}   "
              f"RMSE {a['zone_rmse']:8.1f}   bias {a['zone_bias']:+8.1f} lux")

    # the sentence the whole exercise exists to produce
    d_under = (o["surrogate_vs_target"]["zone_underlit_pct"]
               - o["radiance_vs_target"]["zone_underlit_pct"])
    d_glare = (o["surrogate_vs_target"]["zone_glare_pct"]
               - o["radiance_vs_target"]["zone_glare_pct"])
    print(f"\n  DECISION-LEVEL DISAGREEMENT")
    print(f"    zone underlit : Radiance {o['radiance_vs_target']['zone_underlit_pct']:.2f}%  "
          f"vs surrogate {o['surrogate_vs_target']['zone_underlit_pct']:.2f}%  "
          f"({d_under:+.2f} pp)")
    print(f"    zone glare    : Radiance {o['radiance_vs_target']['zone_glare_pct']:.2f}%  "
          f"vs surrogate {o['surrogate_vs_target']['zone_glare_pct']:.2f}%  "
          f"({d_glare:+.2f} pp)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def verify(binary, per_slot, room, wea_label, out_dir, opt=None,
           n_conditions=32, workers=None, dpi=150, tag=""):
    """Re-simulate `binary` in Radiance and compare against the surrogate and the targets.

    binary    : (4,16,32) 1 = WINDOW
    per_slot  : list of dicts with 'hour', 'grid' (32,32 target lux), 'cond_full' (T,8)
    room      : dict(width, depth, height, north_angle)
    Returns the report dict; writes verification_report.json and the two figures.
    """
    gen = _load_generator()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if opt is None:
        opt = FacadeOptimizerAllSidesClimate()

    W, D, H = room["width"], room["depth"], room["height"]
    facades = binary_to_scene_facades(binary)
    workers = int(workers or max(1, (multiprocessing.cpu_count() or 2) - 2))

    # ── pick conditions: evenly spaced through the year so seasons are covered ──
    tasks, index = [], []
    for si, ps in enumerate(per_slot):
        cf = np.asarray(ps["cond_full"], dtype=np.float32)
        T = len(cf)
        if T == 0:
            continue
        k = min(int(n_conditions), T)
        picks = np.unique(np.linspace(0, T - 1, k).round().astype(int))
        for j in picks:
            alt, az, dni, dhi = cond_to_sun(cf[j])
            key = (si, int(j))
            tasks.append((key, alt, az, dni, dhi))
            index.append(key)

    print(f"\n  Radiance verification: {len(tasks)} simulations on {workers} workers")
    print(f"    (isolated room, {gen.RAD_PARAMS})", flush=True)

    results = {}
    if workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=workers, initializer=_init_worker,
                                  initargs=(str(GEN_PY), facades, (W, D, H))) as pool:
            done = 0
            for key, lux in pool.imap_unordered(_simulate, tasks, chunksize=1):
                results[key] = lux
                done += 1
                if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                    print(f"    {done}/{len(tasks)}", flush=True)
    else:
        _init_worker(str(GEN_PY), facades, (W, D, H))
        for t in tasks:
            key, lux = _simulate(t)
            results[key] = lux

    n_failed = sum(1 for v in results.values() if v is None)

    # ── assemble aligned Radiance / surrogate stacks per slot ─────────────────
    slots_report, gal_rad, gal_sur, gal_slots, day_labels = [], [], [], [], []
    pool_rad, pool_sur, pool_grid, zone_err_parts = [], [], [], []

    for si, ps in enumerate(per_slot):
        cf = np.asarray(ps["cond_full"], dtype=np.float32)
        keep = [j for (s, j) in index if s == si and results.get((s, j)) is not None]
        keep.sort()
        if not keep:
            print(f"  [!] {ps['hour']:02d}:00 — every Radiance run failed; slot skipped")
            continue
        rad = np.stack([sensors_to_plan(results[(si, j)]) for j in keep])   # (T,32,32)
        sur = opt.predict_lux(binary, cf[keep])                            # (T,32,32)
        grid = np.asarray(ps["grid"], dtype=np.float32)

        m_rad = zone_metrics(rad, grid)
        m_sur = zone_metrics(sur, grid)
        slots_report.append(dict(
            hour=int(ps["hour"]), n_conditions=len(keep),
            occupied_cells=m_rad["occupied_cells"],
            target_lux_mean=m_rad["target_lux_mean"],
            radiance_vs_target=m_rad,
            surrogate_vs_target=m_sur,
            radiance_vs_surrogate=agreement(rad, sur, grid)))

        gal_rad.append(rad); gal_sur.append(sur)
        gal_slots.append(dict(hour=ps["hour"], grid=grid))
        days = ps.get("days")
        day_labels.append({i: (f"{days[j][0]}/{days[j][1]}" if days else f"cond {j}")
                           for i, j in enumerate(keep)})
        pool_rad.append(rad); pool_sur.append(sur)
        pool_grid.append(np.broadcast_to(grid, rad.shape))
        # Pool the in-zone errors themselves rather than trying to pool a common mask —
        # slots legitimately have different zones (that is the whole point of multi-slot),
        # so there is usually no shared mask to pool.
        if (grid > 0).any():
            occ = grid > 0
            zone_err_parts.append((sur[:, occ] - rad[:, occ]).ravel())

    if not slots_report:
        raise RuntimeError("verification produced no usable slots — every Radiance run failed")

    PR = np.concatenate(pool_rad, 0)
    PS = np.concatenate(pool_sur, 0)
    PG = np.concatenate(pool_grid, 0)

    def _pooled(E):
        occ = PG > 0
        tgt = PG[occ]
        zv = E[occ]
        return dict(
            # cell-CONDITIONS, not cells: slots have different zones and different condition
            # counts, so a single "occupied cells" number would be meaningless here.
            occupied_cell_conditions=int(occ.sum()),
            target_lux_mean=float(tgt.mean()),
            zone_mean_lux=float(zv.mean()),
            zone_underlit_pct=float((zv < tgt * ZONE_MIN_RATIO).mean() * 100.0),
            zone_glare_pct=float((zv > GLARE_LUX).mean() * 100.0),
            floor_glare_pct=float((E > GLARE_LUX).mean() * 100.0),
            floor_mean_lux=float(E.mean()))

    ov_agree = agreement(PR, PS)
    if zone_err_parts:
        ze = np.concatenate(zone_err_parts)
        ov_agree["zone_rmse"] = float(np.sqrt((ze ** 2).mean()))
        ov_agree["zone_mae"] = float(np.abs(ze).mean())
        ov_agree["zone_bias"] = float(ze.mean())

    overall = dict(
        target_lux_mean=float(PG[PG > 0].mean()),
        radiance_vs_target=_pooled(PR),
        surrogate_vs_target=_pooled(PS),
        radiance_vs_surrogate=ov_agree)

    report = dict(
        schema="facade_verification/1",
        city=wea_label,
        room=dict(width=W, depth=D, height=H, north_angle=room.get("north_angle", 0.0)),
        glazing_pct=float(binary.mean() * 100.0),
        scene="isolated room — no exterior context, matching the training assumption",
        rad_params=gen.RAD_PARAMS,
        sensor_height_m=gen.SENSOR_HEIGHT,
        glare_lux=GLARE_LUX, zone_min_ratio=ZONE_MIN_RATIO,
        n_conditions=len(tasks), n_failed=n_failed,
        slots=slots_report, overall=overall)

    suffix = f"_{tag}" if tag else ""
    write_report(report, out_dir / f"verification_report{suffix}.json")
    _gallery(gal_rad, gal_slots, "RADIANCE (ground truth) — optimized facade",
             out_dir / f"verify_radiance{suffix}.png", day_labels, dpi=dpi)
    _gallery(gal_sur, gal_slots, "SURROGATE (prediction) — optimized facade",
             out_dir / f"verify_surrogate{suffix}.png", day_labels, dpi=dpi)
    print_report(report)
    print(f"\n  Written to {out_dir}")
    for f in (f"verification_report{suffix}.json", f"verify_radiance{suffix}.png",
              f"verify_surrogate{suffix}.png"):
        print(f"    {f}")
    return report


# ---------------------------------------------------------------------------
# Standalone: rebuild everything from an existing output directory
# ---------------------------------------------------------------------------

def load_binary_wide(path):
    """facade_wide.csv -> (4,16,32) with 1 = WINDOW."""
    lines = Path(path).read_text().strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    vals = [float(v) for v in lines[1].split(",")]
    col = dict(zip(header, vals))
    out = np.zeros((N_FACADES, FACADE_ROWS, FACADE_COLS), dtype=np.float32)
    for i, side in enumerate(SIDES):
        flat = np.array([col[f"{side}_Pix_{k}"] for k in range(N_CELLS)], dtype=np.float32)
        out[i] = flat.reshape(FACADE_ROWS, FACADE_COLS)
    return out


def from_output_dir(out_dir, n_conditions=32, workers=None, dpi=150):
    out_dir = Path(out_dir)
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    job = res["job"]
    room = res["resolved"]["room"]
    na = float(res["resolved"].get("north_angle", 0.0))
    binary = load_binary_wide(out_dir / "facade_wide.csv")

    wea_path = resolve_city(job["climate"]["city"])
    per_slot = []
    for s in job["slots"]:
        grid = np.asarray(s["grid"], dtype=np.float32).reshape(GRID_N, GRID_N)
        _, _, cond5, days = annual_conditions(wea_path, int(s["hour"]), north_angle=na)
        if len(cond5) == 0:
            continue
        per_slot.append(dict(hour=int(s["hour"]), grid=grid, days=days,
                             cond_full=append_room(cond5, room["width"], room["depth"],
                                                   room["height"])))
    if not per_slot:
        raise RuntimeError("no usable slots — sun never above horizon at the job's hours")

    return verify(binary, per_slot,
                  dict(width=room["width"], depth=room["depth"], height=room["height"],
                       north_angle=na),
                  city_label(wea_path), out_dir,
                  n_conditions=n_conditions, workers=workers, dpi=dpi)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-simulate an optimized facade in Radiance and compare against the "
                    "surrogate and the design targets.")
    ap.add_argument("out_dir", help="an output directory containing result.json + facade_wide.csv")
    ap.add_argument("--conditions", type=int, default=32,
                    help="Radiance runs per time slot, spread evenly through the year "
                         "(default 32)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: cpu_count - 2)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)
    from_output_dir(args.out_dir, n_conditions=args.conditions,
                    workers=args.workers, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
