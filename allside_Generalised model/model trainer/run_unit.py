"""
run_unit.py — drive every room of an apartment unit through the facade optimizer in one go.

A unit folder ("All Units/101/") holds one facade_job/1 JSON per room.  Running them by hand
is how a unit's outputs drift out of sync with its inputs — a room's `sides` gets edited and
its design silently keeps describing the old wall.  This runs the whole set, reports which
rooms actually changed, and writes a per-unit summary table.

    python run_unit.py "../All Units/101"
    python run_unit.py "../All Units/101" --only 101.6 101.7 --no-png
    python run_unit.py "../All Units/101" --only 101.6 --verify

It adds NO optimization logic.  Each room goes through run_facade_job.load_job() and run()
exactly as a single Rhino job would, so a batch design and a hand-run design are the same
design.  Per-room outputs land in their usual outputs/<name>/ directories; only the summary
is new.

One room's failure is recorded and the batch continues — a malformed JSON must not cost six
good designs.
"""

import re
import csv
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_facade_job import load_job, run, JobError               # noqa: E402

SUMMARY_COLUMNS = [
    "room", "width_m", "depth_m", "height_m", "sides", "north_angle_deg",
    "n_slots", "hours", "glazing_pct", "zone_mean_lux", "zone_underlit_pct",
    "zone_glare_pct", "floor_glare_pct", "seconds", "out_dir", "status",
]


def natural_key(path):
    """Sort 101.10 after 101.9, not after 101.1 — a plain string sort reorders a unit of ten
    rooms, and the summary's row order is how the unit gets read."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def is_job_file(path):
    """Is this a facade_job/1, or some other JSON that happens to live in the folder?

    Filtered by SCHEMA rather than by filename.  A unit folder holds `unit.json` beside the
    room jobs, and matching on the name alone would keep working only until the next
    non-job file appears — the schema is the thing that actually decides.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            head = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False, "not readable as JSON"
    if not isinstance(head, dict):
        return False, "not a JSON object"
    schema = head.get("schema", "facade_job/1")
    if schema != "facade_job/1":
        return False, f"schema {schema!r}"
    return True, ""


def discover_jobs(unit_dir, only=None):
    unit_dir = Path(unit_dir)
    if not unit_dir.is_dir():
        raise SystemExit(f"unit folder not found: {unit_dir}")

    paths, skipped = [], []
    for p in sorted(unit_dir.glob("*.json"), key=natural_key):
        ok, why = is_job_file(p)
        (paths if ok else skipped).append(p if ok else (p.name, why))
    for name, why in skipped:
        print(f"  skipping {name} — {why}")

    if only:
        want    = {o.removesuffix(".json") for o in only}
        missing = want - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"--only names no job in {unit_dir}: {sorted(missing)}")
        paths = [p for p in paths if p.stem in want]
    if not paths:
        raise SystemExit(f"no job JSONs in {unit_dir}")
    return paths


def summarize(path, job, out_dir, seconds, status):
    """One summary row.  Metrics are read back from the result.json that run() just wrote
    rather than recomputed here — there is then exactly one source for them, and the summary
    cannot quietly disagree with the per-room file."""
    row = {c: "" for c in SUMMARY_COLUMNS}
    row.update(room=path.stem, status=status, seconds=f"{seconds:.1f}",
               out_dir=str(out_dir) if out_dir else "")
    if job:
        row.update(width_m=f"{job['width']:.3f}", depth_m=f"{job['depth']:.3f}",
                   height_m=f"{job['height']:.3f}", sides=job["sides"],
                   north_angle_deg=f"{job['north_angle']:g}",
                   n_slots=len(job["slots"]),
                   hours=" ".join(f"{s['hour']:02d}" for s in job["slots"]))
    rj = Path(out_dir) / "result.json" if out_dir else None
    if rj and rj.is_file():
        m = json.loads(rj.read_text(encoding="utf-8")).get("metrics", {})
        for k in ("glazing_pct", "zone_mean_lux", "zone_underlit_pct",
                  "zone_glare_pct", "floor_glare_pct"):
            if k in m:
                row[k] = f"{m[k]:.2f}"
    return row


def unit_name_for(unit_dir):
    """The unit's name, from its manifest if there is one.

    NOT the folder name: the jobs live in a folder called `output` under `Units/201/`, so
    the containing directory is named for its role and the header would read "Unit output".
    Duplicated from unit_zone_matrix rather than imported — that module pulls in matplotlib,
    which is not needed to optimize.
    """
    unit_dir = Path(unit_dir)
    man = unit_dir / "unit.json"
    if man.is_file():
        try:
            name = json.loads(man.read_text(encoding="utf-8")).get("name")
            if name:
                return str(name)
        except (OSError, json.JSONDecodeError):
            pass
    d = unit_dir.resolve()
    if d.name.lower() in ("output", "outputs", "jobs") and d.parent != d:
        return d.parent.name
    return d.name


def draw_brief(unit_dir):
    """Render the occupancy heat maps BEFORE optimizing.

    The heat map is the picture of the brief, and the brief is the thing worth checking
    while it is still cheap: an hour of GPU spent on zones wired to the wrong room is an
    hour wasted, and the mistake is obvious at a glance in the figure.  Drawing it first
    also means the file on disk always matches the jobs that were actually run.

    Never fatal.  A missing matplotlib, a missing unit.json, a bad manifest — none of that
    should stop a unit from being optimized, so every failure is reported and stepped over.
    """
    try:
        import unit_zone_matrix as uzm          # imported here: matplotlib is not needed
    except ImportError as e:                    # to optimize, only to draw
        print(f"  [!] heat map skipped — {e}", flush=True)
        return

    for mode in ("room", "unit"):
        try:
            uzm.main([str(unit_dir), "--by", mode])
        except SystemExit as e:                 # uzm.main exits on a missing/bad unit.json
            if str(e) not in ("0", "None"):
                print(f"  [!] heat map --by {mode} skipped: {e}", flush=True)
        except Exception as e:                  # noqa: BLE001 — a picture must not block a run
            print(f"  [!] heat map --by {mode} failed: {type(e).__name__}: {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run every facade_job/1 in a unit folder through the optimizer.")
    ap.add_argument("unit", help="folder of job JSONs, e.g. \"../All Units/101\"")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these room ids, e.g. --only 101.6 101.7")
    ap.add_argument("--out-root", default=None,
                    help="write outputs to <out-root>/<room>/ instead of each job's own "
                         "output.name directory")
    ap.add_argument("--summary", default=None,
                    help="path for the summary CSV (default: unit_summary.csv beside the "
                         "job files)")
    ap.add_argument("--no-png", action="store_true", help="skip each room's result.png")
    ap.add_argument("--no-matrix", action="store_true",
                    help="skip the occupancy heat maps drawn before optimizing")
    ap.add_argument("--seed", type=int, default=None,
                    help="override every job's optimizer.seed")
    ap.add_argument("--verify", action="store_true",
                    help="re-simulate each finished design in Radiance (slow)")
    ap.add_argument("--verify-conditions", type=int, default=32)
    ap.add_argument("--verify-workers", type=int, default=None)
    args = ap.parse_args(argv)

    unit_dir  = Path(args.unit)
    unit_name = unit_name_for(unit_dir)
    paths     = discover_jobs(unit_dir, only=args.only)

    print(f"\n{'=' * 78}\nUnit {unit_name} — {len(paths)} room(s): "
          f"{', '.join(p.stem for p in paths)}\n{'=' * 78}", flush=True)

    # The brief, drawn before a single second of GPU is spent on it.
    if not args.no_matrix:
        print(f"\n{'-' * 78}\nOccupancy heat map — check this before the run gets "
              f"expensive\n{'-' * 78}", flush=True)
        draw_brief(unit_dir)

    rows, failures = [], []
    t_unit = time.time()
    for i, path in enumerate(paths, 1):
        print(f"\n{'-' * 78}\n[{i}/{len(paths)}] {path.stem}\n{'-' * 78}", flush=True)
        t0, job, out_dir = time.time(), None, None
        try:
            job = load_job(path)
            if args.seed is not None:
                job["seed"] = args.seed
            if args.out_root:
                out_dir = Path(args.out_root) / job["out_name"]
            out_dir = run(job, out_dir=out_dir, no_png=args.no_png,
                          verify=args.verify, verify_conditions=args.verify_conditions,
                          verify_workers=args.verify_workers)
            status = "ok"
        except JobError as e:
            status = f"rejected: {e}"
            print(f"\n  Job rejected: {e}", file=sys.stderr, flush=True)
            failures.append((path.stem, str(e)))
        except Exception as e:                       # noqa: BLE001 — one room must not end the batch
            status = f"failed: {type(e).__name__}: {e}"
            traceback.print_exc()
            failures.append((path.stem, f"{type(e).__name__}: {e}"))
        rows.append(summarize(path, job, out_dir, time.time() - t0, status))

    # ---- summary ----
    # Beside the jobs, not in a central outputs/ tree: the summary describes THIS unit, and
    # a unit reads better as one folder holding its brief, its designs and its summary.
    summary = Path(args.summary) if args.summary else (unit_dir / "unit_summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 78}\nUnit {unit_name} — summary  "
          f"({time.time() - t_unit:.0f}s total)\n{'=' * 78}")
    hdr = (f"{'room':<9}{'W x D':>14}{'sides':>7}{'slots':>7}{'glaz%':>8}"
           f"{'zone lux':>10}{'underlit%':>11}{'zglare%':>9}{'fglare%':>9}{'sec':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['room']:<9}{'':>14}{'':>7}{'':>7}   {r['status'][:60]}")
            continue
        print(f"{r['room']:<9}"
              f"{r['width_m'] + ' x ' + r['depth_m']:>14}"
              f"{r['sides']:>7}{r['n_slots']:>7}"
              f"{r['glazing_pct']:>8}{r['zone_mean_lux']:>10}"
              f"{r['zone_underlit_pct']:>11}{r['zone_glare_pct']:>9}"
              f"{r['floor_glare_pct']:>9}{r['seconds']:>7}")

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n  {n_ok}/{len(rows)} rooms optimized")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for name, msg in failures:
            print(f"    {name}: {msg}")
    print(f"  Summary: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
