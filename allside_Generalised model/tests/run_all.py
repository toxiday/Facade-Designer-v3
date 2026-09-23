"""
run_all.py — run every automated check in the repo and print one PASS/FAIL table.

    python run_all.py            # the standard set (~10 min on an RTX-class GPU)
    python run_all.py --full     # also evaluates on the 400k dataset if it is present

Each suite runs as its own process with this interpreter, so an import-order accident in one
cannot hide or cause a failure in another.  Scratch outputs go to a temp folder; nothing is
written into the repo.  Exit code is non-zero if anything failed.

Suites that need data the repository does not ship (the 400k dataset) are SKIPPED when it is
absent, so this runs unchanged on a fresh clone with only the release weights added.
"""

import sys
import json
import time
import argparse
import tempfile
import subprocess
from pathlib import Path

HERE    = Path(__file__).resolve().parent
AGM     = HERE.parent
REPO    = AGM.parent
TRAINER = AGM / "model trainer"
RESP    = AGM / "responsive facade"
BRIDGE  = AGM / "rhino bridge"
ABL     = REPO / "evidence" / "ablation"
DATA400 = AGM / "datasets" / "dataset_allsides_climate_genroom_400k.csv"
STRESS  = AGM / "datasets" / "dataset_allsides_climate_genroom_stresstest.csv"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--full", action="store_true", help="also run evaluate32x32.py on the 400k dataset")
args = ap.parse_args()

SCRATCH = Path(tempfile.mkdtemp(prefix="facade_run_all_"))
RESULTS = []                                      # (name, status, seconds, note)


def run(name, argv, cwd, timeout=1800, expect=None):
    """Run one suite.  PASS = exit code 0 and, if given, every `expect` string in stdout."""
    print(f"\n{'=' * 78}\n{name}\n  $ {' '.join(str(a) for a in argv)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, *map(str, argv)], cwd=str(cwd), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out = p.stdout + p.stderr
        ok = p.returncode == 0 and all(e in out for e in (expect or []))
        note = "" if ok else (f"exit {p.returncode}" if p.returncode else "expected output missing")
    except subprocess.TimeoutExpired as e:
        out, ok, note = (e.stdout or "") if isinstance(e.stdout, str) else "", False, f"timeout {timeout}s"
    dt = time.time() - t0
    tail = out.strip().splitlines()[-25:]
    print("\n".join("  | " + ln for ln in tail))
    RESULTS.append((name, "PASS" if ok else "FAIL", dt, note))
    return ok, out


def skip(name, why):
    print(f"\n{'=' * 78}\n{name}\n  SKIP: {why}\n{'=' * 78}", flush=True)
    RESULTS.append((name, "SKIP", 0.0, why))


# ---- unit + integration suites ------------------------------------------------------------
run("integration: components agree with each other", [HERE / "test_integration.py"], HERE,
    expect=["all integration checks passed"])
run("responsive facade: live engine (--gpu)", [RESP / "test_live_solver.py", "--gpu"], RESP)
run("responsive facade: app UI (Agg)", [RESP / "test_app.py"], RESP)
run("designer GUI, headless, real solve", [HERE / "test_designer_headless.py", "--out", SCRATCH / "designer"],
    HERE, expect=["designer headless test passed"])

# ---- end to end: a job file through the runner, compared with its committed reference ------
job = BRIDGE / "example_job.json"
out = SCRATCH / "example_job"
ok, _ = run("job runner: rhino bridge/example_job.json", [TRAINER / "run_facade_job.py", job, "--out", out],
            TRAINER)
if ok:
    ref = json.loads((AGM / "outputs" / "example_job" / "result.json").read_text(encoding="utf-8"))["metrics"]
    new = json.loads((out / "result.json").read_text(encoding="utf-8"))["metrics"]
    d = {k: new[k] - ref[k] for k in ("glazing_pct", "zone_mean_lux", "zone_underlit_pct", "zone_glare_pct")}
    within = (abs(d["glazing_pct"]) <= 2.0 and abs(d["zone_mean_lux"]) <= 0.05 * ref["zone_mean_lux"]
              and abs(d["zone_underlit_pct"]) <= 3.0 and abs(d["zone_glare_pct"]) <= 3.0)
    import importlib.util
    spec = importlib.util.spec_from_file_location("_v", str(TRAINER / "verify_facade_radiance.py"))
    sys.path.insert(0, str(TRAINER))
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    a = v.load_binary_wide(AGM / "outputs" / "example_job" / "facade_wide.csv")
    b = v.load_binary_wide(out / "facade_wide.csv")
    agree = float((a == b).mean() * 100)
    note = (f"glazing {d['glazing_pct']:+.2f}pp  zone lux {d['zone_mean_lux']:+.1f}  "
            f"underlit {d['zone_underlit_pct']:+.2f}pp  glare {d['zone_glare_pct']:+.2f}pp  "
            f"panels identical {agree:.1f}%")
    print(f"\n  vs committed reference: {note}")
    RESULTS.append(("  ...matches committed reference (tolerance)", "PASS" if within else "FAIL", 0.0, note))

# ---- apps and utilities --------------------------------------------------------------------
run("responsive app: --screenshot solve", [RESP / "responsive_facade.py", "--screenshot",
                                           SCRATCH / "responsive.png"], RESP, expect=["screenshot ->"])
run("rhino bridge: check_sun_coverage (36/36 bins)", [BRIDGE / "check_sun_coverage.py"], BRIDGE,
    expect=["No azimuth hole"])
run("rhino bridge: verify_rotation (north angle sign)", [BRIDGE / "verify_rotation.py"], TRAINER)
run("evidence: ablation parameter counts", [ABL / "ablation_architecture.py", "--params-only"], ABL)
run("evidence: ablation report from saved results", [ABL / "ablation_architecture.py", "--report"], ABL)
run("evaluate32x32 smoke (stress-test CSV)", [TRAINER / "evaluate32x32.py", "--csv", STRESS], TRAINER,
    expect=["SSIM"])
if args.full:
    if DATA400.is_file():
        run("evaluate32x32 on the 400k dataset (60k held-out)", [TRAINER / "evaluate32x32.py"], TRAINER,
            timeout=7200, expect=["SSIM"])
    else:
        skip("evaluate32x32 on the 400k dataset", "dataset not present (not shipped in git)")

# ---- summary -------------------------------------------------------------------------------
print(f"\n\n{'=' * 78}\nSUMMARY   (scratch: {SCRATCH})\n{'=' * 78}")
for name, status, dt, note in RESULTS:
    print(f"  {status:<5} {dt:6.0f}s  {name}" + (f"\n{'':15}{note}" if note else ""))
n_fail = sum(1 for r in RESULTS if r[1] == "FAIL")
print(f"\n  {len(RESULTS) - n_fail} ok / {n_fail} failed")
sys.exit(1 if n_fail else 0)
