"""
seed_sweep.py — is the facade optimizer finding A minimum, or THE minimum?

Runs the same job N times changing ONLY torch.manual_seed (which drives the random init of
the facade logits) and reports three separate things, because they answer different questions
and can disagree:

  1. LOSS SPREAD      — do the seeds land on equally good solutions?
  2. LAYOUT AGREEMENT — do they land on the SAME solution?  (pairwise window IoU)
  3. DAYLIGHT SPREAD  — does any of it matter in lux terms?

The interesting outcome is 1-and-3 tight while 2 is loose: that means the problem is
DEGENERATE — many different facades are equally good — which is not a failure at all. It
means the optimizer is fine and you have design freedom. The failure case is 1 or 3 being
loose, which is genuine local-minimum trapping.

Also reports BEST-OF-N vs a single seed, i.e. what a one-shot run leaves on the table.

Everything except the seed is held identical: same job file, same k-means-reduced conditions
(reduce_conditions has its own fixed internal seed), same schedule.

Run:
    cd "allside_Generalised model/model trainer"
    python seed_sweep.py ../rhino\\ bridge/example_job.json --seeds 0 1 2 3 4 5
    python seed_sweep.py <job.json> --seeds 0 1 2 --repr 16   # faster, coarser
"""

import sys
import json
import time
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_facade_allsides_climate import (      # noqa: E402
    FacadeOptimizerAllSidesClimate, resolve_city, city_label,
    zone_penalty_sum, outside_glare_sum,
    GRID_N, GLARE_LUX, ZONE_MIN_RATIO, SIDES, FACADE_ROWS, FACADE_COLS)
from run_facade_job import load_job, build_conditions       # noqa: E402


def deployed_loss(opt, binary, cond, tgt, w, glare_weight, glare_cap):
    """The exact objective the optimizer minimises, evaluated on a finished binary design.

    Recomputed here rather than scraped from stdout: optimize_coarse_to_fine() prints its
    best binarised loss but does not return it, and a number parsed out of a log is a number
    that silently changes when the log format does.
    """
    E = torch.tensor(opt.predict_lux(binary, cond), dtype=torch.float32)
    t = torch.tensor(tgt, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    n_occ = (wt.view(-1, 1, 1) * (t > 0).float()).sum().clamp(min=1.0)
    n_un = (wt.view(-1, 1, 1) * (t <= 0).float()).sum().clamp(min=1.0)
    L = float(zone_penalty_sum(E, t, wt).item()) / float(n_occ)
    if glare_weight:
        L += glare_weight * float(outside_glare_sum(E, t, glare_cap, wt).item()) / float(n_un)
    return L


def daylight_metrics(opt, binary, per_slot):
    """Zone mean / underlit / glare over the FULL year — what the design actually delivers."""
    zv, zu, glare_hits, cells = [], [], 0, 0
    for ps in per_slot:
        E = opt.predict_lux(binary, ps["cond_full"])
        occ = ps["grid"] > 0
        tgt = ps["grid"][occ]
        v = E[:, occ]
        zv.append(v.ravel())
        zu.append((v < tgt * ZONE_MIN_RATIO).ravel())
        glare_hits += int((E > GLARE_LUX).sum()); cells += int(E.size)
    zv = np.concatenate(zv); zu = np.concatenate(zu)
    return dict(zone_mean=float(zv.mean()),
                underlit=float(zu.mean() * 100),
                zone_glare=float((zv > GLARE_LUX).mean() * 100),
                floor_glare=float(glare_hits / max(cells, 1) * 100))


def iou(a, b):
    """Window intersection-over-union between two designs — 1.0 = identical glazing.

    Returns None when NEITHER design has a window here, because the honest answer is "not
    applicable", not "perfect agreement".  Reporting 1.0 for an empty union is actively
    misleading: a wall that is frozen solid by `sides`, or that every seed independently
    chose to leave blank, would masquerade as the best-converged facade in the sweep.
    """
    inter = int(((a > 0) & (b > 0)).sum())
    union = int(((a > 0) | (b > 0)).sum())
    return (inter / union) if union else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seed-convergence sweep for the all-sides optimizer.")
    ap.add_argument("job")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--repr", type=int, default=None,
                    help="override optimizer.repr_conditions (fewer = faster, coarser)")
    ap.add_argument("--out", default=None, help="write the raw results as JSON")
    args = ap.parse_args(argv)

    job = load_job(args.job)
    if args.repr:
        job["repr_conditions"] = args.repr
    wea = resolve_city(job["city"])
    cond, tgt, w, per_slot = build_conditions(job, wea)

    n_cells = 4 * FACADE_ROWS * FACADE_COLS
    print(f"Job   : {Path(args.job).name}")
    print(f"City  : {city_label(wea)}   room "
          f"{job['width']:.2f}x{job['depth']:.2f}x{job['height']:.2f} m   sides {job['sides']}")
    print(f"Slots : {', '.join('%02d:00' % p['hour'] for p in per_slot)}   "
          f"-> {len(cond)} weighted conditions")
    print(f"Seeds : {args.seeds}\n")

    opt = FacadeOptimizerAllSidesClimate()
    runs = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"--- seed {seed}  ({i+1}/{len(args.seeds)}) " + "-" * 40, flush=True)
        torch.manual_seed(seed)
        binary, _, _ = opt.optimize_coarse_to_fine(
            cond, tgt, w, job["facade_open"],
            glare_weight=job["glare_weight"], glare_cap=job["glare_cap"],
            log_every=10 ** 9, verbose=False)          # quiet: the sweep prints its own table
        L = deployed_loss(opt, binary, cond, tgt, w, job["glare_weight"], job["glare_cap"])
        m = daylight_metrics(opt, binary, per_slot)
        runs.append(dict(seed=seed, loss=L, binary=binary,
                         glazing=float(binary.mean() * 100), secs=time.time() - t0, **m))
        print(f"    loss {L:.5f}   glazing {runs[-1]['glazing']:5.2f}%   "
              f"zone {m['zone_mean']:6.0f} lux   underlit {m['underlit']:5.2f}%   "
              f"({runs[-1]['secs']:.0f}s)", flush=True)

    # ---------------- report ----------------
    def spread(key):
        v = np.array([r[key] for r in runs], dtype=float)
        rng = v.max() - v.min()
        return v, v.mean(), v.std(), rng, (rng / v.mean() * 100 if v.mean() else 0.0)

    print("\n" + "=" * 78)
    print("PER-SEED RESULTS")
    print("=" * 78)
    print(f"  {'seed':>5}{'loss':>11}{'glazing%':>10}{'zone lux':>10}"
          f"{'underlit%':>11}{'floorglare%':>13}")
    for r in sorted(runs, key=lambda r: r["loss"]):
        print(f"  {r['seed']:>5}{r['loss']:>11.5f}{r['glazing']:>10.2f}"
              f"{r['zone_mean']:>10.0f}{r['underlit']:>11.2f}{r['floor_glare']:>13.2f}")

    print("\n" + "=" * 78)
    print("1. LOSS SPREAD — are the seeds equally good?")
    print("=" * 78)
    v, mu, sd, rng, rel = spread("loss")
    print(f"  best {v.min():.5f}   worst {v.max():.5f}   mean {mu:.5f}   sd {sd:.5f}")
    print(f"  range = {rng:.5f}  ({rel:.1f}% of mean)")
    print(f"  worst seed is {(v.max()/v.min() - 1) * 100:.1f}% worse than the best")

    print("\n" + "=" * 78)
    print("2. LAYOUT AGREEMENT — is it the SAME design each time?")
    print("=" * 78)
    pairs = list(itertools.combinations(range(len(runs)), 2))
    ious = [v for v in (iou(runs[a]["binary"], runs[b]["binary"]) for a, b in pairs)
            if v is not None]
    diffs = [int((runs[a]["binary"] != runs[b]["binary"]).sum()) for a, b in pairs]
    print(f"  pairwise window IoU : mean {np.mean(ious)*100:5.1f}%   "
          f"min {min(ious)*100:5.1f}%   max {max(ious)*100:5.1f}%")
    print(f"  cells differing     : mean {np.mean(diffs):5.0f} / {n_cells}  "
          f"({np.mean(diffs)/n_cells*100:.1f}% of the facade)")

    # Per side, report the WINDOW COUNT range alongside the IoU.  A side every seed leaves
    # blank is a real and interesting agreement ("don't glaze this wall"), but it is an
    # agreement about absence — not about a shared pattern — and must not be scored as 100%.
    print(f"\n  {'side':>6}{'windows/512 (min-max)':>24}{'IoU':>10}   note")
    for si, s in enumerate(SIDES):
        counts = [int(r["binary"][si].sum()) for r in runs]
        vals = [v for v in (iou(runs[a]["binary"][si], runs[b]["binary"][si])
                            for a, b in pairs) if v is not None]
        if not vals:
            note = ("frozen solid by `sides`" if s not in job["sides"]
                    else "every seed left it blank — agreement on ABSENCE")
            print(f"  {s:>6}{'0':>24}{'n/a':>10}   {note}")
        else:
            note = ("converged" if np.mean(vals) >= 0.90 else
                    "partly agreed" if np.mean(vals) >= 0.50 else "essentially arbitrary")
            print(f"  {s:>6}{'%d-%d' % (min(counts), max(counts)):>24}"
                  f"{np.mean(vals)*100:9.0f}%   {note}")

    print("\n" + "=" * 78)
    print("3. DAYLIGHT SPREAD — does the difference matter?")
    print("=" * 78)
    for key, lbl, unit in (("zone_mean", "zone mean lux", "lux"),
                           ("underlit", "underlit", "pts"),
                           ("floor_glare", "floor glare", "pts"),
                           ("glazing", "glazing", "pts")):
        v, mu, sd, rng, rel = spread(key)
        print(f"  {lbl:<16} {mu:8.1f} +/- {sd:6.2f}   range {rng:7.2f} {unit}"
              f"   ({rel:5.1f}% of mean)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    lv = np.array([r["loss"] for r in runs])
    zv = np.array([r["zone_mean"] for r in runs])
    uv = np.array([r["underlit"] for r in runs])

    # BOTH criteria matter, but the loss one has to be SCALE-AWARE.  The loss is a
    # normalised penalty with a floor of 0, so once every seed essentially satisfies the
    # brief it sits around 0.005-0.02 and relative differences explode into meaningless
    # percentages: 0.004 vs 0.010 reads as "150% worse" while delivering the same lux to
    # within 1%.  So a small ABSOLUTE loss counts as tight regardless of the ratio — that is
    # the "several good-enough solutions" case, not a convergence failure.  Relative loss
    # only carries information while the loss is still large.
    LOSS_FLOOR = 0.05
    loss_rel = lv.max() / lv.min() - 1
    loss_tight = (loss_rel < 0.15) or (lv.max() < LOSS_FLOOR)
    lux_tight = (zv.max() - zv.min()) / zv.mean() < 0.10   # <10% spread in delivered lux
    und_tight = (uv.max() - uv.min()) < 2.0                # <2 percentage points underlit
    perf_tight = lux_tight and und_tight
    layout_same = np.mean(ious) >= 0.90

    if loss_tight and perf_tight and layout_same:
        v = "CONVERGED — same design, same performance. The optimum looks unique."
    elif loss_tight and perf_tight:
        v = ("DEGENERATE OPTIMUM — different layouts, equivalent performance.\n"
             "  The optimizer is working; the problem simply has many equally good answers.\n"
             "  Practical consequence: pick any seed, or generate several and choose on\n"
             "  grounds the loss does not model (buildability, view, structure).")
    else:
        v = ("LOCAL MINIMA MATTER — seeds differ in PERFORMANCE, not just layout.\n"
             "  A single run is a lottery; run several and keep the best.")
    print("  " + v)

    print(f"\n  loss   : relative spread {loss_rel*100:5.1f}%  |  absolute max {lv.max():.5f}  "
          f"-> {'tight' if loss_tight else 'LOOSE'}")
    if lv.max() < LOSS_FLOOR and loss_rel >= 0.15:
        print(f"           every seed finished under {LOSS_FLOOR}, so the brief is essentially met\n"
              f"           in all of them and the relative figure is a near-zero artefact.")
    print(f"  daylight: zone lux range {zv.max()-zv.min():.0f} lux, underlit range "
          f"{uv.max()-uv.min():.2f} pts  -> {'tight' if perf_tight else 'LOOSE'}")
    print(f"  layout  : mean IoU {np.mean(ious)*100:.1f}%  "
          f"-> {'same design' if layout_same else 'DIFFERENT designs'}")

    best, worst = min(runs, key=lambda r: r["loss"]), max(runs, key=lambda r: r["loss"])
    print(f"\n  best-of-{len(runs)} vs worst single run:")
    print(f"    loss      {best['loss']:.5f}  vs {worst['loss']:.5f}")
    print(f"    zone lux  {best['zone_mean']:.0f}      vs {worst['zone_mean']:.0f}")
    print(f"    underlit  {best['underlit']:.2f}%     vs {worst['underlit']:.2f}%")
    print(f"\n  mean runtime {np.mean([r['secs'] for r in runs]):.0f}s per seed")

    if args.out:
        Path(args.out).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "binary"} for r in runs], indent=2))
        print(f"\n  raw results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
