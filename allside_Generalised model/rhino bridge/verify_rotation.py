"""
verify_rotation.py — does `north_angle` rotate the sun the RIGHT WAY?

This is the load-bearing test for the Rhino/Grasshopper interface.  An inverted north
convention does not crash and does not look wrong: it silently designs glazing for the
wall on the opposite side of the building.  This script makes that failure loud.

THE IDENTITY BEING TESTED
    Daylight is rotation-equivariant about the vertical axis, and multiples of 90 degrees
    are pure wall RELABELLING.  So "S wall glazed, north_angle=0" and "the wall that now
    faces true south, glazed, with the matching north_angle" must describe the same
    physical building and produce the same floor lux field, up to a rotation of the grid.

    north_angle | +Y faces | wall facing true south | facade cols | floor grid
    ------------+----------+------------------------+-------------+------------
          0     |  North   |  S (y=0)               | as-is       | as-is
        270     |  East    |  E (x=W)               | as-is       | rot90 k=1
        180     |  South   |  N (y=D)               | FLIPPED     | rot90 k=2
         90     |  West    |  W (x=0)               | FLIPPED     | rot90 k=3

    The column flips at 180/90 are not a fudge: those rotations reverse the direction the
    wall's local x/y axis runs, so col 0 lands at the far end.  Derive them from the frame
    mapping, do not guess them.

WHY THE TOLERANCE IS NOT ZERO
    The surrogate is a neural net with a SEPARATE side-type embedding per wall — it learned
    each wall from data rather than being architecturally equivariant, so it satisfies this
    identity only approximately.  To judge the residual fairly the script also measures the
    model's error on the horizontal-MIRROR symmetry, which it was explicitly trained on via
    flip augmentation.  If the rotation residual is the same order as the mirror residual,
    the rotation is as good as a symmetry the model was taught.

    An inverted sign, by contrast, is off by ~100x — that is the signal this test exists for.

Run:
    cd "allside_Generalised model/model trainer"
    python "../rhino bridge/verify_rotation.py"
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model trainer"))
import optimize_facade_allsides_climate as O      # noqa: E402

# (north_angle, wall that faces true south, flip its columns?, floor-grid rot90 k)
CASES = [(270, 'E', False, 1),
         (180, 'N', True,  2),
         (90,  'W', True,  3)]


def _facade(side, pattern):
    F = np.zeros((O.N_FACADES, O.FACADE_ROWS, O.FACADE_COLS), dtype=np.int32)
    F[O.SIDES.index(side)] = pattern
    return F


def _rel(a, b):
    """Mean absolute difference, relative to the reference's mean magnitude."""
    return float(np.abs(a - b).mean() / max(np.abs(a).mean(), 1e-9))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify the north_angle rotation sign.")
    ap.add_argument("--city", default="houston")
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--size", type=float, default=6.0,
                    help="square room side (m) — square so rotation cannot swap W/D")
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=0.15,
                    help="max acceptable relative residual (default 0.15)")
    args = ap.parse_args(argv)

    L, H = args.size, args.height
    wea = O.resolve_city(args.city)
    print(f"City {O.city_label(wea)}  |  {L:g}x{L:g}x{H:g} m  |  hour {args.hour}\n")

    rng = np.random.default_rng(7)
    pattern = (rng.random((O.FACADE_ROWS, O.FACADE_COLS)) < 0.25).astype(np.int32)

    def cond(na):
        _, _, c, _ = O.annual_conditions(wea, args.hour, north_angle=na)
        if len(c) == 0:
            raise SystemExit(f"Sun never above the horizon at hour {args.hour} here.")
        return O.append_room(c, L, L, H)

    opt = O.FacadeOptimizerAllSidesClimate()
    ref = opt.predict_lux(_facade('S', pattern), cond(0))     # S faces true south at na=0

    # Reference scale: the mirror symmetry the model WAS trained on (flip S/N cols,
    # negate Sun_X, flip floor cols).  This is the model's own equivariance error.
    cM = cond(0).copy()
    cM[:, 0] *= -1.0
    mirror = _rel(opt.predict_lux(_facade('S', np.fliplr(pattern)), cM), ref[:, :, ::-1])

    print(f"{'case':>22} {'mean lux':>10} {'residual':>10}   verdict")
    print("-" * 60)
    print(f"{'reference S @ na=0':>22} {ref.mean():>10.1f} {'-':>10}")
    print(f"{'mirror (trained sym)':>22} {'-':>10} {mirror:>10.4f}   model's own error\n")

    worst, ok = 0.0, True
    for na, side, flip, k in CASES:
        pat = np.fliplr(pattern) if flip else pattern
        E   = opt.predict_lux(_facade(side, pat), cond(na))
        r   = _rel(E, np.rot90(ref, k=k, axes=(1, 2)))
        worst = max(worst, r)
        good = r <= args.tol
        ok &= good
        print(f"{f'{side} @ na={na}':>22} {E.mean():>10.1f} {r:>10.4f}   "
              f"{'OK' if good else 'FAIL'}")

    # Control: the same geometry under the INVERTED sign must fail loudly, otherwise the
    # test above proves nothing (a no-op rotation would also 'pass' by symmetry).
    ctrl = _rel(opt.predict_lux(_facade('E', pattern), cond(90)),
                np.rot90(ref, k=1, axes=(1, 2)))
    print(f"{'control (inverted)':>22} {'-':>10} {ctrl:>10.4f}   "
          f"{'discriminates' if ctrl > 10 * worst else 'TEST IS BLIND'}")

    print()
    if ok and ctrl > 10 * worst:
        print(f"PASS — worst rotation residual {worst:.4f} vs mirror baseline {mirror:.4f}; "
              f"inverted sign is {ctrl / max(worst, 1e-9):.0f}x worse.")
        return 0
    print(f"FAIL — worst residual {worst:.4f} (tol {args.tol}), control {ctrl:.4f}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
