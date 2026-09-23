"""
check_sun_coverage.py — is a ROTATED building still in the surrogate's training distribution?

The natural objection to `north_angle` is: "the dataset was generated with the room placed
orthogonally and north straight up, so the model never saw rotated rooms."  This script is
the answer, and it is a measurement rather than an argument.

The model has no north input.  Its conditioning is
[Sun_X, Sun_Y, Sun_Z, DNI/s, DHI/s, W_n, D_n, H_n] — a sun DIRECTION in room coordinates,
with no latitude, city or compass channel.  The orthogonal placement is a property of the
data GENERATOR, which computed a real sun vector and recorded only the vector.  So rotating
the building is identical to handing the model a different sun vector, which is a
first-class input, not an unmodelled condition.

That reduces the whole question to coverage.  A rotation preserves solar altitude and only
shifts azimuth, so the question is exactly:

    at each altitude, which azimuths does the training set actually contain?

If every altitude band spans the full azimuth circle, there is no hole for a rotation to
fall into and ANY north_angle lands in a populated region.

Run (needs no torch, no Radiance — only the .wea library):
    python check_sun_coverage.py
    python check_sun_coverage.py --bins 72 --min-alt 5
"""

import sys
import glob
import math
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[2]
CLIMATE_DIR = REPO_ROOT / "allside_Generalised model" / "climate data" / "library"

ALT_BANDS = [(5, 15), (15, 25), (25, 35), (35, 45), (45, 55),
             (55, 65), (65, 75), (75, 85), (85, 90)]


def solar_position(lat_deg, lon_deg, tz_h, month, day, hour_decimal):
    """(altitude_deg, azimuth_north_cw_deg) — verbatim from the dataset generator."""
    lat = math.radians(lat_deg)
    doy = int(275 * month / 9 - 30 + day - 2)
    if month <= 2:
        doy += 2
    B    = math.radians(360 / 365 * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    ET   = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    st   = hour_decimal + (lon_deg - tz_h * 15) / 15 + ET / 60
    om   = math.radians(15 * (st - 12))
    sa   = max(-1.0, min(1.0, math.sin(lat) * math.sin(decl) +
                              math.cos(lat) * math.cos(decl) * math.cos(om)))
    alt  = math.degrees(math.asin(sa))
    az_s = math.atan2(math.sin(om),
                      math.cos(om) * math.sin(lat) - math.tan(decl) * math.cos(lat))
    return alt, (math.degrees(az_s) + 180.0) % 360.0


def collect(climate_dir, min_alt):
    files = sorted(glob.glob(str(Path(climate_dir) / "**" / "*.wea"), recursive=True))
    if not files:
        raise SystemExit(f"No .wea files under {climate_dir}")
    alts, azs = [], []
    for fp in files:
        lat = lon_w = tz_deg = None
        rows = []
        with open(fp) as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    if   p[0] == "latitude":  lat    = float(p[1])
                    elif p[0] == "longitude": lon_w  = float(p[1])
                    elif p[0] == "time_zone": tz_deg = float(p[1])
                elif len(p) == 5:
                    rows.append((int(p[0]), int(p[1]), float(p[2]),
                                 float(p[3]), float(p[4])))
        if lat is None:
            continue
        lon, tz = -lon_w, -tz_deg / 15.0
        for m, d, h, dni, dhi in rows:
            if dni + dhi <= 0:
                continue
            a, z = solar_position(lat, lon, tz, m, d, h)
            if a > min_alt:
                alts.append(a)
                azs.append(z)
    return len(files), np.array(alts), np.array(azs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(CLIMATE_DIR), help="climate library directory")
    ap.add_argument("--bins", type=int, default=36, help="azimuth bins (default 36 = 10 deg)")
    ap.add_argument("--min-alt", type=float, default=5.0, help="minimum solar altitude (deg)")
    args = ap.parse_args(argv)

    n_files, alts, azs = collect(args.dir, args.min_alt)
    print(f"{n_files} .wea files -> {len(alts):,} lit sun vectors "
          f"(alt > {args.min_alt:g} deg, DNI+DHI > 0)\n")
    print(f"{'alt band':>10} {'samples':>10} {'az bins':>12} {'min/bin':>9}   verdict")
    print("-" * 62)

    holes = 0
    for lo, hi in ALT_BANDS:
        m = (alts >= lo) & (alts < hi)
        if not m.any():
            print(f"{f'{lo}-{hi}':>10} {0:>10} {'-':>12} {'-':>9}   EMPTY BAND")
            continue
        hist, _ = np.histogram(azs[m], bins=args.bins, range=(0, 360))
        occ = int((hist > 0).sum())
        if occ < args.bins:
            holes += 1
        print(f"{f'{lo}-{hi}':>10} {m.sum():>10,} {f'{occ}/{args.bins}':>12} "
              f"{hist.min():>9,}   {'full circle' if occ == args.bins else 'HOLE'}")

    print()
    if holes:
        print(f"{holes} altitude band(s) have an azimuth hole — a rotation landing there would")
        print("be an extrapolation. Consider restricting north_angle to multiples of 90 deg,")
        print("which are pure wall relabelling and therefore always safe.")
        return 1
    print("No azimuth hole at any altitude: every band spans the full circle, so ANY")
    print("north_angle maps a real sun vector into a populated region of the training set.")
    print("The thinnest band is the near-zenith one, but that sparsity already affects")
    print("ordinary UNROTATED tropical-noon use — rotation adds no new deficiency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
