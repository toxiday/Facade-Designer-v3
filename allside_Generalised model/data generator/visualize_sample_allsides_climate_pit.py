"""
visualize_sample_allsides_climate_pit.py  (all-sides + generalized room + climate)
Render the first N examples of a generated all-sides climate point-in-time dataset
(dataset_allsides_climate_genroom_*.csv from generate_dataset_allsides_climate_pit.py)
as a gallery, one row per sample:

    [ S facade ] [ N facade ] [ E facade ] [ W facade ] [ Ground-truth floor lux ]

The lux values are read straight from the CSV (no Radiance re-run), so this works on
any machine without a Radiance install — it verifies the dataset's orientation, the
<0.5 = window convention, and the four-facade + variable-room geometry at a glance.

Panels are drawn at PHYSICAL aspect ratios so the sampled room proportions are visible:
    S / N facades : aspect = H / W   (col 0 = West,  row 0 = ceiling)
    E / W facades : aspect = H / D   (col 0 = South, row 0 = ceiling)
    floor plan    : aspect = D / W   (N up, W left) via rot90(k=1) + fliplr, matching
                    the model_allsides lux transform.

Run:
    python visualize_sample_allsides_climate_pit.py [--csv PATH] [--n 5]
"""

import csv, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[2]
FOLDER          = "allside_Generalised model"
DEFAULT_CSV     = REPO_ROOT / FOLDER / "datasets" / "dataset_allsides_climate_genroom_400k.csv"
FACADE_COLS     = 32                     # facade grid columns (constant across models)
DISPLAY_MAX_LUX = 3000
OUT_DIR         = Path(__file__).resolve().parents[1] / "outputs"
# ──────────────────────────────────────────────────────────────────────────────


def read_examples(csv_path, n):
    """
    Read the first `n` data rows from an all-sides climate-pit CSV.
    Returns (examples, facade_rows, grid_n) where each example is
        (facades_dict{'S','N','E','W' -> list}, climate_name, info, lux_list)
    with info = (month, day, hour, alt, dni, dhi, width, depth, height).
    facade_rows is inferred from the S_Pix_ column count (// FACADE_COLS); grid_n
    from the Lux_ column count (sqrt). Sun altitude derived from Sun_Z (sz=-sin alt).
    """
    with open(str(csv_path), "r") as f:
        rows = [r for r in csv.reader(f) if r]
    header = rows[0]
    col    = dict((name, i) for i, name in enumerate(header))

    n_side  = sum(1 for h in header if h.startswith("S_Pix_"))
    n_sens  = sum(1 for h in header if h.startswith("Lux_"))
    fac_rows = n_side // FACADE_COLS
    grid_n   = int(round(math.sqrt(n_sens)))

    examples = []
    for r in rows[1:n + 1]:
        facades = {s: [float(r[col["%s_Pix_%d" % (s, k)]]) for k in range(n_side)]
                   for s in ('S', 'N', 'E', 'W')}
        lux = [float(r[col["Lux_%d" % k]]) for k in range(n_sens)]
        sz  = float(r[col["Sun_Z"]])
        alt = math.degrees(math.asin(max(-1.0, min(1.0, -sz))))
        info = (int(r[col["Month"]]), int(r[col["Day"]]), float(r[col["Hour"]]),
                alt, float(r[col["DNI"]]), float(r[col["DHI"]]),
                float(r[col["Width"]]), float(r[col["Depth"]]), float(r[col["Height"]]))
        examples.append((facades, r[col["Climate"]], info, lux))
    return examples, fac_rows, grid_n


def main():
    ap = argparse.ArgumentParser(description="Visualize all-sides climate-pit dataset samples (no Radiance).")
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="dataset CSV path")
    ap.add_argument("--n",   type=int, default=5, help="first N rows to render")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Reading: {csv_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    samples, FAC_ROWS, GRID_N = read_examples(csv_path, args.n)
    n = len(samples)
    print(f"  {n} examples  |  facades {FACADE_COLS}x{FAC_ROWS}/side  |  floor {GRID_N}x{GRID_N}\n")
    for i, (facades, cname, info, lux) in enumerate(samples):
        month, day, hour, alt, dni, dhi, w, d, h = info
        wwr = {s: np.mean(np.array(facades[s]) < 0.5) * 100 for s in ('S', 'N', 'E', 'W')}
        print(f"[{i}] {cname[:24]:<24} {month}/{day} {hour:.1f}h alt={alt:.0f} "
              f"DNI={dni:.0f} room={w:.1f}x{d:.1f}x{h:.1f}m  "
              f"WWR S/N/E/W={wwr['S']:.0f}/{wwr['N']:.0f}/{wwr['E']:.0f}/{wwr['W']:.0f}%  "
              f"mean={np.mean(lux):.0f}lux")

    ncol = 5   # S, N, E, W facades + floor lux
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.2 * n), squeeze=False)

    # (side, x-end labels, physical aspect key) for the four facade panels.
    fac_spec = [('S', ("W", "E"), "HW"), ('N', ("W", "E"), "HW"),
                ('E', ("S", "N"), "HD"), ('W', ("S", "N"), "HD")]

    for row, (facades, cname, info, lux) in enumerate(samples):
        month, day, hour, alt, dni, dhi, w, d, h = info
        aspect = {"HW": h / w, "HD": h / d}

        for cslot, (side, xends, akey) in enumerate(fac_spec):
            grid   = np.array(facades[side], dtype=np.float32).reshape(FAC_ROWS, FACADE_COLS)
            binary = (grid < 0.5).astype(float)          # window = 1 (white in binary_r)
            ax = axes[row][cslot]
            ax.imshow(binary, cmap="binary_r", vmin=0, vmax=1,
                      interpolation="nearest", aspect=aspect[akey])
            ax.set_xticks([0, FACADE_COLS - 1]); ax.set_xticklabels(list(xends), fontsize=7)
            ax.set_yticks([0, FAC_ROWS - 1]); ax.set_yticklabels(["ceil", "floor"], fontsize=7)
            ax.tick_params(length=2)
            if row == 0:
                ax.set_title(f"{side} facade", fontsize=9)
            if cslot == 0:
                ax.set_ylabel(f"{cname[:22]}\n{month:02d}/{day:02d} {hour:.1f}h  alt={alt:.0f}°\n"
                              f"DNI={dni:.0f} DHI={dhi:.0f}\nW×D×H={w:.1f}×{d:.1f}×{h:.1f}m",
                              fontsize=7, rotation=0, ha="right", va="center", labelpad=32)

        # Floor lux — plan view (N up, W left) via rot90+fliplr, physical aspect D/W.
        lux_arr  = np.array(lux, dtype=np.float32).reshape(GRID_N, GRID_N)
        lux_plan = np.fliplr(np.rot90(lux_arr, k=1))
        ax = axes[row][ncol - 1]
        im = ax.imshow(lux_plan, cmap="inferno", vmin=0, vmax=DISPLAY_MAX_LUX,
                       interpolation="nearest", aspect=d / w)
        ax.set_xticks([0, GRID_N - 1]); ax.set_xticklabels(["W", "E"], fontsize=7)
        ax.set_yticks([0, GRID_N - 1]); ax.set_yticklabels(["N", "S"], fontsize=7)
        ax.tick_params(length=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="lux")
        ax.set_xlabel(f"mean={lux_arr.mean():.0f}  max={lux_arr.max():.0f} lux", fontsize=7.5)
        if row == 0:
            ax.set_title("Floor lux (Radiance)", fontsize=9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{csv_path.stem}_first{n}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGallery saved: {out}")


if __name__ == "__main__":
    main()
