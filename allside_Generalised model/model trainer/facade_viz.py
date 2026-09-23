"""
facade_viz.py — the result figure, shared by the interactive designer and the headless runner.

The UNFOLDED CROSS: the four optimised facades unfolded around the floor plan at true
W x D x H proportions, with a per-time-slot predicted-lux gallery below it.

Extracted from facade_designer_allsides_climate.py so run_facade_job.py can produce the same
picture without importing the GUI — which would drag in matplotlib's TkAgg backend and try to
open a window inside a headless subprocess.

BACKEND IS THE CALLER'S CHOICE.  This module never calls matplotlib.use():
    * the designer sets TkAgg before importing this, and passes interactive=True
    * the job runner sets Agg, and passes interactive=False (no day slider — a PNG cannot
      be dragged, and Slider on a non-interactive backend is dead weight)

Anything drawn here is in the ROOM-LOCAL frame (N = the y=D wall, plan row 0 = the +Y edge).
When north_angle != 0 those labels are NOT compass directions, so the title spells out which
true direction each wall faces.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from optimize_facade_allsides_climate import (
    annual_conditions, append_room, city_label, wall_azimuths,
    SIDES, GRID_N, ZONE_MIN_RATIO, GLARE_LUX)

OPEN_COLOR   = (0.20, 0.63, 0.17)      # green = wall may hold windows
CLOSED_COLOR = (0.55, 0.55, 0.55)      # gray  = solid wall (frozen)

# Four seasonal sample days shown as fixed columns in the result gallery.
SEASON_SAMPLES = [(3, 21, "Mar"), (6, 21, "Jun"), (9, 21, "Sep"), (12, 21, "Dec")]


def _make_lux_cmap():
    colors = [
        (0.000, (0.00, 0.28, 0.32)), (0.167, (0.15, 0.30, 0.95)),
        (0.333, (0.55, 0.05, 0.85)), (0.500, (0.90, 0.05, 0.10)),
        (0.667, (0.97, 0.52, 0.00)), (1.000, (1.00, 0.95, 0.00)),
    ]
    return LinearSegmentedColormap.from_list("lux", colors, N=256)


LUX_CMAP = _make_lux_cmap()


def doy(month, day):
    return datetime.date(2021, month, day).timetuple().tm_yday


def facade_displays(binary_facades):
    """Unfold the 4 facades (index S=0,N=1,E=2,W=3) into cross-panel orientation:
    returns (N_disp, S_disp, W_disp, E_disp) with each wall's floor edge touching the
    center and ceiling pointing outward."""
    S = binary_facades[0]; N = binary_facades[1]; E = binary_facades[2]; W = binary_facades[3]
    return (N,                          # top-center   (rows: ceiling->floor)
            np.flipud(S),               # bottom-center (floor touches center)
            np.flipud(W.T),             # middle-left  (D tall, H wide)
            np.fliplr(np.flipud(E.T)))  # middle-right


def visualize_result_cross(binary_facades, opt, city_path, markers, width, depth, height,
                           facade_open, out_path, north_angle=0.0, interactive=False,
                           dpi=150):
    """Result figure: the unfolded facade cross on top, and below it a per-time-slot lux
    gallery (rows = time slots; columns = Annual mean, ~Mar 21, ~Jun 21, ~Sep 21, ~Dec 21,
    Selected day).

    markers : list of dicts with "hour" (int), "zone" (32x32, >0 = occupied) and
              "target" (scalar lux, for labelling).  The designer's painted markers and the
              runner's per-slot grids both fit this shape.
    interactive : add the day-of-year slider.  Only meaningful on an interactive backend.
    """
    W, D, H = width, depth, height
    uw, uh = W + 2.0 * H, D + 2.0 * H          # unfolded extent (true proportions)
    markers = sorted(markers, key=lambda m: m["hour"])
    n = len(markers)

    # ── Precompute each slot's full-year floor lux + zone stats ───────────────
    per = []
    for m in markers:
        _, _, cond5, days = annual_conditions(city_path, m["hour"],
                                              include_below_horizon=True,
                                              north_angle=north_angle)
        cond = append_room(cond5, W, D, H)
        E    = opt.predict_lux(binary_facades, cond)          # (Dyear, 32, 32)
        dy   = np.array([doy(d[0], d[1]) for d in days])
        dni  = np.array([d[3] for d in days])
        zmask = m["zone"] > 0
        per.append(dict(m=m, E=E, days=days, doy=dy, dni=dni, zmask=zmask, annual=E.mean(0)))

    # ── Figure sizing: cross band on top (true aspect), gallery below ─────────
    GAL_COLS, COL_W, ROW_H = 6, 1.85, 2.05
    left_lab, right_cbar   = 0.85, 0.95
    fig_w      = left_lab + GAL_COLS * COL_W + right_cbar
    CROSS_BAND = 4.2
    cw = CROSS_BAND * uw / uh
    if cw > fig_w * 0.60:
        cw = fig_w * 0.60; ch = cw * uh / uw
    else:
        ch = CROSS_BAND
    title_h, gap = 0.55, 0.45
    slider_h = 0.75 if interactive else 0.30
    fig_h = title_h + ch + gap + n * ROW_H + slider_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    pct_glaz = binary_facades.mean() * 100
    open_lbl = "".join(s for s, o in zip(SIDES, facade_open) if o) or "none"
    title = (f"Unfolded Facade Plan  |  {city_label(city_path)}  |  "
             f"room {W:.1f}×{D:.1f}×{H:.1f} m (W×D×H)  |  windows on {open_lbl}  |  "
             f"{pct_glaz:.1f}% glazing")
    if north_angle:
        az = wall_azimuths(north_angle)
        # S/N/E/W are room-local wall ids; once rotated they are not compass directions, so
        # say outright which way each open wall actually faces.
        title += (f"\nnorth_angle {north_angle:g}°  ->  "
                  + "  ".join(f"{s}:{az[s]:.0f}°" for s, o in zip(SIDES, facade_open) if o))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

    # ── Unfolded cross on top (centered, true W×D×H proportions) ──────────────
    cross_top = 1.0 - title_h / fig_h
    cross_bot = cross_top - ch / fig_h
    cross_l   = (fig_w - cw) / 2.0 / fig_w
    cross_r   = 1.0 - (fig_w - cw) / 2.0 / fig_w
    cgs = fig.add_gridspec(3, 3, height_ratios=[H, D, H], width_ratios=[H, W, H],
                           left=cross_l, right=cross_r, top=cross_top, bottom=cross_bot,
                           hspace=0.04, wspace=0.04)
    n_disp, s_disp, w_disp, e_disp = facade_displays(binary_facades)
    fkw = dict(cmap="binary_r", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    open_map = dict(zip(SIDES, facade_open))
    for (r, c), (side, img) in {(0, 1): ("N", n_disp), (2, 1): ("S", s_disp),
                                (1, 0): ("W", w_disp), (1, 2): ("E", e_disp)}.items():
        ax = fig.add_subplot(cgs[r, c]); ax.imshow(img, **fkw)
        ax.set_xticks([]); ax.set_yticks([])
        col = OPEN_COLOR if open_map[side] else CLOSED_COLOR
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2.0)
        ax.set_title(side if open_map[side] else f"{side} (solid)",
                     fontsize=8, color=col, fontweight="bold", pad=2)
    ax_c = fig.add_subplot(cgs[1, 1])
    occ = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for m in markers:
        occ = np.maximum(occ, (m["zone"] > 0).astype(np.float32))
    ax_c.imshow(occ, cmap="Greens", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax_c.set_xticks([]); ax_c.set_yticks([]); ax_c.set_title("occupancy", fontsize=8, pad=2)
    for sp in ax_c.spines.values(): sp.set_edgecolor("#333333")
    for (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        fig.add_subplot(cgs[r, c]).set_visible(False)

    # ── Lux gallery below: rows = time slots, cols = annual + 4 seasons + selected ──
    gal_top   = cross_bot - gap / fig_h
    gal_bot   = ((slider_h + 0.25) / fig_h) if interactive else (slider_h / fig_h)
    gal_left  = left_lab / fig_w
    gal_right = 1.0 - right_cbar / fig_w
    ggs = fig.add_gridspec(n, GAL_COLS, left=gal_left, right=gal_right,
                           top=gal_top, bottom=gal_bot, hspace=0.5, wspace=0.12)
    headers = ["Annual mean", "~Mar 21", "~Jun 21", "~Sep 21", "~Dec 21", "Selected day"]
    day0 = doy(6, 21)
    sel, im_last = [], None
    for row, p in enumerate(per):
        m, E, dy, days, dni, zmask = p["m"], p["E"], p["doy"], p["days"], p["dni"], p["zmask"]
        lo        = m["target"] * ZONE_MIN_RATIO
        zone_lux  = E[:, zmask] if zmask.any() else E.reshape(len(E), -1)
        underlit  = (zone_lux < lo).mean() * 100
        glare_pct = (E > GLARE_LUX).mean() * 100
        season_i  = [int(np.abs(dy - doy(mth, d)).argmin()) for mth, d, _ in SEASON_SAMPLES]
        i0        = int(np.abs(dy - day0).argmin())
        arrs      = [p["annual"]] + [E[i] for i in season_i] + [E[i0]]
        for col, arr in enumerate(arrs):
            ax = fig.add_subplot(ggs[row, col])
            im = ax.imshow(arr, cmap=LUX_CMAP, vmin=0, vmax=GLARE_LUX,
                           interpolation="nearest", aspect="auto")
            if zmask.any():
                ax.contour(np.arange(GRID_N), np.arange(GRID_N), zmask.astype(float),
                           levels=[0.5], colors=["white"], linewidths=1.0)
            ax.set_xticks([0, GRID_N - 1]); ax.set_xticklabels(["W", "E"], fontsize=5)
            ax.set_yticks([0, GRID_N - 1]); ax.set_yticklabels(["N", "S"], fontsize=5)
            ax.tick_params(length=1.5)
            if col == 0:
                ax.set_ylabel(f"{m['hour']:02d}:00\n@ {m['target']:.0f} lux", fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=18)
                zmean = p["annual"][zmask].mean() if zmask.any() else 0.0
                ax.set_title(f"annual mean\nunderlit {underlit:.0f}% glare {glare_pct:.0f}%",
                             fontsize=6.5, pad=2)
                ax.set_xlabel(f"zone {zmean:.0f} lux", fontsize=6)
            else:
                if row == 0:
                    ax.set_title(headers[col], fontsize=8, pad=2)
                idx = season_i[col - 1] if col <= 4 else i0
                z_day = E[idx][zmask].mean() if zmask.any() else 0.0
                lbl = (f"{days[idx][0]}/{days[idx][1]} DNI {dni[idx]:.0f}\nzone {z_day:.0f}"
                       if col <= 4 else f"{days[idx][0]}/{days[idx][1]}\nzone {z_day:.0f}")
                ax.set_xlabel(lbl, fontsize=6)
                if col == GAL_COLS - 1:
                    sel.append(dict(ax=ax, im=im, E=E, days=days, doy=dy, zmask=zmask))
            im_last = im

    cax = fig.add_axes([gal_right + 0.012, gal_bot, 0.012, gal_top - gal_bot])
    fig.colorbar(im_last, cax=cax, label="lux (max 3000)")

    if interactive:
        from matplotlib.widgets import Slider

        def _update_day(val):
            d = int(val)
            for s in sel:
                i = int(np.abs(s["doy"] - d).argmin())
                s["im"].set_data(s["E"][i])
                z_day = s["E"][i][s["zmask"]].mean() if s["zmask"].any() else 0.0
                s["ax"].set_xlabel(f"{s['days'][i][0]}/{s['days'][i][1]}\nzone {z_day:.0f}",
                                   fontsize=6)
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([gal_left, 0.035, (gal_right - gal_left) * 0.6, 0.02])
        day_slider = Slider(ax_slider, "Day of year", 1, 365, valinit=day0, valstep=1)
        day_slider.on_changed(_update_day)
        fig._day_slider = day_slider

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Visualization saved: {out_path}", flush=True)
    if interactive:
        plt.show()
    else:
        plt.close(fig)      # headless: release the figure, do not leak it into the next run
    return out_path
