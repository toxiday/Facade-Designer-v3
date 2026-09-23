"""
facade_designer_allsides_climate.py  (all-sides + climate + generalized room)
Interactive facade designer for the 4-facade all-sides climate transformer.

Versus the single-facade generalized designer (facade_designer_climate.py):
  * Room W/D/H sliders (same) drive the 8-vector geometry conditioning and deform the
    floor-plan grid to the true W:D ratio.
  * FOUR clickable green/gray BANDS around the floor plan (N=top, S=bottom, W=left,
    E=right) pick which walls may hold windows: green = open (optimised), click -> gray =
    solid (frozen, no windows).
  * The optimiser designs all picked walls jointly across the city's TMYx year.
  * The RESULT is an UNFOLDED "cross" plan — the 4 optimised facades unfolded around the
    floor, scaled to the true room W x D x H proportions; the center shows the occupancy
    zone (design intent), with a companion predicted-lux panel + day-of-year slider.

Floor plan orientation (where you paint): N = top, S = bottom, W = left, E = right.
These are ROOM-LOCAL wall ids (N = the y=D wall), which equal true compass directions only
at north_angle = 0.  The painted grid uses exactly the row-major order the Rhino/Grasshopper
job file's `grid` field uses -- k = row*GRID_N + col, row 0 = the +Y ("N") edge, col 0 = the
x=0 ("W") edge -- so the GUI and the JSON interface are one convention, not two.

Run (from this folder, so the local imports resolve):
    cd "allside_Generalised model/model trainer"
    ../../.venv/Scripts/python facade_designer_allsides_climate.py
"""

import sys
import queue
import threading
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, TextBox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_facade_allsides_climate import (
    FacadeOptimizerAllSidesClimate, discover_cities, city_label, annual_conditions,
    city_location, solar_position, reduce_conditions, append_room, save_facades_csv,
    GRID_N, GLARE_LUX, DEFAULT_TARGET_LUX, ZONE_MIN_RATIO,
    W_MIN, W_MAX, D_MIN, D_MAX, H_MIN, H_MAX,
    DEFAULT_WIDTH, DEFAULT_DEPTH, DEFAULT_HEIGHT,
    N_FACADES, FACADE_ROWS, FACADE_COLS, SIDES)
# matplotlib.use("TkAgg") is already set above, so facade_viz draws on the interactive
# backend here; run_facade_job.py sets Agg before importing the same module.
from facade_viz import (
    visualize_result_cross, facade_displays as _facade_displays,
    LUX_CMAP as _LUX_CMAP, OPEN_COLOR, CLOSED_COLOR, doy as _doy)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

PALETTE = [
    (0.27, 0.50, 0.80), (0.93, 0.55, 0.07), (0.78, 0.20, 0.20), (0.33, 0.66, 0.29),
    (0.70, 0.40, 0.80), (0.15, 0.70, 0.75), (0.90, 0.75, 0.15), (0.55, 0.27, 0.07),
]
TARGET_LUX_MIN, TARGET_LUX_MAX = 50.0, 2000.0
HOUR_MIN, HOUR_MAX = 4, 21
MARKER_SNAP_THRESH = 0.5
BAND        = 2.6                      # facade-selection band thickness (cell units)
# OPEN_COLOR / CLOSED_COLOR are imported from facade_viz so the edge bands here and the
# facade outlines in the result figure cannot drift to different greens.
_GRID_TITLE_IDLE = ("Floor Plan   click the timeline to add a time slot, then paint; "
                    "click the edge bands to toggle which walls have windows")
_GRID_TITLE_PAD  = 18

_REP_DAYS = [(6, 21, "Jun", "#d9772b"), (3, 21, "Mar/Sep", "#3aa657"),
             (12, 21, "Dec", "#3f7fcc")]


# Result visualisation, the palette and the unfolded-cross layout now live in facade_viz.py
# so run_facade_job.py can draw the identical figure headlessly (it must not import this
# module -- doing so would force the TkAgg backend inside a subprocess with no display).
# Imported above; this module keeps only the GUI itself.


# ---------------------------------------------------------------------------
# Designer GUI
# ---------------------------------------------------------------------------

class AllSidesClimateDesigner:
    def __init__(self):
        self.cities    = discover_cities()
        self.city_idx  = 0
        self.markers   = []
        self.active_idx = None

        self.width  = DEFAULT_WIDTH
        self.depth  = DEFAULT_DEPTH
        self.height = DEFAULT_HEIGHT
        self.facade_open = {'S': True, 'N': True, 'E': True, 'W': True}
        self.n_repr = 48         # k-means representative conditions per time slot (speed vs accuracy)

        self._draw_val  = None
        self._last_cell = None
        self._zone_rgba = np.zeros((GRID_N, GRID_N, 4), dtype=np.float32)
        self._tl_vlines = []
        self._band_patches = {}
        # Shift+drag rectangle (rubber-band) fill state.
        self._shift_down = False
        self._rect_start = None
        self._rect_last  = None
        self._rect_val   = None
        self._rect_patch = None
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        fig = plt.figure(figsize=(14, 9), facecolor="white")
        fig.suptitle("All-Sides Annual Climate Facade Designer", fontsize=14, fontweight="bold")
        self.fig = fig

        self.ax_grid = fig.add_axes([0.06, 0.38, 0.50, 0.54])
        self._init_grid()

        self.ax_tl = fig.add_axes([0.06, 0.17, 0.50, 0.15])
        self._draw_timeline()

        ax_city = fig.add_axes([0.13, 0.095, 0.38, 0.025])
        self.sld_city = Slider(ax_city, "City", 0, len(self.cities) - 1, valinit=0, valstep=1)
        self.sld_city.on_changed(self._on_city)
        ax_search = fig.add_axes([0.13, 0.04, 0.20, 0.035])
        self.txt_search = TextBox(ax_search, "Find  ", initial="")
        self.txt_search.on_submit(self._on_search)
        self.ax_cityname = fig.add_axes([0.35, 0.04, 0.23, 0.035]); self.ax_cityname.set_axis_off()
        self._cityname_txt = self.ax_cityname.text(0.0, 0.5, "", fontsize=8, va="center")

        self.ax_info = fig.add_axes([0.62, 0.50, 0.36, 0.42]); self.ax_info.set_axis_off()
        ax_tgt = fig.add_axes([0.80, 0.43, 0.16, 0.045])
        self.txt_target = TextBox(ax_tgt, "Target lux  ", initial=f"{DEFAULT_TARGET_LUX:.0f}")
        self.txt_target.on_submit(self._on_target_submit)
        ax_cz  = fig.add_axes([0.62, 0.43, 0.15, 0.045])
        ax_run = fig.add_axes([0.62, 0.30, 0.22, 0.09])
        ax_cl  = fig.add_axes([0.86, 0.30, 0.12, 0.09])
        self.btn_czn = Button(ax_cz, "Clear Zone", color=(0.80, 0.80, 0.80))
        self.btn_run = Button(ax_run, "Run Optimizer", color=(0.18, 0.68, 0.44))
        self.btn_clr = Button(ax_cl, "Clear All", color=(0.78, 0.22, 0.18))
        for b in (self.btn_run, self.btn_clr): b.label.set_color("white")
        self.btn_czn.on_clicked(self._clear_active_zone)
        self.btn_run.on_clicked(self._run_optimizer)
        self.btn_clr.on_clicked(self._clear_all)

        self.ax_roomlbl = fig.add_axes([0.62, 0.255, 0.36, 0.02]); self.ax_roomlbl.set_axis_off()
        self._room_txt = self.ax_roomlbl.text(0.0, 0.5, "", fontsize=9, va="center",
                                              fontweight="bold", color="#333333")
        ax_w = fig.add_axes([0.71, 0.215, 0.24, 0.02])
        ax_d = fig.add_axes([0.71, 0.180, 0.24, 0.02])
        ax_h = fig.add_axes([0.71, 0.145, 0.24, 0.02])
        self.sld_w = Slider(ax_w, "Width (m)",  W_MIN, W_MAX, valinit=self.width,  valstep=0.1)
        self.sld_d = Slider(ax_d, "Depth (m)",  D_MIN, D_MAX, valinit=self.depth,  valstep=0.1)
        self.sld_h = Slider(ax_h, "Height (m)", H_MIN, H_MAX, valinit=self.height, valstep=0.1)
        self.sld_w.on_changed(lambda v: self._on_room("width",  v))
        self.sld_d.on_changed(lambda v: self._on_room("depth",  v))
        self.sld_h.on_changed(lambda v: self._on_room("height", v))

        # k-means representative-conditions knob: each time slot's ~365 daytime days are
        # reduced to this many weighted reps before optimizing (fewer = faster/coarser).
        ax_r = fig.add_axes([0.71, 0.110, 0.24, 0.02])
        self.sld_repr = Slider(ax_r, "Repr. cond", 8, 120, valinit=self.n_repr, valstep=4)
        self.sld_repr.on_changed(lambda v: setattr(self, "n_repr", int(v)))

        fig.canvas.mpl_connect("button_press_event",   self._on_press)
        fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event",      self._on_key)
        fig.canvas.mpl_connect("key_release_event",    self._on_key)

        self._refresh_cityname(); self._refresh_info()
        plt.show()

    def _init_grid(self):
        ax = self.ax_grid
        ax.set_facecolor("#f0f0f0")
        self._zone_img = ax.imshow(self._zone_rgba, extent=[0, GRID_N, GRID_N, 0],
                                   origin="upper", interpolation="nearest", zorder=2)
        segs = ([[(0, i), (GRID_N, i)] for i in range(GRID_N + 1)] +
                [[(i, 0), (i, GRID_N)] for i in range(GRID_N + 1)])
        ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=0.3, zorder=1))
        ax.set_xlim(-BAND, GRID_N + BAND); ax.set_ylim(GRID_N + BAND, -BAND)

        # Facade-selection bands (N top, S bottom, W left, E right).
        band_defs = {
            'N': Rectangle((0, -BAND), GRID_N, BAND),
            'S': Rectangle((0, GRID_N), GRID_N, BAND),
            'W': Rectangle((-BAND, 0), BAND, GRID_N),
            'E': Rectangle((GRID_N, 0), BAND, GRID_N),
        }
        for side, rect in band_defs.items():
            rect.set_zorder(4)
            ax.add_patch(rect)
            self._band_patches[side] = rect
            cx = rect.get_x() + rect.get_width() / 2
            cy = rect.get_y() + rect.get_height() / 2
            ax.text(cx, cy, side, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=5)
        self._refresh_bands()
        ax.set_title(_GRID_TITLE_IDLE, color="#555555", fontsize=8.5, pad=_GRID_TITLE_PAD)
        self._apply_room_aspect()

    def _refresh_bands(self):
        for side, rect in self._band_patches.items():
            rect.set_facecolor(OPEN_COLOR if self.facade_open[side] else CLOSED_COLOR)
        if getattr(self, "fig", None) is not None:
            self.fig.canvas.draw_idle()

    def _apply_room_aspect(self):
        ax = self.ax_grid
        ax.set_aspect(self.depth / self.width)
        ax.set_xticks([0, GRID_N]); ax.set_xticklabels(["0", f"{self.width:.1f}m"], fontsize=7)
        ax.set_yticks([0, GRID_N]); ax.set_yticklabels(["0", f"{self.depth:.1f}m"], fontsize=7)
        ax.tick_params(length=2)
        if hasattr(self, "_room_txt"):
            self._room_txt.set_text(
                f"Room  {self.width:.1f} × {self.depth:.1f} × {self.height:.1f} m  (W × D × H)")
        if getattr(self, "fig", None) is not None:
            self.fig.canvas.draw_idle()

    def _on_room(self, which, val):
        setattr(self, which, float(val))
        self._apply_room_aspect()

    def _band_hit(self, x, y):
        """Return the side whose band contains (x, y) in grid data coords, else None."""
        if x is None or y is None:
            return None
        if 0 <= x <= GRID_N and -BAND <= y <= 0:            return 'N'
        if 0 <= x <= GRID_N and GRID_N <= y <= GRID_N + BAND: return 'S'
        if -BAND <= x <= 0 and 0 <= y <= GRID_N:            return 'W'
        if GRID_N <= x <= GRID_N + BAND and 0 <= y <= GRID_N: return 'E'
        return None

    # ---------------------------------------------------------------- timeline
    def _compute_sun_curves(self):
        lat, lon_e, tz_h = city_location(self.cities[self.city_idx])
        hours = np.linspace(HOUR_MIN, HOUR_MAX, 200)
        curves = []
        for month, day, label, color in _REP_DAYS:
            alt = np.array([max(0.0, solar_position(lat, lon_e, tz_h, month, day, h)[0])
                            for h in hours])
            curves.append((label, color, alt))
        return hours, curves

    def _draw_timeline(self):
        ax = self.ax_tl; ax.cla()
        ax.set_facecolor("white")
        hours, curves = self._compute_sun_curves()
        for label, color, alt in curves:
            ax.plot(hours, alt, color=color, linewidth=1.3, label=label, zorder=3)
            ax.fill_between(hours, 0, alt, color=color, alpha=0.10, zorder=2)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=1)
        ax.set_xlim(HOUR_MIN, HOUR_MAX); ax.set_ylim(-8, None)
        ax.set_xticks(list(range(HOUR_MIN, HOUR_MAX + 1, 2)))
        ax.set_xticklabels([f"{h}:00" for h in range(HOUR_MIN, HOUR_MAX + 1, 2)],
                           fontsize=7, color="#555555")
        ax.set_yticks([0, 30, 60]); ax.set_ylabel("Sun alt (deg)", fontsize=7, color="#555555")
        ax.tick_params(colors="#888888", length=3)
        ax.text(0.01, 0.96, "left-click = add / select time slot     right-click = delete",
                transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=7.5)
        ax.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.8)
        self._marker_sc = ax.scatter([], [], s=90, marker="^", zorder=10, linewidths=1.5)
        self._tl_vlines = []
        self._refresh_timeline()

    def _refresh_timeline(self):
        for ln in self._tl_vlines:
            ln.remove()
        self._tl_vlines = []
        if not self.markers:
            self._marker_sc.set_offsets(np.empty((0, 2)))
            self.fig.canvas.draw_idle(); return
        xs    = [m["hour"] for m in self.markers]
        cols  = [m["color"] for m in self.markers]
        edges = ["white" if i == self.active_idx else "none" for i in range(len(self.markers))]
        self._marker_sc.set_offsets(np.column_stack([xs, [-5.0] * len(xs)]))
        self._marker_sc.set_facecolor(cols); self._marker_sc.set_edgecolor(edges)
        for i, m in enumerate(self.markers):
            self._tl_vlines.append(self.ax_tl.axvline(
                m["hour"], color=m["color"], linewidth=2.0 if i == self.active_idx else 0.8,
                linestyle="--", alpha=0.85, zorder=5))
        self.fig.canvas.draw_idle()

    @staticmethod
    def _snap_hour(x):
        return int(round(float(np.clip(x, HOUR_MIN, HOUR_MAX))))

    def _find_near_marker(self, hour):
        if not self.markers:
            return -1
        dists = [abs(m["hour"] - hour) for m in self.markers]
        idx   = int(np.argmin(dists))
        return idx if dists[idx] <= MARKER_SNAP_THRESH else -1

    def _add_marker(self, hour):
        tgt = self._parse_target(self.txt_target.text) or DEFAULT_TARGET_LUX
        self.markers.append({
            "hour": hour, "zone": np.zeros((GRID_N, GRID_N), dtype=np.float32),
            "color": PALETTE[len(self.markers) % len(PALETTE)], "target": tgt})
        self.active_idx = len(self.markers) - 1

    # ---------------------------------------------------------------- city
    def _on_city(self, val):
        self.city_idx = int(val)
        self._refresh_cityname()
        self._draw_timeline()

    def _on_search(self, text):
        text = text.strip().lower()
        if not text:
            return
        for i, c in enumerate(self.cities):
            if text in city_label(c).lower():
                self.sld_city.set_val(i)
                return

    def _refresh_cityname(self):
        self._cityname_txt.set_text(city_label(self.cities[self.city_idx]))
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- target
    @staticmethod
    def _parse_target(text):
        try:
            return float(np.clip(float(text), TARGET_LUX_MIN, TARGET_LUX_MAX))
        except (TypeError, ValueError):
            return None

    def _sync_target_box(self):
        if self.active_idx is not None:
            self.txt_target.eventson = False
            self.txt_target.set_val(f"{self.markers[self.active_idx]['target']:.0f}")
            self.txt_target.eventson = True

    def _on_target_submit(self, text):
        val = self._parse_target(text)
        if val is None:
            self._sync_target_box(); return
        self.txt_target.eventson = False
        self.txt_target.set_val(f"{val:.0f}")
        self.txt_target.eventson = True
        if self.active_idx is not None:
            self.markers[self.active_idx]["target"] = val
            self._update_title(); self._refresh_info()

    # ---------------------------------------------------------------- zone edit
    def _clear_active_zone(self, _evt):
        if self.active_idx is not None:
            self.markers[self.active_idx]["zone"][:] = 0
            self._refresh_zone_overlay(); self._refresh_info()

    def _clear_all(self, _evt):
        self.markers = []; self.active_idx = None
        self._refresh_zone_overlay(); self._refresh_timeline()
        self._refresh_info(); self._update_title()

    # ---------------------------------------------------------------- painting
    def _xy_to_cell(self, x, y):
        return (int(np.clip(np.floor(y), 0, GRID_N - 1)),
                int(np.clip(np.floor(x), 0, GRID_N - 1)))

    def _in_grid(self, x, y):
        return x is not None and y is not None and 0 <= x <= GRID_N and 0 <= y <= GRID_N

    def _paint_cell(self, row, col):
        if (row, col) == self._last_cell or self.active_idx is None:
            return
        self._last_cell = (row, col)
        self.markers[self.active_idx]["zone"][row, col] = self._draw_val
        if self._draw_val:
            self._zone_rgba[row, col] = (*self.markers[self.active_idx]["color"], 0.65)
        else:
            self._zone_rgba[row, col] = (0, 0, 0, 0)
            for i, m in enumerate(self.markers):
                if m["zone"][row, col] > 0:
                    self._zone_rgba[row, col] = (*m["color"], 0.65 if i == self.active_idx else 0.22)
        self._zone_img.set_data(self._zone_rgba)
        self._blit_grid()

    def _blit_grid(self):
        try:
            self.ax_grid.draw(self.fig.canvas.get_renderer())
            self.fig.canvas.blit(self.ax_grid.bbox)
        except Exception:
            self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- shift-drag rectangle
    def _start_rect(self, row, col, val):
        """Begin a rubber-band rectangle fill anchored at (row, col)."""
        self._rect_val   = val
        self._rect_start = (row, col)
        self._rect_last  = (row, col)
        edge = self.markers[self.active_idx]["color"] if val else (0.80, 0.13, 0.11)
        self._rect_patch = Rectangle((col, row), 1, 1, facecolor=(*edge, 0.25),
                                     edgecolor=edge, linewidth=1.5, zorder=6)
        self.ax_grid.add_patch(self._rect_patch)
        self._blit_grid()

    def _update_rect(self, row, col):
        """Resize the rubber-band preview to span the anchor and (row, col)."""
        self._rect_last = (row, col)
        r0, c0 = self._rect_start
        rmin, rmax = sorted((r0, row))
        cmin, cmax = sorted((c0, col))
        self._rect_patch.set_bounds(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)
        self._blit_grid()

    def _commit_rect(self):
        """Fill every cell in the dragged rectangle into the active marker's zone."""
        r0, c0 = self._rect_start
        r1, c1 = self._rect_last
        rmin, rmax = sorted((r0, r1))
        cmin, cmax = sorted((c0, c1))
        self.markers[self.active_idx]["zone"][rmin:rmax+1, cmin:cmax+1] = self._rect_val
        if self._rect_patch is not None:
            self._rect_patch.remove()
        self._rect_patch = self._rect_start = self._rect_last = self._rect_val = None
        self._refresh_zone_overlay()
        self._update_title(); self._refresh_info()

    def _refresh_zone_overlay(self):
        self._zone_rgba[:] = 0
        for m in self.markers:
            self._zone_rgba[m["zone"] > 0] = (*m["color"], 0.22)
        if self.active_idx is not None:
            m = self.markers[self.active_idx]
            self._zone_rgba[m["zone"] > 0] = (*m["color"], 0.65)
        self._zone_img.set_data(self._zone_rgba)
        self.fig.canvas.draw_idle()

    def _update_title(self):
        if self.active_idx is None:
            self.ax_grid.set_title(_GRID_TITLE_IDLE, color="#555555", fontsize=8.5, pad=_GRID_TITLE_PAD)
        else:
            m = self.markers[self.active_idx]
            self.ax_grid.set_title(
                f"Floor Plan  [{m['hour']:02d}:00 active | {int(m['zone'].sum())} cells | "
                f"{m['target']:.0f} lux]   drag=paint  right-drag=erase  Shift+drag=fill area  "
                f"edge bands=wall on/off",
                color="black", fontsize=8.5, pad=_GRID_TITLE_PAD)

    def _refresh_info(self):
        ax = self.ax_info; ax.cla(); ax.set_axis_off()
        ax.text(0.02, 0.98, "Time slots  (each evaluated over the full year)",
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.text(0.02, 0.90, "Click the timeline to add/select a slot; edit Target lux or paint its\n"
                            "zone; right-click a marker to delete. Edge bands pick window walls.",
                transform=ax.transAxes, color="#666666", fontsize=8, va="top")
        open_lbl = "".join(s for s in ("S", "N", "E", "W") if self.facade_open[s]) or "none"
        ax.text(0.02, 0.80, f"Window walls: {open_lbl}   (green=open, gray=solid)",
                transform=ax.transAxes, color="#227722", fontsize=8.5, va="top", fontweight="bold")
        if not self.markers:
            ax.text(0.02, 0.70, "No time slots yet.",
                    transform=ax.transAxes, color="#888888", fontsize=9, va="top")
        for i, m in enumerate(self.markers):
            is_a = (i == self.active_idx)
            txt = (f"{'>> ' if is_a else '   '}{m['hour']:02d}:00   "
                   f"{int(m['zone'].sum())} cells   @ {m['target']:.0f} lux")
            bbox = dict(boxstyle="round,pad=0.25", facecolor=m["color"], alpha=0.25) if is_a else None
            ax.text(0.02, 0.66 - i * 0.09, txt, transform=ax.transAxes,
                    color="black" if is_a else "#555555", fontsize=9, va="top", bbox=bbox)
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- mouse / keys
    def _on_key(self, event):
        """Track Shift state (press sets it, release clears it) for rect-drag fill."""
        if event.key and "shift" in event.key:
            self._shift_down = (event.name == "key_press_event")

    def _on_press(self, event):
        self._last_cell = None
        if event.inaxes is self.ax_grid and event.xdata is not None:
            side = self._band_hit(event.xdata, event.ydata)
            if side is not None:                       # toggle a wall on/off
                self.facade_open[side] = not self.facade_open[side]
                self._refresh_bands(); self._refresh_info()
                return
            if not self._in_grid(event.xdata, event.ydata) or self.active_idx is None:
                return
            row, col = self._xy_to_cell(event.xdata, event.ydata)
            val   = 1.0 if event.button == 1 else 0.0
            shift = self._shift_down or bool(event.key and "shift" in event.key)
            if shift:                                  # Shift+drag = rectangle fill
                self._start_rect(row, col, val)
            else:                                      # plain drag = freehand paint
                self._draw_val = val
                self._paint_cell(row, col)
                self._update_title(); self._refresh_info()

        elif event.inaxes is self.ax_tl and event.xdata is not None:
            hour = self._snap_hour(event.xdata)
            near = self._find_near_marker(hour)
            if event.button == 3:
                if near >= 0:
                    self.markers.pop(near)
                    if self.active_idx is not None and self.active_idx >= len(self.markers):
                        self.active_idx = len(self.markers) - 1 if self.markers else None
            else:
                self.active_idx = near if near >= 0 else (self._add_marker(hour) or self.active_idx)
            self._refresh_timeline(); self._refresh_zone_overlay()
            self._sync_target_box(); self._update_title(); self._refresh_info()

    def _on_motion(self, event):
        if event.inaxes is not self.ax_grid or event.xdata is None:
            return
        cell = self._xy_to_cell(event.xdata, event.ydata)   # clamped to valid cells
        if self._rect_start is not None:
            self._update_rect(*cell)
        elif self._draw_val is not None and self._in_grid(event.xdata, event.ydata):
            self._paint_cell(*cell)

    def _on_release(self, _event):
        if self._rect_start is not None:
            self._commit_rect()
        elif self._draw_val is not None:
            self._draw_val = None; self._last_cell = None; self._refresh_info()

    # ---------------------------------------------------------------- optimizer
    def _collect_inputs(self):
        city_path = self.cities[self.city_idx]
        conds, targets, weights, used = [], [], [], []
        for m in self.markers:
            if not m["zone"].any():
                continue
            _, _, cond, days = annual_conditions(city_path, m["hour"])
            if len(cond) == 0:
                print(f"  [!] {m['hour']:02d}:00 — sun never up in {city_label(city_path)}, skipped.")
                continue
            cond_opt, w = reduce_conditions(cond, n_clusters=self.n_repr)
            cond_opt = append_room(cond_opt, self.width, self.depth, self.height)
            conds.append(cond_opt); weights.append(w)
            tgt = (m["zone"] > 0).astype(np.float32) * m["target"]
            targets.append(np.broadcast_to(tgt, (len(cond_opt), GRID_N, GRID_N)))
            used.append((m, len(days), len(cond_opt)))
        if not conds:
            return None
        used.sort(key=lambda u: u[0]["hour"])
        return (np.concatenate(conds, 0), np.concatenate(targets, 0).copy(),
                np.concatenate(weights, 0), city_path, used)

    def _open_progress_window(self, n_steps):
        self.fig = plt.figure(figsize=(11, 5.5), facecolor="white")
        ax_t = self.fig.add_axes([0.04, 0.88, 0.55, 0.09]); ax_t.set_axis_off()
        ax_t.text(0.0, 0.5, "Optimizing 4 facades over the full year...", ha="left",
                  va="center", fontsize=14, fontweight="bold")
        ax_bar = self.fig.add_axes([0.04, 0.72, 0.52, 0.08])
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1); ax_bar.set_axis_off()
        ax_bar.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e0e0e0"))
        bar = ax_bar.add_patch(plt.Rectangle((0, 0), 0, 1, facecolor="#18a44f"))
        pct = ax_bar.text(0.5, 0.5, "0%", ha="center", va="center", color="white", fontweight="bold")
        ax_s = self.fig.add_axes([0.04, 0.55, 0.52, 0.1]); ax_s.set_axis_off()
        txt = ax_s.text(0.0, 0.5, f"Step 0 / {n_steps}", fontsize=11, va="center")

        # live mini unfolded-cross (4 facade previews) on the right
        cross_gs = gridspec.GridSpec(3, 3, figure=self.fig,
                                     height_ratios=[1, 2, 1], width_ratios=[1, 2, 1],
                                     left=0.62, right=0.98, top=0.92,
                                     bottom=max(0.92 - (0.98 - 0.62) * 11 / 5.5, 0.06),
                                     hspace=0.06, wspace=0.06)
        blank_ns = np.zeros((FACADE_ROWS, FACADE_COLS)); blank_ew = np.zeros((FACADE_COLS, FACADE_ROWS))
        kw = dict(cmap="binary_r", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax_n = self.fig.add_subplot(cross_gs[0, 1]); im_n = ax_n.imshow(blank_ns, **kw)
        ax_s2 = self.fig.add_subplot(cross_gs[2, 1]); im_s = ax_s2.imshow(blank_ns, **kw)
        ax_w = self.fig.add_subplot(cross_gs[1, 0]); im_w = ax_w.imshow(blank_ew, **kw)
        ax_e = self.fig.add_subplot(cross_gs[1, 2]); im_e = ax_e.imshow(blank_ew, **kw)
        for a, t in ((ax_n, "N"), (ax_s2, "S"), (ax_w, "W"), (ax_e, "E")):
            a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8)
        cax = self.fig.add_subplot(cross_gs[1, 1]); cax.set_xticks([]); cax.set_yticks([])
        cax.text(0.5, 0.5, "floor", ha="center", va="center", fontsize=8, color="#888")
        for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            self.fig.add_subplot(cross_gs[r, c]).set_visible(False)
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        return bar, pct, txt, (im_n, im_s, im_w, im_e)

    def _run_optimizer(self, _evt):
        if not any(self.facade_open.values()):
            print("[!] Pick at least one window wall (green band) before optimizing.")
            return
        inputs = self._collect_inputs()
        if inputs is None:
            print("[!] Paint at least one occupancy zone on a time slot first.")
            return
        cond, target_lux, weights, city_path, used = inputs
        facade_open = tuple(self.facade_open[s] for s in SIDES)
        print(f"\n  City: {city_label(city_path)}  |  room "
              f"{self.width:.1f}×{self.depth:.1f}×{self.height:.1f} m  |  "
              f"open {[s for s in ('S','N','E','W') if self.facade_open[s]]}")
        for m, nd, nc in used:
            print(f"    {m['hour']:02d}:00  {int(m['zone'].sum())} cells  @ {m['target']:.0f} lux  "
                  f"over {nd} days -> {nc} repr conditions")

        n_steps = 950          # cosmetic init (coarse-to-fine schedule total); callback reports the real total
        plt.close(self.fig)
        bar, pct, txt, ims = self._open_progress_window(n_steps)
        im_n, im_s, im_w, im_e = ims
        opt = FacadeOptimizerAllSidesClimate()

        prog_q = queue.Queue(); result = [None]; err = [None]; done = threading.Event()

        def _cb(step, n, l_zone, best, pct_win, curr_np, best_np):
            prog_q.put((step, n, best, pct_win, best_np.copy()))

        def _thread():
            # Coarse-to-fine (multigrid): escapes the init-dependent local minima the
            # direct optimizer falls into — commits to "which walls, where" at low res,
            # then adds detail. Keeps the best BINARISED design across all stages.
            try:
                result[0] = opt.optimize_coarse_to_fine(cond, target_lux, weights,
                                                         facade_open, progress_callback=_cb)
            except Exception as exc:
                err[0] = exc
            finally:
                done.set()

        threading.Thread(target=_thread, daemon=True).start()
        while not done.is_set():
            plt.pause(0.05)
            try:
                while True:
                    step, n, best, pct_win, best_np = prog_q.get_nowait()
                    frac = (step + 1) / n
                    bar.set_width(frac); pct.set_text(f"{frac*100:.0f}%")
                    txt.set_text(f"Step {step+1} / {n}   best={best:.3f}   windows={pct_win:.0f}%")
                    n_d, s_d, w_d, e_d = _facade_displays(best_np)
                    im_n.set_data(n_d); im_s.set_data(s_d); im_w.set_data(w_d); im_e.set_data(e_d)
                    self.fig.canvas.draw_idle()
            except queue.Empty:
                pass
        if err[0] is not None:
            raise err[0]

        binary_facades, _, _ = result[0]
        plt.close(self.fig)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = (f"{city_label(city_path)[:20]}_"
               f"W{self.width:.1f}D{self.depth:.1f}H{self.height:.1f}")
        csv_path = OUT_DIR / f"facade_allsides_climate_{tag}.csv"
        save_facades_csv(binary_facades, csv_path)
        print(f"  Glazing: {binary_facades.mean()*100:.1f}%   Facade CSV -> {csv_path}")
        markers_used = [m for m, _, _ in used]
        visualize_result_cross(binary_facades, opt, city_path, markers_used,
                               self.width, self.depth, self.height, facade_open,
                               OUT_DIR / f"facade_allsides_climate_{tag}_result.png",
                               interactive=True)
        print("Done.")


if __name__ == "__main__":
    AllSidesClimateDesigner()
