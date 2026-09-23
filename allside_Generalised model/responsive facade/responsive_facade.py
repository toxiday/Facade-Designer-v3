"""
responsive_facade.py — a room whose facade re-solves itself around the people inside it.

THE IDEA.  A building skin made of individually operable panels, and a sensor that knows where
people are and what they are doing.  Each activity implies a required illuminance; the facade
opens exactly the panels that put that much daylight on that patch of floor, for the sun that
is in the sky at that moment, and re-opens a different set when the person moves.  The
building composes its own elevation, continuously, around its occupants.

WHAT IS REAL AND WHAT IS FAKED.  Only the sensing is faked — you are the camera: WASD walks
the selected occupant and a slider sets what they are doing.  Everything downstream is the
production pipeline the annual facade designer uses: the same TMYx climate library, the same
solar position, the same transformer surrogate, the same gradient optimizer, the same comfort
band, the same lux colour ramp.  The facade you watch assemble itself is a real optimized
design, not an animation.

HOW TO READ THE SCREEN.
    centre        floor plan, coloured by PREDICTED illuminance (the same 0-3000 lux ramp the
                  annual result.png uses, so the two are directly comparable)
    the rings     each occupant's footprint.  The ring is drawn in the colour of the lux they
                  ASKED for, against a floor drawn in the lux they are GETTING — when the ring
                  disappears into its background, the facade has solved them
    around it     the four facades unfolded, white = open panel, black = solid.  A green
                  border means the wall is allowed to carry windows; grey means frozen solid

CONTROLS
    W A S D  /  arrows   walk the selected occupant
    click a ring         select that occupant
    shift-click          spawn a new occupant there
    x                    delete the selected occupant
    space                force a fresh cold solve
    p                    save a PNG of the current state
    sliders              city, date, hour, room W/D/H, north angle, and the selected
                         occupant's target lux and footprint radius

Run it:
    "<repo>\\.venv\\Scripts\\python.exe" responsive_facade.py
    ... --city Boston --width 7 --depth 9 --hour 15
"""

import sys
import argparse
import textwrap
from pathlib import Path
from dataclasses import replace

import numpy as np
import matplotlib

# The caller may have pinned a backend already (--screenshot runs under Agg).  Only claim the
# interactive one when nothing has been chosen, so this module stays importable headlessly.
if "--screenshot" not in sys.argv:
    matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Circle, FancyArrow                  # noqa: E402
from matplotlib.widgets import Button, Slider, TextBox, CheckButtons   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_solver import (                                          # noqa: E402
    Occupant, Scene, SolverWorker, LiveSolver,
    single_condition, occupant_owner_grid, zone_stats, days_in_month,
    ACTIVITIES, DEFAULT_RADIUS, TARGET_LUX_MIN, TARGET_LUX_MAX, RADIUS_MIN, RADIUS_MAX,
    discover_cities, city_label, resolve_city,
    SIDES, GRID_N, GLARE_LUX, MIN_ALT,
    W_MIN, W_MAX, D_MIN, D_MAX, H_MIN, H_MAX)
from optimize_facade_allsides_climate import wall_azimuths          # noqa: E402
from facade_viz import LUX_CMAP, OPEN_COLOR, CLOSED_COLOR, facade_displays   # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "captures"

# Occupant identity colours — the designer's PALETTE, copied rather than imported because
# facade_designer_allsides_climate forces the TkAgg backend at import time and this module
# must stay importable under Agg.
PALETTE = [
    (0.27, 0.50, 0.80), (0.93, 0.55, 0.07), (0.78, 0.20, 0.20), (0.33, 0.66, 0.29),
    (0.70, 0.40, 0.80), (0.15, 0.70, 0.75), (0.90, 0.75, 0.15), (0.55, 0.27, 0.07),
]

# m per keypress.  OS key-repeat fires ~30/s, so a HELD key walks at ~3.6 m/s.  A solve takes
# ~1.1 s, which means a held key leaves the facade roughly 4 m behind and the readout shows
# whatever light happens to be where you skidded to — the lag is real, and at 0.15 it was a
# 5 m/s sprint that made the demo look broken.  Tap the key to walk at a pace the building
# can actually track.
MOVE_STEP   = 0.12
POLL_MS     = 80           # how often the UI checks for a finished solve
BG          = "white"
INK         = "#222222"

# Readout panel.  INFO_FONTSIZE is the CEILING — _refresh_info shrinks below it when the table
# is long (see the auto-fit note there).  WRAP_CHARS is what fits the panel width at that
# ceiling, so wrapping stays conservative as the font shrinks.
INFO_FONTSIZE     = 8.4
MIN_INFO_FONTSIZE = 6.0     # below this it stops being readable, so shed lines instead
INFO_LINESPACING  = 1.42
WRAP_CHARS        = 62

# Is the building serving this person?  Ring colour, keyed on the share of their footprint
# inside the optimizer's own comfort band — so the plan and the readout are scored the same
# way the optimizer was.
SERVED_OK, SERVED_NEAR, SERVED_BAD = "#2aa84a", "#e8a020", "#d3352a"


def served_color(in_band):
    if in_band != in_band:          # NaN — nothing solved for them yet
        return "#9a9a9a"
    return SERVED_OK if in_band >= 75 else (SERVED_NEAR if in_band >= 40 else SERVED_BAD)


def nearest_activity(lux):
    return min(ACTIVITIES, key=lambda a: abs(a[1] - lux))[0]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ResponsiveFacadeApp:

    def __init__(self, city="Houston", month=6, day=21, hour=12, north=0.0,
                 width=6.0, depth=8.0, height=3.0, headless=False):
        self.cities = discover_cities()
        try:
            wanted = resolve_city(city)
            self.city_idx = self.cities.index(wanted)
        except Exception:
            self.city_idx = 0

        self.month, self.day, self.hour = int(month), int(day), int(hour)
        self.north  = float(north)
        self.width, self.depth, self.height = float(width), float(depth), float(height)
        self.sides_open = {s: True for s in SIDES}

        self._next_oid = 1
        self.occupants = [
            Occupant(self._take_oid(), self.width * 0.30, self.depth * 0.62, 500.0,
                     DEFAULT_RADIUS, "Reading"),
            Occupant(self._take_oid(), self.width * 0.70, self.depth * 0.28, 200.0,
                     DEFAULT_RADIUS, "Relaxing / TV"),
        ]
        self.selected = 0
        self.auto     = True
        self.seq      = 0

        # Last completed solve.  The facade and the lux field are properties of that solve;
        # the occupant readout is recomputed against CURRENT positions every redraw (see
        # _refresh_info), so walking out of a light patch shows the number falling immediately
        # while the facade is still catching up.
        self.binary  = np.zeros((4, 16, 32), dtype=np.int32)
        self.lux     = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.last    = None
        self.status  = "starting the surrogate..."

        # The lux map is re-predicted independently of the optimizer — see _maybe_predict.
        # _lux_key is the sun/room state the displayed map was computed for; _facade_key is the
        # state the displayed FACADE was designed for.  They differ exactly while a solve is
        # catching up, which is the interesting state and the one the readout names.
        self._lux_dirty  = True
        self._lux_key    = None
        self._lux_note   = None
        self._facade_key = None
        self._preview_L  = None      # binarised loss of the mid-solve facade on screen, if any
        self._converged   = None     # True once a solve stops because its best plateaued
        self._stop_reason = ""
        self._owner  = occupant_owner_grid(self.occupants, self.width, self.depth)
        self._stats  = zone_stats(self.occupants, self._owner, self.lux)

        self.headless = headless
        self.worker   = SolverWorker(verbose=False).start()
        self._build_ui()

    def _take_oid(self):
        self._next_oid += 1
        return self._next_oid - 1

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # WASD collide with matplotlib's own shortcuts (s = save figure, a/f/g/l/k = axis
        # toggles).  Clear those keymaps rather than picking different movement keys —
        # WASD is the thing a user will try first.
        for k in ("keymap.save", "keymap.fullscreen", "keymap.grid", "keymap.grid_minor",
                  "keymap.yscale", "keymap.xscale", "keymap.home", "keymap.back",
                  "keymap.forward", "keymap.pan", "keymap.zoom", "keymap.copy"):
            if k in plt.rcParams:
                plt.rcParams[k] = []

        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig = fig
        fig.suptitle("Responsive Facade  —  the building re-solves itself around its occupants",
                     fontsize=13, fontweight="bold", y=0.975)

        # ---- the unfolded cross (facades around the plan) -----------------
        self.cross_box = (0.030, 0.085, 0.555, 0.855)
        self.ax_plan = fig.add_axes([0.1, 0.1, 0.2, 0.2])
        self.ax_fac  = {s: fig.add_axes([0.1, 0.1, 0.1, 0.1]) for s in SIDES}
        self._init_cross()

        # ---- colour bar ---------------------------------------------------
        ax_cb = fig.add_axes([0.105, 0.046, 0.390, 0.015])
        cb = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(0, GLARE_LUX), cmap=LUX_CMAP),
            cax=ax_cb, orientation="horizontal")
        # Label ABOVE the bar: below it the tick row and the shortcut line have to share
        # ~0.03 of figure height and they overlap.
        cb.ax.xaxis.set_label_position("top")
        cb.set_label("predicted floor illuminance (lux)  —  same ramp as the annual "
                     "result figures", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7, pad=2)

        self.ax_help = fig.add_axes([0.030, 0.000, 0.555, 0.022]); self.ax_help.set_axis_off()
        self.ax_help.text(0.5, 0.35,
                          "WASD / arrows  walk  •  click  select  •  shift-click  spawn  •  "
                          "x  delete  •  space  re-solve  •  p  save png",
                          ha="center", va="center", fontsize=8, color="#666666")

        # ---- readout ------------------------------------------------------
        self.ax_info = fig.add_axes([0.605, 0.645, 0.390, 0.290]); self.ax_info.set_axis_off()
        self._info_txt = self.ax_info.text(
            0.0, 1.0, "", fontsize=INFO_FONTSIZE, family="monospace", va="top", ha="left",
            color=INK, transform=self.ax_info.transAxes, linespacing=INFO_LINESPACING)

        # ---- sliders ------------------------------------------------------
        SX, SW, SH = 0.705, 0.228, 0.019
        def row(y):
            return self.fig.add_axes([SX, y, SW, SH])

        # Three groups — when the sun is, where the room is, who the selected person is —
        # with a wider gap between groups than within them.
        self.sld_city = Slider(row(0.596), "City  ", 0, max(1, len(self.cities) - 1),
                               valinit=self.city_idx, valstep=1)
        self.sld_mon  = Slider(row(0.554), "Month  ", 1, 12, valinit=self.month, valstep=1)
        self.sld_day  = Slider(row(0.512), "Day  ", 1, 31, valinit=self.day, valstep=1)
        self.sld_hr   = Slider(row(0.470), "Hour  ", 4, 21, valinit=self.hour, valstep=1)
        self.sld_nor  = Slider(row(0.428), "North angle  ", 0, 355, valinit=self.north,
                               valstep=5)
        self.sld_w    = Slider(row(0.370), "Width m  ",  W_MIN, W_MAX, valinit=self.width,
                               valstep=0.1)
        self.sld_d    = Slider(row(0.328), "Depth m  ",  D_MIN, D_MAX, valinit=self.depth,
                               valstep=0.1)
        self.sld_h    = Slider(row(0.286), "Height m  ", H_MIN, H_MAX, valinit=self.height,
                               valstep=0.1)
        self.sld_tgt  = Slider(row(0.228), "Target lux  ", TARGET_LUX_MIN, TARGET_LUX_MAX,
                               valinit=500.0, valstep=25)
        self.sld_rad  = Slider(row(0.186), "Footprint m  ", RADIUS_MIN, RADIUS_MAX,
                               valinit=DEFAULT_RADIUS, valstep=0.1)
        for s in (self.sld_city, self.sld_mon, self.sld_day, self.sld_hr, self.sld_nor,
                  self.sld_w, self.sld_d, self.sld_h, self.sld_tgt, self.sld_rad):
            s.label.set_fontsize(8.5)
            s.valtext.set_fontsize(8.5)
        # The city INDEX is meaningless to read and there is no room beside the track for a
        # 40-character TMYx station name; the readout panel names the city in full.
        self.sld_city.valtext.set_text("")
        self.sld_city.valtext.set_visible(False)

        self.fig.text(0.705, 0.622, "when", fontsize=8, color="#888888", fontweight="bold")
        self.fig.text(0.705, 0.402, "room", fontsize=8, color="#888888", fontweight="bold")
        self.fig.text(0.705, 0.260, "selected occupant", fontsize=8, color="#888888",
                      fontweight="bold")

        self.sld_city.on_changed(lambda v: self._on_scene(city_idx=int(v)))
        self.sld_mon.on_changed(lambda v: self._on_scene(month=int(v)))
        self.sld_day.on_changed(lambda v: self._on_scene(day=int(v)))
        self.sld_hr.on_changed(lambda v: self._on_scene(hour=int(v)))
        self.sld_nor.on_changed(lambda v: self._on_scene(north=float(v)))
        self.sld_w.on_changed(lambda v: self._on_room(width=float(v)))
        self.sld_d.on_changed(lambda v: self._on_room(depth=float(v)))
        self.sld_h.on_changed(lambda v: self._on_room(height=float(v)))
        self.sld_tgt.on_changed(self._on_target)
        self.sld_rad.on_changed(self._on_radius)

        ax_find = self.fig.add_axes([0.800, 0.128, 0.130, 0.030])
        self.txt_find = TextBox(ax_find, "Find city  ", initial="")
        self.txt_find.label.set_fontsize(8.5)
        self.txt_find.on_submit(self._on_find)

        # ---- walls + auto -------------------------------------------------
        # These live in their own left sub-column (x < 0.70).  Sliders draw their labels to
        # the LEFT of the track, so anything placed beside a slider row collides with them —
        # which is what "W wall" did with "Footprint m".
        ax_walls = self.fig.add_axes([0.607, 0.055, 0.082, 0.125]); ax_walls.set_frame_on(False)
        self.chk_walls = CheckButtons(ax_walls, [f"{s} wall" for s in SIDES], [True] * 4)
        for t in self.chk_walls.labels:
            t.set_fontsize(8.5)
        self.chk_walls.on_clicked(self._on_wall)

        ax_auto = self.fig.add_axes([0.607, 0.012, 0.082, 0.034]); ax_auto.set_frame_on(False)
        self.chk_auto = CheckButtons(ax_auto, ["auto-solve"], [True])
        self.chk_auto.labels[0].set_fontsize(8.5)
        self.chk_auto.on_clicked(self._on_auto)

        # ---- buttons ------------------------------------------------------
        self.btn_add = Button(self.fig.add_axes([0.730, 0.062, 0.080, 0.038]),
                              "Spawn", color=(0.33, 0.66, 0.29), hovercolor=(0.45, 0.78, 0.40))
        self.btn_del = Button(self.fig.add_axes([0.818, 0.062, 0.080, 0.038]),
                              "Delete", color=(0.78, 0.30, 0.25), hovercolor=(0.88, 0.42, 0.36))
        self.btn_run = Button(self.fig.add_axes([0.906, 0.062, 0.089, 0.038]),
                              "Re-solve", color=(0.27, 0.50, 0.80), hovercolor=(0.40, 0.62, 0.90))
        for b in (self.btn_add, self.btn_del, self.btn_run):
            b.label.set_fontsize(9); b.label.set_color("white"); b.label.set_fontweight("bold")
        self.btn_add.on_clicked(lambda e: self._spawn())
        self.btn_del.on_clicked(lambda e: self._delete())
        self.btn_run.on_clicked(lambda e: self._request(force_cold=True))

        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._layout_cross()
        self._refresh_all()

    # ------------------------------------------------- cross geometry
    def _init_cross(self):
        ax = self.ax_plan
        self.im_plan = ax.imshow(self.lux, cmap=LUX_CMAP, vmin=0, vmax=GLARE_LUX,
                                 extent=[0, self.width, 0, self.depth], origin="upper",
                                 interpolation="nearest", aspect="auto", zorder=0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333"); sp.set_linewidth(1.2)
        self._plan_artists = []

        # The compass lives in the cross's empty top-right corner rather than on the plan.
        # On the plan it had to survive every part of the lux ramp underneath it and still
        # covered floor the occupants might walk onto; out here it is just a legend.  The
        # corner is H x H metres, so it renders square and the arrow needs no aspect fix-up.
        self.ax_north = self.fig.add_axes([0.1, 0.1, 0.05, 0.05])
        self.ax_north.set_axis_off()
        self.ax_north.set_xlim(0, 1); self.ax_north.set_ylim(0, 1)
        self._north_artists = []

        # Optimizer state badge, in the cross's OTHER empty corner (top-left), mirroring the
        # compass.  It cannot live on the plan's title: the cross axes are laid out edge to
        # edge with no gap, so a title on the plan renders underneath the N facade above it and
        # is never seen — which is exactly what happened to the "solving" indicator before.
        self.ax_state = self.fig.add_axes([0.1, 0.1, 0.05, 0.05])
        self.ax_state.set_axis_off()
        self.ax_state.set_xlim(0, 1); self.ax_state.set_ylim(0, 1)
        self._state_txt = self.ax_state.text(
            0.5, 0.5, "", ha="center", va="center", fontsize=7.6, fontweight="bold",
            linespacing=1.5, transform=self.ax_state.transAxes)

        self.im_fac = {}
        for s in SIDES:
            a = self.ax_fac[s]
            self.im_fac[s] = a.imshow(np.zeros((16, 32)), cmap="binary_r", vmin=0, vmax=1,
                                      interpolation="nearest", aspect="auto")
            a.set_xticks([]); a.set_yticks([])

    def _layout_cross(self):
        """Place the five axes at TRUE W x D x H proportions inside `cross_box`.

        Recomputed on every room change rather than rebuilt: the axes and their images
        persist, only their rectangles move, so a width slider drag does not churn artists.
        """
        x0, y0, bw, bh = self.cross_box
        W, D, H = self.width, self.depth, self.height
        uw, uh  = W + 2 * H, D + 2 * H
        fw, fh  = self.fig.get_size_inches()
        ar      = fw / fh                       # figure fractions are anisotropic in metres
        s  = min(bw / uw, bh / (uh * ar))       # figure-width fraction per metre
        sv = s * ar                             # figure-height fraction per metre
        L  = x0 + (bw - s * uw) / 2.0
        B  = y0 + (bh - sv * uh) / 2.0

        self.ax_fac['S'].set_position([L + H * s, B,                    W * s, H * sv])
        self.ax_plan  .set_position([L + H * s, B + H * sv,             W * s, D * sv])
        self.ax_fac['N'].set_position([L + H * s, B + (H + D) * sv,     W * s, H * sv])
        self.ax_fac['W'].set_position([L,         B + H * sv,           H * s, D * sv])
        self.ax_fac['E'].set_position([L + (H + W) * s, B + H * sv,     H * s, D * sv])
        # Inset inside the corner rather than filling it: at full size the compass reads as
        # heavy as the facade strips it sits between, and its label runs into the E facade's
        # title.  0.58 keeps it square (the corner is H x H metres) and clearly secondary.
        cw, ch = H * s * 0.58, H * sv * 0.58
        self.ax_north.set_position([L + (H + W) * s + (H * s - cw) / 2,
                                    B + (H + D) * sv + (H * sv - ch) / 2, cw, ch])
        self.ax_state.set_position([L, B + (H + D) * sv, H * s, H * sv])

        self.ax_plan.set_xlim(0, W)
        self.ax_plan.set_ylim(0, D)
        self.im_plan.set_extent([0, W, 0, D])

    # ------------------------------------------------- state changes
    def _on_scene(self, **kw):
        for k, v in kw.items():
            setattr(self, "city_idx" if k == "city_idx" else k, v)
        if "month" in kw or "day" in kw:
            # Clamp Feb 30 etc. rather than silently missing the .wea row.
            self.day = min(self.day, days_in_month(self.month))
        self._request()

    def _on_room(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.occupants = [o.clamped(self.width, self.depth) for o in self.occupants]
        self._layout_cross()
        self._request()

    def _on_target(self, v):
        o = self._sel()
        if o is None:
            return
        o.target_lux = float(v)
        o.activity   = nearest_activity(o.target_lux)
        self._request()

    def _on_radius(self, v):
        o = self._sel()
        if o is None:
            return
        o.radius_m = float(v)
        self.occupants = [x.clamped(self.width, self.depth) for x in self.occupants]
        self._request()

    def _on_wall(self, label):
        side = label.split()[0]
        self.sides_open[side] = not self.sides_open[side]
        if not self.sides_open[side]:
            # Close those panels NOW rather than waiting for the re-solve.  Honest: shutting a
            # wall is something the physical facade can do immediately, unlike re-deciding
            # where the remaining windows should go.
            self.binary = self.binary.copy()
            self.binary[SIDES.index(side)] = 0
        self._lux_dirty = True
        self._request()

    def _on_auto(self, _label):
        self.auto = not self.auto
        self._refresh_all()

    def _on_find(self, text):
        text = text.strip()
        if not text:
            return
        hits = [i for i, c in enumerate(self.cities) if text.lower() in city_label(c).lower()]
        if not hits:
            self.status = f"no city matching {text!r}"
            self._refresh_all()
            return
        self.sld_city.set_val(hits[0])      # fires _on_scene

    # ------------------------------------------------- occupants
    def _sel(self):
        if not self.occupants:
            return None
        self.selected = int(np.clip(self.selected, 0, len(self.occupants) - 1))
        return self.occupants[self.selected]

    def _sync_occupant_sliders(self):
        """Point the per-occupant sliders at the selection WITHOUT re-triggering a solve —
        Slider.set_val fires on_changed, which would otherwise make selecting someone
        overwrite their target with the previous selection's value."""
        o = self._sel()
        if o is None:
            return
        for sld, val in ((self.sld_tgt, o.target_lux), (self.sld_rad, o.radius_m)):
            sld.eventson = False
            sld.set_val(val)
            sld.eventson = True

    def _spawn(self, x=None, y=None):
        if len(self.occupants) >= len(PALETTE):
            self.status = f"{len(PALETTE)} occupants is the palette limit"
            self._refresh_all(); return
        o = Occupant(self._take_oid(),
                     self.width / 2 if x is None else x,
                     self.depth / 2 if y is None else y,
                     500.0, DEFAULT_RADIUS, "Reading").clamped(self.width, self.depth)
        self.occupants.append(o)
        self.selected = len(self.occupants) - 1
        self._sync_occupant_sliders()
        self._request()

    def _delete(self):
        if not self.occupants:
            return
        self.occupants.pop(self.selected)
        self.selected = max(0, self.selected - 1)
        self._sync_occupant_sliders()
        self._request()

    # ------------------------------------------------- input
    def _on_key(self, event):
        k = (event.key or "").lower()
        moves = {"w": (0, 1), "up": (0, 1), "s": (0, -1), "down": (0, -1),
                 "a": (-1, 0), "left": (-1, 0), "d": (1, 0), "right": (1, 0)}
        if k in moves:
            o = self._sel()
            if o is None:
                return
            dx, dy = moves[k]
            o.x += dx * MOVE_STEP
            o.y += dy * MOVE_STEP
            self.occupants = [x.clamped(self.width, self.depth) for x in self.occupants]
            self._request()
        elif k == "x":
            self._delete()
        elif k == " ":
            self._request(force_cold=True)
        elif k == "tab":
            if self.occupants:
                self.selected = (self.selected + 1) % len(self.occupants)
                self._sync_occupant_sliders()
                self._refresh_all()
        elif k == "p":
            self._save_png()

    def _on_click(self, event):
        if event.inaxes is not self.ax_plan or event.xdata is None:
            return
        if event.key == "shift":
            self._spawn(event.xdata, event.ydata)
            return
        # Nearest occupant whose footprint contains the click; falls back to nearest centre so
        # a click just outside a small ring still selects rather than doing nothing.
        if not self.occupants:
            return
        d = [np.hypot(o.x - event.xdata, o.y - event.ydata) for o in self.occupants]
        inside = [i for i, o in enumerate(self.occupants) if d[i] <= o.radius_m]
        pick = min(inside, key=lambda i: d[i]) if inside else int(np.argmin(d))
        if not inside and d[pick] > 1.5:
            return
        self.selected = pick
        self._sync_occupant_sliders()
        self._refresh_all()

    # ------------------------------------------------- solving
    def _snapshot(self):
        return Scene(
            city_path=self.cities[self.city_idx],
            month=self.month, day=self.day, hour=self.hour, north_angle=self.north,
            width=self.width, depth=self.depth, height=self.height,
            sides_open=tuple(self.sides_open[s] for s in SIDES),
            # Copy every occupant: Occupant is mutable and the UI keeps editing its own list
            # while the worker is mid-solve on this snapshot.
            occupants=tuple(replace(o) for o in self.occupants),
            seq=self.seq)

    def _cond_key(self):
        """Everything the LUX FIELD depends on: the sun and the room.

        Not Scene.structural_key(), which also carries sides_open — a wall toggle changes the
        facade, not the conditions, and is handled by masking the binary in _on_wall.  And
        pointedly not the occupants: where people stand changes what the facade should BE, never
        what light is currently falling on the floor.
        """
        return (self.cities[self.city_idx], self.month, self.day, self.hour, self.north,
                round(self.width, 3), round(self.depth, 3), round(self.height, 3))

    def _request(self, force_cold=False):
        self.seq += 1
        # Whatever settled, settled for the OLD scene.  Clear it the moment anything changes,
        # or the panel would go on claiming the facade is the answer to a question that has
        # already moved on.
        self._converged, self._stop_reason = None, ""
        if self._cond_key() != self._lux_key:
            self._lux_dirty = True
        if self.auto or force_cold:
            self.worker.submit(self._snapshot(), force_cold=force_cold)
        self._refresh_all()

    def _maybe_predict(self):
        """At most ONE forward pass per UI tick, and only when something it depends on moved.

        Not called from the slider callbacks directly: a drag fires 20-30 events/s and each
        would be its own GPU round trip.  Coalescing to the 80 ms tick caps this at ~12/s.
        Being synchronous is what makes it correct for free — the map is by construction the
        light of the state that existed at this instant, so there is no staleness to track.
        """
        if not self._lux_dirty or not self.worker.ready.is_set():
            return False
        key = self._cond_key()
        out = self.worker.predict(self._snapshot(), self.binary)
        if out is None:
            return False
        lux, _sun, note = out
        self.lux        = lux
        self._lux_key   = key
        self._lux_dirty = False
        # Kept OUT of self.status, which means "how the last solve went".  Writing the note
        # there left it on screen after the cause had gone: drag past 19:00 and back to 16:00
        # and the panel still claimed the sun was below the training floor while its own header
        # read alt 35 degrees.  This field is rewritten (to None) by every predict.
        self._lux_note = note
        return True

    def _collect(self):
        """Drain the worker; returns True if anything changed on screen."""
        if self.worker.error is not None:
            self.status = f"solver failed to start: {self.worker.error}"
            self.worker.error = None
            return True
        r = self.worker.poll()
        if r is None:
            # No finished solve — but the optimizer may have improved on its best design
            # mid-flight.  Show it: the facade then visibly converges instead of sitting on the
            # old design for the whole solve and snapping at the end.  The map follows because
            # the facade changed, via the usual dirty flag.
            p = self.worker.poll_preview()
            if p is None:
                return False
            self.binary      = p.binary
            self._lux_dirty  = True
            self._preview_L  = p.best_L
            self._converged  = None      # still searching; nothing has settled yet
            # A preview is being designed for the scene the worker is on, so it retires the
            # previous solve's "designed for 09:00" note the moment it lands.
            self._facade_key = (p.scene.month, p.scene.day, p.scene.hour, p.scene.city_path)
            return True
        self.last = r
        self._preview_L = None
        self._converged = r.converged
        self._stop_reason = r.stop_reason
        if r.mode == "error":
            self.status = r.message
            return True
        self.binary      = r.binary
        self.status      = r.message
        self._facade_key = (r.scene.month, r.scene.day, r.scene.hour, r.scene.city_path)
        # Take the facade but NOT r.lux, and re-predict instead.  r.lux is the light for the
        # scene the solve STARTED on; if the hour slider has moved since, painting it would show
        # 09:00 light under a 15:00 sun.  One extra ~8 ms forward makes that unrepresentable.
        self._lux_dirty = True
        return True

    # ------------------------------------------------- drawing
    def _refresh_all(self):
        # Scored ONCE per redraw and shared by the plan rings and the table, so the two can
        # never tell different stories about the same occupant.
        #
        # Note what is being scored: the occupants' CURRENT positions against the LAST SOLVED
        # facade's lux field.  That is the physically right question — "what light is this
        # person standing in, under the facade the building currently has" — and it is what
        # makes the demo read correctly: walk out of your patch and the number drops at once,
        # then recovers a second later as the facade catches up with you.
        self._owner = occupant_owner_grid(self.occupants, self.width, self.depth)
        self._stats = zone_stats(self.occupants, self._owner, self.lux)
        self._refresh_plan()
        self._refresh_facades()
        self._refresh_info()
        self._draw()

    def _draw(self):
        self.fig.canvas.draw_idle()

    def _refresh_plan(self):
        self.im_plan.set_data(self.lux)
        for a in self._plan_artists:
            a.remove()
        self._plan_artists = []
        ax = self.ax_plan

        for i, (o, st) in enumerate(zip(self.occupants, self._stats)):
            col  = PALETTE[i % len(PALETTE)]
            sel  = (i == self.selected)
            # The ring says SERVED / NOT SERVED, not what was asked for.  Colouring it by the
            # target from LUX_CMAP was the first attempt and it fails at the dark end of the
            # ramp: a 200-lux ring is near-navy, drawn on the dim floor that a 200-lux zone
            # sits in by definition, so the people most easily satisfied were the hardest to
            # see.  Identity moves to the centre dot and the label, which are readable
            # everywhere.
            self._plan_artists.append(ax.add_patch(Circle(
                (o.x, o.y), o.radius_m, fill=False, ec="white", lw=3.0, zorder=3, alpha=0.9)))
            self._plan_artists.append(ax.add_patch(Circle(
                (o.x, o.y), o.radius_m, fill=False, ec=served_color(st["in_band"]), lw=1.6,
                zorder=4)))
            if sel:
                self._plan_artists.append(ax.add_patch(Circle(
                    (o.x, o.y), o.radius_m + 0.14, fill=False, ec=col, lw=1.4, ls=(0, (4, 3)),
                    zorder=5)))
            self._plan_artists.append(ax.add_patch(Circle(
                (o.x, o.y), min(0.15, o.radius_m * 0.32), facecolor=col, ec="white", lw=1.4,
                zorder=6)))
            got = "—" if st["mean"] != st["mean"] else f"{st['mean']:.0f}"
            self._plan_artists.append(ax.text(
                o.x, o.y + o.radius_m + 0.18, f"{i+1}   {o.target_lux:.0f} → {got} lx",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if sel else "normal", color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="white", lw=0.8, alpha=0.94)))

        self._refresh_north()

        # Three states worth telling apart at a glance: still searching, settled on a local
        # minimum, or stopped only because it ran out of steps.  The last is the one that says
        # "more budget would help" rather than "this is as good as this basin gets".
        self._refresh_state_badge()

    def _refresh_state_badge(self):
        """What the optimizer is doing, in the cross's top-left corner.

        Four states worth telling apart.  The two "settled" ones matter most: the search stops
        improving either because it has met the brief or because it is stuck, and only the
        second is worth re-rolling.  "stopped on budget" is neither — it was still gaining when
        the step count ran out.
        """
        if self.worker.busy:
            tag, col = "● solving…", "#b06000"
        elif self._converged is True:
            if "local minimum" in (self._stop_reason or ""):
                tag, col = "■ settled\nlocal minimum", "#b06000"
            else:
                tag, col = "■ settled", SERVED_OK
        elif self._converged is False:
            tag, col = "▸ stopped on\nstep budget", "#777777"
        else:
            tag, col = "", INK
        self._state_txt.set_text(tag)
        self._state_txt.set_color(col)
        self._state_txt.set_bbox(
            dict(boxstyle="round,pad=0.42", fc="#f6f6f6", ec=col, lw=1.1, alpha=0.95)
            if tag else None)

    def _refresh_north(self):
        """Compass, drawn in its own corner axes in 0..1 axes coordinates.

        +Y is at azimuth -north_angle, so true north sits at north_angle CLOCKWISE from +Y —
        plan direction (sin, cos).  Cross-checked against the wall-azimuth table in
        annual_conditions(): at north=90 this points to room-local E, which is the wall that
        table says faces north.
        """
        a = self.ax_north
        for art in self._north_artists:
            art.remove()
        self._north_artists = []

        th = np.radians(self.north)
        dx, dy = np.sin(th), np.cos(th)
        # R and the label offset are set so the "N" stays inside the axes at EVERY angle —
        # text is not clipped to its axes, so an overhanging label would collide with whichever
        # facade title happens to sit that way.
        R = 0.26
        self._north_artists.append(a.add_patch(Circle(
            (0.5, 0.5), R + 0.12, facecolor="#f4f4f4", ec="#d8d8d8", lw=0.8)))
        self._north_artists.append(a.add_patch(FancyArrow(
            0.5 - dx * R, 0.5 - dy * R, dx * 2 * R, dy * 2 * R,
            width=0.035, head_width=0.15, head_length=0.17,
            length_includes_head=True, fc="#1a1a1a", ec="none")))
        self._north_artists.append(a.text(
            0.5 + dx * (R + 0.22), 0.5 + dy * (R + 0.22), "N",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1a1a1a"))

    def _refresh_facades(self):
        n, s, w, e = facade_displays(self.binary)
        az = wall_azimuths(self.north)
        for side, img in (("N", n), ("S", s), ("W", w), ("E", e)):
            a = self.ax_fac[side]
            self.im_fac[side].set_data(img)
            open_ = self.sides_open[side]
            col   = OPEN_COLOR if open_ else CLOSED_COLOR
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.0)
            n_win = int(self.binary[SIDES.index(side)].sum())
            a.set_title(f"{side}  {az[side]:.0f}°" + (f"   {n_win} open" if open_ else "  solid"),
                        fontsize=8, color=col, fontweight="bold", pad=3)

    def _refresh_info(self):
        city = city_label(self.cities[self.city_idx])
        _, sun = single_condition(self.cities[self.city_idx], self.month, self.day,
                                  self.hour, self.north)
        walls = " ".join(s if self.sides_open[s] else f"({s})" for s in SIDES)
        L = [
            f"{city[:46]}",
            f"{self.month:02d}/{self.day:02d}  {self.hour:02d}:00     "
            f"sun  alt {sun['alt']:5.1f}°   az {sun['az']:5.1f}°",
            f"DNI {sun['dni']:6.1f}   DHI {sun['dhi']:6.1f} W/m²",
            f"room {self.width:.1f} × {self.depth:.1f} × {self.height:.1f} m"
            f"    north {self.north:.0f}°    walls {walls}",
            "",
        ]
        if not sun["ok"]:
            L.append(f"!! sun is below the {MIN_ALT:.0f}° training floor — the surrogate has")
            L.append("   never seen a sun this low, so no design is attempted here.")
            L.append("")

        L.append("  #  activity          asked   getting   in band   served")
        L.append("  " + "─" * 52)
        for i, (o, st) in enumerate(zip(self.occupants, self._stats)):
            mark = ">" if i == self.selected else " "
            got  = "   —  " if st["mean"] != st["mean"] else f"{st['mean']:6.0f}"
            band = "  — " if st["in_band"] != st["in_band"] else f"{st['in_band']:3.0f}%"
            tag  = {SERVED_OK: "  yes", SERVED_NEAR: " near",
                    SERVED_BAD: "   NO"}.get(served_color(st["in_band"]), "    —")
            L.append(f" {mark}{i+1}  {o.activity:<15.15s} {o.target_lux:5.0f}   {got}    "
                     f"{band}   {tag}")
        if not self.occupants:
            L.append("     (nobody here — Spawn, or shift-click the plan)")
        L.append("")
        L.append("  ring colour = served / near / not served    dot = who")
        L.append("")
        L.append(f"glazing {self.binary.mean()*100:5.1f}%     "
                 f"{'AUTO' if self.auto else 'MANUAL — press space or Re-solve'}")
        if self._preview_L is not None:
            L.append(f"  optimizer improving — best so far L {self._preview_L:.4f}")
        elif self._converged is True:
            L.append(f"  ■ {self._stop_reason}")
            # Offer the re-roll only when it would buy something.  A cold solve starts from a
            # fresh random init and the layout is non-unique, so another run really can land in
            # a better basin — but suggesting it when everyone is already in band is noise.
            if "local minimum" in self._stop_reason:
                L.append("    press space to re-solve from a new start")
        elif self._converged is False:
            L.append(f"  ▸ {self._stop_reason}")
        # The map is live but the facade lags, so never let the panel imply they agree: say
        # outright when the design on the walls was made for a different moment.  This is the
        # state the whole decoupling exists to make visible — a static facade under a moved sun.
        if self._facade_key is not None:
            f_mon, f_day, f_hr, f_city = self._facade_key
            if (f_mon, f_day, f_hr) != (self.month, self.day, self.hour) \
                    or f_city != self.cities[self.city_idx]:
                L.append(f"  facade designed for {f_mon:02d}/{f_day:02d} {f_hr:02d}:00"
                         f" — showing it under {self.month:02d}/{self.day:02d} "
                         f"{self.hour:02d}:00")
        L.append("")
        # The below-horizon case already has its own two-line block above, so only surface the
        # predict's note when it is saying something that block does not.
        note = self._lux_note if sun["ok"] else None
        for line in (note, self.status):
            if line:
                L.extend(textwrap.wrap(line, WRAP_CHARS) or [""])

        # Fit the panel.  It has to hold anywhere from 14 lines (one occupant) to 27 (eight,
        # plus a below-horizon warning, a wrapped status and a facade-mismatch note), and at a
        # fixed size the long cases ran off the bottom of the axes into the sliders —
        # matplotlib does not clip text to its axes, so it silently overprinted them.
        #
        # Shrink first, but only to MIN_INFO_FONTSIZE; past that, shed DECORATION rather than
        # keep shrinking, because 5pt type is not a readout.  Order matters: the ring legend is
        # learn-once and goes first, then blank separators, and nothing carrying a number is
        # ever dropped.
        h_pt      = self.ax_info.get_position().height * self.fig.get_size_inches()[1] * 72.0
        max_lines = int(h_pt / (MIN_INFO_FONTSIZE * INFO_LINESPACING))
        if len(L) > max_lines:
            L = [x for x in L if not x.lstrip().startswith("ring colour")]
        if len(L) > max_lines:
            L = [x for i, x in enumerate(L) if x or (i and L[i - 1])]   # collapse double blanks
        if len(L) > max_lines:
            L = [x for x in L if x]                                     # last resort: no blanks

        fs = float(np.clip(h_pt / (max(len(L), 1) * INFO_LINESPACING),
                           MIN_INFO_FONTSIZE, INFO_FONTSIZE))
        self._info_txt.set_fontsize(fs)
        self._info_txt.set_text("\n".join(L))

    def _save_png(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        city = city_label(self.cities[self.city_idx]).split(".")[0][:24]
        p = OUT_DIR / (f"responsive_{city}_{self.month:02d}{self.day:02d}"
                       f"_{self.hour:02d}h_{len(self.occupants)}p.png")
        self.fig.savefig(p, dpi=150, facecolor=BG)
        self.status = f"saved {p.name}"
        self._refresh_all()
        print(f"  saved -> {p}", flush=True)

    # ------------------------------------------------- run
    def run(self):
        timer = self.fig.canvas.new_timer(interval=POLL_MS)

        def tick():
            changed = self._collect()
            # Predict AFTER collecting, so a facade that just landed is reflected on this same
            # tick rather than one tick later.
            changed = self._maybe_predict() or changed
            # Repaint whenever anything lands, and also while a solve is in flight so the
            # "solving" indicator and the live occupant readout keep moving.
            if changed or self.worker.busy:
                self._refresh_all()

        timer.add_callback(tick)
        timer.start()
        self._timer = timer
        self.worker.ready.wait(timeout=120)
        self._request()
        plt.show()
        self.worker.stop()


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city",   default="Houston")
    ap.add_argument("--month",  type=int,   default=6)
    ap.add_argument("--day",    type=int,   default=21)
    ap.add_argument("--hour",   type=int,   default=12)
    ap.add_argument("--north",  type=float, default=0.0)
    ap.add_argument("--width",  type=float, default=6.0)
    ap.add_argument("--depth",  type=float, default=8.0)
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--screenshot", metavar="PATH",
                    help="solve once, save the window to PATH and exit (no GUI). Used to "
                         "check the layout without a display.")
    a = ap.parse_args(argv)

    app = ResponsiveFacadeApp(city=a.city, month=a.month, day=a.day, hour=a.hour,
                              north=a.north, width=a.width, depth=a.depth, height=a.height,
                              headless=bool(a.screenshot))
    if a.screenshot:
        app.worker.ready.wait(timeout=300)
        app.worker.submit(app._snapshot())
        app.worker.wait_idle()
        app._collect()
        app._maybe_predict()        # _collect only takes the facade; the map comes from here
        app._refresh_all()
        Path(a.screenshot).parent.mkdir(parents=True, exist_ok=True)
        app.fig.savefig(a.screenshot, dpi=110, facecolor=BG)
        print(f"screenshot -> {a.screenshot}\n{app.status}")
        app.worker.stop()
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
