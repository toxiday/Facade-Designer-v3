"""
unit_zone_matrix.py — the occupancy-brief figure: rooms (or whole units) x time of day, as a
matrix of painted target-lux plans.

Where result.png answers "what does the design deliver?", this answers "what was it asked
for?".  Every cell is a floor plan with its painted behaviour zone drawn in the SAME lux
colour ramp result.png uses — same LUX_CMAP, same 0..GLARE_LUX limits — so a colour means the
same illuminance in both pictures and the pair can be read against each other.

    python unit_zone_matrix.py "../All Units/101"                  # one row per ROOM
    python unit_zone_matrix.py "../All Units" --by unit            # one row per UNIT
    python unit_zone_matrix.py "../All Units/101" --check-layout   # validate, draw nothing

TWO MODES
    --by room   rows are rooms, each cell that room's own 32x32 plan.  Needs only the job
                JSONs, so it works before anything has been placed or optimized.
    --by unit   rows are UNITS, each cell the whole apartment with every room composited
                into one image at its real position.  Needs a `unit/1` manifest
                (unit.json) written by gh_write_unit_jobs.py — room placement cannot be
                recovered from the per-room `placement` blocks, which do not agree with
                one another (measured on unit 101: two overlapping pairs, one room 10 m
                adrift).  --check-layout reports exactly that, on any manifest.

TWO CONVENTIONS WORTH KNOWING
    * A cell's value is a TARGET lux, not a prediction.  `0` therefore means "nobody is
      here", which is a different statement from result.png's "0 lux predicted" — so
      unoccupied floor is masked to neutral grey rather than painted with the bottom of the
      ramp.
    * Slots that fall in the same column are merged with `max`, the same rule the pipeline
      already uses for overlapping zones (see FACADE_JOB_SCHEMA.md).  The cell caption always
      names the hours that were folded in, so the merge is visible rather than silent.

Frames are room-local throughout, exactly as in facade_viz: grid row 0 = the +Y ("N") edge,
col 0 = the x=0 ("W") edge.  Those letters are wall ids, not compass directions.
"""

import re
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np

# Agg before anything imports pyplot: this is a figure-writing script, and importing
# facade_viz (which pulls in pyplot) must not try to open a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Rectangle, Polygon                        # noqa: E402
from matplotlib.colors import ListedColormap                             # noqa: E402
from matplotlib.path import Path as MplPath                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facade_viz import LUX_CMAP                                          # noqa: E402
from optimize_facade_allsides_climate import GRID_N, GLARE_LUX           # noqa: E402

HERE      = Path(__file__).resolve().parent
BRIDGE    = HERE.parent / "rhino bridge" / "gh_write_unit_jobs.py"

# ---------------------------------------------------------------------------
# Time ranges
# ---------------------------------------------------------------------------
# Half-open [start, end).  The first bin's start is 0, not 8: three rooms in unit 101 are
# occupied at 07:00, and a literal 8am edge would silently drop a real slot from a figure
# whose whole job is to show the complete brief.  The header states the widening, and each
# cell names the hours it actually holds.
TIME_RANGES = [
    (0,  11, "Morning",   "to 11am"),
    (11, 14, "Midday",    "11am - 2pm"),
    (14, 17, "Afternoon", "2pm - 5pm"),
    (17, 24, "Evening",   "5pm - 8pm"),
]

UNOCCUPIED_COLOR = "#e9e9e9"     # neutral grey: "nobody here", NOT "0 lux"
ZONE_EDGE_COLOR  = "white"       # matches facade_viz's white zone contour
ROOM_EDGE_COLOR  = "#666666"
MARGIN_IN        = 0.13          # blank border around each plan, INCHES — constant across
                                 # rows, so it does not scale with the room and distort the
                                 # size comparison
UNIT_RES_M       = 0.04          # composite raster resolution in unit mode


def load_layout_checker():
    """Borrow check_layout/rect_corners from the Grasshopper writer.

    Imported rather than reimplemented: two implementations of "do these rooms overlap"
    would eventually disagree, and the disagreement would surface as a drawing that looks
    fine while the jobs it came from do not tile.
    """
    if not BRIDGE.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_ghunit", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SUN_CACHE = {}


def hours_without_sun(city, hours):
    """Which of `hours` never see the sun above 5 deg in this climate.

    The optimizer SKIPS such a slot outright — it has no daylight to design for — so the
    zone painted at that hour is silently ignored.  Houston has no sun at 19:00 on any day
    of the year, and three rooms across units 102/201/202 were briefed at exactly that hour.

    Flagged here, on the picture of the brief, because this is the last point before an hour
    of GPU is spent: the alternative is discovering it afterwards by counting rows in
    result.png.
    """
    hours = sorted(set(hours))
    if not city or not hours:
        return set()
    cache = _SUN_CACHE.setdefault(str(city), {})
    todo = [h for h in hours if h not in cache]
    if todo:
        try:
            from optimize_facade_allsides_climate import resolve_city, annual_conditions
            wea = resolve_city(city)
        except Exception as e:                       # noqa: BLE001 — advisory only
            print(f"  [!] could not check sun hours for {city!r}: {e}")
            return set()
        for h in todo:
            try:
                _, _, cond, _ = annual_conditions(wea, h)
                cache[h] = (len(cond) == 0)
            except Exception:                        # noqa: BLE001
                cache[h] = False
    return {h for h in hours if cache.get(h)}


def dead_hour_note(dead):
    """The footer line explaining the asterisks, or empty when every hour has sun."""
    if not dead:
        return ""
    return ("   * %s — the sun is never above 5° at %s hour in this climate, so the "
            "optimizer SKIPS that slot and its zone is not designed for."
            % (", ".join("%02d:00" % h for h in sorted(dead)),
               "that" if len(dead) == 1 else "those"))


def natural_key(path):
    """Sort 101.10 after 101.9 rather than after 101.1 — a plain string sort puts a
    ten-room unit in the wrong order, and the figure's row order is the reading order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_slots(slots, label=""):
    """Job slots -> one merged entry per time range (None where there is no activity)."""
    cells = [None] * len(TIME_RANGES)
    for s in slots:
        hour = int(s["hour"])
        col = next((i for i, (lo, hi, *_) in enumerate(TIME_RANGES) if lo <= hour < hi), None)
        if col is None:                      # unreachable with the ranges above
            print(f"  [!] {label} {hour:02d}:00 falls in no time range — skipped")
            continue
        g = np.asarray(s["grid"], dtype=np.float32).reshape(GRID_N, GRID_N)
        if cells[col] is None:
            cells[col] = dict(grid=g, hours=[hour])
        else:
            # Same rule the pipeline uses for overlapping zones: max, never sum — two
            # activities in one room do not make the room brighter, they make its
            # brightest demand the one to satisfy.
            cells[col]["grid"] = np.maximum(cells[col]["grid"], g)
            cells[col]["hours"].append(hour)

    for c in cells:
        if c:
            c["hours"].sort()
            occ = c["grid"] > 0
            c["levels"] = sorted({round(float(v)) for v in c["grid"][occ]})
            c["n_cells"] = int(occ.sum())
    return cells


def load_rooms(unit_dir, only=None):
    """Read every facade_job/1 in the unit folder into a light dict."""
    unit_dir = Path(unit_dir)
    if not unit_dir.is_dir():
        raise SystemExit(f"unit folder not found: {unit_dir}")

    paths = sorted((p for p in unit_dir.glob("*.json") if p.name != "unit.json"),
                   key=natural_key)
    if only:
        want = {o.removesuffix(".json") for o in only}
        paths = [p for p in paths if p.stem in want]
    if not paths:
        raise SystemExit(f"no job JSONs in {unit_dir}"
                         + (f" matching {sorted(only)}" if only else ""))

    rooms = []
    for p in paths:
        job = json.loads(p.read_text(encoding="utf-8"))
        if job.get("schema") not in (None, "facade_job/1"):
            continue
        r = job.get("room") or {}
        rooms.append(dict(
            id=p.stem, path=p,
            width=float(r.get("width", 0.0)), depth=float(r.get("depth", 0.0)),
            height=float(r.get("height", 0.0)),
            north_angle=float(r.get("north_angle", 0.0)),
            sides=str(job.get("sides", "SNEW")).upper(),
            city=(job.get("climate") or {}).get("city", ""),
            slots=job.get("slots", []),
            cells=bin_slots(job.get("slots", []), p.stem)))
    return rooms


def load_unit(unit_dir):
    """Read a `unit/1` manifest plus the job files it names."""
    unit_dir = Path(unit_dir)
    man = unit_dir / "unit.json"
    if not man.is_file():
        raise SystemExit(
            f"no unit.json in {unit_dir}.\n"
            "A unit row needs to know where each room sits inside the apartment, and that\n"
            "cannot be recovered from the per-room `placement` blocks — they were written\n"
            "one room at a time and do not agree (unit 101 has two overlapping pairs).\n"
            "Produce it with the gh_write_unit_jobs.py Grasshopper component, or hand-write\n"
            "a unit/1 manifest and check it with --check-layout.")
    m = json.loads(man.read_text(encoding="utf-8"))
    if m.get("schema") != "unit/1":
        raise SystemExit(f"{man} is not a unit/1 manifest (schema={m.get('schema')!r})")

    u = m.get("unit") or {}
    rooms = []
    for e in m.get("rooms", []):
        jp = unit_dir / e.get("job_file", f"{e['name']}.json")
        if not jp.is_file():
            raise SystemExit(f"{man.name} names {jp.name}, which does not exist")
        job = json.loads(jp.read_text(encoding="utf-8"))
        rooms.append(dict(
            name=e["name"],
            width=float(e["width"]), depth=float(e["depth"]),
            sides=e.get("sides", ""),
            origin_uv=tuple(e["origin_uv"]),
            x_axis_uv=tuple(e["x_axis_uv"]),
            y_axis_uv=tuple(e["y_axis_uv"]),
            slots=job.get("slots", []),
            cells=bin_slots(job.get("slots", []), e["name"])))

    # The apartment outline. `unit.width`/`depth` are only its bounding box — flats are
    # L-shaped and notched far more often than rectangular, so the polygon is the shape and
    # the box is just the raster's extent. Older manifests without it fall back to the box.
    boundary = [tuple(p) for p in m.get("boundary_uv", [])]
    if len(boundary) < 3:
        w, d = float(u.get("width", 0.0)), float(u.get("depth", 0.0))
        boundary = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
        print(f"  [!] {man.name} has no boundary_uv — assuming a rectangular "
              f"{w:.2f} x {d:.2f} m apartment")

    return dict(id=m.get("name", unit_dir.name), path=unit_dir,
                width=float(u.get("width", 0.0)), depth=float(u.get("depth", 0.0)),
                city=(m.get("climate") or {}).get("city", ""),
                boundary=boundary, rooms=rooms, manifest=m)


# ---------------------------------------------------------------------------
# Compositing a unit
# ---------------------------------------------------------------------------

def composite_unit(unit, col, res=UNIT_RES_M):
    """(lux, in_room) rasters of the whole apartment for one time range.

    Built by INVERSE mapping: for each unit pixel, transform into each room's own frame and
    read the cell it lands in.  Nearest-neighbour, never interpolated — these are per-cell
    illuminance TARGETS, and blending two of them would invent a level nobody asked for.
    """
    nx = max(1, int(np.ceil(unit["width"] / res)))
    ny = max(1, int(np.ceil(unit["depth"] / res)))
    lux = np.zeros((ny, nx), dtype=np.float32)
    inr = np.zeros((ny, nx), dtype=bool)

    # Pixel centres in unit-local metres.  Row 0 is drawn at the TOP, which is v = depth.
    us = (np.arange(nx) + 0.5) * res
    vs = unit["depth"] - (np.arange(ny) + 0.5) * res

    for room in unit["rooms"]:
        cell = room["cells"][col]
        W, D = room["width"], room["depth"]
        ox, oy = room["origin_uv"]
        xx, xy = room["x_axis_uv"]
        yx, yy = room["y_axis_uv"]

        # Only visit the pixels inside this room's unit-space bounding box.
        cx = [ox, ox + xx * W, ox + xx * W + yx * D, ox + yx * D]
        cy = [oy, oy + xy * W, oy + xy * W + yy * D, oy + yy * D]
        j0 = max(0, int(np.floor(min(cx) / res)) - 1)
        j1 = min(nx, int(np.ceil(max(cx) / res)) + 1)
        i0 = max(0, int(np.floor((unit["depth"] - max(cy)) / res)) - 1)
        i1 = min(ny, int(np.ceil((unit["depth"] - min(cy)) / res)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue

        U, V = np.meshgrid(us[j0:j1], vs[i0:i1])
        dx, dy = U - ox, V - oy
        x = dx * xx + dy * xy                 # room-local metres
        y = dx * yx + dy * yy

        inside = (x >= 0) & (x < W) & (y >= 0) & (y < D)
        if not inside.any():
            continue
        inr[i0:i1, j0:j1] |= inside

        if cell is None:
            continue                          # room exists here but is unoccupied now
        # Grid convention: col 0 = the x=0 edge, row 0 = the +Y edge, so y runs the other way.
        cc = np.clip((x * GRID_N / W).astype(np.int32), 0, GRID_N - 1)
        rr = np.clip(GRID_N - 1 - (y * GRID_N / D).astype(np.int32), 0, GRID_N - 1)
        vals = cell["grid"][rr, cc]
        block = lux[i0:i1, j0:j1]
        np.copyto(block, np.maximum(block, vals), where=inside)
        lux[i0:i1, j0:j1] = block

    return lux, inr


def boundary_mask(unit, res=UNIT_RES_M):
    """Which raster pixels fall inside the apartment outline.

    Cached on the unit: it is the same for all four time ranges, and point-in-polygon over
    a ~1400x1150 raster is not worth repeating.
    """
    if unit.get("_mask") is not None:
        return unit["_mask"]
    nx = max(1, int(np.ceil(unit["width"] / res)))
    ny = max(1, int(np.ceil(unit["depth"] / res)))
    us = (np.arange(nx) + 0.5) * res
    vs = unit["depth"] - (np.arange(ny) + 0.5) * res
    U, V = np.meshgrid(us, vs)
    inside = MplPath(np.asarray(unit["boundary"], dtype=float)).contains_points(
        np.column_stack([U.ravel(), V.ravel()])).reshape(ny, nx)
    unit["_mask"] = inside
    return inside


def unit_cell_summary(unit, col):
    """Caption data for a unit cell, or None when no room is occupied in this range."""
    hours, levels, n_cells, n_rooms = set(), set(), 0, 0
    for room in unit["rooms"]:
        c = room["cells"][col]
        if not c:
            continue
        n_rooms += 1
        hours.update(c["hours"])
        levels.update(c["levels"])
        n_cells += c["n_cells"]
    if not n_rooms:
        return None
    return dict(hours=sorted(hours), levels=sorted(levels),
                n_cells=n_cells, n_rooms=n_rooms)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def fmt_levels(levels):
    """Lux levels for a caption.

    A room asks for one or two levels and they are worth naming.  A whole unit can ask for
    nine, and spelling all of them out makes the caption wider than the column it belongs to
    — so past four, the span and the count say the same thing in a third of the width.
    """
    if len(levels) <= 4:
        return " · ".join(f"{v:g}" for v in levels) + " lux"
    return f"{min(levels):g}–{max(levels):g} lux ({len(levels)} levels)"


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _edge_letters(ax, W, D, margin_m):
    """Room-local wall ids on the plan edges — the same W/E, N/S labelling facade_viz uses,
    so the two figures are read the same way round."""
    tkw = dict(fontsize=5.5, color="#777777", ha="center", va="center")
    ax.text(-margin_m / 2, D / 2, "W", **tkw)
    ax.text(W + margin_m / 2, D / 2, "E", **tkw)
    ax.text(W / 2, D + margin_m / 2, "N", **tkw)
    ax.text(W / 2, -margin_m / 2, "S", **tkw)


def draw_room_cell(ax, room, col, margin_m):
    """One room x one time range.

    An empty cell is a PLAIN WHITE image — no frame, no outline, no label.  The gaps in the
    grid are then just gaps, and the eye goes straight to where the unit is actually occupied.
    """
    _blank(ax)
    cell = room["cells"][col]
    if cell is None:
        return None

    W, D = room["width"], room["depth"]
    g = cell["grid"]

    # Mask rather than paint: 0 here means unoccupied, and the bottom of LUX_CMAP already
    # means "0 lux" in result.png.  Giving the two different renderings keeps that honest.
    cmap = LUX_CMAP.copy()
    cmap.set_bad(UNOCCUPIED_COLOR)
    im = ax.imshow(np.ma.masked_where(g <= 0.0, g), cmap=cmap, vmin=0.0, vmax=GLARE_LUX,
                   origin="upper", extent=[0.0, W, 0.0, D], aspect="equal",
                   interpolation="nearest")

    xs = (np.arange(GRID_N) + 0.5) * W / GRID_N
    ys = (np.arange(GRID_N) + 0.5) * D / GRID_N
    ax.contour(xs, D - ys, (g > 0).astype(float), levels=[0.5],
               colors=[ZONE_EDGE_COLOR], linewidths=0.9)

    ax.add_patch(Rectangle((0, 0), W, D, fill=False, edgecolor=ROOM_EDGE_COLOR,
                           linewidth=0.8, zorder=5))
    # An equal margin in METRES on all four sides.  The caller sized the axes to
    # (W + 2*margin) x (D + 2*margin) at the shared scale, so the plan lands at exactly the
    # right physical size and `aspect="equal"` has nothing left to stretch.
    ax.set_xlim(-margin_m, W + margin_m)
    ax.set_ylim(-margin_m, D + margin_m)
    _edge_letters(ax, W, D, margin_m)
    return im


def draw_unit_cell(ax, unit, col, margin_m):
    """One unit x one time range: every room composited into one apartment plan."""
    _blank(ax)
    if unit_cell_summary(unit, col) is None:
        return None

    W, D = unit["width"], unit["depth"]
    lux, inr = composite_unit(unit, col)
    inside = boundary_mask(unit)

    # THREE background tones, because these are three different statements and must not
    # share a colour: outside the flat entirely, inside the flat but not a room
    # (circulation, voids), and in a room that has no target right now.  Then the ramp,
    # only where a target actually exists.
    bg = np.zeros_like(lux)
    bg[inside] = 1.0
    bg[inr] = 2.0
    ax.imshow(bg, cmap=ListedColormap(["white", "#f6f6f6", UNOCCUPIED_COLOR]),
              vmin=0, vmax=2, origin="upper", extent=[0.0, W, 0.0, D],
              aspect="equal", interpolation="nearest")
    cmap = LUX_CMAP.copy()
    cmap.set_bad(color=(0, 0, 0, 0))                      # transparent, so grey shows through
    im = ax.imshow(np.ma.masked_where(lux <= 0.0, lux), cmap=cmap, vmin=0.0, vmax=GLARE_LUX,
                   origin="upper", extent=[0.0, W, 0.0, D], aspect="equal",
                   interpolation="nearest")

    gh = load_layout_checker()
    for room in unit["rooms"]:
        corners = (gh.rect_corners(room["origin_uv"], room["x_axis_uv"], room["y_axis_uv"],
                                   room["width"], room["depth"]) if gh else None)
        if corners is None:
            continue
        occupied = room["cells"][col] is not None
        ax.add_patch(Polygon(corners, closed=True, fill=False,
                             edgecolor=ROOM_EDGE_COLOR if occupied else "#bbbbbb",
                             linewidth=0.7, zorder=6))
        cx = sum(p[0] for p in corners) / 4.0
        cy = sum(p[1] for p in corners) / 4.0
        ax.text(cx, cy, room["name"].split(".")[-1], fontsize=5.0, zorder=7,
                ha="center", va="center",
                color="#333333" if occupied else "#aaaaaa")

    # The apartment outline as drawn, not its bounding box.
    ax.add_patch(Polygon(unit["boundary"], closed=True, fill=False, edgecolor="#333333",
                         linewidth=1.0, zorder=8))
    ax.set_xlim(-margin_m, W + margin_m)
    ax.set_ylim(-margin_m, D + margin_m)
    _edge_letters(ax, W, D, margin_m)
    return im


# ---------------------------------------------------------------------------
# Figure assembly (shared by both modes)
# ---------------------------------------------------------------------------

def build_figure(rows, title, subtitle, draw, caption, all_levels, dpi=200, note=""):
    """Lay the matrix out with explicit axes rectangles rather than a uniform gridspec.

    ONE SHARED SCALE.  Every row is drawn at the same metres-per-inch, so plan size on the
    page is size in reality: a 2.7 x 2.6 m room reads as a third of the area of a 5.4 x 7.9 m
    one.  A uniform cell size would make a bathroom and a living room look alike, which is
    exactly the comparison this figure exists to support.

    Two consequences handled here: rows are sized to their own depth (no dead band above and
    below a shallow row), and captions are placed on the fixed COLUMN width rather than on
    the axes, so a small room's caption still has room to sit.

    `rows` are dicts with id/label/width/depth; `draw(ax, row, col, margin_m)` renders one
    cell and returns its image (or None for an empty cell); `caption(row, col)` returns the
    text above it (or None).
    """
    n_rows, n_cols = len(rows), len(TIME_RANGES)

    MAX_PLAN_W, MAX_PLAN_H = 1.80, 2.95   # inches available to the largest row's plan
    COL_GAP  = 0.30
    CAP_H    = 0.46                       # caption band above each plan
    ROW_GAP  = 0.34

    left_lab, cbar_w = 1.45, 1.20
    title_h, head_h  = 1.00, 0.70
    foot_h           = 0.56

    max_w = max((r["width"] for r in rows), default=1.0) or 1.0
    max_d = max((r["depth"] for r in rows), default=1.0) or 1.0
    scale = min(MAX_PLAN_W / max_w, MAX_PLAN_H / max_d)      # inches per metre
    margin_m = MARGIN_IN / scale

    ax_w = [r["width"] * scale + 2 * MARGIN_IN for r in rows]
    ax_h = [r["depth"] * scale + 2 * MARGIN_IN for r in rows]

    col_w = MAX_PLAN_W + 2 * MARGIN_IN + COL_GAP
    row_h = [CAP_H + h for h in ax_h]

    fig_w = left_lab + n_cols * col_w + cbar_w
    fig_h = title_h + head_h + sum(row_h) + ROW_GAP * (n_rows - 1) + foot_h
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.0 - 0.30 / fig_h)
    fig.text(0.5, 1.0 - 0.62 / fig_h, subtitle, ha="center", va="top", fontsize=8.0,
             color="#444444")

    gs_top    = 1.0 - (title_h + head_h) / fig_h
    gs_bottom = foot_h / fig_h
    gs_left   = left_lab / fig_w
    gs_right  = 1.0 - cbar_w / fig_w

    def col_center(c):
        return (left_lab + c * col_w + (col_w - COL_GAP) / 2.0 + COL_GAP / 2.0) / fig_w

    for c, (lo, hi, name, span) in enumerate(TIME_RANGES):
        x = col_center(c)
        held = sorted({h for r in rows for h in (caption(r, c) or {}).get("hours", [])})
        lbl = ", ".join(f"{h:02d}" for h in held) or "—"
        fig.text(x, gs_top + 0.50 / fig_h, name, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold")
        fig.text(x, gs_top + 0.26 / fig_h, span, ha="center", va="bottom",
                 fontsize=8.5, color="#333333")
        fig.text(x, gs_top + 0.07 / fig_h, f"({lbl})", ha="center", va="bottom",
                 fontsize=6.8, color="#777777")

    im_last = None
    y_cursor = gs_top
    for r, row in enumerate(rows):
        plan_top = y_cursor - CAP_H / fig_h
        plan_bot = plan_top - ax_h[r] / fig_h
        for c in range(n_cols):
            ax = fig.add_axes([col_center(c) - ax_w[r] / 2.0 / fig_w, plan_bot,
                               ax_w[r] / fig_w, ax_h[r] / fig_h])
            im = draw(ax, row, c, margin_m)
            if im is None:
                continue
            im_last = im
            cap = caption(row, c)
            if cap:
                # Caption on the COLUMN centre at full column width, not on the axes: a
                # small room's plan is narrower than its caption, and an axes title would
                # spill into the neighbouring column.
                fig.text(col_center(c), plan_top + 0.04 / fig_h, cap["text"],
                         ha="center", va="bottom", fontsize=6.4, color="#222222",
                         linespacing=1.35)
        fig.text(gs_left - 0.14 / fig_w, (plan_top + plan_bot) / 2.0, row["label"],
                 ha="right", va="center", fontsize=8.5, linespacing=1.55)
        y_cursor = plan_bot - ROW_GAP / fig_h

    if im_last is not None:
        cax = fig.add_axes([gs_right + 0.020, gs_bottom, 0.013, gs_top - gs_bottom])
        cb  = fig.colorbar(im_last, cax=cax)
        # Tick-marked at the levels this brief actually asks for, so the compressed lower
        # third of the 0-3000 ramp stays readable.
        keep = []
        for t in [0.0] + sorted(all_levels) + [GLARE_LUX]:
            if not keep or t - keep[-1] >= GLARE_LUX * 0.035:
                keep.append(t)
        cb.set_ticks(keep)
        cb.set_ticklabels([f"{t:g}" for t in keep])
        cb.ax.tick_params(labelsize=7)
        cb.set_label(f"target lux  (scale 0–{GLARE_LUX:.0f}, as result.png)", fontsize=8)

    # A 1 m scale bar, drawn once.  The shared scale is the whole point of the layout, so it
    # gets stated rather than left for the reader to infer from the row labels.
    bar_y  = gs_bottom - 0.20 / fig_h
    bar_x0 = gs_left
    bar_x1 = bar_x0 + 1.0 * scale / fig_w
    sax = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=10)
    sax.set_xlim(0, 1); sax.set_ylim(0, 1); sax.axis("off")
    sax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#333333", linewidth=1.6,
             solid_capstyle="butt")
    for x in (bar_x0, bar_x1):
        sax.plot([x, x], [bar_y - 0.04 / fig_h, bar_y + 0.04 / fig_h],
                 color="#333333", linewidth=1.0)
    sax.text((bar_x0 + bar_x1) / 2.0, bar_y + 0.07 / fig_h, "1 m", ha="center", va="bottom",
             fontsize=7, color="#333333")

    fig.text(bar_x1 + 0.16 / fig_w, bar_y,
             "All rows share this scale — plan size on the page is size in reality.   "
             "Plans are room-local: row 0 = the +Y (\"N\") wall, col 0 = the x=0 (\"W\") wall; "
             "these are wall ids, not compass directions.\n"
             "Grey = unoccupied (no target), not 0 lux; the palest tone is apartment "
             "outside any room.   "
             "Blank cell = no activity in that range.   "
             "Slots sharing a column are merged by max."
             + ("\n" + note if note else ""),
             ha="left", va="center", fontsize=6.8, color="#666666", linespacing=1.6)
    return fig


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def unit_name_for(unit_dir):
    """The unit's name, from its manifest if there is one.

    NOT the folder name: the jobs live in a folder called `output` under `Units/101/`, so
    the containing directory is named for its role, not for the unit.  The manifest is the
    only place that actually knows.
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
    # No manifest: fall back to the folder, stepping up out of a role-named one.
    d = unit_dir.resolve()
    if d.name.lower() in ("output", "outputs", "jobs") and d.parent != d:
        return d.parent.name
    return d.name


def room_mode(unit_dir, only):
    rooms = load_rooms(unit_dir, only=only)
    unit_name = unit_name_for(unit_dir)

    rows = [dict(id=r["id"], width=r["width"], depth=r["depth"], room=r,
                 label=f"{r['id']}\n{r['width']:.2f} × {r['depth']:.2f} m\n"
                       f"sides {r['sides']} · N {r['north_angle']:g}°")
            for r in rooms]

    def draw(ax, row, col, margin_m):
        return draw_room_cell(ax, row["room"], col, margin_m)

    city = next((r["city"] for r in rooms if r["city"]), "")
    dead = hours_without_sun(city, [h for r in rooms for c in r["cells"] if c
                                    for h in c["hours"]])

    def caption(row, col):
        c = row["room"]["cells"][col]
        if not c:
            return None
        hours = " + ".join(f"{h:02d}:00" + ("*" if h in dead else "") for h in c["hours"])
        return dict(hours=c["hours"],
                    text=f"{hours}\n{fmt_levels(c['levels'])}  ({c['n_cells']} cells)")

    n_slots = sum(len(c["hours"]) for r in rooms for c in r["cells"] if c)
    levels = {v for r in rooms for c in r["cells"] if c for v in c["levels"]}
    n_blank = len(rooms) * len(TIME_RANGES) - sum(1 for r in rooms for c in r["cells"] if c)

    print(f"Unit {unit_name}: {len(rooms)} rooms x {len(TIME_RANGES)} ranges = "
          f"{len(rooms) * len(TIME_RANGES)} cells "
          f"({len(rooms) * len(TIME_RANGES) - n_blank} drawn, {n_blank} blank)")
    for r in rooms:
        bits = ["--" if c is None else "+".join(f"{h:02d}" for h in c["hours"])
                for c in r["cells"]]
        print(f"  {r['id']:<8} {r['width']:5.2f} x {r['depth']:5.2f} m  "
              f"sides {r['sides']:<4} | " + "  ".join(f"{b:>9}" for b in bits))
    if dead:
        print(f"  [!] no sun at {', '.join('%02d:00' % h for h in sorted(dead))} in this "
              f"climate — those slots will be SKIPPED by the optimizer")

    return rows, draw, caption, levels, (
        f"Unit {unit_name}  |  occupancy behaviour zones by room and time of day  |  "
        f"target lux"), (
        f"{city}  |  {len(rooms)} rooms, {n_slots} time slots  |  "
        f"lux ramp and 0–{GLARE_LUX:.0f} scale identical to result.png, so the brief and "
        f"the design read on one colour language"), unit_name, dead_hour_note(dead)


def unit_mode(root, only):
    """Rows are units.  `root` is a folder of unit folders, or one unit folder."""
    root = Path(root)
    if (root / "unit.json").is_file():
        dirs = [root]
    else:
        # Units live at <root>/<unit>/output/, so a single level is not enough: the folder
        # holding the manifest is named for its role, not for the unit.
        found = {p.parent for p in root.glob("*/unit.json")}
        found |= {p.parent for p in root.glob("*/*/unit.json")}
        dirs = sorted(found, key=lambda d: natural_key(Path(unit_name_for(d))))
        if not dirs:
            raise SystemExit(
                f"no unit.json found in {root}, or one or two levels inside it.\n"
                "Produce one with the gh_write_unit_jobs.py Grasshopper component "
                "(see --check-layout).")
    if only:
        want = set(only)
        dirs = [d for d in dirs if d.name in want]

    gh = load_layout_checker()
    units = [load_unit(d) for d in dirs]
    rows = []
    for u in units:
        # Floor area, not the bounding box — for a stepped or L-shaped flat the two differ
        # substantially, and the bounding box is not a number anyone wants about a flat.
        area = abs(gh.polygon_area(u["boundary"])) if gh else u["width"] * u["depth"]
        rows.append(dict(id=u["id"], width=u["width"], depth=u["depth"], unit=u,
                         label=f"Unit {u['id']}\n{area:.0f} m² floor area\n"
                               f"{len(u['rooms'])} rooms"))

    def draw(ax, row, col, margin_m):
        return draw_unit_cell(ax, row["unit"], col, margin_m)

    city = next((u["city"] for u in units if u["city"]), "")
    dead = hours_without_sun(city, [h for u in units for r in u["rooms"]
                                    for c in r["cells"] if c for h in c["hours"]])

    def caption(row, col):
        s = unit_cell_summary(row["unit"], col)
        if not s:
            return None
        hours = ", ".join(f"{h:02d}" + ("*" if h in dead else "") for h in s["hours"])
        return dict(hours=s["hours"],
                    text=f"{s['n_rooms']} room{'' if s['n_rooms'] == 1 else 's'} "
                         f"at {hours}h\n{fmt_levels(s['levels'])}  "
                         f"({s['n_cells']} cells)")

    levels = {v for u in units for r in u["rooms"] for c in r["cells"] if c
              for v in c["levels"]}
    n_rooms = sum(len(u["rooms"]) for u in units)

    for u in units:
        print(f"Unit {u['id']}: {u['width']:.2f} x {u['depth']:.2f} m, {len(u['rooms'])} rooms")
        for c, (lo, hi, name, span) in enumerate(TIME_RANGES):
            s = unit_cell_summary(u, c)
            print(f"  {name:<10} {span:<12} " +
                  ("--" if not s else
                   f"{s['n_rooms']} room(s), hours "
                   f"{','.join(f'{h:02d}' for h in s['hours'])}"))

    if dead:
        print(f"  [!] no sun at {', '.join('%02d:00' % h for h in sorted(dead))} in this "
              f"climate — those slots will be SKIPPED by the optimizer")

    return rows, draw, caption, levels, (
        f"Occupancy behaviour zones by unit and time of day  |  target lux"), (
        f"{city}  |  {len(units)} unit{'' if len(units) == 1 else 's'}, {n_rooms} rooms  |  "
        f"lux ramp and 0–{GLARE_LUX:.0f} scale identical to result.png, so the brief and "
        f"the design read on one colour language"), (
        units[0]["id"] if len(units) == 1 else "all"), dead_hour_note(dead)


def check_layout_mode(path):
    """Validate a unit/1 manifest's room placement and print the report."""
    gh = load_layout_checker()
    if gh is None:
        raise SystemExit(f"cannot load the layout checker from {BRIDGE}")

    p = Path(path)
    man = p if p.is_file() else p / "unit.json"
    if not man.is_file():
        raise SystemExit(f"no unit/1 manifest at {man}")
    m = json.loads(man.read_text(encoding="utf-8"))
    u = m.get("unit") or {}
    rooms = [dict(name=e["name"], width=float(e["width"]), depth=float(e["depth"]),
                  origin_uv=tuple(e["origin_uv"]), x_axis_uv=tuple(e["x_axis_uv"]),
                  y_axis_uv=tuple(e["y_axis_uv"])) for e in m.get("rooms", [])]

    boundary = [tuple(p) for p in m.get("boundary_uv", [])]
    if len(boundary) < 3:
        w, d = float(u.get("width", 0)), float(u.get("depth", 0))
        boundary = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
        print("  [!] no boundary_uv — assuming a rectangular apartment")

    print(f"{man}\n  apartment {abs(gh.polygon_area(boundary)):.1f} m2, "
          f"{len(boundary)} corners, "
          f"{u.get('width', 0):.2f} x {u.get('depth', 0):.2f} m bounding box, "
          f"{len(rooms)} rooms")
    for r in rooms:
        print(f"    {r['name']:<8} {r['width']:5.2f} x {r['depth']:5.2f} m  "
              f"at ({r['origin_uv'][0]:7.2f}, {r['origin_uv'][1]:7.2f})")

    errors, notes = gh.check_layout(rooms, boundary)
    print()
    for n in notes:
        print(f"  note : {n}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print(f"\n  {len(errors)} problem(s) — this layout cannot be composited.")
        return 1
    print("\n  Layout is valid: no overlaps, every room inside the apartment outline.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Matrix of painted occupancy target lux, by room or by unit.")
    ap.add_argument("path", help="a unit folder of job JSONs, or (with --by unit) a folder "
                                 "of unit folders")
    ap.add_argument("--by", choices=("room", "unit"), default="room",
                    help="rows are rooms (default) or whole units")
    ap.add_argument("-o", "--out", default=None, help="output PNG")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these room (or unit) ids")
    ap.add_argument("--check-layout", action="store_true",
                    help="validate a unit/1 manifest's room placement and exit")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    if args.check_layout:
        return check_layout_mode(args.path)

    # Figures land BESIDE the jobs they describe, so a unit is one self-contained folder
    # rather than a brief here and its picture in a central outputs/ tree.  In unit mode
    # over many units there is no single unit folder, so the root they share is used.
    if args.by == "unit":
        rows, draw, caption, levels, title, sub, tag, note = unit_mode(args.path, args.only)
        home = rows[0]["unit"]["path"] if len(rows) == 1 else Path(args.path)
        default_out = home / f"unit_{tag}_zone_matrix_composite.png"
    else:
        rows, draw, caption, levels, title, sub, tag, note = room_mode(args.path, args.only)
        default_out = Path(args.path) / f"unit_{tag}_zone_matrix.png"

    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(rows, title, sub, draw, caption, levels, dpi=args.dpi, note=note)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
