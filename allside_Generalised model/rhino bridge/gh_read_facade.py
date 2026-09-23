"""
gh_read_facade.py — Grasshopper script component: build the optimized facade as Rhino geometry.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib + RhinoCommon.

Closes the loop.  Reads facade_panels.csv and builds the design as a SMALL NUMBER OF MERGED
SURFACES on the room box's exterior walls — not one surface per grid cell.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    room            (none) Item   a Room from gh_make_room — carries the box AND the
                                  exterior walls. Type hint MUST be "No type hint".
    results_dir     str    Item   with `room`, the folder holding <room name>/ — the CSV
                                  and result.json paths are then built from the room name
    csv_path        str    Item   facade_panels.csv — wire from the writer's `panels_csvs`
                                  (an alternative to room + results_dir)
    room_box        Box    Item   the SAME box that authored the job (not needed with `room`)
    result_json     str    Item   optional; supplies `sides_open` and a fallback frame
    sides           str    Item   optional; force the exterior walls, e.g. "SE"
    include_walls   bool   Item   optional, default True; False builds only the windows
    merge           bool   Item   optional, default True; False returns one surface per cell

TWO WAYS TO WIRE IT, AND WHY THE FIRST IS SAFER

    ONE ROOM, EXPLICIT:      csv_path + room_box (+ result_json)
    A WHOLE UNIT:            room + results_dir

    This component reads ONE room per run; Grasshopper iterates it when its Item inputs are
    fed lists.  That makes list ORDER load-bearing — and a separate list of boxes kept in
    step by hand with a list of CSVs is the fragile part, because two similarly-sized rooms
    swapped between them would pass check_room_match and build a plausible, WRONG facade.

    Wiring `room` removes the problem structurally: the box and its exterior walls travel
    inside one object, and with `results_dir` the CSV path is derived from the room's own
    name.  Feed the SAME merged `rooms` list that fed gh_write_unit_jobs and nothing has to
    be kept in order at all.

COMPONENT OUTPUTS
    windows         one planar Brep per connected window REGION (holes included)
    walls           one planar Brep per connected solid region, exterior walls only
    window_curves   the region outlines (bake / draft / offset)
    window_sides    parallel list of "S"/"N"/"E"/"W" — Dispatch on it to split by facade
    info            which sides were built, px → region counts, glazing %
────────────────────────────────────────────────────────────────────────────────────────

TWO THINGS THIS COMPONENT NOW REFUSES TO DO

1.  BUILD INTERIOR WALLS.  In a unit, most of a room's walls face other rooms.  The
    optimizer freezes any wall not in `sides` fully solid, so those walls are entirely
    black — and building them produced 512 dead surfaces each, 1024–1536 per room, 14,336
    for a seven-room unit.  The exterior wall list is now honoured: explicit `sides` input,
    else `resolved.sides_open` from result.json, else inference from which sides hold any
    window at all.  `info` says which rule fired.

2.  EMIT ONE SURFACE PER PIXEL.  Adjacent window cells are merged into a single region and
    built from that region's OUTLINE.  Measured on unit 101's results:

        101.1   79 window px ->  3 regions    (26x fewer)
        101.2  303 window px ->  2 regions   (151x fewer)
        101.3  155 window px ->  4 regions    (39x fewer)
        101.5  155 window px ->  6 regions    (26x fewer)

    Set `merge` to False for the old per-cell behaviour.

WHY THIS FILE STILL DOES NOT RE-DERIVE AN INDEX CONVENTION
    facade_panels.csv carries EXPLICIT CORNER COORDINATES.  The merge works in the grid's
    integer (row, col) lattice, but every coordinate it emits is looked up from the CSV's
    own panel corners — the lattice is BUILT from them.  "row 0 is the ceiling", "col 0 is
    the x=0 end for S/N but the y=0 end for E/W", "1 = window" all stay the optimizer's
    problem.  Resist any urge to compute a corner from row/col arithmetic here; that is how
    the two sides drift apart.

UNITS RUN THE OPPOSITE WAY FROM THE WRITER
    The CSV is in METRES.  The writer divides doc units down into metres; this multiplies
    back up.  In a millimetre document that is x1000 — get it backwards and you get a facade
    1000x too small parked at the box corner.  Visible, but only if you look.
"""

import os
import csv
import json
from collections import deque

SIDES = ("S", "N", "E", "W")
REQUIRED_COLS = ("side", "row", "col", "is_window", "x0", "y0", "z0", "x1", "y1", "z1")


class FacadeReadError(Exception):
    """The CSV or the frame cannot be used."""


def room_attr(room, name):
    """Read one field off a Room — by attribute, falling back to dict access.

    DUCK-TYPED ON PURPOSE, never isinstance(): gh_make_room.py is loaded in its own module
    instance, so its Room class is a different class object from anything this file could
    import, and an identity check would be False for every real room.  A plain
    {"room_box": ..., "name": ..., "sides": ...} dict works too.
    """
    room = getattr(room, "Value", room)               # unwrap a GH object wrapper
    v = getattr(room, name, None)
    if v is None and isinstance(room, dict):
        v = room.get(name)
    return v


# ---------------------------------------------------------------------------
# Pure logic (no Rhino dependency — unit-testable outside Grasshopper)
# ---------------------------------------------------------------------------

def read_panels(csv_path):
    """facade_panels.csv -> list of dicts.  Raises FacadeReadError with something actionable."""
    if not csv_path:
        raise FacadeReadError("`csv_path` is empty — wire it from the writer's `panels_csv` "
                              "output, or point it at a facade_panels.csv")
    p = str(csv_path).strip().strip('"')
    if not os.path.isfile(p):
        raise FacadeReadError(
            "no file at %r.\nIf this path came from the writer's `panels_csv` it is a "
            "PREDICTION — the file appears only after the optimizer has run. Has it "
            "finished?" % p)

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate the HEADER before filtering rows.  Filtering first would drop every row of
        # a wrong-format file (nothing has a `side` column) and then report "no data rows",
        # which points at the file being empty rather than at it being the wrong file.
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise FacadeReadError(
                "%s is missing column(s) %s — is this really a facade_panels.csv? "
                "(facade_wide.csv is a different, single-row {S,N,E,W}_Pix_k format.)"
                % (os.path.basename(p), ", ".join(missing)))
        rows = [r for r in reader if r and r.get("side")]
    if not rows:
        raise FacadeReadError("%s has no data rows" % os.path.basename(p))

    panels = []
    for i, r in enumerate(rows):
        try:
            panels.append(dict(
                side=r["side"].strip(), row=int(r["row"]), col=int(r["col"]),
                is_window=int(r["is_window"]),
                p0=(float(r["x0"]), float(r["y0"]), float(r["z0"])),
                p1=(float(r["x1"]), float(r["y1"]), float(r["z1"]))))
        except (TypeError, ValueError) as e:
            raise FacadeReadError("row %d of %s is malformed: %s"
                                  % (i + 2, os.path.basename(p), e)) from None
        if panels[-1]["side"] not in SIDES:
            raise FacadeReadError("row %d has unknown side %r"
                                  % (i + 2, panels[-1]["side"]))
    return panels


def csv_extents(panels):
    """(W, D, H) in metres, from the panels themselves.

    The panels tile all four walls, so the largest corner IS the room's dimensions.  That is
    what makes the mismatch check below possible without trusting any metadata.
    """
    return (max(max(q["p0"][0], q["p1"][0]) for q in panels),
            max(max(q["p0"][1], q["p1"][1]) for q in panels),
            max(max(q["p0"][2], q["p1"][2]) for q in panels))


def panel_corners(panel):
    """The four room-local corners, in order, for a planar axis-aligned wall panel.

    One horizontal axis is degenerate: S/N hold y constant, E/W hold x constant.  So the
    min/max pair in the CSV determines the rectangle completely.
    """
    (x0, y0, z0), (x1, y1, z1) = panel["p0"], panel["p1"]
    if panel["side"] in ("S", "N"):
        y = y0
        return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    x = x0
    return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]


def check_room_match(panels, w_m, d_m, h_m, tol=0.01):
    """Guard against reading a CSV from a DIFFERENT run than the box describes.

    Nothing stops the two being wired independently, and a mismatch does not fail loudly on
    its own — it silently builds a correctly-shaped facade for the wrong room, scaled and
    offset, which looks entirely plausible in the viewport.  So compare and refuse.
    """
    cw, cd, ch = csv_extents(panels)
    bad = [(n, c, b) for n, c, b in (("width", cw, w_m), ("depth", cd, d_m),
                                     ("height", ch, h_m)) if abs(c - b) > tol]
    if bad:
        raise FacadeReadError(
            "this CSV is not for this box.\n"
            "  CSV : %.3f x %.3f x %.3f m\n  box : %.3f x %.3f x %.3f m\n"
            "  differs on: %s\n"
            "Wire the box that authored the job, or re-run the optimizer for this box."
            % (cw, cd, ch, w_m, d_m, h_m, ", ".join(b[0] for b in bad)))
    return cw, cd, ch


# ---------------------------------------------------------------------------
# Which sides are exterior
# ---------------------------------------------------------------------------

def sides_with_windows(panels):
    """Sides holding at least one window cell.

    The optimizer freezes every wall outside `sides` fully solid, so a side with any window
    at all is necessarily one that was declared open.  The converse does not hold — an open
    side can end up with zero windows — which is why this is the LAST fallback, not the rule.
    """
    return tuple(s for s in SIDES
                 if any(q["is_window"] for q in panels if q["side"] == s))


def sides_from_result_json(path):
    """`resolved.sides_open` — the authoritative exterior-wall list, as run by the optimizer."""
    p = str(path).strip().strip('"')
    if not os.path.isfile(p):
        raise FacadeReadError("no result.json at %r" % p)
    data = json.loads(open(p, encoding="utf-8").read())
    so = (data.get("resolved") or {}).get("sides_open")
    if not so:
        return None
    return tuple(s for s in SIDES if s in so)


def resolve_exterior_sides(panels, sides_input=None, result_json=None,
                           sides_source="the `sides` input"):
    """(sides tuple, explanation).  Precedence: explicit -> result.json -> inference."""
    if sides_input:
        s = str(sides_input).upper().strip()
        bad = sorted(set(s) - set("SNEW"))
        if bad:
            raise FacadeReadError(
                "`sides` contains invalid wall id(s) %s — allowed are S, N, E, W "
                "(room-local wall ids, not compass directions)"
                % ", ".join(repr(b) for b in bad))
        got = tuple(x for x in SIDES if x in s)
        if not got:
            raise FacadeReadError("`sides` is empty")
        return got, "exterior sides %s — from %s" % ("".join(got), sides_source)

    if result_json:
        got = sides_from_result_json(result_json)
        if got:
            return got, ("exterior sides %s — from result.json `sides_open`"
                         % "".join(got))

    got = sides_with_windows(panels)
    if not got:
        raise FacadeReadError(
            "no side holds a single window, so the exterior walls cannot be inferred. "
            "Wire `result_json` (its `sides_open` is authoritative) or type the walls into "
            "`sides`. A design with zero windows everywhere usually means the optimizer ran "
            "with no usable time slot.")
    return got, ("exterior sides %s — INFERRED from which sides hold windows; wire "
                 "`result_json` to use the design's own `sides_open`" % "".join(got))


# ---------------------------------------------------------------------------
# Merging cells into regions
# ---------------------------------------------------------------------------

def side_lattice(panels, side):
    """(grid, u_lo, u_hi, z_lo, z_hi, const, rows, cols) for one wall.

    `u` is the wall's horizontal axis — x for S/N, y for E/W — and `const` is the coordinate
    the wall holds fixed on the other axis.  Every number here is READ OFF the CSV's own
    panel corners, never computed from row/col, so the lattice cannot drift from the file.
    """
    sp = [q for q in panels if q["side"] == side]
    if not sp:
        raise FacadeReadError("no panels for side %r in the CSV" % side)
    rows = max(q["row"] for q in sp) + 1
    cols = max(q["col"] for q in sp) + 1

    ui = 0 if side in ("S", "N") else 1          # which world axis is the wall's horizontal
    ci = 1 if side in ("S", "N") else 0          # the axis the wall holds constant

    grid  = [[0] * cols for _ in range(rows)]
    u_lo  = [None] * cols
    u_hi  = [None] * cols
    z_lo  = [None] * rows
    z_hi  = [None] * rows
    const = sp[0]["p0"][ci]

    for q in sp:
        r, c = q["row"], q["col"]
        grid[r][c] = 1 if q["is_window"] else 0
        a, b = q["p0"], q["p1"]
        u_lo[c] = min(a[ui], b[ui])
        u_hi[c] = max(a[ui], b[ui])
        z_lo[r] = min(a[2], b[2])
        z_hi[r] = max(a[2], b[2])

    if any(v is None for v in u_lo + u_hi + z_lo + z_hi):
        raise FacadeReadError("side %s is missing panels — the grid is not complete" % side)
    return grid, u_lo, u_hi, z_lo, z_hi, const, rows, cols


def connected_regions(grid, want=1):
    """4-connected components of cells equal to `want`.  Each is a set of (row, col).

    FOUR-connected, deliberately.  Two cells touching only at a corner share a point, not an
    edge; fusing them would produce a pinched, non-manifold surface.  They stay separate
    regions, which is also what they look like.
    """
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    out = []
    for r0 in range(rows):
        for c0 in range(cols):
            if grid[r0][c0] != want or seen[r0][c0]:
                continue
            comp, q = set(), deque([(r0, c0)])
            seen[r0][c0] = True
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < rows and 0 <= cc < cols
                            and grid[rr][cc] == want and not seen[rr][cc]):
                        seen[rr][cc] = True
                        q.append((rr, cc))
            out.append(comp)
    return out


def region_loops(cells):
    """Closed boundary loops of a set of cells, as lists of (row, col) LATTICE VERTICES.

    Method: emit each cell's four edges as DIRECTED edges wound counter-clockwise in the
    (col, -row) plane, then delete every pair that appears in both directions — those are
    interior, shared by two cells of the same region.  What survives is the boundary, already
    consistently oriented, and it chains into loops with no special case for holes: an
    enclosed void simply comes back as an extra loop wound the other way.

    Returns (outer_loops, hole_loops); a well-formed region yields exactly one outer loop.
    """
    directed = set()
    for (r, c) in cells:
        # CCW around the cell in (x=col, y=-row): bottom edge, right, top, left.
        for a, b in (((r + 1, c), (r + 1, c + 1)),
                     ((r + 1, c + 1), (r, c + 1)),
                     ((r, c + 1), (r, c)),
                     ((r, c), (r + 1, c))):
            if (b, a) in directed:
                directed.discard((b, a))        # interior edge: both cells cancel it
            else:
                directed.add((a, b))

    nxt = {}
    for a, b in directed:
        nxt.setdefault(a, []).append(b)

    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, cur = [start], start
        while True:
            outs = nxt.get(cur)
            if not outs:
                break                            # malformed; bail rather than spin
            nb = outs.pop()
            if not outs:
                del nxt[cur]
            if nb == start:
                break
            loop.append(nb)
            cur = nb
        if len(loop) >= 4:
            loops.append(loop)

    outer, holes = [], []
    for lp in loops:
        # Signed area in (col, -row): CCW (positive) is the outer boundary, CW a hole.
        a = 0.0
        for i in range(len(lp)):
            r0, c0 = lp[i]
            r1, c1 = lp[(i + 1) % len(lp)]
            a += c0 * (-r1) - c1 * (-r0)
        (outer if a > 0 else holes).append(lp)
    return outer, holes


def rect_decompose(cells):
    """A set of cells -> a list of (row0, row1, col0, col1) rectangles covering it exactly.

    WHY THIS EXISTS.  A region with an enclosed opening needs a TRIMMED surface with inner
    loops, and Rhino would not reliably build one: CreatePlanarBreps' nesting inference,
    CreateTrimmedPlane and boolean difference all failed on real wall regions (measured:
    102.5 N wall, 1 hole; 102.6 N wall, 4 holes), leaving the opening filled by a second
    surface.  Rather than keep guessing at which Rhino call cuts a hole, a holed region is
    split into rectangles — which have no holes, so nothing has to be cut at all.

    Greedy row-run merging: maximal horizontal runs per row, then merge vertically adjacent
    runs with the same column span.  Not the theoretical minimum number of rectangles, but
    deterministic, exact, and a handful of surfaces rather than hundreds.
    """
    by_row = {}
    for (r, c) in cells:
        by_row.setdefault(r, []).append(c)

    runs_by_row = {}
    for r, cs in by_row.items():
        cs.sort()
        runs, start, prev = [], cs[0], cs[0]
        for c in cs[1:]:
            if c == prev + 1:
                prev = c
            else:
                runs.append((start, prev))
                start = prev = c
        runs.append((start, prev))
        runs_by_row[r] = runs

    rects, open_r = [], {}          # span -> (first_row, last_row)
    for r in sorted(runs_by_row):
        cur = set(runs_by_row[r])
        for span in list(open_r):
            r0, rl = open_r[span]
            if span in cur and rl == r - 1:
                open_r[span] = (r0, r)      # still growing downwards
            else:
                rects.append((r0, rl, span[0], span[1]))
                del open_r[span]
        for span in cur:
            open_r.setdefault(span, (r, r))
    for span, (r0, rl) in open_r.items():
        rects.append((r0, rl, span[0], span[1]))
    return rects


def rect_loop(r0, r1, c0, c1):
    """A rectangle's lattice vertices, wound like the cell edges in region_loops:
    counter-clockwise in (col, -row), starting bottom-left."""
    return [(r1 + 1, c0), (r1 + 1, c1 + 1), (r0, c1 + 1), (r0, c0)]


def simplify_loop(loop):
    """Drop vertices that sit mid-edge on a straight run.

    A 32-cell-wide opening otherwise arrives as 32 collinear segments per edge.  Merging
    them is what turns a merged region into an actual rectangle downstream.
    """
    n = len(loop)
    if n < 3:
        return loop
    out = []
    for i in range(n):
        r0, c0 = loop[(i - 1) % n]
        r1, c1 = loop[i]
        r2, c2 = loop[(i + 1) % n]
        # Cross product of the two segments; zero means r1,c1 is redundant.
        if (c1 - c0) * (r2 - r1) - (r1 - r0) * (c2 - c1) != 0:
            out.append(loop[i])
    return out or loop


def loop_to_points(loop, side, u_lo, u_hi, z_lo, z_hi, const, rows, cols):
    """Lattice vertices -> room-local 3D metres, via the CSV-derived lattice."""
    def vu(c):
        return u_lo[c] if c < cols else u_hi[cols - 1]

    def vz(r):
        # Row 0 is the CEILING, so z DECREASES as row increases: a vertex's z is the top of
        # its row, and the final lattice line is the bottom of the last row.
        return z_hi[r] if r < rows else z_lo[rows - 1]

    pts = []
    for (r, c) in loop:
        u, z = vu(c), vz(r)
        pts.append((u, const, z) if side in ("S", "N") else (const, u, z))
    return pts


def pinched_holes(outer_loop, hole_loops):
    """Holes that TOUCH the outer boundary at a vertex.

    Such a region is pinched, and a trimmed surface through the pinch point is genuinely
    ill-defined — not merely awkward for Rhino.  Measured on real data: 102.5's N wall has
    exactly this, which is why no amount of coaxing would trim it.  Those regions must be
    split; the clean donuts around them need not be.
    """
    ov = set(outer_loop)
    return [h for h in hole_loops if ov & set(h)]


def merge_side(panels, side, want=1):
    """One entry per region: dict(outer=pts, holes=[pts...], rects=[pts...], pinched=bool).

    `outer`/`holes` describe the region exactly and are what a trimmed surface is built
    from.  `rects` is the SAME region cut into hole-free rectangles, carried along as the
    fallback for when that trimmed surface cannot be built — computed here because this is
    where the cell set lives.

    The caller decides which to use, and prefers the outline: a region that trims
    successfully is ONE surface, where its rectangles would be a dozen or more.
    """
    grid, u_lo, u_hi, z_lo, z_hi, const, rows, cols = side_lattice(panels, side)
    lat = (u_lo, u_hi, z_lo, z_hi, const, rows, cols)

    out = []
    for cells in connected_regions(grid, want):
        outer, holes = region_loops(cells)
        if not outer:
            continue
        # A 4-connected region has exactly one outer boundary, so the holes belong to it.
        # Should tracing ever return more, the extras are emitted as their own surfaces
        # rather than silently dropped — and they would not own these holes.
        hp = [loop_to_points(simplify_loop(h), side, *lat) for h in holes]
        rects = ([loop_to_points(rect_loop(*r), side, *lat)
                  for r in rect_decompose(cells)] if holes else [])
        pin = bool(holes and pinched_holes(outer[0], holes))
        for i, o in enumerate(outer):
            out.append(dict(outer=loop_to_points(simplify_loop(o), side, *lat),
                            holes=hp if i == 0 else [],
                            rects=rects if i == 0 else [],
                            pinched=pin if i == 0 else False))
    return out


def summarize(panels, exterior, merged_counts=None):
    n = len(panels)
    win = [q for q in panels if q["is_window"]]
    lines = ["%d panels in the CSV, %d windows (%.1f%% glazing overall)"
             % (n, len(win), 100.0 * len(win) / max(n, 1))]
    for s in SIDES:
        sp = [q for q in panels if q["side"] == s]
        sw = sum(1 for q in sp if q["is_window"])
        if s not in exterior:
            lines.append("  %s: INTERIOR — not built (%d solid panels skipped)" % (s, len(sp)))
            continue
        tail = ""
        if merged_counts and s in merged_counts:
            tail = "  ->  %d region%s" % (merged_counts[s],
                                          "" if merged_counts[s] == 1 else "s")
        lines.append("  %s: %4d/%4d glazed%s%s"
                     % (s, sw, len(sp), "   (solid)" if not sw else "", tail))
    return lines


# ---------------------------------------------------------------------------
# Rhino-dependent
# ---------------------------------------------------------------------------

def metres_to_doc():
    """Scale factor from metres into the active document's units — the inverse of the
    writer's doc->metres.  1000 in a millimetre document."""
    import Rhino
    doc = Rhino.RhinoDoc.ActiveDoc
    if doc is None:
        return 1.0
    return Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Meters, doc.ModelUnitSystem)


def frame_from_box(box):
    """(base_plane, w_m, d_m, h_m) — the room-local frame, in DOC units for the plane.

    DUPLICATED from gh_write_facade_job.room_from_box().  These are separately pasted
    components with no shared module, so the derivation cannot be imported.  If either copy
    changes, change both — a divergence here silently misplaces every panel.
    """
    import Rhino
    import Rhino.Geometry as rg
    if box is None:
        return None
    if not isinstance(box, rg.Box):
        raise FacadeReadError(
            "`room_box` must be a Box (got %s). Set the input's type hint to Box, and build "
            "it with the same Domain Box component that fed the writer." % type(box).__name__)
    if not box.IsValid:
        raise FacadeReadError("`room_box` is not a valid Box")

    pl = box.Plane
    origin = pl.PointAt(box.X.Min, box.Y.Min, box.Z.Min)
    base = rg.Plane(origin, pl.XAxis, pl.YAxis)
    s = Rhino.RhinoMath.UnitScale(Rhino.RhinoDoc.ActiveDoc.ModelUnitSystem,
                                  Rhino.UnitSystem.Meters) if Rhino.RhinoDoc.ActiveDoc else 1.0
    return base, box.X.Length * s, box.Y.Length * s, box.Z.Length * s


def frame_from_result_json(path):
    """(base_plane, None, None, None) from the runner's echoed `placement` block.

    Lets someone with only the outputs rebuild the facade without the original definition.
    Room dimensions come back None, so the mismatch check is skipped — the placement and the
    panels came from the same run, so there is nothing to cross-check against.
    """
    import Rhino.Geometry as rg
    p = str(path).strip().strip('"')
    if not os.path.isfile(p):
        raise FacadeReadError("no result.json at %r" % p)
    data = json.loads(open(p, encoding="utf-8").read())
    pm = (data.get("job") or {}).get("placement") or data.get("placement")
    if not pm:
        raise FacadeReadError("%s has no `placement` block" % os.path.basename(p))
    o = pm["origin"]; xa = pm["x_axis"]; ya = pm["y_axis"]
    base = rg.Plane(rg.Point3d(*o), rg.Vector3d(*xa), rg.Vector3d(*ya))
    return base, None, None, None


def _to_world_curve(pts, base, scale):
    import Rhino.Geometry as rg
    wp = [base.PointAt(x * scale, y * scale, z * scale) for (x, y, z) in pts]
    return rg.PolylineCurve(wp + [wp[0]])


def _crv_area(c):
    import Rhino.Geometry as rg
    a = rg.AreaMassProperties.Compute(c)
    return abs(a.Area) if a else 0.0


def _brep_area(b):
    import Rhino.Geometry as rg
    a = rg.AreaMassProperties.Compute(b)
    return abs(a.Area) if a else 0.0


def _planar_brep(outer_crv, hole_crvs, tol, warn=None, want=None, label=""):
    """Trimmed planar surface from an outer loop plus any inner loops.

    THE HOLE MUST ACTUALLY BE A HOLE.  Brep.CreatePlanarBreps is given the outer loop and
    the inner ones together and is *supposed* to nest them into one trimmed face — but it
    can also hand back the outer region AND each inner loop as its own separate patch, or
    one untrimmed face.  Either way the opening ends up filled by a second surface sitting
    exactly in it, which in a shaded viewport is invisible and in the model is two
    overlapping surfaces.

    `want` is the expected area, computed from the CELL LATTICE by the caller rather than
    measured off Rhino curves — one less thing that can quietly return None and turn the
    check into a no-op.

    Three construction strategies are tried in order, each verified by area:

      1. CreatePlanarBreps with all loops        — the documented nesting path
      2. CreateTrimmedPlane on the loops' plane  — explicit, and the one that works when
                                                   nesting inference does not
      3. boolean difference                      — last, because Rhino's booleans want
                                                   CLOSED SOLIDS and generally refuse a
                                                   pair of open planar surfaces
    """
    import Rhino.Geometry as rg

    if not hole_crvs:
        b = rg.Brep.CreatePlanarBreps([outer_crv], tol)
        return list(b) if b else []

    if want is None:
        want = _crv_area(outer_crv) - sum(_crv_area(h) for h in hole_crvs)
    slack = max(1e-9, 0.002 * max(want, 1e-9))
    loops = [outer_crv] + list(hole_crvs)

    def ok(breps):
        if not breps:
            return None
        got = sum(_brep_area(b) for b in breps)
        return list(breps) if abs(got - want) <= slack else None

    # 1 — the documented path.
    got = ok(rg.Brep.CreatePlanarBreps(loops, tol))
    if got and len(got) == 1:
        return got

    # 2 — state the plane instead of letting it be inferred.
    try:
        has_plane, plane = outer_crv.TryGetPlane(tol)
    except Exception:                              # noqa: BLE001 — fall through to 3
        has_plane, plane = False, None
    if has_plane:
        try:
            tp = rg.Brep.CreateTrimmedPlane(plane, loops)
        except Exception:                          # noqa: BLE001
            tp = None
        got = ok([tp] if tp else None)
        if got:
            if warn is not None:
                warn.append("  [i] %s: %d hole(s) cut by CreateTrimmedPlane"
                            % (label, len(hole_crvs)))
            return got

    # 3 — booleans, which usually refuse open planar breps but cost nothing to try.
    outer_b = rg.Brep.CreatePlanarBreps([outer_crv], tol)
    hole_b = []
    for h in hole_crvs:
        hb = rg.Brep.CreatePlanarBreps([h], tol)
        if hb:
            hole_b.extend(list(hb))
    if outer_b and hole_b:
        try:
            got = ok(rg.Brep.CreateBooleanDifference(list(outer_b), hole_b, tol))
        except Exception:                          # noqa: BLE001
            got = None
        if got:
            if warn is not None:
                warn.append("  [i] %s: %d hole(s) cut by boolean difference"
                            % (label, len(hole_crvs)))
            return got

    # Nothing cut it.  Return None — NOT the untrimmed outline, which would fill the
    # opening with a second surface.  The caller falls back to rectangles, which need no
    # trimming at all.
    return None


def facade_rect_points(side, u_lo, u_hi, z_lo, z_hi, const, rows, cols):
    """The WHOLE wall as one rectangle, in room-local metres.

    Row 0 is the ceiling, so the top of the wall is z_hi[0] and the bottom is
    z_lo[rows-1] — taken from the lattice, like everything else, never computed.
    """
    u0, u1 = u_lo[0], u_hi[cols - 1]
    z0, z1 = z_lo[rows - 1], z_hi[0]
    quad = [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]
    return [((u, const, z) if side in ("S", "N") else (const, u, z)) for (u, z) in quad]


def cell_at(local_pt, side, u_lo, u_hi, z_lo, z_hi, rows, cols, grid, eps=1e-9):
    """Which grid cell does a room-local point fall in?  Returns its value, or None."""
    u = local_pt[0] if side in ("S", "N") else local_pt[1]
    z = local_pt[2]
    col = next((c for c in range(cols) if u_lo[c] - eps <= u <= u_hi[c] + eps), None)
    row = next((r for r in range(rows) if z_lo[r] - eps <= z <= z_hi[r] + eps), None)
    if col is None or row is None:
        return None
    return grid[row][col]


def loop_area(pts, side):
    """Shoelace of a loop in the WALL's own 2D plane (u, z), in room-local metres.

    Computed here rather than asked of Rhino: this number is the yardstick the hole check
    is measured against, and AreaMassProperties returning None would silently turn that
    check into "anything passes".
    """
    ui = 0 if side in ("S", "N") else 1
    a = 0.0
    for i in range(len(pts)):
        u0, z0 = pts[i][ui], pts[i][2]
        u1, z1 = pts[(i + 1) % len(pts)][ui], pts[(i + 1) % len(pts)][2]
        a += u0 * z1 - u1 * z0
    return abs(a / 2.0)


def cell_centres(side, u_lo, u_hi, z_lo, z_hi, const, rows, cols):
    """Every cell's centre in room-local metres, with its grid value's (row, col).

    These are the probe points that label the split pieces.  They work where blind
    parameter sampling does not, for a structural reason: Split cuts along cell EDGES, so
    every piece is a union of whole cells and therefore contains at least one centre — and
    no centre ever lands on a piece boundary.
    """
    out = []
    for r in range(rows):
        z = (z_lo[r] + z_hi[r]) / 2.0
        for c in range(cols):
            u = (u_lo[c] + u_hi[c]) / 2.0
            out.append(((u, const, z) if side in ("S", "N") else (const, u, z), r, c))
    return out


def world_probes(side, lat, base, scale):
    """cell_centres() mapped into WORLD Point3d.

    The conversion has to happen before anything is handed to Rhino: cell_centres works in
    room-local metres (so it stays testable outside Grasshopper), and Brep.ClosestPoint
    takes a Point3d in document space.
    """
    out = []
    for (pt, r, c) in cell_centres(side, *lat):
        out.append((base.PointAt(pt[0] * scale, pt[1] * scale, pt[2] * scale), r, c))
    return out


def _label_piece(piece, probes, grid, tol):
    """The grid value of the first probe point lying ON this piece, or None.

    `Brep.ClosestPoint` rather than `IsPointOnFace`: after a Split the face's parameter
    DOMAIN is still that of the whole underlying rectangle, so sampling that domain spreads
    guesses across the entire wall and misses every small piece.  Measured: only the sides
    whose regions were large enough to be hit by a coarse sweep classified at all.
    Distance-to-the-piece asks the question directly instead.
    """
    for pt, r, c in probes:
        cp = piece.ClosestPoint(pt)
        if cp is not None and cp.DistanceTo(pt) <= tol:
            return grid[r][c]
    return None


def build_split_side(panels, side, base, scale, tol):
    """Split the WHOLE facade rectangle with the window outlines, then label the pieces.

    This is the approach that actually works, and it is simpler than everything it
    replaces.  `BrepFace.Split` operates on a PLANAR FACE, so unlike a boolean it needs no
    solid, and unlike CreatePlanarBreps it never has to infer which loop is a hole: the
    window boundaries cut the rectangle into pieces, and each piece is then simply asked
    which grid cell it sits in.

    Openings, pinched regions and multiply-holed walls all fall out of this with no special
    case — the rectangle is partitioned, so every piece is either window or wall.

    Returns (window_breps, wall_breps, reason).  On failure the breps are None and `reason`
    says WHICH step gave up — splitting or labelling — because those need different fixes.
    """
    import Rhino.Geometry as rg

    grid, u_lo, u_hi, z_lo, z_hi, const, rows, cols = side_lattice(panels, side)
    lat = (u_lo, u_hi, z_lo, z_hi, const, rows, cols)

    rect_c = _to_world_curve(facade_rect_points(side, *lat), base, scale)
    rect_b = rg.Brep.CreatePlanarBreps([rect_c], tol)
    if not rect_b:
        return None, None, "could not build the facade rectangle"

    # Every window boundary, outer loops and hole loops alike: together they are exactly
    # the window/wall frontier, so they partition the rectangle.
    cutters = []
    for reg in merge_side(panels, side, want=1):
        cutters.append(_to_world_curve(reg["outer"], base, scale))
        cutters.extend(_to_world_curve(h, base, scale) for h in reg["holes"])
    if not cutters:
        return [], [rect_b[0]], "no windows — the whole wall is one piece"

    try:
        split = rect_b[0].Faces[0].Split(cutters, tol)
    except Exception as e:                           # noqa: BLE001
        return None, None, "Split raised %s" % type(e).__name__
    if split is None:
        return None, None, "Split returned nothing for %d cutter(s)" % len(cutters)
    if split.Faces.Count < 2:
        return None, None, ("Split produced %d face(s) from %d cutter(s) — the curves did "
                            "not cut the rectangle" % (split.Faces.Count, len(cutters)))

    probes = world_probes(side, lat, base, scale)
    windows, walls, unknown = [], [], 0
    for f in split.Faces:
        piece = f.DuplicateFace(False)
        if piece is None:
            unknown += 1
            continue
        val = _label_piece(piece, probes, grid, tol)
        if val is None:
            unknown += 1
        elif val:
            windows.append(piece)
        else:
            walls.append(piece)

    # A piece we could not place is a piece we cannot label window or wall; rather than
    # guess, hand the whole side back to the region builder.
    if unknown:
        return None, None, ("%d of %d split piece(s) could not be labelled"
                            % (unknown, split.Faces.Count))
    if not (windows or walls):
        return None, None, "split produced no usable pieces"
    return windows, walls, "%d piece(s)" % split.Faces.Count


def build_merged(panels, exterior, base, scale, tol, include_walls=True, warn=None):
    """Merged regions -> world Breps.  Returns (windows, walls, curves, sides, counts)."""
    windows, walls, curves, sides, counts = [], [], [], [], {}
    n_split = 0

    def build(reg, s, what, idx):
        """(outer_curve, hole_curves, breps) for one region.

        The outline is tried FIRST and kept whenever it works — that is one surface for the
        whole region.  Rectangles are the fallback, not the default: always splitting cost
        102.2 four window surfaces instead of sixteen, and 102.4's wall three instead of
        thirty-one, on regions Rhino had trimmed perfectly well.
        """
        outer, holes = reg["outer"], reg["holes"]
        lbl = "%s %s region %d" % (s, what, idx)
        oc = _to_world_curve(outer, base, scale)
        hc = [_to_world_curve(h, base, scale) for h in holes]

        breps = None
        if not reg["pinched"]:
            want = (loop_area(outer, s)
                    - sum(loop_area(h, s) for h in holes)) * scale * scale
            breps = _planar_brep(oc, hc, tol, warn, want, lbl)
        if breps is not None:
            return oc, hc, breps

        # Could not trim (or the region is pinched, where a trimmed face is ill-defined).
        # Rectangles have no holes, so nothing needs cutting.
        out = []
        for pts in reg["rects"]:
            rc = _to_world_curve(pts, base, scale)
            got = _planar_brep(rc, [], tol)
            if got:
                out.extend(got)
        if warn is not None:
            warn.append("  [i] %s: %d opening(s)%s — could not be trimmed, so the region "
                        "was split into %d rectangle(s)"
                        % (lbl, len(holes),
                           " TOUCHING the outer edge (pinched)" if reg["pinched"] else "",
                           len(reg["rects"])))
        return oc, hc, out

    for s in exterior:
        regions = merge_side(panels, s, want=1)
        counts[s] = len(regions)

        # Preferred: split the whole facade rectangle with the window outlines and label
        # the pieces.  No holes to cut, no pinch cases, no rectangle fallback — the
        # rectangle is partitioned, so every piece is one or the other.
        win_b, wall_b, reason = build_split_side(panels, s, base, scale, tol)
        if win_b is not None:
            windows.extend(win_b)
            sides.extend([s] * len(win_b))
            if include_walls:
                walls.extend(wall_b)
            for reg in regions:
                curves.append(_to_world_curve(reg["outer"], base, scale))
                curves.extend(_to_world_curve(h, base, scale) for h in reg["holes"])
            if warn is not None:
                warn.append("  [i] %s: facade rectangle split -> %d window + %d wall "
                            "piece(s)" % (s, len(win_b), len(wall_b)))
            continue

        # Fallback: build each region from its own outline, splitting into rectangles only
        # where a trimmed surface cannot be made.
        if warn is not None:
            warn.append("  [!] %s: %s — falling back to per-region outlines" % (s, reason))
        for i, reg in enumerate(regions):
            oc, hc, breps = build(reg, s, "window", i)
            n_split += 1 if (reg["holes"] and len(breps) > 1) else 0
            for b in breps:
                windows.append(b)
                sides.append(s)
            curves.append(oc)
            curves.extend(hc)
        if include_walls:
            for i, reg in enumerate(merge_side(panels, s, want=0)):
                _, _, breps = build(reg, s, "wall", i)
                n_split += 1 if (reg["holes"] and len(breps) > 1) else 0
                walls.extend(breps)

    if warn is not None and n_split:
        warn.insert(0, "  %d region(s) could not be trimmed and were split into rectangles"
                       % n_split)
    return windows, walls, curves, sides, counts


def build_per_cell(panels, exterior, base, scale, tol, include_walls=True, warn=None):
    """The original one-surface-per-cell behaviour, kept behind `merge=False`.

    Takes `tol` it does not use, so the two builders are interchangeable at the call site.
    """
    import Rhino.Geometry as rg

    windows, walls, curves, sides = [], [], [], []
    for q in panels:
        if q["side"] not in exterior:
            continue
        if not q["is_window"] and not include_walls:
            continue
        pts = [base.PointAt(x * scale, y * scale, z * scale)
               for (x, y, z) in panel_corners(q)]
        brep = rg.Brep.CreateFromCornerPoints(pts[0], pts[1], pts[2], pts[3], 1e-6)
        if brep is None:
            continue                     # degenerate cell; nothing sensible to build
        if q["is_window"]:
            windows.append(brep)
            sides.append(q["side"])
            curves.append(rg.PolylineCurve(pts + [pts[0]]))
        else:
            walls.append(brep)
    return windows, walls, curves, sides, None


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`.

    PASTE this file into a Python 3 script component; the trailer at the bottom calls this
    automatically, since component inputs ARE module globals there.
    """
    import Rhino
    tol = (Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance
           if Rhino.RhinoDoc.ActiveDoc else 0.001)

    # A Room carries its box AND its exterior walls in one object, so wiring `room` removes
    # the one genuinely fragile thing about the per-room wiring: keeping a separate list of
    # boxes in the same order as the list of CSVs.  Two similar rooms swapped between those
    # lists would pass check_room_match and build a plausible, wrong facade.
    room = g.get("room")
    box       = g.get("room_box")
    sides_in  = g.get("sides")
    csv_path  = g.get("csv_path")
    rjson     = g.get("result_json")
    notes     = []
    sides_src = "the `sides` input"

    if room is not None:
        box = box or room_attr(room, "room_box")
        if not sides_in:
            sides_in  = room_attr(room, "sides")
            sides_src = "the Room object"
        name     = room_attr(room, "name")
        if name is None:
            raise FacadeReadError(
                "`room` has no `name` — expected a Room from gh_make_room (or a dict with "
                "room_box/name/sides). Check the input's type hint is 'No type hint': a "
                "typed input coerces the object and destroys it.")
        notes.append("room %s — box and sides taken from the Room object" % name)

        # With a results root, the paths follow from the name, so nothing has to be kept in
        # step by hand at all.
        rd = g.get("results_dir")
        if rd:
            rd = str(rd).strip().strip('"')
            csv_path = csv_path or os.path.join(rd, str(name), "facade_panels.csv")
            cand = os.path.join(rd, str(name), "result.json")
            if not rjson and os.path.isfile(cand):
                rjson = cand

    if not csv_path:
        raise FacadeReadError(
            "nothing to read: wire `csv_path` (from the writer's `panels_csvs`), or wire "
            "`room` together with `results_dir` so the path can be built from the room name.")

    panels = read_panels(csv_path)

    frame = frame_from_box(box)
    if frame is None:
        if not rjson:
            raise FacadeReadError("wire `room` or `room_box` (preferred), or `result_json` — "
                                  "the panels are in room-local metres and need a frame to "
                                  "be placed in")
        frame = frame_from_result_json(rjson)
    base, w_m, d_m, h_m = frame

    exterior, why = resolve_exterior_sides(panels, sides_in, rjson, sides_src)

    iw = g.get("include_walls")
    iw = True if iw is None else bool(iw)
    mg = g.get("merge")
    mg = True if mg is None else bool(mg)

    builder = build_merged if mg else build_per_cell
    warn = []
    windows, walls, curves, sides, counts = builder(
        panels, exterior, base, metres_to_doc(), tol, include_walls=iw, warn=warn)

    info = notes + [why] + list(summarize(panels, exterior, counts))
    if w_m is not None:
        check_room_match(panels, w_m, d_m, h_m)
        info.append("room %.2f x %.2f x %.2f m — matches the box" % (w_m, d_m, h_m))
    else:
        info.append("frame from result.json placement (no box to cross-check against)")

    n_cells = sum(1 for q in panels if q["side"] in exterior and q["is_window"])
    if mg:
        info.append("built %d window + %d wall surface(s) — merged from %d window cell(s)"
                    % (len(windows), len(walls), n_cells))
    else:
        info.append("built %d window + %d wall surface(s) — one per cell (`merge` is False)"
                    % (len(windows), len(walls)))
    info.extend(warn)

    return {"windows": windows, "walls": walls, "window_curves": curves,
            "window_sides": sides, "info": info}


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

# Either entry point is valid now — `room` (+ results_dir) or a bare `csv_path` — so the
# trailer fires on whichever of them the component actually has.
if _IN_RHINO and ("room" in globals() or "csv_path" in globals()):
    _r = gh_main(globals())
    windows       = _r["windows"]
    walls         = _r["walls"]
    window_curves = _r["window_curves"]
    window_sides  = _r["window_sides"]
    info          = _r["info"]
