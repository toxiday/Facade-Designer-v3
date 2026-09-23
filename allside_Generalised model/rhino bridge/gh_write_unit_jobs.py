"""
gh_write_unit_jobs.py — Grasshopper script component: export a WHOLE APARTMENT UNIT.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib + RhinoCommon only.

Takes the apartment outline and a list of Rooms, and writes one `facade_job/1` per room
PLUS a `unit/1` manifest recording where each room sits inside the unit.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    unit_boundary   Brep    Item   the apartment as a CLOSED POLYSURFACE — its floor
                                   polygon extruded and capped. A Box or an Extrusion is
                                   accepted too; a bare closed planar Curve gives the
                                   footprint but carries no height.
    rooms           (none)  List   Room objects from gh_make_room.py.
                                   Type hint MUST be "No type hint".
    unit_name       str     Item   e.g. "101" — names the folder and the manifest
    unit_plane      Plane   Item   optional; orientation of the unit-local frame.
                                   Default: world XY. A polygon has no intrinsic axes,
                                   so wire this if the flat is not axis-aligned.
    unit_height     float   Item   optional; OVERRIDES the solid's own height. Normally
                                   leave it empty — the polysurface already states it.
    city            str     Item   city NAME — wire from gh_pick_city.py
    north           float   Item   Ladybug north for world +Y (default 0)
    repr_conditions int     Item   optional, default 48
    seed            int     Item   optional, default 0
    glare_weight    float   Item   optional (0.25)
    glare_cap       float   Item   optional (3000)
    out_dir         str     Item   optional; where the job JSONs go
                                   (default <project>/allside_Generalised model/All Units/<unit_name>)
    results_dir     str     Item   optional; where the optimizer writes its results.
                                   Default: BESIDE THE JOBS, at <out_dir>/<room name>/,
                                   so a unit is one self-contained folder.
    write           bool    Item   True to write the files
    matrix          bool    Item   optional, default True — draw the occupancy heat maps
                                   as soon as the jobs are written, so the figures on disk
                                   always match them
    run             bool    Item   True to ALSO launch run_unit.py over the whole unit
    console         bool    Item   optional, default True — own console window, so a
                                   multi-hour unit run is visible live
    repo_root       str     Item   optional; needed only if this script was PASTED
    writer_path     str     Item   optional; explicit path to gh_write_facade_job.py

COMPONENT OUTPUTS
    json_paths      the written job files, one per room
    unit_json       the written unit/1 manifest
    result_dirs     where each room's results WILL be   -> parallel to room_names
    panels_csvs     where each facade_panels.csv WILL be -> wire into gh_read_facade.py
    result_jsons    where each result.json WILL be       -> wire into gh_read_facade.py
    room_sides      each room's exterior walls           -> parallel to room_names
    room_names      parallel list of room names

    The four lists above are ORDERED LIKE `rooms`.  gh_read_facade.py reads ONE room at a
    time, so feed them into its Item-access inputs and Grasshopper iterates it per room —
    one reader component for the whole unit, not one per room.
    unit_dims       apartment area, corner count and bounding box
    sides_compass   which TRUE direction each room-local wall faces
    room_rects      the measured room rectangles — PREVIEW THE LAYOUT before writing
    unit_outline    the BASE FOOTPRINT the component extracted from the solid — preview it
                    too, to confirm the cap it picked is the one you meant
    log             validation notes, overlap report, coverage
────────────────────────────────────────────────────────────────────────────────────────

THE APARTMENT IS A SOLID WHOSE FOOTPRINT IS A POLYGON

    `unit_boundary` is a closed polysurface — a floor polygon extruded and capped — for the
    same reason a room is a Box rather than a rectangle: one piece of geometry carries both
    the plan and the storey height, so the two cannot disagree.  The footprint is read from
    the BOTTOM cap (rooms sit on the floor, so the floor outline is what their placement is
    checked against) and the height from the cap separation.

    Flats are L-shaped, notched and stepped far more often than they are rectangular, so
    every check that depends on the footprint works on the POLYGON, never its bounding box:

      * containment is point-in-polygon plus an edge-crossing test.  Across a concave notch
        all four corners of a room can sit inside the flat while the room still bulges out
        through the gap — corners alone would pass it.
      * coverage is measured against the polygon's true area, so a notch does not make a
        fully-planned flat read as half empty.
      * a room sitting FLUSH against an apartment wall is the normal case, so shared edges
        and touching corners are explicitly not treated as crossings or as "outside".

    The solid is checked for being a real extrusion: an open (uncapped) polysurface is
    refused with the fix named, a flat one is refused as a surface, and a top cap whose area
    differs from the base is reported — there the footprint varies with height and using the
    base would silently pick one of several answers.

    A polygon has no intrinsic axes the way a Box does, so the IN-PLANE orientation is stated
    rather than inferred: `unit_plane` if wired, else world XY, with the origin moved to the
    footprint's bounding-box min corner.  `unit.width`/`depth` in the manifest are that
    bounding box and exist only to size the composite raster; `boundary_uv` is the shape.

WHY THIS EXISTS: THE PLACEMENT PROBLEM

    Every per-room job already carried a `placement` block, so it looked like room positions
    were known.  They were not.  Measured across unit 101's seven files, written one at a
    time from whatever box happened to be on the canvas:

        101.2 x 101.7   overlap 2.95 x 1.65 m = 4.88 m2
        101.3 x 101.6   overlap 2.71 x 1.59 m = 4.30 m2
        101.4           sits 10 m away from every other room
        union bbox      268 m2 of bounding box for 130 m2 of rooms

    Nothing checked one room against another, because no component could see more than one
    room at a time.  That is the gap this file closes: rooms are measured INTO one shared
    unit frame, and every pair is checked against every other.  A layout that cannot be
    built is refused here rather than becoming a plausible-looking composite drawing later.

FRAMES
    Unit-local (u, v): metres in the unit box's own plane, origin at its min corner.
    Room-local (x, y): metres in the room box's own plane — unchanged, and still what the
    job files and all optimizer outputs speak.  `origin_uv` / `x_axis_uv` / `y_axis_uv` in
    the manifest are the transform between them, so a room's 32x32 floor grid can be placed
    into a unit-wide raster without guessing.

NO OPTIMIZER LOGIC, AND NO SECOND RASTERIZER
    This component LOADS gh_write_facade_job.py from disk and calls its build_job() once per
    room.  Copying that file's rasterize()/build_slots() here would create a second
    implementation of the binary-centre-sampling rule, which is precisely the kind of
    divergence that produces two different briefs from one drawing.  One rasterizer, reused.
"""

import os
import json
import subprocess

# ---------------------------------------------------------------------------
# EDIT THESE if you move things.  They are two DIFFERENT locations:
#
#   PROJECT_ROOT  the code — .venv, run_unit.py, gh_write_facade_job.py.
#                 Same constant as gh_write_facade_job.py.
#   MODEL_ROOT    the Rhino model and its units — where briefs and results live.
#
# They are separate because the code is a repository and the model is a project; keeping a
# building's data inside the optimizer's source tree is how it ends up committed, or lost.
# ---------------------------------------------------------------------------
PROJECT_ROOT = r"C:\Computation Project\Facade Designer v3"
MODEL_ROOT   = r"C:\Computation Project\Building Model"

# Under MODEL_ROOT: <MODEL_ROOT>/<UNITS_SUBDIR>/<unit_name>/<JOBS_SUBDIR>/
UNITS_SUBDIR = "Units"
JOBS_SUBDIR  = "output"

SIDES = ("S", "N", "E", "W")
OVERLAP_TOL_M2 = 0.01           # ignore hairline overlaps from box snapping
CONTAIN_TOL_M  = 0.01


class UnitInputError(Exception):
    """Something about the unit makes it unsafe to export."""


# ---------------------------------------------------------------------------
# Pure layout geometry — no Rhino, no numpy.
#
# Kept importable so unit_zone_matrix.py --check-layout can run the SAME overlap and
# containment checks on a manifest outside Rhino.  Two implementations of "do these rooms
# overlap" would eventually disagree, and the disagreement would show up as a drawing that
# looks fine while the jobs it came from do not tile.
# ---------------------------------------------------------------------------

def rect_corners(origin_uv, x_axis_uv, y_axis_uv, width, depth):
    """The room's four unit-local corners, counter-clockwise from its local origin."""
    ox, oy = origin_uv
    xx, xy = x_axis_uv
    yx, yy = y_axis_uv
    return [(ox, oy),
            (ox + xx * width, oy + xy * width),
            (ox + xx * width + yx * depth, oy + xy * width + yy * depth),
            (ox + yx * depth, oy + yy * depth)]


def polygon_area(poly):
    """Shoelace, signed; callers take abs()."""
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def polygon_bbox(poly):
    """(u_min, v_min, u_max, v_max)."""
    us = [p[0] for p in poly]
    vs = [p[1] for p in poly]
    return min(us), min(vs), max(us), max(vs)


def point_on_segment(p, a, b, tol):
    """Is p within `tol` of segment ab?  Rooms sit flush against apartment walls constantly,
    so "exactly on the boundary" is the normal case, not an edge case."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-18:
        return (px - ax) ** 2 + (py - ay) ** 2 <= tol * tol
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2 <= tol * tol


def point_in_polygon(p, poly, tol=1e-9):
    """Ray casting, with points ON the boundary counted as INSIDE.

    The on-boundary case has to be decided explicitly: a room corner sitting exactly on an
    apartment wall is ambiguous to a crossing count, and getting it wrong would reject every
    normal layout.
    """
    n = len(poly)
    for i in range(n):
        if point_on_segment(p, poly[i], poly[(i + 1) % n], tol):
            return True
    x, y = p
    inside = False
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xi > x:
                inside = not inside
    return inside


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_properly_cross(a, b, c, d, eps=1e-9):
    """True only for a TRANSVERSAL crossing — touching endpoints and collinear overlap are
    deliberately False.

    A room flush against an apartment wall shares a whole edge with the boundary, and a room
    in a corner shares two.  Treating that as a crossing would reject every real plan, so
    only a segment that genuinely passes through counts.
    """
    d1, d2 = _orient(a, b, c), _orient(a, b, d)
    d3, d4 = _orient(c, d, a), _orient(c, d, b)
    if abs(d1) < eps or abs(d2) < eps or abs(d3) < eps or abs(d4) < eps:
        return False                      # collinear or touching: not a crossing
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def rect_inside_polygon(rect, poly, tol):
    """(ok, offending_point).  A rectangle is inside a polygon iff every corner is inside
    and no edge crosses the boundary.

    The second half is what a corners-only test misses: across a concave notch, all four
    corners can sit inside the apartment while the room still bulges out through the gap.
    An L-shaped apartment makes that a realistic mistake, not a theoretical one.
    """
    for c in rect:
        if not point_in_polygon(c, poly, tol):
            return False, c
    n, m = len(rect), len(poly)
    for i in range(n):
        a, b = rect[i], rect[(i + 1) % n]
        for j in range(m):
            c, d = poly[j], poly[(j + 1) % m]
            if segments_properly_cross(a, b, c, d):
                return False, ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return True, None


def clip_convex(subject, clipper):
    """Sutherland-Hodgman: the intersection polygon of two CONVEX polygons.

    Rooms are rectangles, so convexity holds even when they are rotated relative to each
    other — which an axis-aligned min/max overlap test would get wrong.  Used only for
    room-vs-room overlap; the apartment boundary may be concave and is handled by
    rect_inside_polygon instead.
    """
    if polygon_area(clipper) < 0:
        clipper = clipper[::-1]
    out = list(subject)
    for i in range(len(clipper)):
        if not out:
            return []
        ax, ay = clipper[i]
        bx, by = clipper[(i + 1) % len(clipper)]
        ex, ey = bx - ax, by - ay

        def inside(p):
            return ex * (p[1] - ay) - ey * (p[0] - ax) >= -1e-12

        new, prev = [], out[-1]
        for cur in out:
            if inside(cur):
                if not inside(prev):
                    new.append(_line_isect(prev, cur, (ax, ay), (bx, by)))
                new.append(cur)
            elif inside(prev):
                new.append(_line_isect(prev, cur, (ax, ay), (bx, by)))
            prev = cur
        out = new
    return out


def _line_isect(p, q, a, b):
    x1, y1 = p; x2, y2 = q; x3, y3 = a; x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-15:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def overlap_area(poly_a, poly_b):
    inter = clip_convex(poly_a, poly_b)
    return abs(polygon_area(inter)) if len(inter) >= 3 else 0.0


def check_layout(rooms, boundary,
                 overlap_tol=OVERLAP_TOL_M2, contain_tol=CONTAIN_TOL_M):
    """Validate a set of placed rooms against the apartment outline.

    `rooms`    : list of dicts with name, width, depth, origin_uv, x_axis_uv, y_axis_uv.
    `boundary` : the apartment outline as [(u, v), ...] in unit-local metres — a POLYGON,
                 not a rectangle.  Apartments are L-shaped, notched and stepped far more
                 often than they are rectangular, and a bounding box would accept rooms
                 sitting in a notch that is not part of the flat.

    Returns (errors, notes) — both lists of strings.  Errors mean the layout cannot be
    built; notes are informational.
    """
    errors, notes = [], []

    if len(boundary) < 3:
        return (["the unit boundary has fewer than 3 points — it is not a polygon"], notes)

    seen = {}
    for r in rooms:
        if r["name"] in seen:
            errors.append("duplicate room name %r — names become filenames, so the second "
                          "would overwrite the first" % r["name"])
        seen[r["name"]] = True

    polys = {}
    for r in rooms:
        polys[r["name"]] = rect_corners(r["origin_uv"], r["x_axis_uv"], r["y_axis_uv"],
                                        r["width"], r["depth"])

    # Containment in the apartment POLYGON
    u0, v0, u1, v1 = polygon_bbox(boundary)
    for r in rooms:
        ok, bad = rect_inside_polygon(polys[r["name"]], boundary, contain_tol)
        if not ok:
            errors.append(
                "room %s falls outside the apartment outline (near %.2f, %.2f; the outline "
                "spans %.2f-%.2f x %.2f-%.2f m) — is this room part of this unit, and does "
                "the outline cover it?"
                % (r["name"], bad[0], bad[1], u0, u1, v0, v1))

    # Pairwise overlap
    names = [r["name"] for r in rooms]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = overlap_area(polys[names[i]], polys[names[j]])
            if a > overlap_tol:
                errors.append(
                    "rooms %s and %s overlap by %.2f m2 — two rooms cannot occupy the same "
                    "floor. Fix the boxes in Rhino." % (names[i], names[j], a))

    # Coverage, against the polygon's TRUE area rather than its bounding box — an L-shaped
    # flat would otherwise look half empty purely because of the notch.
    room_area = sum(r["width"] * r["depth"] for r in rooms)
    unit_area = abs(polygon_area(boundary))
    cov = 100.0 * room_area / unit_area if unit_area > 0 else 0.0
    notes.append("coverage: %.1f m2 of rooms in a %.1f m2 apartment (%.0f%%)"
                 % (room_area, unit_area, cov))
    if cov < 40.0:
        # Plain append, not a % format — so the percent sign is written once, not doubled.
        notes.append("  [!] under 40% of the apartment is room — is the outline the flat "
                     "itself, or are rooms still missing?")
    if cov > 100.0 + 1e-6:
        errors.append("rooms total %.1f m2 but the apartment outline encloses only %.1f m2"
                      % (room_area, unit_area))
    return errors, notes


# ---------------------------------------------------------------------------
# Loading the per-room writer
# ---------------------------------------------------------------------------

def resolve_repo(repo_root=None):
    """Locate the project folder, or None.  Mirrors gh_write_facade_job.resolve_repo."""
    if repo_root:
        p = os.path.abspath(str(repo_root).strip().strip('"'))
        return p if os.path.isdir(p) else None
    if PROJECT_ROOT and os.path.isdir(PROJECT_ROOT):
        return os.path.abspath(PROJECT_ROOT)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return None                     # pasted script: nothing to walk up from
    p = os.path.abspath(os.path.join(here, "..", ".."))
    return p if os.path.isdir(p) else None


def load_writer(repo_root=None, writer_path=None):
    """Import gh_write_facade_job.py from disk.

    Deliberately loaded rather than duplicated: build_job/rasterize/build_slots encode the
    binary-centre-sampling rule and the max-merge slot rule, and a second copy of those
    would be a second definition of what the drawing MEANS.
    """
    import importlib.util

    if writer_path:
        p = os.path.abspath(str(writer_path).strip().strip('"'))
    else:
        repo = resolve_repo(repo_root)
        if repo is None:
            raise UnitInputError(
                "cannot locate the project, so gh_write_facade_job.py cannot be loaded. "
                "If this script was pasted it has no __file__ to walk up from — set the "
                "PROJECT_ROOT constant at the top (currently %r), or wire `repo_root` or "
                "`writer_path`." % PROJECT_ROOT)
        p = os.path.join(repo, "allside_Generalised model", "rhino bridge",
                         "gh_write_facade_job.py")
    if not os.path.isfile(p):
        raise UnitInputError("no gh_write_facade_job.py at %r" % p)

    spec = importlib.util.spec_from_file_location("_ghwriter", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _room_field(r, name, ri):
    """Read one field off a Room — by attribute, falling back to dict access.

    DUCK-TYPED ON PURPOSE, never isinstance(): gh_make_room.py is loaded in its own module
    instance, so its Room class is a different class object from anything this file could
    import.  See the note in gh_make_room.py.
    """
    v = getattr(r, name, None)
    if v is None and isinstance(r, dict):
        v = r.get(name)
    if v is None:
        raise UnitInputError(
            "rooms[%d] has no `%s` — expected a Room from the gh_make_room component (or a "
            "dict with room_box/name/sides/zones). Got %r. Check the input's type hint is "
            "'No type hint': a typed input coerces the object and destroys it."
            % (ri, name, type(r).__name__))
    return v


# ---------------------------------------------------------------------------
# Rhino-dependent
# ---------------------------------------------------------------------------

def coerce_boundary_solid(obj, tol):
    """Whatever Grasshopper handed us -> (Brep solid or None, fallback Curve or None).

    The apartment is a CLOSED POLYSURFACE: a polygon extruded and capped.  That is the same
    kind of object a room is — a solid, not an outline — so the unit carries its footprint
    AND its storey height in one piece of geometry, and the two cannot disagree.

    A Box is a capped extrusion of a rectangle, so it is accepted unchanged.  A bare closed
    planar Curve is also accepted as a convenience, but it has no height: `unit_height` or
    the tallest room then supplies it, and the log says so.
    """
    import Rhino
    import Rhino.Geometry as rg

    obj = getattr(obj, "Value", obj)                 # unwrap a GH object wrapper

    if type(obj).__name__ == "Guid":
        doc = Rhino.RhinoDoc.ActiveDoc
        ro = doc.Objects.FindId(obj) if doc is not None else None
        if ro is None:
            raise UnitInputError("`unit_boundary` is a Guid that is not in the active Rhino "
                                 "document — was the object deleted?")
        obj = ro.Geometry

    if isinstance(obj, rg.Box):
        return obj.ToBrep(), None
    if isinstance(obj, rg.Extrusion):
        b = obj.ToBrep(True)                         # True = split kinked faces
        if b is None:
            raise UnitInputError("`unit_boundary` is an Extrusion that could not be "
                                 "converted to a polysurface")
        return b, None
    if isinstance(obj, rg.Brep):
        if not obj.IsSolid:
            # An open Brep is almost always a surface or an uncapped extrusion.  Say which,
            # because the fix differs: cap it, or wire the footprint curve instead.
            raise UnitInputError(
                "`unit_boundary` is an OPEN polysurface (%d face(s)) — the apartment must be "
                "a CLOSED solid. If this is an extrusion, cap it (Rhino: Cap, or "
                "Grasshopper: Extrude with a closed profile then Cap Holes). If it is a flat "
                "region, wire its boundary curve instead." % obj.Faces.Count)
        return obj, None

    # Curve-ish fallbacks: Rectangle3d, Polyline, Circle, Arc, or a real Curve.
    if not isinstance(obj, rg.Curve):
        to_nurbs = getattr(obj, "ToNurbsCurve", None)
        if callable(to_nurbs):
            obj = to_nurbs()
    if isinstance(obj, rg.Curve):
        if not obj.IsClosed:
            raise UnitInputError("`unit_boundary` is an open curve — the apartment outline "
                                 "must be closed")
        if not obj.IsPlanar(tol):
            raise UnitInputError("`unit_boundary` is a non-planar curve")
        return None, obj

    raise UnitInputError(
        "`unit_boundary` must be a closed polysurface — a polygon extruded and capped (got "
        "%s). Set the input's type hint to Brep. A Box, or a closed planar Curve for the "
        "footprint alone, is accepted too." % type(obj).__name__)


def _polyline_points(curve, tol):
    """A closed curve -> its corner points, discretizing anything genuinely curved.

    Every consumer downstream — containment, area, the composite raster mask — works on a
    polygon, so the discretization happens once, here.
    """
    ok, pl = curve.TryGetPolyline()
    if not ok:
        pc = curve.ToPolyline(tol, 0.02, 0.0, 0.0)
        if pc is None:
            raise UnitInputError("could not convert the apartment outline into a polygon")
        pl = pc.ToPolyline()
    pts = list(pl)
    if len(pts) > 1 and pts[0].DistanceTo(pts[-1]) < tol:
        pts = pts[:-1]                               # drop the repeated closing point
    if len(pts) < 3:
        raise UnitInputError("the apartment outline has fewer than 3 corners")
    return pts


def footprint_from_solid(brep, plane, tol):
    """(footprint points, height, notes) — the base cap outline and the extrusion height.

    Takes the BOTTOM cap: the lowest face whose plane is horizontal in the unit frame.  The
    rooms sit on the floor, so the floor outline is the one their placement must be checked
    against.
    """
    import Rhino.Geometry as rg

    notes = []
    caps = []
    for f in brep.Faces:
        ok, fpl = f.TryGetPlane(tol)
        if not ok:
            continue
        if abs(rg.Vector3d.Multiply(fpl.Normal, plane.ZAxis)) < 0.999:
            continue                                 # a wall, not a cap
        h = rg.Vector3d.Multiply(rg.Vector3d(fpl.Origin - plane.Origin), plane.ZAxis)
        caps.append((h, f))

    if not caps:
        raise UnitInputError(
            "`unit_boundary` has no horizontal cap face, so it is not a vertical extrusion "
            "in this frame. Build it by extruding the apartment polygon straight up and "
            "capping it — or wire `unit_plane` if the flat is deliberately tilted.")

    caps.sort(key=lambda t: t[0])
    base_h, base = caps[0]
    top_h = caps[-1][0]
    height = top_h - base_h

    loop = base.OuterLoop.To3dCurve()
    if loop is None or not loop.IsClosed:
        raise UnitInputError("the base cap of `unit_boundary` has no closed outer loop")
    pts = _polyline_points(loop, tol)

    if base.Loops.Count > 1:
        notes.append("[!] the apartment base has %d inner loop(s) (holes) — only the OUTER "
                     "boundary is used, so a courtyard or shaft will read as apartment"
                     % (base.Loops.Count - 1))
    if height <= tol:
        raise UnitInputError("`unit_boundary` is flat (height %.4f) — it is a surface, not "
                             "an extruded solid" % height)

    # A genuine extrusion has the same footprint top and bottom.  If it does not, the
    # footprint varies with height and using the base silently picks one of several answers.
    a_base = rg.AreaMassProperties.Compute(loop)
    a_top = rg.AreaMassProperties.Compute(caps[-1][1].OuterLoop.To3dCurve())
    if a_base and a_top and abs(a_base.Area) > 1e-9:
        if abs(abs(a_top.Area) - abs(a_base.Area)) / abs(a_base.Area) > 0.01:
            notes.append("[!] the top cap encloses %.1f%% of the base area — this is not a "
                         "straight extrusion, and the BASE outline is the one being used"
                         % (100.0 * abs(a_top.Area) / abs(a_base.Area)))
    return pts, height, notes


def unit_frame(unit_boundary, tol, unit_plane=None, unit_height=None):
    """(base_plane, boundary_uv, width_m, depth_m, height_m, scale, notes).

    The apartment solid supplies both the footprint and the storey height, so the two come
    from one piece of geometry and cannot drift apart.

    A polygon has no intrinsic axes the way a Box does, so the in-plane orientation is
    stated rather than inferred: the optional `unit_plane` if wired, otherwise world X/Y.
    `width`/`depth` are the footprint's bounding box — used only to size the composite
    raster; containment and area use the polygon itself.
    """
    import Rhino
    import Rhino.Geometry as rg

    if unit_boundary is None:
        raise UnitInputError("`unit_boundary` is empty — draw the apartment as a closed "
                             "polysurface: extrude its floor polygon and cap it")

    doc = Rhino.RhinoDoc.ActiveDoc
    s = (Rhino.RhinoMath.UnitScale(doc.ModelUnitSystem, Rhino.UnitSystem.Meters)
         if doc else 1.0)

    up = getattr(unit_plane, "Value", unit_plane)
    if up is not None and not isinstance(up, rg.Plane):
        raise UnitInputError("`unit_plane` must be a Plane (got %s)" % type(up).__name__)
    frame = up if up is not None else rg.Plane.WorldXY

    solid, curve = coerce_boundary_solid(unit_boundary, tol)
    notes = []
    if solid is not None:
        pts, h_doc, notes = footprint_from_solid(solid, frame, tol)
        notes.insert(0, "apartment solid: %d faces, base footprint %d corners, "
                        "extruded %.3f (doc units)"
                     % (solid.Faces.Count, len(pts), h_doc))
    else:
        pts = _polyline_points(curve, tol)
        h_doc = 0.0
        notes.append("[!] `unit_boundary` is a bare curve, so it carries no height — "
                     "wire `unit_height`, or a capped polysurface, to state the storey")

    xa, ya = frame.XAxis, frame.YAxis
    anchor = frame.Origin if up is not None else pts[0]

    # Measure the outline in that orientation, then move the origin to its min corner so
    # unit-local coordinates start at (0, 0).
    raw = [(rg.Vector3d.Multiply(rg.Vector3d(p - anchor), xa),
            rg.Vector3d.Multiply(rg.Vector3d(p - anchor), ya)) for p in pts]
    umin = min(u for u, _ in raw)
    vmin = min(v for _, v in raw)
    base = rg.Plane(anchor + xa * umin + ya * vmin, xa, ya)

    boundary_uv = [((u - umin) * s, (v - vmin) * s) for u, v in raw]
    uu0, vv0, uu1, vv1 = polygon_bbox(boundary_uv)

    # An explicit `unit_height` overrides the solid, but say so — silently ignoring the
    # geometry someone drew is worse than disagreeing with it out loud.
    h = h_doc * s
    if unit_height:
        stated = float(unit_height) * s
        if h and abs(stated - h) > 0.01:
            notes.append("[!] `unit_height` %.2f m overrides the solid's own %.2f m"
                         % (stated, h))
        h = stated
    return base, boundary_uv, uu1 - uu0, vv1 - vv0, h, s, notes


def room_in_unit(room_base, unit_base, scale):
    """(origin_uv, x_axis_uv, y_axis_uv) — the room's frame expressed in unit-local metres.

    Measured by projection onto the unit plane's axes rather than read off the room box
    independently.  That is the whole fix: a room's position is now a MEASUREMENT INTO a
    shared frame, so two rooms cannot disagree about where they are.
    """
    import Rhino.Geometry as rg

    v = rg.Vector3d(room_base.Origin - unit_base.Origin)
    ux, uy = unit_base.XAxis, unit_base.YAxis
    origin_uv = (rg.Vector3d.Multiply(v, ux) * scale,
                 rg.Vector3d.Multiply(v, uy) * scale)
    # Axes are unit vectors, so their components in the unit frame need no scaling.
    x_axis_uv = (rg.Vector3d.Multiply(room_base.XAxis, ux),
                 rg.Vector3d.Multiply(room_base.XAxis, uy))
    y_axis_uv = (rg.Vector3d.Multiply(room_base.YAxis, ux),
                 rg.Vector3d.Multiply(room_base.YAxis, uy))
    return origin_uv, x_axis_uv, y_axis_uv


def rect_polyline(corners, unit_base, scale):
    """Unit-local metres -> a closed world polyline, for previewing the layout in Rhino."""
    import Rhino.Geometry as rg
    pts = [unit_base.PointAt(x / scale, y / scale, 0.0) for (x, y) in corners]
    return rg.PolylineCurve(pts + [pts[0]])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_unit(g):
    """Everything except writing files.  Returns a dict of results + the manifest."""
    import Rhino

    doc = Rhino.RhinoDoc.ActiveDoc
    tol = doc.ModelAbsoluteTolerance if doc else 0.001

    W = load_writer(g.get("repo_root"), g.get("writer_path"))

    rooms_in = [r for r in (g.get("rooms") or []) if r is not None]
    if not rooms_in:
        raise UnitInputError("no `rooms` — wire at least one Room from gh_make_room "
                             "(input Access must be List, type hint 'No type hint')")

    unit_name = str(g.get("unit_name") or "unit").strip()
    unit_base, boundary_uv, unit_w, unit_d, unit_h, scale, unotes = unit_frame(
        g.get("unit_boundary"), tol, g.get("unit_plane"), g.get("unit_height"))

    out_dir = g.get("out_dir")
    if out_dir:
        out_dir = os.path.abspath(str(out_dir).strip().strip('"'))
    elif MODEL_ROOT and os.path.isdir(MODEL_ROOT):
        out_dir = os.path.join(MODEL_ROOT, UNITS_SUBDIR, unit_name, JOBS_SUBDIR)
    else:
        raise UnitInputError(
            "cannot default `out_dir`: MODEL_ROOT is %r, which does not exist. Set the "
            "MODEL_ROOT constant at the top of this script to your Rhino model folder, or "
            "wire `out_dir`." % MODEL_ROOT)

    # Results live BESIDE THE JOBS by default: <out_dir>/<room name>/.  A unit is then one
    # self-contained folder — the brief, the designs and the summary in the same place —
    # rather than jobs here and results in a central outputs/ tree that has no idea which
    # unit a room belongs to.  Wire `results_dir` to send them somewhere else.
    results_root = g.get("results_dir")
    results_root = (os.path.abspath(str(results_root).strip().strip('"'))
                    if results_root else out_dir)

    north = g.get("north")
    city  = g.get("city")

    entries, notes, all_pts, all_vals, names = [], list(unotes), [], [], []
    for ri, r in enumerate(rooms_in):
        r = getattr(r, "Value", r)                  # unwrap a GH object wrapper
        box   = _room_field(r, "room_box", ri)
        name  = str(_room_field(r, "name", ri))
        sides = str(_room_field(r, "sides", ri))
        zones = _room_field(r, "zones", ri)

        # Stamping `output.dir` into the job is what actually redirects the results:
        # run_facade_job.py reads it and writes there, so the job file carries its own
        # destination rather than depending on a runner flag being remembered.
        room_results = os.path.join(results_root, name)

        # One rasterizer, reused — see the module docstring.
        #
        # The per-room writer reports zones as "zones[0]" etc., which is unambiguous when
        # ONE room is being written and useless when seven are: it names an index into a
        # list you cannot see, on a room it does not name.  Re-raise with the room attached
        # so the message points at a component on the canvas.
        try:
            job, pts, vals, city_s, north_angle, rnotes = W.build_job(
                box, north, sides, city, zones,
                g.get("repr_conditions"), g.get("seed"), tol,
                g.get("glare_weight"), g.get("glare_cap"),
                out_name=name, result_dir=room_results)
        except Exception as e:                    # noqa: BLE001 — re-raised immediately
            zsum = []
            for zi, z in enumerate(zones or []):
                z = getattr(z, "Value", z)
                h = getattr(z, "hour", None)
                lx = getattr(z, "lux", None)
                if h is None and isinstance(z, dict):
                    h, lx = z.get("hour"), z.get("lux")
                zsum.append("zones[%d] = %s @ %s lux"
                            % (zi, "%02d:00" % int(h) if h is not None else "?",
                               ("%g" % lx) if lx is not None else "?"))
            raise UnitInputError(
                "room %s (rooms[%d], sides %s) could not be written:\n\n%s\n\n"
                "That room's zones, in the order the writer sees them:\n  %s\n\n"
                "Each `zones[i]` above is one gh_make_zone component feeding THIS room's "
                "gh_make_room.\nIf an hour or lux here is not one you set for %s, that is "
                "the mis-wired zone."
                % (name, ri, sides, e, "\n  ".join(zsum) or "(none)", name)) from None

        room_base = W.room_from_box(box, north)[3]
        origin_uv, x_axis_uv, y_axis_uv = room_in_unit(room_base, unit_base, scale)

        res_dir, panels_csv, res_json = W.predict_outputs(
            name, room_results, g.get("repo_root"))

        entries.append(dict(
            name=name, job=job,
            width=job["room"]["width"], depth=job["room"]["depth"],
            height=job["room"]["height"], north_angle=north_angle, sides=job["sides"],
            origin_uv=[round(origin_uv[0], 6), round(origin_uv[1], 6)],
            x_axis_uv=[round(x_axis_uv[0], 6), round(x_axis_uv[1], 6)],
            y_axis_uv=[round(y_axis_uv[0], 6), round(y_axis_uv[1], 6)],
            job_file="%s.json" % name,
            result_dir=res_dir, panels_csv=panels_csv, result_json=res_json))
        names.append(name)
        all_pts.extend(pts)
        all_vals.extend(vals)
        notes.extend("%s: %s" % (name, n) for n in rnotes)
        notes.append("%s: %d slot(s) %s | sides %s | %.2f x %.2f m @ (%.2f, %.2f) in unit"
                     % (name, len(job["slots"]),
                        ",".join("%02d" % s["hour"] for s in job["slots"]),
                        job["sides"], job["room"]["width"], job["room"]["depth"],
                        origin_uv[0], origin_uv[1]))

    errors, cnotes = check_layout(entries, boundary_uv)
    notes.extend(cnotes)
    if errors:
        raise UnitInputError(
            "the unit layout cannot be built:\n  " + "\n  ".join(errors) +
            "\n\nEvery room is measured into the apartment outline's frame, so these are "
            "real conflicts between the geometry on the canvas, not a units or convention "
            "problem.")

    if not unit_h:
        unit_h = max((e["height"] for e in entries), default=0.0)
        notes.append("unit height fell back to the tallest room (%.2f m) — draw the "
                     "apartment as a capped extrusion, or wire `unit_height`" % unit_h)

    manifest = dict(
        schema="unit/1", units="m", name=unit_name,
        unit=dict(width=round(unit_w, 6), depth=round(unit_d, 6),
                  height=round(unit_h, 6),
                  area=round(abs(polygon_area(boundary_uv)), 6)),
        # The apartment outline in unit-local metres.  width/depth above are only its
        # bounding box; THIS is the shape.
        boundary_uv=[[round(u, 6), round(v, 6)] for u, v in boundary_uv],
        climate=dict(city=entries[0]["job"]["climate"]["city"] if entries else ""),
        placement={"origin": [unit_base.Origin.X, unit_base.Origin.Y, unit_base.Origin.Z],
                   "x_axis": [unit_base.XAxis.X, unit_base.XAxis.Y, unit_base.XAxis.Z],
                   "y_axis": [unit_base.YAxis.X, unit_base.YAxis.Y, unit_base.YAxis.Z]},
        rooms=[{k: e[k] for k in ("name", "width", "depth", "height", "north_angle",
                                  "sides", "origin_uv", "x_axis_uv", "y_axis_uv",
                                  "job_file", "result_dir")} for e in entries])

    rects = [rect_polyline(rect_corners(e["origin_uv"], e["x_axis_uv"], e["y_axis_uv"],
                                        e["width"], e["depth"]), unit_base, scale)
             for e in entries]

    return dict(entries=entries, manifest=manifest, notes=notes, out_dir=out_dir,
                unit_w=unit_w, unit_d=unit_d, unit_h=unit_h, names=names, rects=rects,
                boundary_uv=boundary_uv,
                boundary_curve=rect_polyline(boundary_uv, unit_base, scale),
                unit_area=abs(polygon_area(boundary_uv)),
                grid_pts=all_pts, grid_vals=all_vals,
                north_angle=entries[0]["north_angle"] if entries else 0.0,
                writer=W)


def launch_matrix(unit_dir, repo_root=None):
    """Draw the occupancy heat maps for the jobs just written.

    Drawn HERE, at export time, not only before an optimizer run: a figure produced by the
    writer is by construction the picture of the jobs it just wrote.  Unit 301 showed why
    that matters — it was optimized, then re-exported an hour later, leaving results that
    answered a brief nobody could see any more.

    Shells out to the project venv for the same reason the optimizer does: Rhino's Python
    has no numpy and no matplotlib.  It takes a couple of seconds, so unlike the optimizer
    it is run inline and its output captured into `log` rather than given its own console.
    """
    repo = resolve_repo(repo_root)
    if repo is None:
        return "heat map skipped: cannot locate the project (set PROJECT_ROOT)"
    python = os.path.join(repo, ".venv", "Scripts", "python.exe")
    script = os.path.join(repo, "allside_Generalised model", "model trainer",
                          "unit_zone_matrix.py")
    if not os.path.isfile(python):
        return "heat map skipped: no venv python at %s" % python
    if not os.path.isfile(script):
        return "heat map skipped: no unit_zone_matrix.py at %s" % script

    # Rhino exports PYTHONHOME/PYTHONPATH at its own embedded install; a child venv python
    # would boot against Rhino's stdlib and die inside `import re`.  Scrub them.
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)

    out = []
    for mode in ("room", "unit"):
        try:
            # encoding is stated: the script prints em-dashes and degree signs, and the
            # Windows default (cp1252) turns those into mojibake in the Grasshopper panel.
            p = subprocess.run([python, script, unit_dir, "--by", mode],
                               cwd=os.path.dirname(script), env=env,
                               capture_output=True, text=True, timeout=300,
                               encoding="utf-8", errors="replace")
        except Exception as e:                   # noqa: BLE001 — a picture must not block
            out.append("heat map --by %s failed: %s" % (mode, e))
            continue
        if p.returncode == 0:
            wrote = [ln for ln in (p.stdout or "").splitlines() if ln.startswith("Written:")]
            out.append(wrote[-1] if wrote else "heat map --by %s: done" % mode)
            # Surface the no-sun warning — it is the one thing here worth acting on.
            out.extend(ln.strip() for ln in (p.stdout or "").splitlines() if "[!]" in ln)
        else:
            tail = ((p.stderr or p.stdout or "").strip().splitlines() or ["(no output)"])[-1]
            out.append("heat map --by %s failed: %s" % (mode, tail))
    return "\n".join(out)


def launch_unit_run(unit_dir, repo_root=None, console=True):
    """Shell out to run_unit.py in the project venv — the whole unit in one process.

    Same reasoning as gh_write_facade_job.launch_optimizer: torch cannot run inside Rhino,
    and a Grasshopper component blocks while it runs, so a separate console is the only way
    to watch a multi-hour unit run.
    """
    repo = resolve_repo(repo_root)
    if repo is None:
        return "run skipped: cannot locate the project (set PROJECT_ROOT or wire repo_root)"
    python = os.path.join(repo, ".venv", "Scripts", "python.exe")
    runner = os.path.join(repo, "allside_Generalised model", "model trainer", "run_unit.py")
    if not os.path.isfile(python):
        return "run skipped: no venv python at %s" % python
    if not os.path.isfile(runner):
        return "run skipped: no runner at %s" % runner

    # Rhino exports PYTHONHOME/PYTHONPATH at its own embedded install; a child venv python
    # would boot against Rhino's stdlib and die inside `import re`.  Scrub them.
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)

    cmd = [python, runner, unit_dir]
    if console:
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(runner), env=env,
                                creationflags=0x00000010)     # CREATE_NEW_CONSOLE
        proc.wait()
        return ("unit run finished in its own console window (exit code %d)"
                % proc.returncode)
    proc = subprocess.run(cmd, cwd=os.path.dirname(runner),
                          capture_output=True, text=True, env=env)
    return (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`."""
    r = build_unit(g)
    W = r["writer"]
    az = W.wall_azimuths(r["north_angle"])

    out = {
        "json_paths": [],
        "unit_json": None,
        # Parallel lists, all ordered like `rooms`.  gh_read_facade.py takes ONE room at a
        # time, so these are what Grasshopper iterates it over — feed them into its Item
        # inputs and it runs once per room.  room_sides and result_jsons exist so the reader
        # can be told which walls are exterior rather than having to infer it.
        "result_dirs": [e["result_dir"] for e in r["entries"]],
        "panels_csvs": [e["panels_csv"] for e in r["entries"]],
        "result_jsons": [e["result_json"] for e in r["entries"]],
        "room_sides": [e["sides"] for e in r["entries"]],
        "room_names": r["names"],
        "unit_dims": "%.1f m2 apartment, %d corners, %.2f m storey  "
                     "(%.2f x %.2f m bounding box, %d rooms)"
                     % (r["unit_area"], len(r["boundary_uv"]), r["unit_h"],
                        r["unit_w"], r["unit_d"], len(r["entries"])),
        "sides_compass": ["%s -> %s (%.0f deg)" % (s, W.compass_name(az[s]), az[s])
                          for s in SIDES],
        "room_rects": r["rects"],
        "unit_outline": r["boundary_curve"],
        "log": list(r["notes"]),
    }

    if not g.get("write"):
        out["log"].append("layout is valid — set `write` to True to write %d job file(s) "
                          "+ unit.json into %s" % (len(r["entries"]), r["out_dir"]))
        return out

    os.makedirs(r["out_dir"], exist_ok=True)
    for e in r["entries"]:
        p = os.path.join(r["out_dir"], e["job_file"])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(e["job"], f, indent=2)
        out["json_paths"].append(p)
    up = os.path.join(r["out_dir"], "unit.json")
    with open(up, "w", encoding="utf-8") as f:
        json.dump(r["manifest"], f, indent=2)
    out["unit_json"] = up
    out["log"].append("wrote %d job file(s) + unit.json into %s"
                      % (len(r["entries"]), r["out_dir"]))

    # Draw the brief straight away, so the figure on disk always matches the jobs that were
    # just written.  `matrix` defaults ON; set it False to skip the couple of seconds.
    mx = g.get("matrix")
    if mx is None or bool(mx):
        out["log"].append(launch_matrix(r["out_dir"], g.get("repo_root")))

    if g.get("run"):
        console = g.get("console")
        out["log"].append(launch_unit_run(r["out_dir"], g.get("repo_root"),
                                          True if console is None else bool(console)))
    return out


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

if _IN_RHINO and "unit_boundary" in globals():
    _r = gh_main(globals())
    json_paths    = _r["json_paths"]
    unit_json     = _r["unit_json"]
    result_dirs   = _r["result_dirs"]
    panels_csvs   = _r["panels_csvs"]
    result_jsons  = _r["result_jsons"]
    room_sides    = _r["room_sides"]
    room_names    = _r["room_names"]
    unit_dims     = _r["unit_dims"]
    sides_compass = _r["sides_compass"]
    room_rects    = _r["room_rects"]
    unit_outline  = _r["unit_outline"]
    log           = _r["log"]
