"""
gh_write_facade_job.py — Grasshopper script component: export a `facade_job/1` JSON.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib + RhinoCommon only — no external packages.

You draw a BOX for the room and closed CURVES for the occupancy zones; this component
derives every parameter the optimizer needs, rasterizes the zones onto the model's fixed
32x32 floor grid, and writes the job file.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    room_box        Box        Item   the room INTERIOR (walls have no thickness)
    north           float      Item   Ladybug north for world +Y (default 0)
    sides           str        Item   which walls may hold windows, e.g. "SNEW"
    city            str        Item   city NAME — wire from gh_pick_city.py
    zones           (none)     List   Zone objects — wire from gh_make_zone.py.
                                      Type hint MUST be "No type hint": a typed input
                                      coerces the object and destroys it.
    repr_conditions int        Item   default 48
    seed            int        Item   default 0 (reproducible designs)
    glare_weight    float      Item   optional; unoccupied-floor glare penalty (0.25)
    glare_cap       float      Item   optional; lux threshold for it (3000)
    out_path        str        Item   destination .json
    result_dir      str        Item   optional; where the optimizer should write its results
                                      (default <project>/allside_Generalised model/outputs/<name>)
    write           bool       Item   True to write the file
    run             bool       Item   True to ALSO launch the optimizer (needs the venv)
    console         bool       Item   optional, default True — run the optimizer in its own
                                      console window so progress is visible LIVE.  Grasshopper
                                      blocks while a component runs, so Rhino's output panel
                                      cannot repaint until the run ends; a separate console is
                                      the only way to watch it.  False captures into `log`
                                      instead, showing nothing until it finishes.
    repo_root       str        Item   optional; only needed if this script was PASTED
                                      rather than loaded from disk — see launch_optimizer

HOW TO USE: paste this whole file into a Grasshopper Python 3 script component, add the
inputs and outputs above, and set PROJECT_ROOT (top of the file) to your project folder.

The .gh file can live anywhere — nothing resolves relative to the Grasshopper document or
the working directory.  PROJECT_ROOT is used ONLY by the optional `run` toggle.

This component runs inside RHINO's Python, which has no numpy and no torch — hence stdlib +
RhinoCommon only.  `run` does not change that: it launches the project's .venv as a
SEPARATE PROCESS, so the heavy dependencies stay outside Rhino entirely.

COMPONENT OUTPUTS
    json_path       the written file
    panels_csv      where facade_panels.csv WILL be -> wire into gh_read_facade.py
    result_json     where result.json WILL be
    result_dir      the folder holding both
    grid_pts        1024 cell-centre points, in world space — preview the raster
    grid_vals       matching target lux per point (0 = unoccupied)
    city_resolved   the city name actually written into the job
    room_dims       "W x D x H m  (north_angle=...)" as derived from the box
    sides_compass   which TRUE direction each room-local wall faces
    log             validation notes / optimizer stdout

`panels_csv` / `result_json` / `result_dir` are PREDICTED — they exist only once the
optimizer has actually run.  They are emitted anyway so the reader component can be wired up
front rather than having its path typed by hand.

CITY SELECTION lives in gh_pick_city.py, not here — so you can see which city you picked,
and whether its sun is even up at your chosen hours, before exporting anything.  This
component just carries the NAME through; resolve_city() on the Python side does the
authoritative lookup (exact stem, exact label, or unique substring) and raises listing
candidates if it is ambiguous.  So a typed name works too, it just fails at run time
instead of in Grasshopper.
────────────────────────────────────────────────────────────────────────────────────────

TWO CONVENTIONS THAT WILL BITE IF IGNORED

1.  `sides` is in ROOM-LOCAL wall ids, not compass directions.  `S` means "the wall at
    y=0 in the box's own frame", never "the wall facing south".  Once the box is rotated
    they differ — read `sides_compass` before choosing.  At north_angle=270 the wall
    facing true south is `E`, so asking for "S" would glaze the west-facing wall instead.

2.  Rasterization is BINARY CENTRE SAMPLING: a cell is occupied iff its centre falls
    inside a curve.  That is not a shortcut — the surrogate's floor output is 1024 point
    sensors at exactly those centres, and the loss scores each point against its own
    target, so the question really is yes/no per point.  Fractional coverage would be
    WRONG: an edge cell 40% covered would become a 0.4*400 = 160 lux TARGET, and the
    optimizer would dutifully darken the zone edge.  Consequences: zone edges quantize to
    +/- half a cell (W/32 x D/32), and a zone thinner than one cell can miss every centre
    — which is why a curve capturing zero cells is a hard error here, not a warning.
"""

import os
import json
import math
import subprocess

# ---------------------------------------------------------------------------
# EDIT THIS if you move the project.
# ---------------------------------------------------------------------------
# Used ONLY by the optional `run` toggle, to find .venv and run_facade_job.py.
# Everything else in this component works without it.
#
# It is a hard-coded constant because the normal way to use this file is to PASTE it into a
# Grasshopper script component — and a pasted script has no __file__ to locate itself from.
# The `repo_root` input overrides it if you would rather wire the path than edit the code.
PROJECT_ROOT = r"C:\Computation Project\Facade Designer v3"

GRID_N      = 32
N_GRID_VALS = GRID_N * GRID_N
SIDES       = ("S", "N", "E", "W")
# Must match model_allsides_climate.py — the model CLAMPS outside these, silently.
W_MIN, W_MAX = 2.5, 10.0
D_MIN, D_MAX = 2.5, 12.0
H_MIN, H_MAX = 2.5, 8.0


class JobInputError(Exception):
    """Something about the Grasshopper inputs makes the job unsafe to export."""


# ---------------------------------------------------------------------------
# Pure logic (no Rhino dependency — unit-testable outside Grasshopper)
# ---------------------------------------------------------------------------

def cell_center_local(row, col, width, depth):
    """Room-local (x, y) centre of floor grid cell (row, col), in the SAME units as
    width/depth.

    Grid convention, identical to the surrogate's plan-view floor and to the interactive
    designer's painted grid:  k = row*GRID_N + col,  row 0 = the +Y ("N") edge,
    col 0 = the x=0 ("W") edge.  So y DECREASES as row increases.
    """
    x = (col + 0.5) * width / GRID_N
    y = (GRID_N - 1 - row + 0.5) * depth / GRID_N
    return x, y


def wall_azimuths(north_angle):
    """{side: true compass azimuth its OUTWARD face points at}.  Mirrors
    optimize_facade_allsides_climate.wall_azimuths — keep the two in step."""
    base = -float(north_angle)
    return {"N": base % 360.0, "E": (base + 90.0) % 360.0,
            "S": (base + 180.0) % 360.0, "W": (base + 270.0) % 360.0}


def compass_name(az):
    return ["North", "NE", "East", "SE", "South", "SW", "West", "NW"][
        int((az % 360.0) / 45.0 + 0.5) % 8]


def validate_sides(sides):
    s = str(sides or "").upper().strip()
    if not s:
        raise JobInputError("`sides` is empty — at least one wall must hold windows, e.g. 'SNEW'")
    bad = sorted(set(s) - set("SNEW"))
    if bad:
        raise JobInputError(
            "`sides` contains invalid wall id(s) %s — allowed are S, N, E, W. "
            "A typo here does not error downstream, it silently freezes a wall solid."
            % ", ".join(repr(b) for b in bad))
    return "".join(ch for ch in "SNEW" if ch in s)


def validate_room(w, d, h):
    notes = []
    for name, val, lo, hi in (("width", w, W_MIN, W_MAX),
                              ("depth", d, D_MIN, D_MAX),
                              ("height", h, H_MIN, H_MAX)):
        if val <= 0:
            raise JobInputError("room %s is %.3f m — the box has no extent on that axis" % (name, val))
        if not (lo - 1e-9 <= val <= hi + 1e-9):
            raise JobInputError(
                "room %s = %.2f m is outside the model's trained range %.1f-%.1f m. "
                "The optimizer CLAMPS instead of failing, so it would return a confident "
                "design for a %.1f m %s. Resize the box (or check document units)."
                % (name, val, lo, hi, min(max(val, lo), hi), name))
    return notes


def _zone_field(z, name, zi):
    """Read one field off a Zone — by attribute, falling back to dict access.

    DUCK-TYPED ON PURPOSE, never isinstance().  gh_make_zone.py is loaded through
    importlib.spec_from_file_location, which does NOT consult sys.modules, so that
    component holds a Zone class that is a DIFFERENT class object from anything this file
    could import — isinstance() would be False for every real zone.  Duck typing also lets
    a hand-built {"curves": [...], "hour": 9, "lux": 400} work unchanged.
    """
    v = getattr(z, name, None)
    if v is None and isinstance(z, dict):
        v = z.get(name)
    if v is None:
        raise JobInputError(
            "zones[%d] has no `%s` — expected a Zone from the gh_make_zone component "
            "(or a dict with curves/hour/lux). Got %r. Check the input's type hint is "
            "'No type hint': a typed input coerces the object and destroys it."
            % (zi, name, type(z).__name__))
    return v


def expand_zones(zones):
    """Zones -> the parallel (curves, hours, luxes) arrays the raster already expects.

    Everything downstream of this function is unchanged by the Zone refactor: rasterize()
    and build_slots() still see flat per-curve lists.  `owner` maps each curve back to its
    zone so errors can name it the way the canvas does.
    """
    zs = [z for z in (zones or []) if z is not None]
    if not zs:
        raise JobInputError("no `zones` — wire at least one Zone from gh_make_zone "
                            "(input Access must be List, type hint 'No type hint')")

    curves, hours, luxes, owner = [], [], [], []
    for zi, z in enumerate(zs):
        # Grasshopper may box an object crossing a wire; unwrap before reading.
        z = getattr(z, "Value", z)
        cs = _zone_field(z, "curves", zi)
        h  = int(_zone_field(z, "hour", zi))
        lx = float(_zone_field(z, "lux", zi))
        if not (0 <= h <= 23):
            raise JobInputError("zones[%d] hour %d is outside 0-23" % (zi, h))
        if lx <= 0:
            raise JobInputError("zones[%d] lux %g must be > 0 (0 means 'unoccupied', which "
                                "is not scored at all)" % (zi, lx))
        cs = [c for c in cs if c is not None]
        if not cs:
            raise JobInputError("zones[%d] (%02d:00 @ %g lux) has no curves" % (zi, h, lx))
        for c in cs:
            curves.append(c); hours.append(h); luxes.append(lx); owner.append(zi)
    return curves, hours, luxes, owner


def _curve_ref(owner, hours, luxes, i):
    """Name a curve the way the canvas does — by its zone, not by a flat index."""
    zi = owner[i]
    n_before = sum(1 for j in range(i) if owner[j] == zi)
    return "zones[%d] (%02d:00 @ %g lux) curve %d" % (zi, hours[i], luxes[i], n_before)


def build_slots(hours, luxes, masks):
    """Group per-curve masks into one slot per distinct hour.  Overlapping curves resolve
    by MAX lux, so nesting a bright zone inside a dim one expresses both levels."""
    by_hour = {}
    for hour, lux, mask in zip(hours, luxes, masks):
        grid = by_hour.setdefault(int(hour), [0.0] * N_GRID_VALS)
        for k in mask:
            if lux > grid[k]:
                grid[k] = float(lux)
    return [{"hour": h, "grid": by_hour[h]} for h in sorted(by_hour)]


def validate_city(city):
    """The city is carried through as a NAME and resolved on the Python side.

    Deliberately not validated against the library here: gh_pick_city.py owns all
    climate-library logic, and duplicating discovery in this component is what made the
    two inputs (`library_dir` + `city_index`) necessary in the first place.  A name that
    does not resolve fails loudly in resolve_city(), which raises listing the candidates.
    """
    c = str(city or "").strip()
    if not c:
        raise JobInputError("`city` is empty — wire it from the gh_pick_city component "
                            "(or type a name/substring such as 'Houston')")
    return c


# ---------------------------------------------------------------------------
# Rhino-dependent
# ---------------------------------------------------------------------------

def unit_scale_to_metres():
    """Scale factor from the active document's units to metres.

    THE quiet killer: a box drawn in a millimetre document reports width 6000, which the
    optimizer clamps to 10 m and then designs for — plausibly, and completely wrong.
    """
    import Rhino
    doc = Rhino.RhinoDoc.ActiveDoc
    if doc is None:
        return 1.0
    return Rhino.RhinoMath.UnitScale(doc.ModelUnitSystem, Rhino.UnitSystem.Meters)


def room_from_box(box, north):
    """(width_m, depth_m, height_m, base_plane, north_angle, placement) from a GH Box.

    A Box carries its own Plane and X/Y/Z intervals, so width-vs-depth is unambiguous. A
    Brep would need an oriented bounding box whose axis order is arbitrary, which would
    silently swap which walls are S/N vs E/W — so a Box is required, not merely preferred.

    north_angle follows Ladybug: the CLOCKWISE angle from the room's +Y to true North.
    With `north` = that same angle for world +Y, and phi = the COUNTER-clockwise rotation
    of the box's +Y from world +Y, north_angle = north + phi.  Check: north=0 and a box
    turned 90 deg CCW points its +Y West and yields 90 — Ladybug's "90 is West".
    """
    import Rhino.Geometry as rg
    if box is None:
        raise JobInputError("`room_box` is empty — draw a Box for the room interior")
    if not isinstance(box, rg.Box):
        raise JobInputError(
            "`room_box` must be a Box (got %s). Set the input's type hint to Box; a Brep "
            "would need an oriented bounding box whose axis order is arbitrary, which can "
            "silently swap the S/N and E/W walls." % type(box).__name__)
    if not box.IsValid:
        raise JobInputError("`room_box` is not a valid Box")

    s = unit_scale_to_metres()
    pl = box.Plane
    origin = pl.PointAt(box.X.Min, box.Y.Min, box.Z.Min)
    base   = rg.Plane(origin, pl.XAxis, pl.YAxis)

    py  = rg.Vector3d(pl.YAxis.X, pl.YAxis.Y, 0.0)
    if py.Length < 1e-9:
        raise JobInputError("`room_box` is tipped on its side — its Y axis is vertical, so "
                            "there is no floor plan to rasterize")
    py.Unitize()
    phi = math.degrees(math.atan2(-py.X, py.Y))     # CCW angle world +Y -> box +Y
    north_angle = (float(north or 0.0) + phi) % 360.0

    placement = {"origin": [origin.X, origin.Y, origin.Z],
                 "x_axis": [pl.XAxis.X, pl.XAxis.Y, pl.XAxis.Z],
                 "y_axis": [pl.YAxis.X, pl.YAxis.Y, pl.YAxis.Z]}
    return (box.X.Length * s, box.Y.Length * s, box.Z.Length * s,
            base, box.X.Length, box.Y.Length, north_angle, placement)


def coerce_curve(obj, label):
    """Whatever Grasshopper handed us -> a real Rhino.Geometry.Curve.

    Zones travel between components as untyped objects, so their curves arrive in whatever
    form the upstream input produced.  Three cases are common and none of them is a Curve:

      * a Guid — a REFERENCED Rhino document object, which is what you get unless the
        gh_make_zone `curves` input is type-hinted "Curve".  This is the usual cause of
        "'Guid' object has no attribute 'IsClosed'".
      * a Rectangle3d / Polyline / Circle / Arc / Line — GH value types, not Curves.  The
        Rectangle component emits Rectangle3d, which is exactly what you draw zones with.
      * a GH object wrapper around any of the above.

    Coercing here rather than demanding the right type hint keeps the component working
    however it is wired — and the wiring is untyped by necessity, since a type hint on
    `zones` would destroy the Zone objects themselves.
    """
    import Rhino
    import Rhino.Geometry as rg

    obj = getattr(obj, "Value", obj)             # unwrap a GH object wrapper
    if isinstance(obj, rg.Curve):
        return obj

    if type(obj).__name__ == "Guid":
        doc = Rhino.RhinoDoc.ActiveDoc
        ro  = doc.Objects.FindId(obj) if doc is not None else None
        if ro is None:
            raise JobInputError("%s is a Guid that is not in the active Rhino document — "
                                "was the object deleted?" % label)
        obj = ro.Geometry
        if isinstance(obj, rg.Curve):
            return obj

    # Rectangle3d, Polyline, Circle, Arc, Ellipse, Line: value types that convert.
    to_nurbs = getattr(obj, "ToNurbsCurve", None)
    if callable(to_nurbs):
        c = to_nurbs()
        if isinstance(c, rg.Curve):
            return c

    raise JobInputError(
        "%s is a %s, not a curve. Set the gh_make_zone `curves` input type hint to Curve "
        "(right-click the input -> Type hint -> Curve)." % (label, type(obj).__name__))


def zero_cell_reason(label, flat, base, a_crv, w_doc, d_doc):
    """Why did this curve capture no cells?  There are TWO causes and they need
    different fixes, so the message has to tell them apart.

    A curve big enough to cover many cells that still captures none is not thin — it is
    somewhere else.  Blaming thinness there sends you off resizing a zone that was never
    the problem, when the real fault is almost always a zone wired to the WRONG ROOM: with
    seven gh_make_room components on the canvas, a mis-ordered Merge pairs room A's curves
    with room B's box, and nothing else complains.
    """
    import Rhino.Geometry as rg

    cw, cd = w_doc / GRID_N, d_doc / GRID_N
    could_cover = a_crv / (cw * cd) if cw * cd > 0 else 0.0

    # The curve's extent in the ROOM's own frame: the room is x 0..w_doc, y 0..d_doc.
    try:
        bb = rg.Box(base, flat)
        x0, x1 = bb.X.Min, bb.X.Max
        y0, y1 = bb.Y.Min, bb.Y.Max
    except Exception:                                  # noqa: BLE001 — diagnosis only
        x0 = x1 = y0 = y1 = None

    head = ("%s captured ZERO grid cells (area %.4g doc-units^2, and the grid is %dx%d "
            "with cells of %.3g x %.3g).\n" % (label, a_crv, GRID_N, GRID_N, cw, cd))

    if x0 is not None:
        outside = (x1 < 0.0 or x0 > w_doc or y1 < 0.0 or y0 > d_doc)
        where = ("  curve spans x %.3f..%.3f, y %.3f..%.3f in the room's frame\n"
                 "  the room is  x %.3f..%.3f, y %.3f..%.3f\n"
                 % (x0, x1, y0, y1, 0.0, w_doc, 0.0, d_doc))
        if outside:
            return (head + where +
                    "It lies OUTSIDE this room entirely, so no sample point can fall in it.\n"
                    "This is almost always a zone wired to the wrong room — check that the\n"
                    "Merge feeding `rooms` and the curves feeding each gh_make_room are in\n"
                    "the same order, and that this room's box is the one you think it is.")
        if could_cover >= 1.0:
            return (head + where +
                    "It overlaps the room but still caught no cell CENTRE. At %.0f cells' "
                    "worth of\narea it is not too small, so it is probably a thin sliver, or "
                    "it sits in the\ngap between centre points. Nudge it onto a centre, or "
                    "fatten it slightly."
                    % could_cover)
        return (head + where +
                "It is smaller than one cell (%.2f cells' worth of area), so it can fall "
                "between\nevery sample point. Enlarge it or move it onto a cell centre."
                % could_cover)

    # No usable bounding box — fall back to the area-only reading.
    if could_cover >= 1.0:
        return (head +
                "At %.0f cells' worth of area it is NOT too small, so it is very likely over "
                "a\ndifferent room — check the zone-to-room wiring before resizing anything."
                % could_cover)
    return (head + "A zone thinner than one cell can fall between every sample point. "
                   "Enlarge it or move it onto a cell centre.")


def rasterize(curves, base, w_doc, d_doc, tol, ref=None):
    """Centre-sample each curve onto the 32x32 grid.

    Returns (masks, points, notes).  `masks[i]` is the set of flat cell indices captured by
    curve i.  Curves are projected onto the room's base plane first, so zone outlines may
    be drawn at any height rather than having to sit exactly on the floor.

    `ref(i) -> str` names curve i for error messages.  The writer passes one that reports
    the owning ZONE ("zones[2] (09:00 @ 400 lux) curve 1"), because a flat index into an
    expanded list means nothing to someone looking at the canvas.
    """
    if ref is None:
        ref = lambda i: "curves[%d]" % i        # noqa: E731 — trivial default
    import Rhino.Geometry as rg

    pts = []
    for row in range(GRID_N):
        for col in range(GRID_N):
            x, y = cell_center_local(row, col, w_doc, d_doc)
            pts.append(base.PointAt(x, y, 0.0))

    masks, notes = [], []
    for i, crv in enumerate(curves):
        if crv is None:
            raise JobInputError("%s is null" % ref(i))
        crv = coerce_curve(crv, ref(i))
        if not crv.IsClosed:
            raise JobInputError("%s is not closed — occupancy zones must be closed "
                                "outlines" % ref(i))
        if not crv.IsPlanar(tol):
            raise JobInputError("%s is not planar" % ref(i))

        flat = crv.DuplicateCurve()
        flat.Transform(rg.Transform.PlanarProjection(base))

        hit = set()
        for k, p in enumerate(pts):
            if flat.Contains(p, base, tol) == rg.PointContainment.Inside:
                hit.add(k)

        area  = rg.AreaMassProperties.Compute(flat)
        a_crv = area.Area if area else 0.0
        a_cel = len(hit) * (w_doc / GRID_N) * (d_doc / GRID_N)
        if not hit:
            raise JobInputError(zero_cell_reason(
                ref(i), flat, base, a_crv, w_doc, d_doc))
        ratio = (a_cel / a_crv) if a_crv > 0 else float("nan")
        notes.append("%s: %d cells, rasterized/true area = %.2f" % (ref(i), len(hit), ratio))
        masks.append(hit)

    return masks, pts, notes


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_job(room_box, north, sides, city, zones,
              repr_conditions, seed, tol,
              glare_weight=None, glare_cap=None,
              out_name=None, result_dir=None):
    sides_s = validate_sides(sides)

    (w_m, d_m, h_m, base, w_doc, d_doc, north_angle, placement) = room_from_box(room_box, north)
    notes = validate_room(w_m, d_m, h_m)

    city_s = validate_city(city)

    curves, hours, luxes, owner = expand_zones(zones)
    masks, pts, rnotes = rasterize(curves, base, w_doc, d_doc, tol,
                                   ref=lambda i: _curve_ref(owner, hours, luxes, i))
    notes.extend(rnotes)
    slots = build_slots(hours, luxes, masks)

    # Omitting glare_weight/glare_cap leaves them to run_facade_job.py's job defaults
    # (0.25 / 3000), so wiring nothing changes nothing.
    optimizer = {"repr_conditions": int(repr_conditions or 48),
                 "mode": "coarse_to_fine", "seed": int(seed or 0)}
    if glare_weight is not None:
        gw = float(glare_weight)
        if gw < 0:
            raise JobInputError("`glare_weight` must be >= 0 (0 disables the penalty)")
        optimizer["glare_weight"] = gw
    if glare_cap is not None:
        gc = float(glare_cap)
        if gc <= 0:
            raise JobInputError("`glare_cap` must be > 0 lux")
        optimizer["glare_cap"] = gc

    job = {
        "schema": "facade_job/1",
        "units": "m",
        "room": {"width": round(w_m, 6), "depth": round(d_m, 6),
                 "height": round(h_m, 6), "north_angle": round(north_angle, 6)},
        "sides": sides_s,
        "climate": {"city": city_s},
        "slots": slots,
        "optimizer": optimizer,
        "placement": placement,
        # `name` is pinned rather than left null so the writer's predicted output paths and
        # the runner's actual choice read the SAME value out of the job file, instead of
        # independently reproducing the runner's "fall back to the json stem" default.
        "output": {"dir": (str(result_dir).strip().strip('"') if result_dir else None),
                   "name": out_name or None},
    }

    merged = [0.0] * N_GRID_VALS
    for s in slots:
        for k, v in enumerate(s["grid"]):
            if v > merged[k]:
                merged[k] = v
    return job, pts, merged, city_s, north_angle, notes


def resolve_repo(repo_root=None):
    """Locate the project folder, or None.

    THE .gh FILE'S LOCATION IS IRRELEVANT, and so is Rhino's working directory.  Order:

        1. the `repo_root` input, if wired
        2. the PROJECT_ROOT constant at the top of this file   <- edit that one
        3. walking up from __file__ (only exists if this module was LOADED from disk
           rather than pasted into a script component)

    Shared by `run` and by the predicted output paths, deliberately: if those two resolved
    the project independently they could disagree about where results land, and the wire
    into the reader component would point at nothing while the run succeeded.
    """
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


def predict_outputs(out_name, out_dir=None, repo_root=None):
    """Where run_facade_job.py WILL write its results — (result_dir, panels_csv, result_json).

    These paths do not exist until the optimizer has actually run; they are emitted so the
    reader component can be wired up front instead of having the path typed by hand.  The
    reader is responsible for saying "has the optimizer finished?" rather than "bad path".

    Mirrors the runner's own choice: `out.get("dir") or OUT_ROOT / out_name`, with OUT_ROOT
    = <repo>/allside_Generalised model/outputs.  The writer pins `output.name` explicitly in
    the job file so both sides read the same value rather than separately reproducing the
    runner's "fall back to the json stem" default.
    """
    if out_dir:
        d = os.path.abspath(str(out_dir).strip().strip('"'))
    else:
        repo = resolve_repo(repo_root)
        if repo is None:
            return "", "", ""
        d = os.path.join(repo, "allside_Generalised model", "outputs", out_name)
    return (d,
            os.path.join(d, "facade_panels.csv"),
            os.path.join(d, "result.json"))


def launch_optimizer(json_path, repo_root=None, console=True):
    """Shell out to the project venv — the optimizer needs torch and a GPU, so it cannot
    run inside Rhino's interpreter.  Only possible this cleanly because we target CPython 3.

    It must run in the project's .venv, NOT in Rhino's Python — which is why this launches a
    separate process rather than importing anything.  Nothing heavier than the standard
    library ever enters Rhino.
    """
    repo = resolve_repo(repo_root)
    if repo is None:
        return ("run skipped: cannot locate the project. If this script was pasted it has no "
                "__file__ to walk up from, so set the PROJECT_ROOT constant at the top of the "
                "script (currently %r) or wire `repo_root`." % PROJECT_ROOT)

    python = os.path.join(repo, ".venv", "Scripts", "python.exe")
    runner = os.path.join(repo, "allside_Generalised model", "model trainer", "run_facade_job.py")
    if not os.path.isfile(python):
        return "run skipped: no venv python at %s" % python
    if not os.path.isfile(runner):
        return "run skipped: no runner at %s" % runner
    # Rhino's Python exports PYTHONHOME (and can export PYTHONPATH) pointing at its own
    # embedded 3.9 install.  A child process inherits those, so the project's python.exe
    # would boot against RHINO'S standard library instead of its own — which fails as
    # "AssertionError: SRE module mismatch" deep inside `import re`, with a traceback full
    # of .rhinocode paths that looks nothing like an environment problem.
    # Scrub them so the venv resolves its own stdlib from its pyvenv.cfg.
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)

    cmd = [python, runner, json_path]

    # WHY A SEPARATE CONSOLE WINDOW.
    # A Grasshopper component blocks while it runs, and Rhino's UI (including the script
    # editor's output panel) does not repaint until it returns.  So capturing stdout gives
    # you NOTHING for the whole run — several minutes of a frozen canvas — and then the
    # entire log at once.  Printing as lines arrive does not help either: the panel still
    # cannot repaint.  The only way to watch progress live is to give the child its own
    # console, which Windows repaints independently of Rhino.
    if console:
        CREATE_NEW_CONSOLE = 0x00000010
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(runner), env=env,
                                creationflags=CREATE_NEW_CONSOLE)
        proc.wait()
        return ("optimizer ran in its own console window (exit code %d).\n"
                "Set `console` to False to capture the log here instead — but then nothing "
                "is visible until the run finishes." % proc.returncode)

    proc = subprocess.run(cmd, cwd=os.path.dirname(runner),
                          capture_output=True, text=True, env=env)
    return (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`.

    When this file is PASTED into a script component (the normal case), the trailer at the
    bottom calls this with globals() automatically — component inputs ARE module globals
    there, so nothing else is needed.

    Taking the inputs as an argument rather than reading this module's own globals also
    allows an alternative wiring, where a small loader component keeps the file on disk as
    the source of truth so edits are picked up without re-pasting:

        import importlib.util
        spec = importlib.util.spec_from_file_location("ghw", r"...\\gh_write_facade_job.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        r = m.gh_main(globals())
        json_path, grid_pts, grid_vals = r["json_path"], r["grid_pts"], r["grid_vals"]
        ...
    """
    import Rhino

    tol = Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance if Rhino.RhinoDoc.ActiveDoc else 0.001

    path = g.get("out_path")
    path = str(path).strip().strip('"') if path else None
    # The runner names the output folder after the job file, so the job's `output.name` is
    # pinned from the same stem here — that is what keeps the predicted paths honest.
    out_name = os.path.splitext(os.path.basename(path))[0] if path else "facade_job"

    job, pts, vals, city, north_angle, notes = build_job(
        g.get("room_box"), g.get("north"), g.get("sides"), g.get("city"),
        g.get("zones"), g.get("repr_conditions"), g.get("seed"), tol,
        g.get("glare_weight"), g.get("glare_cap"),
        out_name=out_name, result_dir=g.get("result_dir"))

    res_dir, panels_csv, res_json = predict_outputs(
        out_name, g.get("result_dir"), g.get("repo_root"))

    az   = wall_azimuths(north_angle)
    room = job["room"]
    out = {
        "grid_pts": pts,
        "grid_vals": vals,
        "city_resolved": city,
        "room_dims": "%.2f x %.2f x %.2f m  (north_angle=%.1f deg)" % (
            room["width"], room["depth"], room["height"], north_angle),
        "sides_compass": ["%s -> %s (%.0f deg)" % (s, compass_name(az[s]), az[s]) for s in SIDES],
        "json_path": None,
        "result_dir": res_dir,
        "panels_csv": panels_csv,
        "result_json": res_json,
        "log": notes + ["%d slot(s): %s" % (len(job["slots"]),
                                            ", ".join("%02d:00" % s["hour"] for s in job["slots"]))],
    }
    if not res_dir:
        out["log"].append("cannot predict the output paths — set PROJECT_ROOT at the top of "
                          "this script, or wire `repo_root` or `result_dir`")

    if g.get("write") and path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        out["json_path"] = path
        out["log"].append("wrote %s" % path)
        if g.get("run"):
            console = g.get("console")
            out["log"].append(launch_optimizer(
                path, g.get("repo_root"), True if console is None else bool(console)))
    elif not path:
        out["log"].append("set `out_path` to write the job file")
    else:
        out["log"].append("set `write` to True to write the job file")
    return out


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

if _IN_RHINO and "room_box" in globals():
    _r = gh_main(globals())
    json_path     = _r["json_path"]
    grid_pts      = _r["grid_pts"]
    grid_vals     = _r["grid_vals"]
    city_resolved = _r["city_resolved"]
    result_dir    = _r["result_dir"]
    panels_csv    = _r["panels_csv"]
    result_json   = _r["result_json"]
    room_dims     = _r["room_dims"]
    sides_compass = _r["sides_compass"]
    log           = _r["log"]
