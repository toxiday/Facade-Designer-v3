"""
gh_make_zone.py — Grasshopper script component: bundle curves + hour + lux into a Zone.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib only — no RhinoCommon, no numpy.

Feeds `zones` into gh_write_facade_job.py.  Replaces three parallel list inputs
(`zone_curves` / `zone_hour` / `zone_lux`) whose correspondence was positional and
therefore invisible on the canvas: a Merge in the wrong order silently reassigned hours and
lux values to the wrong curves, and nothing errored — you just got a confident design for
the wrong brief.  One object, one wire, correspondence structural.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    curves          Curve   List   closed planar occupancy outlines for THIS zone
    hour            int     Item   0-23, local clock hour
    lux             float   Item   target illuminance for every cell this zone covers

    Set `curves` type hint to Curve.  Without it, geometry REFERENCED from the Rhino
    document arrives as a Guid, and GH's own Rectangle component emits Rectangle3d —
    neither is a Curve.  The writer's coerce_curve() repairs both rather than failing, so
    a wrong hint is survivable, but hinting Curve keeps the object on the wire honest.

COMPONENT OUTPUTS
    zone            the Zone object -> wire into the writer's `zones` input (List access)
    info            readable summary, for a Panel
────────────────────────────────────────────────────────────────────────────────────────

MULTI-LEVEL ZONES
    Make TWO zones at the SAME hour with different lux.  build_slots() in the writer
    merges by hour taking MAX, so nesting a 600 lux zone inside a 200 lux one expresses
    both levels — and now reads as two labelled objects rather than four positionally
    aligned list entries.

WHY THE WRITER DUCK-TYPES THIS INSTEAD OF isinstance()
    Each component loads its .py through importlib.spec_from_file_location +
    module_from_spec, which does NOT consult sys.modules.  So the writer holds its own
    module object with its own distinct Zone class, and isinstance(z, ThatZone) would be
    False for every zone produced here even though the object is structurally perfect.
    The writer therefore reads .hour/.lux/.curves via getattr with a dict fallback.  Keep
    this class a plain attribute container so that stays true.
"""


class ZoneInputError(Exception):
    """The zone inputs are not usable."""


class Zone:
    """One occupancy zone: a set of outlines, all at one hour and one target lux.

    Deliberately a plain attribute container — see the duck-typing note in the module
    docstring.  Do not add behaviour the writer would need to call.
    """

    __slots__ = ("curves", "hour", "lux")

    def __init__(self, curves, hour, lux):
        self.curves = list(curves)
        self.hour = int(hour)
        self.lux = float(lux)

    def __repr__(self):
        # A Panel wired to `zone` renders str(obj), so this is the whole point of the
        # object: the wire becomes self-documenting on the canvas.
        n = len(self.curves)
        return "Zone %02d:00 @ %g lux (%d curve%s)" % (
            self.hour, self.lux, n, "" if n == 1 else "s")


def make_zone(curves, hour, lux):
    """Validate and build.  Raises ZoneInputError with an actionable message.

    Only what can be checked WITHOUT the room is checked here.  The geometry checks
    (closed, planar, captures at least one grid cell) stay in the writer's rasterize(),
    which needs the room box — one home for them, so the two cannot drift apart.
    """
    cs = [c for c in (curves or []) if c is not None]
    if not cs:
        raise ZoneInputError("`curves` is empty — a zone needs at least one closed "
                             "occupancy outline")

    if hour is None:
        raise ZoneInputError("`hour` is required (0-23, local clock hour)")
    try:
        h = int(hour)
    except (TypeError, ValueError):
        raise ZoneInputError("`hour` must be a whole number 0-23, got %r" % (hour,)) from None
    if not (0 <= h <= 23):
        raise ZoneInputError("`hour` %d is outside 0-23" % h)

    if lux is None:
        raise ZoneInputError("`lux` is required (target illuminance for this zone)")
    try:
        lx = float(lux)
    except (TypeError, ValueError):
        raise ZoneInputError("`lux` must be a number, got %r" % (lux,)) from None
    if lx <= 0:
        raise ZoneInputError(
            "`lux` must be > 0, got %g. Zero means 'unoccupied', and unoccupied cells are "
            "not scored by the optimizer at all — so a 0 lux zone is the same as not "
            "drawing it. To constrain circulation space, give it a low target (100-150)."
            % lx)

    return Zone(cs, h, lx)


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`.

    PASTE this file into a Python 3 script component and the trailer at the bottom calls
    this automatically — component inputs ARE module globals there.  You will paste this
    one once per zone; each copy is an independent component with its own hour and lux.

    Alternative loader wiring, which keeps the file on disk as the source of truth:

        import importlib.util
        spec = importlib.util.spec_from_file_location("mkzone", r"...\\gh_make_zone.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        r = m.gh_main(globals())
        zone, info = r["zone"], r["info"]
    """
    z = make_zone(g.get("curves"), g.get("hour"), g.get("lux"))
    return {"zone": z, "info": repr(z)}


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

if _IN_RHINO and "curves" in globals():
    _r = gh_main(globals())
    zone = _r["zone"]
    info = _r["info"]
