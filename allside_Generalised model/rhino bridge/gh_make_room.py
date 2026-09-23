"""
gh_make_room.py — Grasshopper script component: bundle box + name + sides + zones into a Room.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib only — no RhinoCommon, no numpy.

Feeds `rooms` into gh_write_unit_jobs.py, which exports a WHOLE APARTMENT UNIT in one go.
Stands in the same relation to the unit writer as gh_make_zone.py does to the per-room
writer, and for the same reason: with seven rooms on the canvas, four parallel lists
(`boxes` / `names` / `sides` / `zone_groups`) correspond only by position, and a Merge in
the wrong order silently gives room 101.3 room 101.6's occupancy and nothing errors.  One
object, one wire, correspondence structural.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    room_box        Box     Item   the room INTERIOR (walls have no thickness)
    name            str     Item   e.g. "101.3" — becomes the job filename AND output.name
    sides           str     Item   room-local wall ids that may hold windows, e.g. "SE"
    zones           (none)  List   Zone objects from gh_make_zone.py.
                                   Type hint MUST be "No type hint": a typed input
                                   coerces the object and destroys it.

COMPONENT OUTPUTS
    room            the Room object -> wire into gh_write_unit_jobs `rooms` (List access)
    info            readable summary, for a Panel
────────────────────────────────────────────────────────────────────────────────────────

`sides` IS THE EXTERIOR-WALL DECLARATION
    In a unit, most of a room's walls face other rooms.  A wall left out of `sides` is
    frozen fully solid by the optimizer and can never receive a window — which is exactly
    what an interior party wall should be.  So `sides` is not merely "where windows are
    allowed", it is the statement of which walls are on the building envelope, and
    gh_read_facade.py uses it downstream to skip building interior walls at all.

    These are ROOM-LOCAL wall ids, not compass directions: `S` is the wall at y=0 in the
    box's own frame.  Read `sides_compass` off the unit writer before choosing.

WHY THE WRITER DUCK-TYPES THIS INSTEAD OF isinstance()
    Each component loads its .py through importlib.spec_from_file_location +
    module_from_spec, which does NOT consult sys.modules.  So the writer holds its own
    module object with its own distinct Room class, and isinstance(r, ThatRoom) would be
    False for every room produced here even though the object is structurally perfect.
    The writer therefore reads .room_box/.name/.sides/.zones via getattr with a dict
    fallback.  Keep this class a plain attribute container so that stays true.
"""


class RoomInputError(Exception):
    """The room inputs are not usable."""


class Room:
    """One room of a unit: its box, its name, its exterior walls and its occupancy zones.

    Deliberately a plain attribute container — see the duck-typing note in the module
    docstring.  Do not add behaviour the writer would need to call.
    """

    __slots__ = ("room_box", "name", "sides", "zones")

    def __init__(self, room_box, name, sides, zones):
        self.room_box = room_box
        self.name = str(name)
        self.sides = str(sides)
        self.zones = list(zones)

    def __repr__(self):
        # A Panel wired to `room` renders str(obj), so this is the whole point of the
        # object: the wire becomes self-documenting on the canvas.
        hours = []
        for z in self.zones:
            h = getattr(z, "hour", None)
            if h is None and isinstance(z, dict):
                h = z.get("hour")
            if h is not None:
                hours.append(int(h))
        span = ", ".join("%02d:00" % h for h in sorted(set(hours))) or "no hours"
        return "Room %s | sides %s | %d zone%s | %s" % (
            self.name, self.sides, len(self.zones),
            "" if len(self.zones) == 1 else "s", span)


def validate_name(name):
    """A room name becomes a FILENAME (`<name>.json`) and the optimizer's output folder, so
    reject anything that would not survive the trip to disk."""
    if name is None:
        raise RoomInputError("`name` is required — it becomes the job filename and the "
                             "optimizer's output folder, e.g. \"101.3\"")
    n = str(name).strip()
    if not n:
        raise RoomInputError("`name` is empty")
    bad = [ch for ch in n if ch in '<>:"/\\|?*' or ord(ch) < 32]
    if bad:
        raise RoomInputError(
            "`name` %r contains character(s) %s that cannot be used in a filename — the "
            "name becomes <name>.json and the output folder."
            % (n, ", ".join(repr(b) for b in sorted(set(bad)))))
    if n in (".", ".."):
        raise RoomInputError("`name` %r is not a usable filename" % n)
    return n


def validate_sides(sides):
    """Room-local wall ids.  DUPLICATED from gh_write_facade_job.validate_sides so this
    component stays stdlib-only and independently pasteable; the writer re-validates, so a
    divergence would be caught there rather than producing a wrong design."""
    s = str(sides or "").upper().strip()
    if not s:
        raise RoomInputError(
            "`sides` is empty — at least one wall must be able to hold windows. In a unit "
            "this is the room's EXTERIOR wall list, e.g. \"E\" for a room with one wall on "
            "the envelope, \"SE\" for a corner room.")
    bad = sorted(set(s) - set("SNEW"))
    if bad:
        raise RoomInputError(
            "`sides` contains invalid wall id(s) %s — allowed are S, N, E, W. A typo here "
            "does not error downstream, it silently freezes a wall solid."
            % ", ".join(repr(b) for b in bad))
    return "".join(ch for ch in "SNEW" if ch in s)


def make_room(room_box, name, sides, zones):
    """Validate and build.  Raises RoomInputError with an actionable message.

    Only what can be checked WITHOUT the unit is checked here.  Box geometry, room
    dimension ranges, zone rasterization and — critically — room-vs-room overlap all stay
    in gh_write_unit_jobs, which is the only place that can see the other rooms.
    """
    if room_box is None:
        raise RoomInputError("`room_box` is empty — draw a Box for the room interior")

    n  = validate_name(name)
    sd = validate_sides(sides)

    zs = [z for z in (zones or []) if z is not None]
    if not zs:
        raise RoomInputError(
            "room %r has no `zones` — wire at least one Zone from gh_make_zone (input "
            "Access must be List, type hint 'No type hint'). A room with no occupancy "
            "cannot be optimized: with nothing scored there is no gradient." % n)

    return Room(room_box, n, sd, zs)


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`.

    PASTE this file into a Python 3 script component and the trailer at the bottom calls
    this automatically — component inputs ARE module globals there.  You will paste this one
    once per room; each copy is an independent component with its own box, name and sides.

    Alternative loader wiring, which keeps the file on disk as the source of truth:

        import importlib.util
        spec = importlib.util.spec_from_file_location("mkroom", r"...\\gh_make_room.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        r = m.gh_main(globals())
        room, info = r["room"], r["info"]
    """
    r = make_room(g.get("room_box"), g.get("name"), g.get("sides"), g.get("zones"))
    return {"room": r, "info": repr(r)}


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

if _IN_RHINO and "room_box" in globals():
    _r = gh_main(globals())
    room = _r["room"]
    info = _r["info"]
