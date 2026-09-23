"""
gh_pick_city.py — Grasshopper script component: browse the climate library and pick a city.

Rhino 8, CPython 3 (`#! python 3`).  Stdlib only — no RhinoCommon, no numpy.

Feeds `city` into gh_write_facade_job.py.  Split out from the writer so the choice is
INSPECTABLE BEFORE YOU COMMIT: with a bare index you cannot see which city 137 is, nor
whether the sun is even up at the hours you plan to design for, until you export a job and
run the optimizer.  This turns that into a glance.

────────────────────────────────────────────────────────────────────────────────────────
COMPONENT INPUTS      (right-click each → Type hint / Access as noted)
    library_dir     str   Item   ...\\allside_Generalised model\\climate data\\library
    search          str   Item   case-insensitive substring filter (blank = all)
    city_index      int   Item   index into the FILTERED list

COMPONENT OUTPUTS
    city            resolved city name -> wire into the writer's `city` input
    matches         labels surviving `search`, for a Panel
    n_matches       how many — so the slider range is obvious
    city_list       every label in the library
    location        "lat 29.98, lon -95.36, tz UTC-6.0"
    sun_hours       per-hour day counts — which hours are usable, see below
    wea_path        full path to the resolved .wea
────────────────────────────────────────────────────────────────────────────────────────

WHY `sun_hours` IS HERE
    run_facade_job.py raises on a slot whose sun never clears the horizon, and the
    interactive designer silently skips it.  Either way you find out late.  This reports,
    for every hour 0-23, how many days of the year the sun is above MIN_ALT (5 deg) — the
    exact filter annual_conditions() applies — so choosing `zone_hour` becomes a decision
    made with the data in front of you.  It is also where a high-latitude winter afternoon
    reveals itself as nearly unusable.

DUPLICATED SOLAR CODE — DELIBERATE
    solar_position() and the .wea header parse below are copies.  They cannot be imported:
    check_sun_coverage.py is the natural source but imports numpy at module scope, and
    Rhino's CPython has no numpy unless the user installs it — which the whole Rhino side
    is built to avoid.  This is the project's FOURTH copy.  The source of truth is
    generate_dataset_allsides_climate_pit.py:solar_position; fix bugs in all of them.
"""

import os
import glob
import math

MIN_ALT = 5.0        # deg — must match optimize_facade_allsides_climate.MIN_ALT


class CityPickError(Exception):
    """The library or the selection cannot be resolved."""


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def discover_cities(library_dir):
    """Sorted .wea paths.  MUST match optimize_facade_allsides_climate.discover_cities()
    ordering, so an index seen here means the same thing on the Python side."""
    if not library_dir:
        raise CityPickError("`library_dir` is empty — point it at "
                            "'allside_Generalised model/climate data/library'")
    library_dir = str(library_dir).strip().strip('"')
    if not os.path.isdir(library_dir):
        raise CityPickError("`library_dir` is not a directory: %r" % library_dir)
    files = sorted(glob.glob(os.path.join(library_dir, "**", "*.wea"), recursive=True))
    if not files:
        raise CityPickError("no .wea files under %r" % library_dir)
    return files


def city_label(path):
    """Mirrors optimize_facade_allsides_climate.city_label() — the name written into the
    job file, and what resolve_city() matches against."""
    return os.path.splitext(os.path.basename(path))[0].split("_TMYx")[0]


def filter_cities(files, search):
    """Case-insensitive substring filter on the label.  Blank search = everything."""
    s = (search or "").strip().lower()
    if not s:
        return list(files)
    hits = [f for f in files if s in city_label(f).lower()]
    if not hits:
        raise CityPickError("no city matches %r — clear `search` to see all %d, or check "
                            "the spelling" % (search, len(files)))
    return hits


def select(files, search, city_index):
    hits = filter_cities(files, search)
    i = int(city_index or 0)
    if not (0 <= i < len(hits)):
        raise CityPickError("`city_index` %d is out of range for %d match(es)%s — set the "
                            "slider max to %d" % (i, len(hits),
                                                  (" for search %r" % search) if search else "",
                                                  len(hits) - 1))
    return hits[i], hits


# ---------------------------------------------------------------------------
# .wea parsing + solar position   (see DUPLICATED SOLAR CODE note above)
# ---------------------------------------------------------------------------

def parse_wea(path):
    """(latitude, longitude_east, tz_hours, rows) with rows = [(month, day, hour, dni, dhi)].

    The .wea stores longitude and time_zone WEST-POSITIVE; both are converted here, exactly
    as annual_conditions() does (lon_east = -lon_w, tz_h = -tz_deg/15).
    """
    lat = lon_w = tz_deg = None
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                if   p[0] == "latitude":  lat    = float(p[1])
                elif p[0] == "longitude": lon_w  = float(p[1])
                elif p[0] == "time_zone": tz_deg = float(p[1])
            elif len(p) == 5:
                rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
    if lat is None or lon_w is None or tz_deg is None:
        raise CityPickError("%s is missing latitude/longitude/time_zone in its header"
                            % os.path.basename(path))
    return lat, -lon_w, -tz_deg / 15.0, rows


def solar_altitude(lat_deg, lon_deg, tz_h, month, day, hour_decimal):
    """Solar altitude in degrees.  Altitude-only slice of the shared solar_position()."""
    lat = math.radians(lat_deg)
    doy = int(275 * month / 9 - 30 + day - 2)
    if month <= 2:
        doy += 2
    B    = math.radians(360 / 365 * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    ET   = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    st   = hour_decimal + (lon_deg - tz_h * 15) / 15 + ET / 60
    om   = math.radians(15 * (st - 12))
    sa   = max(-1.0, min(1.0, math.sin(lat) * math.sin(decl) +
                              math.cos(lat) * math.cos(decl) * math.cos(om)))
    return math.degrees(math.asin(sa))


def hour_day_counts(path):
    """{hour: days of the year with sun above MIN_ALT}, for hours 0-23.

    Replicates annual_conditions()'s filter exactly — `int(hour) == hour_slot` and
    `alt > MIN_ALT` — so these counts are the same numbers the optimizer will act on.
    """
    lat, lon, tz, rows = parse_wea(path)
    counts = dict((h, 0) for h in range(24))
    for month, day, hour, _dni, _dhi in rows:
        slot = int(hour)
        if 0 <= slot <= 23 and solar_altitude(lat, lon, tz, month, day, hour) > MIN_ALT:
            counts[slot] += 1
    return counts, lat, lon, tz


def format_sun_hours(counts):
    """One readable line per hour that has any sun, plus an explicit dead-hours summary."""
    live = [h for h in range(24) if counts[h] > 0]
    if not live:
        return ["no hour of the year has sun above %.0f deg here" % MIN_ALT]
    out = ["%02d:00  %3d days" % (h, counts[h]) for h in live]
    dead = [h for h in range(24) if counts[h] == 0]
    if dead:
        out.append("no sun at: " + ", ".join("%02d" % h for h in dead))
    return out


# ---------------------------------------------------------------------------
# Grasshopper glue
# ---------------------------------------------------------------------------

def gh_main(g):
    """Run the component. `g` is the mapping of component inputs — pass `globals()`.

    PASTE this file into a Python 3 script component and the trailer at the bottom calls
    this automatically — component inputs ARE module globals there.  This component needs
    no project path at all: `library_dir` is wired directly.

    Alternative loader wiring, which keeps the file on disk as the source of truth:

        import importlib.util
        spec = importlib.util.spec_from_file_location("pick", r"...\\gh_pick_city.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        r = m.gh_main(globals())
        city, matches, n_matches = r["city"], r["matches"], r["n_matches"]
        city_list, location, sun_hours, wea_path = (
            r["city_list"], r["location"], r["sun_hours"], r["wea_path"])
    """
    files = discover_cities(g.get("library_dir"))
    path, hits = select(files, g.get("search"), g.get("city_index"))
    counts, lat, lon, tz = hour_day_counts(path)

    return {
        "city":      city_label(path),
        "matches":   [city_label(f) for f in hits],
        "n_matches": len(hits),
        "city_list": [city_label(f) for f in files],
        "location":  "lat %.2f, lon %.2f, tz UTC%+.1f" % (lat, lon, tz),
        "sun_hours": format_sun_hours(counts),
        "wea_path":  path,
    }


try:
    import Rhino  # noqa: F401
    _IN_RHINO = True
except ImportError:
    _IN_RHINO = False

if _IN_RHINO and "library_dir" in globals():
    _r = gh_main(globals())
    city      = _r["city"]
    matches   = _r["matches"]
    n_matches = _r["n_matches"]
    city_list = _r["city_list"]
    location  = _r["location"]
    sun_hours = _r["sun_hours"]
    wea_path  = _r["wea_path"]
