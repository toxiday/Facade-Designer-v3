"""
download_climate_library.py
Build a library of TMYx weather files for the cities listed in "library planner.csv"
by downloading from climate.onebuilding.org.

For each city the script:
  1. resolves the onebuilding country folder (globally — onebuilding's region
     grouping does not always match the planner's WMO column),
  2. fetches that country's index (one page lists every zip, with sub-state paths),
  3. picks the best station match (prefer TMYx, international airport, latest year
     range),
  4. downloads the .zip and extracts the .wea (and .epw) into ./library/,
  5. records the result in download_manifest.csv.

Re-runnable: HTML directory listings are cached under a temp folder, and cities
whose .wea is already present are skipped.

No third-party dependencies (urllib + zipfile from the stdlib).
"""

import csv, os, re, ssl, sys, time, zipfile, hashlib, urllib.request, urllib.parse
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
PLANNER_CSV = HERE / "library planner.csv"
LIBRARY_DIR = HERE / "library"
MANIFEST    = HERE / "download_manifest.csv"
CACHE_DIR   = Path(os.environ.get("TEMP", "/tmp")) / "onebuilding_cache"

BASE = "https://climate.onebuilding.org/"
REGION_FOLDERS = [
    "WMO_Region_1_Africa",
    "WMO_Region_2_Asia",
    "WMO_Region_3_South_America",
    "WMO_Region_4_North_and_Central_America",
    "WMO_Region_5_Southwest_Pacific",
    "WMO_Region_6_Europe",
    "WMO_Region_7_Antarctica",
]

# Country folders carrying alternative / projected datasets — never preferred.
VARIANT_TOKENS = ("future", "rmy", "niwa", "bbsr", "meteireann", "brazfuture",
                  "climate_zones")

# Planner country string -> onebuilding country folder, for names that don't
# resolve by ISO3 code or by a normalized-name match.
COUNTRY_ALIASES = {
    "russia": "RUS_Russian_Federation",
}

# Planner city -> search token, for stations stored under a localized name.
# (Seeded with the obvious ones; extend as the unmatched report reveals more.)
CITY_ALIASES = {
    # localized station names (verified against onebuilding listings)
    "Rome": "Roma",
    "Athens": "Athina",
    "Lisbon": "Lisboa",
    "Vienna": "Wien",
    "Warsaw": "Warszawa",
    # "City"-suffixed names — match on the bare city token
    "Mexico City": "Mexico.City",
    "Panama City": "Panama",
    "Guatemala City": "Guatemala",
    # Ambiguous US city names shared across states — pin via the state code so the
    # right station is chosen (USA_<ST>_<City>...). Country resolution already limits
    # to the USA folder; these just disambiguate within it.
    "Portland": "OR_Portland",       # Oregon, not Portland ME
    "Richmond": "VA_Richmond",       # Virginia
    "Columbus": "OH_Columbus",       # Ohio
    "Madison": "WI_Madison",         # Wisconsin
    "Riverside": "CA_Riverside",     # California
    "Gent": "Ghent",                 # onebuilding stores Gent (BE) under the English "Ghent"
    "Prague": "Praha",               # Czech stations use the local name Praha
    # Antarctic stations stored without the "Station" suffix
    "Amundsen-Scott": "Amundsen",
    "McMurdo Station": "McMurdo",
    "Palmer Station": "Palmer",
    "Vostok Station": "Vostok",
}

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _cache_path(url):
    h = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def fetch(url, use_cache=True, retries=3):
    cp = _cache_path(url)
    if use_cache and cp.exists():
        return cp.read_text("utf-8", "replace")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=90) as r:
                html = r.read().decode("utf-8", "replace")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(html, "utf-8")
            return html
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url} :: {last}")


def download(url, dest, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            return len(data)
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} :: {last}")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def links(html, pattern):
    return re.findall(pattern, html)


# ── COUNTRY RESOLUTION ────────────────────────────────────────────────────────

def build_country_index():
    """Return {country_folder: [region_folder, ...]} — a country may live in
    several WMO regions (e.g. ESP_Spain in both Africa and Europe)."""
    index = {}
    for reg in REGION_FOLDERS:
        html = fetch(BASE + reg + "/")
        for href in links(html, r'href="([^"]+/index\.html)"'):
            if href.startswith(".."):
                continue
            folder = href.split("/")[0]
            index.setdefault(folder, [])
            if reg not in index[folder]:
                index[folder].append(reg)
    return index


def resolve_country(country, region_label, country_index):
    """Resolve a planner (country, region) to a country_folder (or None)."""
    if region_label.startswith("Region 7"):           # Antarctica special case
        if "ATA_Antarctica" in country_index:
            return "ATA_Antarctica"

    nc = norm(country)
    if nc in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[nc]

    def is_variant(folder):
        return any(t in norm(folder) for t in VARIANT_TOKENS)

    candidates = []
    for folder in country_index:
        iso3, _, name = folder.partition("_")
        if country.strip().upper() == iso3:           # planner gave an ISO3 (e.g. USA)
            candidates.append((folder, 0))
        elif norm(name) == nc:                         # exact normalized-name match
            candidates.append((folder, 1))
        elif nc and (nc in norm(name) or norm(name) in nc):
            candidates.append((folder, 2))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[1], is_variant(c[0]), len(c[0])))
    return candidates[0][0]


def country_zip_urls(country_folder, regions):
    """Absolute URLs of every zip for a country, unioned across its regions."""
    urls = []
    for reg in regions:
        base = f"{BASE}{reg}/{country_folder}/"
        for h in links(fetch(base), r'href="([^"]+\.zip)"'):
            urls.append(urllib.parse.urljoin(base, h))
    return urls


# ── STATION SELECTION ─────────────────────────────────────────────────────────

def years_in(name):
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", name)]
    return max(yrs) if yrs else 0


def pick_zip(zip_hrefs, city):
    """Choose the best zip href for a city; returns (href, matched_token) or (None, None)."""
    token = CITY_ALIASES.get(city, city)
    nt = norm(token)

    def matches(href, key):
        return key in norm(os.path.basename(href))

    cands = [h for h in zip_hrefs if matches(h, nt)]
    if not cands:                                      # fallback: first significant word
        first = re.split(r"[ \-]", token)[0]
        if len(first) >= 4:
            nf = norm(first)
            cands = [h for h in zip_hrefs if matches(h, nf)]
    if not cands:
        return None, None

    def score(href):
        b = os.path.basename(href).lower()
        is_tmyx = "tmyx" in b
        intl = any(k in b for k in ("intl", "intercontinental", "international"))
        ap = ".ap." in b or b.endswith(".ap.zip") or "_ap" in b
        # prefer: TMYx > international > airport > most recent > shorter name
        return (is_tmyx, intl, ap, years_in(b), -len(b))

    cands.sort(key=score, reverse=True)
    return cands[0], token


# ── EXTRACTION ────────────────────────────────────────────────────────────────

def extract_wea(zip_path, dest_dir):
    """Extract .wea and .epw members; return list of extracted .wea filenames."""
    weas = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            low = member.lower()
            if low.endswith(".wea") or low.endswith(".epw"):
                target = dest_dir / os.path.basename(member)
                with z.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
                if low.endswith(".wea"):
                    weas.append(target.name)
    return weas


# ── MAIN ──────────────────────────────────────────────────────────────────────

def read_planner():
    rows = []
    with open(PLANNER_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("City/Station"):
                rows.append(row)
    return rows


def main():
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    cities = read_planner()
    print(f"Planner cities : {len(cities)}")
    print("Building country index from 7 region listings...")
    country_index = build_country_index()
    print(f"Countries found: {len(country_index)}\n")

    zips_cache = {}        # country_folder -> list of zip hrefs
    manifest = []
    ok = fail = skip = 0

    for i, row in enumerate(cities, 1):
        region = row["WMO Region"].strip()
        city   = row["City/Station"].strip()
        country = row["Country/Location Type"].strip()
        tag = f"[{i:3}/{len(cities)}] {city}, {country}"

        country_folder = resolve_country(country, region, country_index)
        if not country_folder:
            print(f"{tag:<48} NO COUNTRY FOLDER")
            manifest.append({"City": city, "Country": country, "Region": region,
                             "Status": "no_country", "File": "", "URL": ""})
            fail += 1
            continue

        if country_folder not in zips_cache:
            zips_cache[country_folder] = country_zip_urls(
                country_folder, country_index[country_folder])
        zip_urls = zips_cache[country_folder]

        zip_url, _ = pick_zip(zip_urls, city)
        if not zip_url:
            print(f"{tag:<48} NO MATCH in {country_folder}")
            sample = [os.path.basename(h) for h in zip_urls[:14]]
            print("        candidates:", ", ".join(sample))
            manifest.append({"City": city, "Country": country, "Region": region,
                             "Status": "no_match", "File": "", "URL": country_folder})
            fail += 1
            continue

        zip_name = os.path.basename(zip_url)
        stem     = zip_name[:-4]
        if (LIBRARY_DIR / f"{stem}.wea").exists():
            print(f"{tag:<48} skip (have {stem}.wea)")
            manifest.append({"City": city, "Country": country, "Region": region,
                             "Status": "exists", "File": f"{stem}.wea", "URL": zip_url})
            skip += 1
            continue

        try:
            tmp_zip = CACHE_DIR / zip_name
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            nbytes = download(zip_url, tmp_zip)
            weas = extract_wea(tmp_zip, LIBRARY_DIR)
            tmp_zip.unlink(missing_ok=True)
            if not weas:
                raise RuntimeError("no .wea inside zip")
            print(f"{tag:<48} OK  {weas[0]}  ({nbytes//1024} KB)")
            manifest.append({"City": city, "Country": country, "Region": region,
                             "Status": "ok", "File": weas[0], "URL": zip_url})
            ok += 1
        except Exception as e:
            print(f"{tag:<48} FAIL {e}")
            manifest.append({"City": city, "Country": country, "Region": region,
                             "Status": f"error:{e}", "File": "", "URL": zip_url})
            fail += 1

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["City", "Country", "Region", "Status", "File", "URL"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nDone: {ok} downloaded, {skip} already present, {fail} unresolved.")
    print(f"Library : {LIBRARY_DIR}")
    print(f"Manifest: {MANIFEST}")
    if fail:
        print("\nUnresolved cities:")
        for m in manifest:
            if m["Status"] in ("no_country", "no_match") or m["Status"].startswith("error"):
                print(f"  - {m['City']}, {m['Country']}  [{m['Status']}]")


if __name__ == "__main__":
    main()
