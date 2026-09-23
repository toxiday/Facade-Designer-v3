"""
generate_dataset_allsides_climate_pit.py  (all-sides + generalized room + climate)
Climate-conditioned POINT-IN-TIME training data for an ALL-SIDES surrogate that
GENERALIZES across room dimensions AND climate.

This fuses two existing generators:
  * generate_dataset_allsides.py       — windows on all four walls (S, N, E, W),
    but a FIXED 12x12x3 m room, a clear CIE gensky sky, and sun-only conditioning.
  * sDA_Generalized Model/data generator/generate_dataset_climate_pit.py — a single
    south facade, but VARIABLE room W/D/H, real .wea climate via gendaylit Perez
    skies (DNI/DHI), and per-hour TMYx sampling.

The combination: a random 1-4 of the four walls (each FACADE_ROWS x FACADE_COLS
= 16 x 32) are glazed (uniform count, random orientations, remaining walls solid) on a
variable W(X) x D(Y) x H(Z) box, lit by a gendaylit Perez sky from a random city +
random daytime hour of its TMYx year.  The room dimensions and the
weather (Sun vector, DNI, DHI) are recorded as columns so the model can learn the
floor-lux field as a function of facade x sun x weather x room geometry.

Physical cell sizes derive per-axis from the sampled room (the all-sides original
reused one cell size for all walls, valid only for a square room):
    S/N facades: col -> X, cell width  = W / FACADE_COLS
    E/W facades: col -> Y, cell width  = D / FACADE_COLS
    all facades: row -> Z, cell height = H / FACADE_ROWS
    sensors    : cx = W / GRID_N (X), cy = D / GRID_N (Y), plane at SENSOR_HEIGHT.

  sky:  gendaylit -ang <alt> <az> -W <DNI> <DHI>      (Perez all-weather + sun disc)
  lux:  179 * (0.265 R + 0.670 G + 0.065 B)           (same photometric scale)

Dataset row (CSV):
  Iter, Climate, Month, Day, Hour, Sun_X, Sun_Y, Sun_Z, DNI, DHI,
  Width, Depth, Height,                       room dimensions in metres
  S_Pix_0..511, N_Pix_0..511, E_Pix_0..511, W_Pix_0..511   (4 x 16x32, <0.5 = window)
  Lux_0..1023                                 floor illuminance (lux)

Facade convention (matches model_allsides): index = row * FACADE_COLS + col,
row 0 = top (high Z), row 15 = bottom.

S/N/E/W are ROOM-LOCAL wall ids, NOT compass directions:
    S = wall at y=0    N = wall at y=D    E = wall at x=W    W = wall at x=0
    col 0 = the x=0 end for S/N;  col 0 = the y=0 end for E/W.
This generator places the room orthogonally with +Y = true north, so here the labels DO
coincide with compass directions.  Downstream that is no longer guaranteed: the optimizer
supports a rotated building via north_angle, which rotates the sun vector rather than
relabelling walls (optimize_facade_allsides_climate.annual_conditions).

The Radiance scene builder below (build_scene, the S/N/E/W polygon block) is the GEOMETRY
GROUND TRUTH for where a facade cell physically sits.  run_facade_job.py's facade_panels.csv
is a direct port of those corner formulas — keep the two in sync.

Sensor ordering (X-outer east->west, Y-inner south->north) is identical to both
source generators, so the model_allsides rot90+fliplr plan-view transform still holds.

Run (Radiance required):
    python generate_dataset_allsides_climate_pit.py                 # full N_SAMPLES
    python generate_dataset_allsides_climate_pit.py --samples 8 --workers 2 --out test.csv
"""

import os, csv, glob, random, math, argparse, subprocess, multiprocessing, tempfile, shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[2]
FOLDER        = "allside_Generalised model"
OUTPUT_CSV    = str(REPO_ROOT / FOLDER / "datasets" / "dataset_allsides_climate_genroom_stresstest.csv")
CLIMATE_DIRS  = [str(REPO_ROOT / FOLDER / "climate data" / "library")]

N_SAMPLES     = 50000
N_WORKERS     = 22          # leave 2 of 24 cores free for other tasks
RANDOM_SEED   = 451

# Room dimensions (metres) — sampled independently & uniformly per sample.
W_RANGE       = (2.5, 10.0)   # width  (X) — facade span (S/N), floor
D_RANGE       = (2.5, 12.0)   # depth  (Y) — facade span (E/W), light penetration, floor
H_RANGE       = (2.5,  8.0)   # height (Z) — all facades & ceiling

# Facade grid: 16 rows x 32 cols per side, four sides.
FACADE_COLS   = 32
FACADE_ROWS   = 16
N_CELLS       = FACADE_COLS * FACADE_ROWS      # 512 per facade
N_FACADES     = 4

# Floor sensor grid: 32 x 32 = 1024 points at SENSOR_HEIGHT.
GRID_N        = 32
SENSOR_HEIGHT = 0.6            # metres above floor — fixed regardless of room size
N_SENSORS     = GRID_N * GRID_N                # 1024 floor sensors

# High-accuracy diffuse set (-ab 6): full inter-reflection, key for overcast Perez
# skies; -aa 0 point-exact ambient.  Same params as the generalized climate generator.
RAD_PARAMS    = ("-I+ -ab 6 -ad 1024 -as 512 -aa 0 -lw 0.00005 -lr 8 "
                 "-dc 1 -dj 0 -ds 0.2 -dt 0 -dr 1 -dp 512 -ss 0.7 -st 0.5")

WALL_REFL     = 0.5
FLOOR_REFL    = 0.2
CEIL_REFL     = 0.8
GLASS_TRANS   = 0.6975761815384331
WINDOW_THRESH = 0.5
MIN_ALT       = 5.0           # degrees — minimum solar altitude to sample an hour

LUM           = (0.265, 0.670, 0.065)   # photopic weights; * 179 lm/W -> lux
# ─────────────────────────────────────────────────────────────────────────────


# ── SOLAR POSITION (identical to both source generators) ───────────────────────

def solar_position(lat_deg, lon_deg, tz_h, month, day, hour_decimal):
    """
    (altitude_deg, azimuth_north_cw_deg, (sun_x, sun_y, sun_z)).
    sun_xyz = illumination direction (from sun toward scene), Z<0 when sun is up.
    azimuth: 0=N, 90=E, 180=S, 270=W (clockwise from North).
    lon_deg is EAST-positive; tz_h is the UTC offset in hours.
    """
    lat = math.radians(lat_deg)
    doy = int(275 * month / 9 - 30 + day - 2)
    if month <= 2:
        doy += 2
    B    = math.radians(360 / 365 * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    ET   = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    solar_time = hour_decimal + (lon_deg - tz_h * 15) / 15 + ET / 60
    omega      = math.radians(15 * (solar_time - 12))
    sin_alt = max(-1.0, min(1.0,
        math.sin(lat) * math.sin(decl) +
        math.cos(lat) * math.cos(decl) * math.cos(omega)))
    altitude_deg = math.degrees(math.asin(sin_alt))
    az_south_rad = math.atan2(
        math.sin(omega),
        math.cos(omega) * math.sin(lat) - math.tan(decl) * math.cos(lat))
    az = (math.degrees(az_south_rad) + 180.0) % 360.0
    alt_r, az_r = math.radians(altitude_deg), math.radians(az)
    sx = -math.sin(az_r) * math.cos(alt_r)
    sy = -math.cos(az_r) * math.cos(alt_r)
    sz = -math.sin(alt_r)
    return altitude_deg, az, (sx, sy, sz)


# ── CLIMATE FILES (from generate_dataset_climate_pit.py) ───────────────────────

def parse_wea(path):
    """
    Read a .wea climate file. Returns dict with location and the daytime hours
    (sun above MIN_ALT) precomputed as a list of records ready to simulate:
        (month, day, hour, dni, dhi, alt, az, (sx, sy, sz))
    The .wea stores longitude and time_zone West-positive; convert for solar_position
    (East-positive longitude, UTC-hour offset).
    """
    lat = lon_w = tz_deg = None
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and p[0] == "latitude":
                lat = float(p[1])
            elif len(p) == 2 and p[0] == "longitude":
                lon_w = float(p[1])
            elif len(p) == 2 and p[0] == "time_zone":
                tz_deg = float(p[1])
            elif len(p) == 5:
                rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
    lon_east = -lon_w                 # West-positive -> East-positive
    tz_h     = -tz_deg / 15.0          # standard meridian (deg W) -> UTC hour offset

    daytime = []
    for month, day, hour, dni, dhi in rows:
        alt, az, vec = solar_position(lat, lon_east, tz_h, month, day, hour)
        if alt > MIN_ALT:
            daytime.append((month, day, hour, dni, dhi, alt, az, vec))
    return {"name": Path(path).stem, "latitude": lat, "daytime": daytime}


def discover_climates():
    files = []
    for d in CLIMATE_DIRS:
        files.extend(glob.glob(os.path.join(d, "**", "*.wea"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No .wea files found under {CLIMATE_DIRS}")
    climates = [parse_wea(f) for f in files]
    return [c for c in climates if c["daytime"]]


# ── FACADE GENERATION (5-mode, from generate_dataset_allsides.py) ──────────────

def _pseudo_noise(x, y, seed):
    nx = x + seed * 13.7
    ny = y + seed * 19.3
    return (math.sin(nx) + math.cos(ny) +
            math.sin(nx * 1.42 + ny * 0.73) * 0.5 +
            math.cos(nx * 0.81 - ny * 1.14) * 0.5)


def generate_facade(rng):
    """
    Flat list of N_CELLS values in [0, 1] for ONE facade.
    Index = row * FACADE_COLS + col; row 0 = top, col 0 = left.  <0.5 = window.

    mode 0 — random scatter        mode 1 — clustered noise     mode 2 — smooth noise
    mode 3 — uniform rectangles    mode 4 — irregular rectangles
    """
    seed       = rng.randint(1, 9999999)
    target_wwr = rng.uniform(0.01, 0.80)
    mode       = seed % 5

    if mode == 0:
        n_win  = int(target_wwr * N_CELLS)
        n_wall = N_CELLS - n_win
        cells  = ([round(rng.uniform(0.0, 0.499), 3) for _ in range(n_win)] +
                  [round(rng.uniform(0.5, 1.0),   3) for _ in range(n_wall)])
        rng.shuffle(cells)
        return cells

    if mode == 3:
        # One window template size for this facade -> consistent proportions
        # (punched grid / ribbon / vertical panels / full-height bays).
        w_h = rng.randint(1, FACADE_ROWS)
        w_w = rng.randint(1, FACADE_COLS)
        mask   = [False] * N_CELLS
        placed = 0
        target_cells = int(target_wwr * N_CELLS)
        for _ in range(500):
            if placed >= target_cells:
                break
            r0 = rng.randint(0, FACADE_ROWS - w_h)
            c0 = rng.randint(0, FACADE_COLS - w_w)
            for r in range(r0, r0 + w_h):
                for c in range(c0, c0 + w_w):
                    idx = r * FACADE_COLS + c
                    if not mask[idx]:
                        mask[idx] = True
                        placed += 1
        return [round(rng.uniform(0.0, 0.499), 3) if mask[i]
                else round(rng.uniform(0.5, 1.0), 3)
                for i in range(N_CELLS)]

    if mode == 4:
        # Independent random size per rectangle -> mixed-size fenestration.
        mask         = [False] * N_CELLS
        placed       = 0
        target_cells = int(target_wwr * N_CELLS)
        for _ in range(500):
            if placed >= target_cells:
                break
            w_h = rng.randint(1, FACADE_ROWS)
            w_w = rng.randint(1, FACADE_COLS)
            r0  = rng.randint(0, FACADE_ROWS - w_h)
            c0  = rng.randint(0, FACADE_COLS - w_w)
            for r in range(r0, r0 + w_h):
                for c in range(c0, c0 + w_w):
                    idx = r * FACADE_COLS + c
                    if not mask[idx]:
                        mask[idx] = True
                        placed += 1
        return [round(rng.uniform(0.0, 0.499), 3) if mask[i]
                else round(rng.uniform(0.5, 1.0), 3)
                for i in range(N_CELLS)]

    # modes 1 & 2: noise-based (clustered vs smooth). Scales tuned on 0.375 m cells.
    scale = rng.uniform(1.0, 2.0) if mode == 1 else rng.uniform(0.08, 0.98)
    raw = []
    for row in range(FACADE_ROWS):
        for col in range(FACADE_COLS):
            raw.append(_pseudo_noise(col * scale, row * scale, seed))
    thresh_idx = min(int(target_wwr * N_CELLS), N_CELLS - 1)
    threshold  = sorted(raw)[thresh_idx]
    return [round(rng.uniform(0.0, 0.499), 3) if v < threshold
            else round(rng.uniform(0.5, 1.0), 3)
            for v in raw]


def solid_facade(rng):
    """A fully opaque wall (no windows): every cell >= WINDOW_THRESH.
    Same value texture as the wall cells in generate_facade, so the CSV carries no
    special sentinel — the scene's <0.5=window test renders it as a solid wall."""
    return [round(rng.uniform(0.5, 1.0), 3) for _ in range(N_CELLS)]


# ── RADIANCE (from generate_dataset_climate_pit.py) ────────────────────────────

def find_radiance_bin():
    cands = [
        Path(os.environ.get("USERPROFILE", "")) / "ladybug_tools" / "radiance" / "bin",
        Path("C:/Radiance/bin"),
        Path("C:/ladybug_tools/radiance/bin"),
        Path("C:/Program Files/Radiance/bin"),
    ]
    for p in cands:
        if (p / "rtrace.exe").exists() and (p / "gendaylit.exe").exists():
            return str(p)
    import shutil as sh
    if sh.which("rtrace") and sh.which("gendaylit"):
        return ""
    raise RuntimeError("Radiance (rtrace + gendaylit) not found.")

RADIANCE_BIN = find_radiance_bin()


def make_env():
    env = os.environ.copy()
    if RADIANCE_BIN:
        env["PATH"] = RADIANCE_BIN + os.pathsep + env.get("PATH", "")
    rad_lib = str(Path(RADIANCE_BIN).parent / "lib") if RADIANCE_BIN else ""
    if rad_lib and Path(rad_lib).exists():
        env["RAYPATH"] = rad_lib + os.pathsep + env.get("RAYPATH", ".")
    return env


def polygon_rad(name, mat, vertices):
    lines = [f"{mat} polygon {name}", "0", "0", str(len(vertices) * 3)]
    for v in vertices:
        lines.append(f"  {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    return "\n".join(lines) + "\n"


SKY_GLOW = ("\nskyfunc glow sky_glow\n0\n0\n4 1 1 1 0\n"
            "sky_glow source sky\n0\n0\n4 0 0 1 180\n\n"
            "skyfunc glow ground_glow\n0\n0\n4 1 1 1 0\n"
            "ground_glow source ground\n0\n0\n4 0 0 -1 180\n")


def write_scene(work_dir, facades, width, depth, height,
                alt_deg, az_north_cw_deg, dni, dhi, env):
    """Build the Radiance scene for one weather hour + room with four facades.

    facades : dict {'S','N','E','W'} -> flat list of N_CELLS floats (row 0 = top).
              S/N col 0 = West;  E/W col 0 = South.  <0.5 = window.
    The room is a width(X) x depth(Y) x height(Z) box.  All four walls ARE the
    facade cells (no solid wall polygons); only floor + ceiling are fixed polys.
    Returns (oct_path, pts_path).
    """
    W, D, H = width, depth, height
    cw_sn = W / FACADE_COLS     # S/N facade column width (along X)
    cw_ew = D / FACADE_COLS     # E/W facade column width (along Y)
    ch    = H / FACADE_ROWS     # facade row height (along Z)
    cx, cy = W / GRID_N, D / GRID_N   # sensor spacing (X, Y)

    # 1. Sky — gendaylit Perez all-weather from this hour's DNI/DHI + sun.
    #    -ang takes (altitude, azimuth-from-South positive-West).
    gd_az = az_north_cw_deg - 180.0
    cmd   = f"gendaylit -ang {alt_deg:.4f} {gd_az:.4f} -W {dni:.3f} {dhi:.3f}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"gendaylit failed: {r.stderr}")
    (work_dir / "sky.rad").write_text(r.stdout + SKY_GLOW)

    # 2. Materials (HB defaults)
    (work_dir / "materials.rad").write_text(
        f"void plastic wall_mat\n0\n0\n5 {WALL_REFL} {WALL_REFL} {WALL_REFL} 0 0\n\n"
        f"void plastic floor_mat\n0\n0\n5 {FLOOR_REFL} {FLOOR_REFL} {FLOOR_REFL} 0 0\n\n"
        f"void plastic ceil_mat\n0\n0\n5 {CEIL_REFL} {CEIL_REFL} {CEIL_REFL} 0 0\n\n"
        f"void glass window_mat\n0\n0\n3 {GLASS_TRANS} {GLASS_TRANS} {GLASS_TRANS}\n")

    # 3. Geometry — W x D x H box; floor + ceiling fixed, four walls = facade cells.
    #    Vertex winding gives each polygon an inward normal (S:+Y N:-Y E:-X W:+X).
    geom = [
        polygon_rad("floor",   "floor_mat", [(0,0,0),(W,0,0),(W,D,0),(0,D,0)]),
        polygon_rad("ceiling", "ceil_mat",  [(0,0,H),(0,D,H),(W,D,H),(W,0,H)]),
    ]
    for row in range(FACADE_ROWS):            # row 0 = top (high Z)
        z0 = (FACADE_ROWS - 1 - row) * ch
        z1 = z0 + ch
        for col in range(FACADE_COLS):
            idx = row * FACADE_COLS + col
            # South facade  Y=0  col0=West  normal +Y
            x0 = col * cw_sn; x1 = x0 + cw_sn
            mat = "window_mat" if facades['S'][idx] < WINDOW_THRESH else "wall_mat"
            geom.append(polygon_rad(f"S_{col}_{row}", mat,
                [(x0,0,z0),(x0,0,z1),(x1,0,z1),(x1,0,z0)]))
            # North facade  Y=D  col0=West  normal -Y
            mat = "window_mat" if facades['N'][idx] < WINDOW_THRESH else "wall_mat"
            geom.append(polygon_rad(f"N_{col}_{row}", mat,
                [(x0,D,z0),(x1,D,z0),(x1,D,z1),(x0,D,z1)]))
            # East facade  X=W  col0=South  normal -X
            y0 = col * cw_ew; y1 = y0 + cw_ew
            mat = "window_mat" if facades['E'][idx] < WINDOW_THRESH else "wall_mat"
            geom.append(polygon_rad(f"E_{col}_{row}", mat,
                [(W,y0,z0),(W,y0,z1),(W,y1,z1),(W,y1,z0)]))
            # West facade  X=0  col0=South  normal +X
            mat = "window_mat" if facades['W'][idx] < WINDOW_THRESH else "wall_mat"
            geom.append(polygon_rad(f"W_{col}_{row}", mat,
                [(0,y0,z0),(0,y1,z0),(0,y1,z1),(0,y0,z1)]))
    (work_dir / "geometry.rad").write_text("".join(geom))

    # 4. Sensors — X-outer east->west, Y-inner south->north (matches both sources).
    pts = []
    for i in range(GRID_N - 1, -1, -1):
        for j in range(GRID_N):
            pts.append(f"{(i+0.5)*cx:.6f} {(j+0.5)*cy:.6f} {SENSOR_HEIGHT:.6f} 0 0 1")
    pts_path = work_dir / "sensors.pts"
    pts_path.write_text("\n".join(pts))

    # 5. Octree
    oct_path = work_dir / "scene.oct"
    with open(oct_path, "wb") as fh:
        rr = subprocess.run("oconv sky.rad materials.rad geometry.rad",
                            shell=True, stdout=fh, stderr=subprocess.PIPE,
                            env=env, cwd=work_dir)
    if rr.returncode != 0:
        raise RuntimeError(f"oconv failed: {rr.stderr.decode()}")
    return oct_path, pts_path


def run_simulation(oct_path, pts_path, env):
    """rtrace -> list of N_SENSORS lux values."""
    r = subprocess.run(f"rtrace {RAD_PARAMS} \"{oct_path.name}\"",
                       shell=True, input=pts_path.read_text(),
                       capture_output=True, text=True, env=env, cwd=oct_path.parent)
    lux = []
    for line in r.stdout.strip().splitlines():
        p = line.split()
        if len(p) >= 3:
            try:                       # rtrace echoes the octree header first; skip it
                R, G, B = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue
            lux.append(max(0.0, 179.0 * (LUM[0]*R + LUM[1]*G + LUM[2]*B)))
    return lux


# ── WORKER ────────────────────────────────────────────────────────────────────
# Climates and the Radiance env are loaded once per worker (via the pool
# initializer) instead of pickled with every task.

_CLIMATES = None
_ENV = None


def _init_worker(climates):
    global _CLIMATES, _ENV
    _CLIMATES = climates
    _ENV = make_env()


def generate_sample(args):
    sample_idx, seed, climate_idx = args
    climate = _CLIMATES[climate_idx]
    env = _ENV
    work_dir = Path(tempfile.mkdtemp(prefix=f"allpit_{sample_idx}_"))
    try:
        rng = random.Random(seed)
        # Glaze a random 1-4 of the four walls (uniform) at random orientations; the
        # rest are fully solid.  Keeps each glazed wall's WWR ceiling wide while
        # dropping the mean glazed-wall count 4 -> 2.5 (shifts interiors toward the
        # usable daylight band; still yields 25% four-sided glass boxes).
        sides      = ('S', 'N', 'E', 'W')
        n_open     = rng.randint(1, 4)                    # uniform 1-4 glazed walls
        open_sides = set(rng.sample(sides, n_open))       # random which orientations
        facades = {s: (generate_facade(rng) if s in open_sides else solid_facade(rng))
                   for s in sides}

        # Random room — drawn from the same seeded RNG so the sample is reproducible.
        width  = round(rng.uniform(*W_RANGE), 3)
        depth  = round(rng.uniform(*D_RANGE), 3)
        height = round(rng.uniform(*H_RANGE), 3)

        month, day, hour, dni, dhi, alt, az, (sx, sy, sz) = \
            rng.choice(climate["daytime"])

        oct_path, pts_path = write_scene(work_dir, facades, width, depth, height,
                                         alt, az, dni, dhi, env)
        lux = run_simulation(oct_path, pts_path, env)
        if len(lux) != N_SENSORS:
            return None

        row = {"Iter": sample_idx, "Climate": climate["name"],
               "Month": month, "Day": day, "Hour": round(hour, 4),
               "Sun_X": round(sx, 6), "Sun_Y": round(sy, 6), "Sun_Z": round(sz, 6),
               "DNI": round(dni, 3), "DHI": round(dhi, 3),
               "Width": width, "Depth": depth, "Height": height}
        for side in ('S', 'N', 'E', 'W'):
            for k, v in enumerate(facades[side]):
                row[f"{side}_Pix_{k}"] = v
        for k, v in enumerate(lux):
            row[f"Lux_{k}"] = round(v, 4)
        return row
    except Exception as e:
        print(f"[WARN] Sample {sample_idx} failed: {e}")
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def build_header():
    cols = ["Iter", "Climate", "Month", "Day", "Hour",
            "Sun_X", "Sun_Y", "Sun_Z", "DNI", "DHI",
            "Width", "Depth", "Height"]
    for side in ('S', 'N', 'E', 'W'):
        cols += [f"{side}_Pix_{i}" for i in range(N_CELLS)]
    cols += [f"Lux_{i}" for i in range(N_SENSORS)]
    return cols


def main():
    ap = argparse.ArgumentParser(description="All-sides + generalized-room + climate PIT dataset generator.")
    ap.add_argument("--samples", type=int, default=N_SAMPLES, help=f"number of samples (default {N_SAMPLES})")
    ap.add_argument("--workers", type=int, default=N_WORKERS, help=f"worker processes (default {N_WORKERS})")
    ap.add_argument("--seed",    type=int, default=RANDOM_SEED, help=f"master RNG seed (default {RANDOM_SEED})")
    ap.add_argument("--out",     type=str, default=OUTPUT_CSV, help="output CSV path")
    args = ap.parse_args()

    n_samples, n_workers, out_csv = args.samples, args.workers, args.out

    print(f"Radiance bin : {RADIANCE_BIN or '(from PATH)'}")
    print(f"Room         : variable  W{W_RANGE} x D{D_RANGE} x H{H_RANGE} m")
    print(f"Facades      : 4 x {FACADE_COLS}x{FACADE_ROWS} = {N_FACADES*N_CELLS} cells "
          f"| sensors {N_SENSORS} @ {SENSOR_HEIGHT}m")
    print(f"Sky model    : gendaylit Perez all-weather (real DNI/DHI per hour)")

    climates = discover_climates()
    n_hours = sum(len(c["daytime"]) for c in climates)
    print(f"Climates     : {len(climates)}  |  daytime hours available: {n_hours}")
    print(f"Samples      : {n_samples}  |  Workers: {n_workers}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    print(f"Output       : {out_csv}\n")

    rng = random.Random(args.seed)
    arg_list = []
    for i in range(n_samples):
        seed = rng.randint(0, 2**31 - 1)
        c_idx = rng.randrange(len(climates))   # random city per sample
        arg_list.append((i, seed, c_idx))

    header = build_header()
    done = failed = 0
    report_every = max(1, n_samples // 100)   # ~100 progress prints, any n_samples
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        with multiprocessing.Pool(processes=n_workers,
                                  initializer=_init_worker,
                                  initargs=(climates,)) as pool:
            for row in pool.imap_unordered(generate_sample, arg_list, chunksize=4):
                if row is not None:
                    writer.writerow(row); f.flush(); done += 1
                else:
                    failed += 1
                total = done + failed
                if total % report_every == 0 or total == n_samples:
                    print(f"  {total}/{n_samples} ({total/n_samples*100:.0f}%)  "
                          f"done={done} failed={failed}")

    print(f"\nComplete: {done} generated, {failed} failed -> {out_csv}")


if __name__ == "__main__":
    main()
