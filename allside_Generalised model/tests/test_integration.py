"""
test_integration.py — do the project's components still agree with EACH OTHER?

Each component has its own checks; this file checks the seams between them.  Several pieces
of logic are deliberately DUPLICATED across components — Grasshopper's Python cannot import
torch-side code, and the dataset generator predates the optimizer — so a drift between two
copies would leave every component passing on its own while the pipeline quietly answers a
different question than the one it was trained on.  The copies checked here:

    solar position     generator / optimizer / check_sun_coverage / gh_pick_city
    city list + order  optimizer.discover_cities / gh_pick_city.discover_cities
    sun-hour filter    optimizer.annual_conditions / gh_pick_city.hour_day_counts
    wall azimuths      optimizer / gh_write_facade_job
    room limits        model / generator / gh_write_facade_job
    panel geometry     optimizer.facade_panel_boxes / generator.write_scene (Radiance)
    floor transform    trainer dataset / verify_facade_radiance.sensors_to_plan
    output contract    run_facade_job outputs / gh_read_facade

    python test_integration.py                     # everything below
    python test_integration.py --results DIR ...   # also round-trip these result folders
    python test_integration.py --no-gpu            # skip the model checks (weights + torch)
    python test_integration.py --no-radiance       # skip the Radiance scene check

Nothing here writes into the project: the one Radiance scene is built in a temp folder, and
the training step runs in memory on a fresh model — the trainer's __main__, which would
overwrite models/, is never called.
"""

import os
import sys
import json
import math
import argparse
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE    = Path(__file__).resolve().parent
AGM     = HERE.parent                                   # allside_Generalised model/
REPO    = AGM.parent
TRAINER = AGM / "model trainer"
BRIDGE  = AGM / "rhino bridge"
GENDIR  = AGM / "data generator"
RESP    = AGM / "responsive facade"
ABL     = REPO / "evidence" / "ablation"
for p in (TRAINER, RESP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--results", nargs="*", default=[],
                help="extra run_facade_job output folders to round-trip through the reader")
ap.add_argument("--no-gpu", action="store_true", help="skip the model/weights checks")
ap.add_argument("--no-radiance", action="store_true", help="skip the Radiance scene check")
ARGS = ap.parse_args()

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def load(path, name):
    """Import a file by path — the way the Rhino components and verify load their peers."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def in_repo(p):
    try:
        return Path(p).resolve().is_relative_to(REPO)
    except (TypeError, ValueError):
        return False


def same_path(a, b):
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


# ---------------------------------------------------------------------------
print("\n[1] every .py in the repo compiles")
# ---------------------------------------------------------------------------
pys = [p for p in REPO.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]
bad = []
for p in pys:
    try:
        compile(p.read_bytes(), str(p), "exec")          # syntax only; writes no .pyc
    except SyntaxError as e:
        bad.append(f"{p.relative_to(REPO)}:{e.lineno}: {e.msg}")
check(f"{len(pys)} files compile", not bad, "; ".join(bad))

# ---------------------------------------------------------------------------
print("\n[2] every module imports from THIS repo, and every path it resolves is inside it")
# ---------------------------------------------------------------------------
import model_allsides_climate as M                                  # noqa: E402
import optimize_facade_allsides_climate as O                        # noqa: E402
import facade_viz as FV                                             # noqa: E402
import run_facade_job as RFJ                                        # noqa: E402
import run_unit as RU                                               # noqa: E402
import unit_zone_matrix as UZM                                      # noqa: E402
import verify_facade_radiance as V                                  # noqa: E402
import evaluate32x32 as EV                                          # noqa: E402
import seed_sweep as SS                                             # noqa: E402
import live_solver as LS                                            # noqa: E402
import responsive_facade as RF                                      # noqa: E402  (claims TkAgg)
import facade_designer_allsides_climate as FD                       # noqa: E402  (claims TkAgg)
import matplotlib.pyplot as plt                                     # noqa: E402
plt.switch_backend("Agg")                                          # nothing may open a window

GW  = load(BRIDGE / "gh_write_facade_job.py", "t_gh_write_facade_job")
GU  = load(BRIDGE / "gh_write_unit_jobs.py",  "t_gh_write_unit_jobs")
GR  = load(BRIDGE / "gh_read_facade.py",      "t_gh_read_facade")
PC  = load(BRIDGE / "gh_pick_city.py",        "t_gh_pick_city")
GMR = load(BRIDGE / "gh_make_room.py",        "t_gh_make_room")
GMZ = load(BRIDGE / "gh_make_zone.py",        "t_gh_make_zone")
CSC = load(BRIDGE / "check_sun_coverage.py",  "t_check_sun_coverage")
VR  = load(BRIDGE / "verify_rotation.py",     "t_verify_rotation")
VIS = load(GENDIR / "visualize_sample_allsides_climate_pit.py", "t_visualize_sample")
DLC = load(AGM / "climate data" / "download_climate_library.py", "t_download_climate")
GEN = None
if ARGS.no_radiance:
    print("  skip  generator import (--no-radiance; it locates Radiance at import time)")
else:
    GEN = load(GENDIR / "generate_dataset_allsides_climate_pit.py", "t_generator")
sys.path.insert(0, str(ABL))
ABLA = load(ABL / "ablation_architecture.py", "t_ablation")

mods = [M, O, FV, RFJ, RU, UZM, V, EV, SS, LS, RF, FD, GW, GU, GR, PC, GMR, GMZ, CSC, VR,
        VIS, DLC, ABLA] + ([GEN] if GEN else [])
outside = [m.__name__ for m in mods if not in_repo(m.__file__)]
check(f"{len(mods)} modules loaded from this repo", not outside, ", ".join(outside))
for dep in ("model20x20_v2", "model_transformer_v2"):
    check(f"ablation dependency {dep} resolved inside evidence/ablation",
          dep in sys.modules and same_path(Path(sys.modules[dep].__file__).parent, ABL))

must_exist = {
    "optimizer CLIMATE_DIR": O.CLIMATE_DIR, "model BEST_MODEL_PATH": M.BEST_MODEL_PATH,
    "verify GEN_PY": V.GEN_PY, "unit_zone_matrix BRIDGE": UZM.BRIDGE,
    "live_solver _TRAINER": LS._TRAINER, "check_sun_coverage CLIMATE_DIR": CSC.CLIMATE_DIR,
    "ablation OUT_DIR": ABLA.OUT_DIR,
}
if GEN:
    must_exist["generator CLIMATE_DIRS[0]"] = GEN.CLIMATE_DIRS[0]
may_be_created = {
    "run_facade_job OUT_ROOT": RFJ.OUT_ROOT, "designer OUT_DIR": FD.OUT_DIR,
    "responsive OUT_DIR": RF.OUT_DIR, "visualize OUT_DIR": VIS.OUT_DIR,
    "visualize DEFAULT_CSV": VIS.DEFAULT_CSV, "model CHECKPOINT_PATH": M.CHECKPOINT_PATH,
    "ablation DATA_CSV (gitignored; only needed to re-train)": ABLA.DATA_CSV,
}
if GEN:
    may_be_created["generator OUTPUT_CSV"] = GEN.OUTPUT_CSV
for k, p in must_exist.items():
    check(f"{k} exists inside the repo", in_repo(p) and Path(p).exists(), str(p))
for k, p in may_be_created.items():
    check(f"{k} points inside the repo", in_repo(p), str(p))

# ---------------------------------------------------------------------------
print("\n[3] shared constants agree across components")
# ---------------------------------------------------------------------------
check("room limits: bridge == model",
      (GW.W_MIN, GW.W_MAX, GW.D_MIN, GW.D_MAX, GW.H_MIN, GW.H_MAX)
      == (M.W_MIN, M.W_MAX, M.D_MIN, M.D_MAX, M.H_MIN, M.H_MAX))
if GEN:
    check("room limits: generator sampling ranges == model normalisation",
          (GEN.W_RANGE, GEN.D_RANGE, GEN.H_RANGE)
          == ((M.W_MIN, M.W_MAX), (M.D_MIN, M.D_MAX), (M.H_MIN, M.H_MAX)))
    check("grid sizes: generator == model",
          (GEN.FACADE_ROWS, GEN.FACADE_COLS, GEN.GRID_N, GEN.N_FACADES)
          == (M.FACADE_ROWS, M.FACADE_COLS, M.GRID_N, M.N_FACADES))
    check("window threshold 0.5 separates the optimizer's two input values",
          O.WIN_VAL < GEN.WINDOW_THRESH <= O.WALL_VAL, f"{O.WIN_VAL} < {GEN.WINDOW_THRESH} <= {O.WALL_VAL}")
alts = {"optimizer": O.MIN_ALT, "gh_pick_city": PC.MIN_ALT, "live_solver": LS.MIN_ALT}
if GEN:
    alts["generator"] = GEN.MIN_ALT
check("MIN_ALT identical everywhere", len(set(alts.values())) == 1, str(alts))
check("floor grid: bridge == model", GW.GRID_N == M.GRID_N)

# ---------------------------------------------------------------------------
print("\n[4] solar position: four copies, identical answers")
# ---------------------------------------------------------------------------
cities = O.discover_cities()
worst = dict(alt=0.0, az=0.0, vec=0.0)
n_eval = 0
hdr_bad = []
for path in cities:
    lat, lon, tz, rows = PC.parse_wea(path)
    if (lat, lon, tz) != O.city_location(path):
        hdr_bad.append(Path(path).stem)
    for (mo, d, h, _dni, _dhi) in rows:
        if d != 21:
            continue
        a_o, z_o, v_o = O.solar_position(lat, lon, tz, mo, d, h)
        a_c, z_c = CSC.solar_position(lat, lon, tz, mo, d, h)
        a_p = PC.solar_altitude(lat, lon, tz, mo, d, h)
        alts_ = [a_c, a_p]
        azs_ = [z_c]
        if GEN:
            a_g, z_g, v_g = GEN.solar_position(lat, lon, tz, mo, d, h)
            alts_.append(a_g); azs_.append(z_g)
            worst["vec"] = max(worst["vec"], max(abs(x - y) for x, y in zip(v_o, v_g)))
        worst["alt"] = max(worst["alt"], max(abs(a - a_o) for a in alts_))
        worst["az"] = max(worst["az"], max(min(abs(z - z_o), 360 - abs(z - z_o)) for z in azs_))
        n_eval += 1
check(f"{len(cities)} .wea headers parse identically (gh_pick_city vs optimizer)", not hdr_bad,
      ", ".join(hdr_bad[:5]))
check(f"altitude/azimuth/vector identical over {n_eval:,} city-hours",
      max(worst.values()) <= 1e-12,
      f"max diff alt {worst['alt']:.1e} az {worst['az']:.1e} vec {worst['vec']:.1e}"
      + ("" if GEN else "  (generator copy skipped)"))

# ---------------------------------------------------------------------------
print("\n[5] city list, city lookup and sun-hour filter: Rhino picker == optimizer")
# ---------------------------------------------------------------------------
pc_files = PC.discover_cities(str(O.CLIMATE_DIR))
check("same cities in the same order", len(pc_files) == len(cities)
      and all(same_path(a, b) for a, b in zip(pc_files, cities)), f"{len(pc_files)} vs {len(cities)}")
check("same labels", [PC.city_label(f) for f in pc_files] == [O.city_label(c) for c in cities])
for name in ("Houston", "Seattle", "Lisboa", "Seoul", "Calgary", "Luanda", "New.York"):
    hits = PC.filter_cities(pc_files, name)
    if len(hits) == 1:
        check(f"'{name}' -> same file", same_path(PC.select(pc_files, name, 0)[0], O.resolve_city(name)),
              O.city_label(hits[0]))
    else:
        try:
            O.resolve_city(name)
            check(f"'{name}' is ambiguous ({len(hits)} hits) and resolve_city refuses it", False)
        except ValueError:
            check(f"'{name}' is ambiguous ({len(hits)} hits) and resolve_city refuses it", True)
lats = [(abs(O.city_location(c)[0]), c) for c in cities]
probe = [O.resolve_city(n) for n in ("Houston", "Seattle", "Luanda")] + [max(lats)[1], min(lats)[1]]
mism = []
for c in probe:
    counts, *_ = PC.hour_day_counts(c)
    for h in range(24):
        n_opt = len(O.annual_conditions(c, h)[3])
        if counts[h] != n_opt:
            mism.append(f"{O.city_label(c)[:18]} {h:02d}h {counts[h]} vs {n_opt}")
check(f"per-hour sunny-day counts identical ({len(probe)} cities incl. highest/lowest latitude x 24 h)",
      not mism, "; ".join(mism[:4]))

# ---------------------------------------------------------------------------
print("\n[6] wall azimuths: bridge == optimizer")
# ---------------------------------------------------------------------------
worst_az = 0.0
for k in range(-720, 721):
    n = k * 0.5
    a, b = GW.wall_azimuths(n), O.wall_azimuths(n)
    worst_az = max(worst_az, max(abs(a[s] - b[s]) for s in "SNEW"))
check("north angles -360..360 step 0.5", worst_az < 1e-9, f"max diff {worst_az:.1e}")

# ---------------------------------------------------------------------------
print("\n[7] floor orientation: trainer dataset transform == verify's sensors_to_plan")
# ---------------------------------------------------------------------------
import torch                                                        # noqa: E402
raw = np.arange(M.N_SENSORS, dtype=np.float32)
t = torch.tensor(raw).view(1, 1, M.GRID_N, M.GRID_N)
t = torch.flip(torch.rot90(t, k=1, dims=[2, 3]), dims=[3])[0, 0].numpy()
check("identical 32x32 mapping", np.array_equal(t, V.sensors_to_plan(raw)))

# ---------------------------------------------------------------------------
print("\n[8] Rhino bridge resolves THIS repo, its venv and its runners")
# ---------------------------------------------------------------------------
for mod, fname in ((GW, "gh_write_facade_job.py"), (GU, "gh_write_unit_jobs.py")):
    got = mod.resolve_repo()
    check(f"{fname}: resolve_repo() == this checkout", same_path(got, REPO),
          str(got) if same_path(got, REPO) else
          f"resolves to {got!r} — edit PROJECT_ROOT at the top of rhino bridge/{fname} to {str(REPO)!r}")
saved = (GW.PROJECT_ROOT, GU.PROJECT_ROOT)
GW.PROJECT_ROOT = GU.PROJECT_ROOT = ""                   # a fresh clone elsewhere: walk up instead
check("with PROJECT_ROOT blank, both walk up from __file__ to the repo",
      same_path(GW.resolve_repo(), REPO) and same_path(GU.resolve_repo(), REPO))
GW.PROJECT_ROOT, GU.PROJECT_ROOT = saved
w = GU.load_writer()
check("unit writer loads THIS repo's gh_write_facade_job.py",
      same_path(w.__file__, BRIDGE / "gh_write_facade_job.py"), w.__file__)
d, pcsv, rjs = GW.predict_outputs("probe_job")
check("predicted output dir == run_facade_job.OUT_ROOT / name",
      same_path(d, RFJ.OUT_ROOT / "probe_job") and same_path(pcsv, Path(d) / "facade_panels.csv"))
check("venv python the bridge launches exists", (REPO / ".venv" / "Scripts" / "python.exe").is_file())
for runner in ("run_facade_job.py", "run_unit.py"):
    check(f"bridge's runner {runner} exists", (TRAINER / runner).is_file())

# ---------------------------------------------------------------------------
print("\n[9] unit layout validator")
# ---------------------------------------------------------------------------
def room(name, u, v, w_, d_):
    return dict(name=name, width=w_, depth=d_, origin_uv=(u, v), x_axis_uv=(1.0, 0.0),
                y_axis_uv=(0.0, 1.0))


flat = [(0, 0), (13.089, 0), (13.089, 14.104), (0, 14.104)]
err, _ = GU.check_layout([room("a", 0, 0, 4, 5), room("b", 4, 0, 4, 5)], flat)
check("flush, non-overlapping rooms pass", not err, "; ".join(err))
err, _ = GU.check_layout([room("a", 0, 0, 4, 5), room("b", 2, 2.56, 4, 5)], flat)
check("the measured unit-101 overlap (4.88 m2) is flagged", any("overlap by 4.88" in e for e in err),
      "; ".join(err))
err, _ = GU.check_layout([room("a", 12, 0, 4, 5)], flat)
check("a room outside the flat is flagged", any("outside" in e for e in err))
u_shape = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
err, _ = GU.check_layout([room("bulge", 1, 5, 8, 3)], u_shape)
check("corners inside but bulging through a notch is flagged (edge-crossing test)",
      any("outside" in e for e in err))
err, _ = GU.check_layout([room("a", 0, 0, 4, 5), room("a", 5, 0, 4, 5)], flat)
check("duplicate room names are flagged", any("duplicate" in e for e in err))

# ---------------------------------------------------------------------------
print("\n[10] climate library is well-formed")
# ---------------------------------------------------------------------------
malformed = []
for c in cities:
    try:
        _lat, _lon, _tz, rows = PC.parse_wea(c)
        if len(rows) != 8760:
            malformed.append(f"{Path(c).stem}: {len(rows)} rows")
    except Exception as e:                                          # noqa: BLE001
        malformed.append(f"{Path(c).stem}: {e}")
check(f"{len(cities)} .wea files, each with a header and 8760 hourly rows", not malformed and len(cities) == 170,
      "; ".join(malformed[:3]))

# ---------------------------------------------------------------------------
print("\n[11] optimizer output -> Rhino reader round trip")
# ---------------------------------------------------------------------------
boxes = {}
result_dirs = [AGM / "outputs" / "example_job", AGM / "outputs" / "rotated_multislot"] \
    + [Path(r) for r in ARGS.results]
for rd in result_dirs:
    tag = rd.name
    if not (rd / "facade_panels.csv").is_file():
        check(f"{tag}: result folder present", False, str(rd))
        continue
    res = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    room_ = res["resolved"]["room"]
    W, D, H = room_["width"], room_["depth"], room_["height"]
    panels = GR.read_panels(str(rd / "facade_panels.csv"))
    binary = V.load_binary_wide(rd / "facade_wide.csv")
    check(f"{tag}: 2048 panels", len(panels) == 2048)
    same = all(np.array_equal(np.array(GR.side_lattice(panels, s)[0]), binary[i])
               for i, s in enumerate(O.SIDES))
    check(f"{tag}: reader's per-side lattice == facade_wide.csv, cell for cell", same)
    try:
        GR.check_room_match(panels, W, D, H)
        check(f"{tag}: panel extents match the job's room {W}x{D}x{H}", True)
    except GR.FacadeReadError as e:
        check(f"{tag}: panel extents match the job's room", False, str(e).splitlines()[0])
    ref = {(s, r, c): (p0, p1) for s, r, c, _i, p0, p1, _n in O.facade_panel_boxes(W, D, H)}
    geo = max(max(abs(a - b) for a, b in zip(q["p0"] + q["p1"], sum(ref[(q["side"], q["row"], q["col"])], ())))
              for q in panels)
    check(f"{tag}: every panel rectangle == facade_panel_boxes", geo < 1e-4, f"max diff {geo:.1e} m")
    so = tuple(res["resolved"]["sides_open"])
    check(f"{tag}: sides holding windows are a subset of sides_open {''.join(so)}",
          set(GR.sides_with_windows(panels)) <= set(so))
    check(f"{tag}: reader's sides_from_result_json == sides_open",
          GR.sides_from_result_json(str(rd / "result.json")) == so)
    check(f"{tag}: glazing_pct metric == the shipped binary",
          abs(res["metrics"]["glazing_pct"] - binary.mean() * 100) < 1e-6)

# ---------------------------------------------------------------------------
print("\n[12] trained weights, dataset schema, one forward pass, one in-memory training step")
# ---------------------------------------------------------------------------
if ARGS.no_gpu:
    print("  skip  (--no-gpu)")
else:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtime0 = Path(M.BEST_MODEL_PATH).stat().st_mtime_ns
    model = M.AllSidesClimateTransformer(**M.MODEL_CONFIG)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("parameter count == 31,181,572", n_par == 31_181_572, f"{n_par:,}")
    state = torch.load(M.BEST_MODEL_PATH, weights_only=True, map_location="cpu")
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    r = model.load_state_dict(state, strict=True)
    check("weights load with strict=True", not r.missing_keys and not r.unexpected_keys)
    model.to(dev).eval()

    csv_ = AGM / "datasets" / "dataset_allsides_climate_genroom_stresstest.csv"
    ds = M.AllSidesClimateDataset(str(csv_))
    c0, f0, l0 = ds[0]
    check("dataset item shapes (8,) (4,16,32) (1,32,32)",
          tuple(c0.shape) == (8,) and tuple(f0.shape) == (4, 16, 32) and tuple(l0.shape) == (1, 32, 32))
    check("facade inputs in [0,1], room channels normalised to [0,1]",
          float(ds.facade_tensors.min()) >= 0 and float(ds.facade_tensors.max()) <= 1
          and float(ds.cond_tensors[:, 5:].min()) >= 0 and float(ds.cond_tensors[:, 5:].max()) <= 1)

    with torch.no_grad():
        pred = torch.cat([model(ds.cond_tensors[i:i + 32].to(dev), ds.facade_tensors[i:i + 32].to(dev)).float().cpu()
                          for i in range(0, len(ds), 32)])
    check("forward pass finite, shape (N,1,32,32)",
          torch.isfinite(pred).all().item() and tuple(pred.shape[1:]) == (1, 32, 32))
    t_lux, p_lux = (ds.lux_tensors ** 2).numpy(), (pred ** 2).numpy()
    from skimage.metrics import structural_similarity as ssim
    rng_ = float(t_lux.max() - t_lux.min()) or 1.0
    s_mean = float(np.mean([ssim(a[0], b[0], data_range=rng_, win_size=5) for a, b in zip(t_lux, p_lux)]))
    usable = (t_lux >= 300) & (t_lux <= 3000)
    # NOT a held-out score: the stress-test CSV was generated with the generator's default
    # seed, the same one as the 400k training set, so its rows duplicate training samples.
    # This only proves the weights and the data pipeline are wired together correctly.
    print(f"        (info) stress-test CSV, {len(ds)} samples (overlaps training data): "
          f"SSIM {s_mean:.4f}, usable-zone MAE {np.abs(t_lux - p_lux)[usable].mean():.1f} lux, "
          f"all-pixel MAE {np.abs(t_lux - p_lux).mean():.1f} lux")

    fresh = M.AllSidesClimateTransformer(**M.MODEL_CONFIG).to(dev).train()
    crit = M.FullRangeDaylightLoss(huber_delta=2.0, grad_weight=0.5, tversky_weight=0.5,
                                   tversky_alpha=0.3, tversky_beta=0.7)
    optim_ = torch.optim.AdamW(fresh.parameters(), lr=1.5e-4, weight_decay=0.01)
    first = next(fresh.parameters())
    before = first.detach().clone()
    loss = crit(fresh(ds.cond_tensors[:8].to(dev), ds.facade_tensors[:8].to(dev)), ds.lux_tensors[:8].to(dev))
    optim_.zero_grad(); loss.backward(); optim_.step()
    moved = not torch.equal(before, first.detach())
    check("training step: finite loss, weights update", math.isfinite(loss.item()) and moved,
          f"loss {loss.item():.3f}")
    check("models/ untouched by the test", Path(M.BEST_MODEL_PATH).stat().st_mtime_ns == mtime0)
    del model, fresh
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
print("\n[13] Radiance scene geometry == optimizer panel geometry  (the 1 = window trap)")
# ---------------------------------------------------------------------------
if GEN is None:
    print("  skip  (--no-radiance)")
else:
    gv = V._load_generator()
    check("verify loads THIS repo's generator", same_path(gv.__file__, GENDIR / "generate_dataset_allsides_climate_pit.py"))
    W, D, H = 5.3, 7.9, 3.4
    binary = (np.random.default_rng(0).random((4, 16, 32)) < 0.3).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        GEN.write_scene(Path(td), V.binary_to_scene_facades(binary), W, D, H,
                        35.0, 200.0, 600.0, 120.0, GEN.make_env())
        text = (Path(td) / "geometry.rad").read_text().splitlines()
    polys = {}
    for i, line in enumerate(text):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "polygon" and parts[2][0] in "SNEW" and "_" in parts[2]:
            side, col, row = parts[2].split("_")
            pts = [tuple(map(float, text[i + 4 + k].split())) for k in range(4)]
            polys[(side, int(row), int(col))] = (parts[0], pts)
    check("scene has 2048 facade polygons", len(polys) == 2048, str(len(polys)))
    worst_g, wrong_mat = 0.0, 0
    for s, r, c, _i, p0, p1, _n in O.facade_panel_boxes(W, D, H):
        mat, pts = polys[(s, r, c)]
        lo = tuple(min(p[k] for p in pts) for k in range(3))
        hi = tuple(max(p[k] for p in pts) for k in range(3))
        worst_g = max(worst_g, max(abs(a - b) for a, b in zip(lo + hi, p0 + p1)))
        want = "window_mat" if binary[O.SIDES.index(s), r, c] == 1 else "wall_mat"
        wrong_mat += mat != want
    check("every Radiance polygon sits exactly where facade_panel_boxes says", worst_g < 1e-5,
          f"max diff {worst_g:.1e} m")
    check("binary 1 -> window_mat, 0 -> wall_mat for every cell", wrong_mat == 0, f"{wrong_mat} wrong")

# ---------------------------------------------------------------------------
print()
if FAILS:
    print(f"{len(FAILS)} FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("all integration checks passed")
