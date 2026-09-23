"""
test_designer_headless.py — drive the annual facade designer GUI end to end, with no display.

Every input goes through the designer's own event handlers (timeline click, edge-band
toggle, Shift+drag fill, freehand paint, target box, city search, sliders) using synthetic
mouse events, then presses Run.  The real optimizer runs on the GPU and the real result
files are written — into a scratch folder, not the repo's outputs/.

    python test_designer_headless.py                 # outputs to a temp folder
    python test_designer_headless.py --out DIR       # keep the CSV + PNG for inspection
"""

import sys
import argparse
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
# The designer calls plt.show()/plt.pause() as a live GUI would; on Agg those are no-ops.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

HERE    = Path(__file__).resolve().parent
TRAINER = HERE.parent / "model trainer"
sys.path.insert(0, str(TRAINER))

import facade_designer_allsides_climate as FD                        # noqa: E402  (claims TkAgg)
import matplotlib.pyplot as plt                                      # noqa: E402
plt.switch_backend("Agg")                                           # ...so take it back
from optimize_facade_allsides_climate import (                       # noqa: E402
    FacadeOptimizerAllSidesClimate, resolve_city, city_label, annual_conditions,
    reduce_conditions, append_room, SIDES)
from verify_facade_radiance import load_binary_wide                  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--out", default=None, help="output folder (default: a temp folder)")
args = ap.parse_args()
out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="designer_test_"))
out_dir.mkdir(parents=True, exist_ok=True)
FD.OUT_DIR = out_dir                                                 # never the repo's outputs/

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def ev(ax, x, y, button=1, key=None):
    return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=button, key=key,
                           name="button_press_event")


print("\n[1] the window builds")
d = FD.AllSidesClimateDesigner()                                     # plt.show() is a no-op on Agg
check("170 cities in the city slider", len(d.cities) == 170 and d.sld_city.valmax == 169)

print("\n[2] city search, room sliders, repr slider")
d.txt_search.set_val("seattle"); d._on_search("seattle")
check("search 'seattle' selects Seattle", "Seattle" in city_label(d.cities[d.city_idx]),
      city_label(d.cities[d.city_idx]))
d._on_search("houston")
check("search 'houston' selects Houston", "Houston" in city_label(d.cities[d.city_idx]))
d.sld_w.set_val(5.0); d.sld_d.set_val(7.0); d.sld_h.set_val(3.2)
check("room sliders set W x D x H", (d.width, d.depth, d.height) == (5.0, 7.0, 3.2),
      f"{d.width} x {d.depth} x {d.height}")
d.sld_repr.set_val(16)
check("repr-condition slider", d.n_repr == 16)

print("\n[3] edge bands toggle walls")
B = FD.BAND
d._on_press(ev(d.ax_grid, 16.0, -B / 2))                             # N band
d._on_press(ev(d.ax_grid, -B / 2, 16.0))                             # W band
check("clicking N and W bands closes them",
      d.facade_open == {'S': True, 'N': False, 'E': True, 'W': False}, str(d.facade_open))

print("\n[4] timeline slots, target box, Shift+drag fill, freehand paint")
d._on_press(ev(d.ax_tl, 10.2, 20.0))                                 # add 10:00
check("timeline click adds a 10:00 slot", len(d.markers) == 1 and d.markers[0]["hour"] == 10)
d._on_target_submit("500")
check("target box sets 500 lux", d.markers[0]["target"] == 500.0)
d._on_press(ev(d.ax_grid, 8.5, 10.5, key="shift"))
d._on_motion(ev(d.ax_grid, 23.5, 19.5))
d._on_release(None)
check("Shift+drag fills a 10x16 rectangle", int(d.markers[0]["zone"].sum()) == 160,
      str(int(d.markers[0]["zone"].sum())))
d._on_press(ev(d.ax_tl, 15.1, 20.0))                                 # add 15:00
d._on_target_submit("300")
d._on_press(ev(d.ax_grid, 4.5, 24.5))
for x in np.arange(4.5, 28.0, 1.0):
    d._on_motion(ev(d.ax_grid, x, 24.5))
d._on_release(None)
check("freehand drag paints a row of 24 cells on the 15:00 slot",
      len(d.markers) == 2 and int(d.markers[1]["zone"].sum()) == 24 and d.markers[1]["target"] == 300.0,
      str(int(d.markers[1]["zone"].sum())))

print("\n[5] Run Optimizer (real GPU solve)")
d._run_optimizer(None)
csvs = sorted(out_dir.glob("facade_allsides_climate_*.csv"))
pngs = sorted(out_dir.glob("facade_allsides_climate_*_result.png"))
check("facade CSV written", len(csvs) == 1, str(csvs))
check("unfolded-cross result PNG written", len(pngs) == 1 and pngs[0].stat().st_size > 20_000,
      str(pngs))
if csvs:
    binary = load_binary_wide(csvs[0])
    per = {s: int(binary[i].sum()) for i, s in enumerate(SIDES)}
    check("closed walls N and W have 0 windows", per["N"] == 0 and per["W"] == 0, str(per))
    check("open walls carry the glazing", per["S"] + per["E"] > 0, str(per))
    city = resolve_city("houston")
    opt = FacadeOptimizerAllSidesClimate()
    for m in d.markers:
        _, _, cond, _ = annual_conditions(city, m["hour"])
        c, w = reduce_conditions(cond, n_clusters=16)
        lux = opt.predict_lux(binary, append_room(c, 5.0, 7.0, 3.2))
        zone = m["zone"] > 0
        mean = float(np.average(lux[:, zone].mean(axis=1), weights=w))
        print(f"        (info) {m['hour']:02d}:00 zone mean {mean:.0f} lux against {m['target']:.0f}")

print(f"\n  outputs: {out_dir}")
print()
if FAILS:
    print(f"{len(FAILS)} FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("designer headless test passed")
