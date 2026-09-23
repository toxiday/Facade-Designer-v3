"""Benchmark: how fast is a SINGLE-condition facade optimize?  That number decides
whether a live reactive app is possible at all."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model trainer"))
import numpy as np, torch
from optimize_facade_allsides_climate import (
    FacadeOptimizerAllSidesClimate, resolve_city, annual_conditions, append_room, GRID_N)

opt = FacadeOptimizerAllSidesClimate()
city = resolve_city("Houston")
label, lat, cond5, days = annual_conditions(city, 12, north_angle=0.0)
print(f"{label}  lat {lat}  {len(cond5)} days at 12:00")

# ONE condition = one specific date+time
i = [k for k, d in enumerate(days) if d[0] == 6 and d[1] == 21]
i = i[0] if i else 0
cond1 = append_room(cond5[i:i+1], 6.0, 8.0, 3.0)
print("cond:", cond1, "day:", days[i])

# a small occupancy zone: one person standing at (0.5W, 0.4D), 1.2 m radius, 500 lux
tgt = np.zeros((1, GRID_N, GRID_N), dtype=np.float32)
yy, xx = np.mgrid[0:GRID_N, 0:GRID_N]
cx, cy = 16, 13
tgt[0][((xx - cx) ** 2 + (yy - cy) ** 2) <= 5 ** 2] = 500.0
print("occupied cells:", int((tgt > 0).sum()))

def bench(name, fn):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.time()
    binary, P, hist = fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.time() - t
    lux = opt.predict_lux(binary, cond1)[0]
    occ = tgt[0] > 0
    print(f"\n### {name}: {dt:.2f}s   glazing {binary.mean()*100:.1f}%   "
          f"in-zone mean {lux[occ].mean():.0f} lux (target 500)   "
          f"in-band {np.mean((lux[occ] >= 375) & (lux[occ] <= 625))*100:.0f}%\n", flush=True)
    return dt

# 1. raw forward speed
fac = np.zeros((4, 16, 32), dtype=np.int32); fac[0, 4:10, 8:24] = 1
for _ in range(3): opt.predict_lux(fac, cond1)
torch.cuda.synchronize() if torch.cuda.is_available() else None
t = time.time()
for _ in range(50): opt.predict_lux(fac, cond1)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"\n### forward (T=1): {(time.time()-t)/50*1000:.1f} ms/call\n")

bench("c2f default schedule", lambda: opt.optimize_coarse_to_fine(
    cond1, tgt, facade_open=(True,)*4, glare_weight=0.25, verbose=False))

bench("c2f short (2:60,4:60,8:80,16:60)", lambda: opt.optimize_coarse_to_fine(
    cond1, tgt, facade_open=(True,)*4, glare_weight=0.25,
    schedule=((2, 60), (4, 60), (8, 80), (16, 60)), eval_every=10, verbose=False))

bench("c2f tiny (2:30,4:30,8:40,16:30)", lambda: opt.optimize_coarse_to_fine(
    cond1, tgt, facade_open=(True,)*4, glare_weight=0.25,
    schedule=((2, 30), (4, 30), (8, 40), (16, 30)), eval_every=10, verbose=False))

bench("direct optimize 200 steps", lambda: opt.optimize(
    cond1, tgt, facade_open=(True,)*4, glare_weight=0.25, n_steps=200, log_every=10**9))
