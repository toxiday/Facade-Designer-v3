"""Step 0: how long does one predict_lux take while a cold solve is hammering the GPU?
Under ~40 ms -> the predict can run synchronously on the UI thread."""
import sys, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from live_solver import (LiveSolver, Occupant, Scene, resolve_city, single_condition,
                         append_room, SHORT_SCHEDULE)

s = LiveSolver(verbose=False)
city = resolve_city("Houston")
W, D, H = 6.0, 8.0, 3.0
cond5, _ = single_condition(city, 6, 21, 12)
cond = append_room(cond5, W, D, H)
fac = (np.random.default_rng(0).random((4, 16, 32)) > 0.85).astype(np.int32)

def timed(n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        s.opt.predict_lux(fac, cond)
        out.append((time.perf_counter() - t) * 1000)
    return np.array(out)

for _ in range(10):
    s.opt.predict_lux(fac, cond)

idle = timed(60)
print(f"\nIDLE      median {np.median(idle):6.1f} ms   p95 {np.percentile(idle,95):6.1f} ms   "
      f"max {idle.max():6.1f} ms")

# Now hammer the GPU from another thread with a real cold solve, and predict from THIS thread.
scene = Scene(city_path=city, month=6, day=21, hour=12, north_angle=0.0,
              width=W, depth=D, height=H, sides_open=(True,)*4,
              occupants=(Occupant(0, 2.0, 4.0, 500.0), Occupant(1, 4.5, 6.0, 200.0)))
done = threading.Event()

def solve():
    try:
        s.solve(scene, force_cold=True)
    finally:
        done.set()

th = threading.Thread(target=solve, daemon=True); th.start()
time.sleep(0.4)                      # let the solve get going
busy = []
while not done.is_set():
    t = time.perf_counter()
    s.opt.predict_lux(fac, cond)
    busy.append((time.perf_counter() - t) * 1000)
    time.sleep(0.08)                 # the app's UI tick rate
th.join()
busy = np.array(busy)
print(f"DURING    median {np.median(busy):6.1f} ms   p95 {np.percentile(busy,95):6.1f} ms   "
      f"max {busy.max():6.1f} ms   ({len(busy)} samples)")
print(f"\n=> {'SYNCHRONOUS on the UI thread is fine' if np.percentile(busy,95) < 40 else 'NEEDS its own thread'}")
