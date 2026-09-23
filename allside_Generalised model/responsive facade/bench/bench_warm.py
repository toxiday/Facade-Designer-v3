"""Does a warm start (previous facade as init) let a SHORT refine match a cold full solve?
If yes, moving one step costs ~0.5s instead of 2.6s and the facade morphs instead of jumping."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model trainer"))
import numpy as np, torch, torch.nn.functional as F
from optimize_facade_allsides_climate import (
    FacadeOptimizerAllSidesClimate, resolve_city, annual_conditions, append_room,
    facade_in, zone_penalty_sum, outside_glare_sum, tv_loss,
    GRID_N, N_FACADES, FACADE_ROWS, FACADE_COLS, GLARE_LUX)

opt  = FacadeOptimizerAllSidesClimate()
city = resolve_city("Houston")
_, _, cond5, days = annual_conditions(city, 12, north_angle=0.0)
i = [k for k, d in enumerate(days) if d[0] == 6 and d[1] == 21][0]
cond1 = append_room(cond5[i:i+1], 6.0, 8.0, 3.0)

def zones(people):
    """people: list of (col, row, radius_cells, lux)"""
    t = np.zeros((1, GRID_N, GRID_N), dtype=np.float32)
    yy, xx = np.mgrid[0:GRID_N, 0:GRID_N]
    for cx, cy, r, lux in people:
        t[0][((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r] = lux
    return t

def score(binary, tgt):
    lux = opt.predict_lux(binary, cond1)[0]
    occ = tgt[0] > 0
    per = {}
    for lv in np.unique(tgt[0][occ]):
        m = (tgt[0] == lv) & occ
        per[float(lv)] = (float(lux[m].mean()),
                          float(np.mean((lux[m] >= .75*lv) & (lux[m] <= 1.25*lv)) * 100))
    return per

SHORT = ((2, 30), (4, 30), (8, 40), (16, 30))

def run(tgt, sched=SHORT, tag=""):
    t = time.time()
    b, P, h = opt.optimize_coarse_to_fine(cond1, tgt, facade_open=(True,)*4,
                                          glare_weight=0.25, schedule=sched,
                                          eval_every=10, verbose=False)
    dt = time.time() - t
    print(f"  {tag:<34} {dt:5.2f}s  glazing {b.mean()*100:4.1f}%  " +
          "  ".join(f"[{int(k)}lx -> {v[0]:5.0f}, {v[1]:3.0f}% in-band]"
                    for k, v in sorted(score(b, tgt).items())), flush=True)
    return b, dt

print("\n=== 1 person ===")
run(zones([(16, 13, 5, 500)]), tag="cold, short")

print("\n=== 3 people, different targets (T is still 1 - same cost) ===")
p3 = zones([(7, 8, 4, 300), (16, 20, 4, 750), (26, 11, 4, 500)])
run(p3, tag="cold, short")
run(p3, sched=((2, 250), (4, 250), (8, 300), (16, 150)), tag="cold, DEFAULT schedule")

print("\n=== 5 people ===")
p5 = zones([(5, 6, 3, 300), (14, 6, 3, 500), (24, 8, 3, 1000),
            (9, 22, 3, 200), (25, 24, 3, 750)])
run(p5, tag="cold, short")

# ---- warm start: refine at full 16x32 from the previous solution -------------
def refine(X0, tgt, n_steps=60, lr=0.05, tau=0.6, tau_final=0.3, lambda_tv=0.02,
           glare_weight=0.25):
    """Single-stage refine at full res, warm-started from logits X0."""
    cond_t = torch.tensor(cond1, dtype=torch.float32, device=opt.device)
    tgt_t  = torch.tensor(tgt,   dtype=torch.float32, device=opt.device)
    n_occ   = (tgt_t > 0).float().sum().clamp(min=1.0)
    n_unocc = (tgt_t <= 0).float().sum().clamp(min=1.0)
    X = X0.clone().detach().requires_grad_(True)
    o = torch.optim.Adam([X], lr=lr)
    best_L, best_X = float("inf"), X.detach().clone()
    for s in range(n_steps):
        o.zero_grad()
        cur_tau = tau + (tau_final - tau) * (s / max(1, n_steps - 1))
        P   = torch.sigmoid(X / cur_tau)
        lux = opt._lux(cond_t, facade_in(P).unsqueeze(0))
        L = zone_penalty_sum(lux, tgt_t) / n_occ \
            + glare_weight * outside_glare_sum(lux, tgt_t, GLARE_LUX) / n_unocc \
            + lambda_tv * tv_loss(P)
        L.backward(); o.step()
        if s % 10 == 0 or s == n_steps - 1:
            with torch.no_grad():
                hard = (X > 0).float()
                lx = opt._lux(cond_t, facade_in(hard).unsqueeze(0))
                bl = float((zone_penalty_sum(lx, tgt_t) / n_occ
                            + glare_weight * outside_glare_sum(lx, tgt_t, GLARE_LUX) / n_unocc).item())
            if bl < best_L:
                best_L, best_X = bl, X.detach().clone()
    return (best_X > 0).float().cpu().numpy().astype(np.int32), best_X

print("\n=== warm start: person walks one step at a time ===")
tgt0 = zones([(8, 8, 4, 500)])
b0, _ = run(tgt0, tag="cold solve at start pos")
# recover logits from the binary (proxy: +-2 logit) as the warm start
X = torch.tensor(np.where(b0 > 0, 2.0, -2.0), dtype=torch.float32, device=opt.device)
for step, cx in enumerate(range(10, 20, 2)):
    tgt = zones([(cx, 8, 4, 500)])
    t = time.time(); bw, X = refine(X, tgt, n_steps=60); dt = time.time() - t
    ch = int((bw != b0).sum()) if step == 0 else int((bw != prev).sum())
    prev = bw
    print(f"  warm refine -> col {cx:2d}         {dt:5.2f}s  glazing {bw.mean()*100:4.1f}%  " +
          "  ".join(f"[{int(k)}lx -> {v[0]:5.0f}, {v[1]:3.0f}% in-band]"
                    for k, v in sorted(score(bw, tgt).items())) +
          f"  cells changed {ch}", flush=True)
print("\n  (compare: cold short solve at the final position)")
run(zones([(18, 8, 4, 500)]), tag="cold, short")
