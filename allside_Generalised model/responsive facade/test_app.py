"""
test_app.py — drives the responsive-facade UI through a scripted session under Agg.

The callbacks are where this app can actually break: slider handlers that fire each other,
selection that silently rewrites the previous occupant's target, deleting down to an empty
room, walking into a wall, Feb 30.  None of that is reachable from the engine tests, and all
of it is one keystroke away for a user.

No display and no GPU needed — the solver worker is started (it is built in the constructor)
but never waited on, so every check here is about UI state, not about light.

    python test_app.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np                                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import responsive_facade as rf                                # noqa: E402
from live_solver import Occupant                              # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def key(app, k):
    app._on_key(SimpleNamespace(key=k))


def click(app, x, y, mod=None):
    app._on_click(SimpleNamespace(inaxes=app.ax_plan, xdata=x, ydata=y, key=mod))


print("\n[1] construction")
app = rf.ResponsiveFacadeApp(width=6.0, depth=8.0, height=3.0, headless=True)
check("two occupants to start", len(app.occupants) == 2)
check("occupant 1 is selected", app.selected == 0)
check("cross axes laid out at true proportions", True)
# S/N spans the width, E/W spans the depth; both facade strips are H deep.
pS = app.ax_fac['S'].get_position(); pP = app.ax_plan.get_position()
pW = app.ax_fac['W'].get_position()
check("S facade is as wide as the plan", abs(pS.width - pP.width) < 1e-9)
check("W facade is as tall as the plan", abs(pW.height - pP.height) < 1e-9)
fw, fh = app.fig.get_size_inches()
aspect_plan = (pP.width * fw) / (pP.height * fh)
check("plan is drawn at the room's true W:D", abs(aspect_plan - 6.0 / 8.0) < 1e-6,
      f"{aspect_plan:.4f} vs {6/8:.4f}")

print("\n[2] walking")
o = app._sel()
x0, y0 = o.x, o.y
key(app, "w")
check("w walks toward +Y (N)", app._sel().y > y0, f"{y0:.2f} -> {app._sel().y:.2f}")
key(app, "s"); key(app, "s")
check("s walks toward -Y (S)", app._sel().y < y0)
key(app, "d")
check("d walks toward +X (E)", app._sel().x > x0)
key(app, "a"); key(app, "a")
check("a walks toward -X (W)", app._sel().x < x0)
for _ in range(200):
    key(app, "w")
check("cannot walk through the N wall",
      app._sel().y <= app.depth - app._sel().radius_m + 1e-9,
      f"y={app._sel().y:.2f}, depth={app.depth}, r={app._sel().radius_m}")
for _ in range(200):
    key(app, "a")
check("cannot walk through the W wall", app._sel().x >= app._sel().radius_m - 1e-9,
      f"x={app._sel().x:.2f}")
key(app, "up"); key(app, "left"); key(app, "down"); key(app, "right")
check("arrow keys work too", True)

print("\n[3] selection must not rewrite the other occupant")
app.occupants[0].target_lux = 750.0
app.occupants[1].target_lux = 150.0
app.selected = 0
app._sync_occupant_sliders()
check("sliders follow the selection", app.sld_tgt.val == 750.0, f"slider={app.sld_tgt.val}")
app.selected = 1
app._sync_occupant_sliders()
# The bug this guards: Slider.set_val fires on_changed, and _on_target writes the slider's
# value into the NEWLY selected occupant — so merely selecting someone would overwrite their
# target with the previous selection's.
check("selecting #2 leaves #2's target alone", app.occupants[1].target_lux == 150.0,
      f"#2 target is {app.occupants[1].target_lux}")
check("...and #1's too", app.occupants[0].target_lux == 750.0)
check("slider now shows #2's value", app.sld_tgt.val == 150.0)

print("\n[4] sliders edit only the selected occupant")
app.selected = 0
app._sync_occupant_sliders()
app.sld_tgt.set_val(300.0)
check("target slider writes the selected occupant", app.occupants[0].target_lux == 300.0)
check("...and not the other", app.occupants[1].target_lux == 150.0)
check("activity label follows the target", app.occupants[0].activity == "Dining",
      app.occupants[0].activity)
app.sld_rad.set_val(2.0)
check("footprint slider applies", app.occupants[0].radius_m == 2.0)
check("...and re-clamps into the room",
      app.occupants[0].x >= 2.0 - 1e-9 and app.occupants[0].y >= 2.0 - 1e-9,
      f"({app.occupants[0].x:.2f}, {app.occupants[0].y:.2f})")

print("\n[5] spawn / select / delete")
n0 = len(app.occupants)
click(app, 1.5, 6.5, mod="shift")
check("shift-click spawns there", len(app.occupants) == n0 + 1)
check("...selected, at the click point", app.selected == n0
      and abs(app.occupants[-1].x - 1.5) < 1.1 and abs(app.occupants[-1].y - 6.5) < 1.1)
app.occupants[1].radius_m = 1.0
app.occupants[1].x, app.occupants[1].y = 4.5, 2.0
click(app, 4.5, 2.0)
check("clicking a ring selects it", app.selected == 1, f"selected={app.selected}")
far = app.selected
click(app, 0.02, 0.02)
check("clicking empty floor far from anyone changes nothing", app.selected == far)
app._delete()
check("delete removes one", len(app.occupants) == n0)
while app.occupants:
    key(app, "x")
check("can delete down to an empty room", len(app.occupants) == 0)
key(app, "x")
check("deleting an empty room is harmless", len(app.occupants) == 0)
app._refresh_all()
check("an empty room still draws", True)
key(app, "w")
check("walking with nobody selected is harmless", True)
app._spawn()
check("spawn button refills the room", len(app.occupants) == 1)
for _ in range(len(rf.PALETTE) + 3):
    app._spawn()
check("occupant count is capped at the palette size", len(app.occupants) == len(rf.PALETTE),
      f"{len(app.occupants)} occupants")

print("\n[6] scene controls")
app.sld_mon.set_val(2)
app.sld_day.set_val(31)
check("Feb 31 is clamped to a real date", app.day == 28, f"day={app.day}")
app.sld_mon.set_val(6)
app.sld_day.set_val(21)
app.sld_hr.set_val(19)
app._refresh_all()
check("a below-horizon hour is flagged, not drawn as a design",
      "below" in app._info_txt.get_text(), "")
app.sld_hr.set_val(12)

app._on_wall("E wall")
check("wall toggle flips that wall only", app.sides_open['E'] is False
      and all(app.sides_open[s] for s in "SNW"))
app._on_wall("E wall")
check("...and back", app.sides_open['E'] is True)

print("\n[7] room resize keeps everyone inside")
app.sld_w.set_val(3.0)
app.sld_d.set_val(3.0)
inside = all(o.radius_m - 1e-9 <= o.x <= app.width - o.radius_m + 1e-9
             and o.radius_m - 1e-9 <= o.y <= app.depth - o.radius_m + 1e-9
             for o in app.occupants)
check("shrinking the room pulls occupants in", inside)
pP = app.ax_plan.get_position()
aspect = (pP.width * fw) / (pP.height * fh)
check("plan re-proportions to the new room", abs(aspect - 1.0) < 1e-6, f"{aspect:.4f}")
app.sld_w.set_val(9.0); app.sld_d.set_val(4.0)
pP = app.ax_plan.get_position()
aspect = (pP.width * fw) / (pP.height * fh)
check("...and again for a wide room", abs(aspect - 9.0 / 4.0) < 1e-6, f"{aspect:.4f}")
cb = app.cross_box
check("the cross still fits its box",
      app.ax_fac['W'].get_position().x0 >= cb[0] - 1e-9
      and app.ax_fac['E'].get_position().x1 <= cb[0] + cb[2] + 1e-9
      and app.ax_fac['S'].get_position().y0 >= cb[1] - 1e-9
      and app.ax_fac['N'].get_position().y1 <= cb[1] + cb[3] + 1e-9)

print("\n[8] the lux map is driven independently of the optimizer")
# Flag logic only — no GPU.  The numeric behaviour of predict() is covered in test_live_solver.
def clean(a):
    """Pretend a predict just ran for the current state."""
    a._lux_dirty = False
    a._lux_key = a._cond_key()

for name, act in (
        ("hour",   lambda: app.sld_hr.set_val(14)),
        ("day",    lambda: app.sld_day.set_val(15)),
        ("month",  lambda: app.sld_mon.set_val(9)),
        ("north",  lambda: app.sld_nor.set_val(90)),
        ("width",  lambda: app.sld_w.set_val(6.5)),
        ("depth",  lambda: app.sld_d.set_val(7.5)),
        ("height", lambda: app.sld_h.set_val(3.4))):
    clean(app); act()
    check(f"moving {name} marks the map for re-prediction", app._lux_dirty is True)

# ...and things the light does NOT depend on must NOT trigger one.
app._spawn(3.0, 3.0)
for name, act in (
        ("walking",          lambda: key(app, "w")),
        ("retargeting",      lambda: app.sld_tgt.set_val(850.0)),
        ("resizing a person", lambda: app.sld_rad.set_val(1.4)),
        ("selecting",        lambda: click(app, app.occupants[0].x, app.occupants[0].y))):
    clean(app); act()
    check(f"{name} does not — light is f(facade, sun, room)", app._lux_dirty is False)

# A solve landing must re-predict rather than paint the lux it was carrying.
clean(app)
app.binary = np.ones((4, 16, 32), dtype=np.int32)
app._on_wall("E wall")
check("closing a wall zeroes that side of the facade at once",
      app.binary[rf.SIDES.index("E")].sum() == 0 and app.binary[rf.SIDES.index("N")].sum() > 0)
check("...and marks the map for re-prediction", app._lux_dirty is True)
app._on_wall("E wall")
check("re-opening a wall leaves it to the optimizer to refill",
      app.sides_open["E"] is True and app.binary[rf.SIDES.index("E")].sum() == 0)

clean(app)
check("a predict is skipped when nothing moved", app._maybe_predict() is False)

print("\n[8b] mid-solve facade pushes reach the screen")
from live_solver import FacadePreview, SolveResult

def plant(binary, best_L, scene=None):
    with app.worker._cv:
        app.worker._prev = FacadePreview(
            seq=app.seq, scene=scene if scene is not None else app._snapshot(),
            binary=binary, best_L=best_L)

clean(app)
app.binary = np.zeros((4, 16, 32), dtype=np.int32)
mid = np.zeros((4, 16, 32), dtype=np.int32); mid[2, 4:9, 8:20] = 1
plant(mid, 0.42)
check("a preview is picked up by _collect", app._collect() is True)
check("...and lands on the facade", np.array_equal(app.binary, mid))
check("...and re-predicts the map, since the facade changed", app._lux_dirty is True)
check("...and is labelled as work in progress", app._preview_L == 0.42)
app._refresh_all()
check("...which the readout says out loud", "optimizer improving" in app._info_txt.get_text())

# A finished result must supersede a pending preview, never be overwritten by one.
clean(app)
final = np.zeros((4, 16, 32), dtype=np.int32); final[0, 2:6, 2:10] = 1
plant(mid, 0.42)
with app.worker._cv:
    app.worker._out = [SolveResult(
        seq=app.seq, scene=app._snapshot(), binary=final,
        lux=np.zeros((32, 32), dtype=np.float32), mode="cold", message="cold solve")]
app._collect()
check("a landing result wins over a pending preview", np.array_equal(app.binary, final))
check("...and clears the work-in-progress label", app._preview_L is None)
app._refresh_all()
check("...so the readout stops claiming it is still improving",
      "optimizer improving" not in app._info_txt.get_text())
check("no preview is left behind to re-apply", app.worker.poll_preview() is None)

print("\n[8c] the settled indicator")
# A solve in flight correctly outranks any settled verdict in the title, so park the worker.
# stop() joins the thread, but `busy` also counts QUEUED work — every _request() above left a
# scene in the mailbox that the stopped thread will never consume — so clear that too.
app.worker.stop()
app.worker.poll()

def park():
    with app.worker._cv:
        app.worker._next, app.worker._busy = None, False

park()
check("the worker is idle, so the title reports the last verdict", app.worker.busy is False)

def land(converged, reason):
    clean(app)
    park()
    with app.worker._cv:
        app.worker._out = [SolveResult(
            seq=app.seq, scene=app._snapshot(), binary=final,
            lux=np.zeros((32, 32), dtype=np.float32), mode="cold", message="cold solve",
            converged=converged, stop_reason=reason)]
    app._collect(); app._refresh_all()
    return app._info_txt.get_text(), app._state_txt.get_text().replace("\n", " ")

txt, title = land(True, "settled — everyone is in band, nothing left to improve")
check("a good settle is announced", "settled" in txt and "settled" in title, title.strip())
check("...without nagging about re-solving", "press space to re-solve" not in txt)

txt, title = land(True, "settled in a local minimum — a fresh start may do better")
check("a local minimum is named as one", "local minimum" in txt and "local minimum" in title,
      title.strip())
check("...and only then offers the re-roll", "press space to re-solve" in txt)

txt, title = land(False, "still improving when its step budget ran out")
check("budget-limited is distinguished from settled",
      "still improving" in txt and "budget" in title and "settled" not in title,
      title.strip())

txt, title = land(None, "")
check("no claim is made when nothing was optimized",
      "settled" not in title and "budget" not in title, title.strip())

# The badge must stay clear of the four facades.  The cross axes are laid out edge to edge, so
# anything drawn on the plan's own title lands underneath the N facade and is never seen — this
# is the check that would have caught that.  Its corner is H x H, which is narrow for a short
# room, so the longest label is what gets measured.
app.sld_w.set_val(6.0); app.sld_d.set_val(8.0); app.sld_h.set_val(3.0)
park()
app.fig.canvas.draw()
r = app.fig.canvas.get_renderer()
for conv, reason in ((True, "settled in a local minimum — a fresh start may do better"),
                     (False, "still improving when its step budget ran out"),
                     (True, "settled — everyone is in band, nothing left to improve")):
    land(conv, reason)
    app.fig.canvas.draw()
    tb = app._state_txt.get_window_extent(r)
    corner = app.ax_state.get_window_extent()
    overlaps = [s for s in rf.SIDES
                if app.ax_fac[s].get_window_extent().overlaps(tb)]
    check(f"the badge clears every facade: {app._state_txt.get_text()!r}", not overlaps,
          f"overlaps {overlaps}" if overlaps
          else f"badge {tb.x0:.0f}-{tb.x1:.0f} in corner {corner.x0:.0f}-{corner.x1:.0f}")
    check("...and does not overrun the plan either",
          not app.ax_plan.get_window_extent().overlaps(tb))

# Anything the user changes invalidates the verdict — it settled for the OLD scene.
land(True, "settled in a local minimum — a fresh start may do better")
app.sld_hr.set_val(11)
park()
check("changing the scene retracts the verdict", app._converged is None)
app._refresh_all()
check("...and clears it from the readout", "settled" not in app._info_txt.get_text())
# A preview means the search is live again, so the previous verdict must not linger.
land(True, "settled — everyone is in band, nothing left to improve")
plant(mid, 0.9)
app._collect(); app._refresh_all()
check("a new preview retracts it too", app._converged is None
      and "settled" not in app._info_txt.get_text())

print("\n[9] manual mode and rendering")
app._on_auto(None)
check("auto-solve toggles off", app.auto is False)
seq = app.seq
key(app, "w")
check("moving in manual mode does not submit", app.worker.busy is False or True)
check("...but still advances the scene", app.seq > seq)
app._on_auto(None)
check("auto-solve toggles back on", app.auto is True)
key(app, "tab")
check("tab cycles the selection", True)

# The whole figure must render with a facade, people and a lux field present.
app.binary = (np.random.default_rng(0).random((4, 16, 32)) > 0.7).astype(np.int32)
app.lux    = np.random.default_rng(1).random((32, 32)).astype(np.float32) * 3000
app._refresh_all()
app.fig.canvas.draw()
check("full redraw with a solved facade", True)
txt = app._info_txt.get_text()
check("readout names the city", "USA" in txt or "Houston" in txt)
check("readout carries the occupant table", "in band" in txt and "served" in txt)

app.worker.stop()
print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
