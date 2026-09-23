# Responsive Facade

A room whose facade re-solves itself around the people inside it.

The rest of this project designs a facade **once**, for a whole year, from a brief of scheduled
occupancy zones. This inverts that: the occupancy is **live**, and the facade is re-optimized as
it changes. The premise is a building skin made of individually operable panels plus a sensor
that knows where people are and what they are doing. Each activity implies a required
illuminance; the facade opens exactly the panels that put that much daylight on that patch of
floor, for the sun that is in the sky at that moment, and opens a different set when the person
moves.

Only the **sensing** is faked — you are the camera. WASD walks the selected occupant, a slider
says what they are doing. Everything downstream is the production pipeline: the same TMYx
climate library, the same solar position, the same transformer surrogate, the same gradient
optimizer, the same comfort band, the same lux colour ramp as `result.png`. The facade you watch
assemble itself is a real optimized design, not an animation.

---

## Run it

```
cd "<repo>\allside_Generalised model\responsive facade"
"<repo>\.venv\Scripts\python.exe" responsive_facade.py
```

Options (all optional — everything is also a slider once the window is up):

```
--city Boston --month 3 --day 21 --hour 15 --north 30
--width 7 --depth 9 --height 3.2
--screenshot out.png       solve once, save the window, exit (no display needed)
```

First launch takes a few seconds to load the surrogate onto the GPU.

## Controls

| | |
|---|---|
| `W A S D` / arrows | walk the selected occupant |
| click a ring | select that occupant |
| shift-click the plan | spawn a new occupant there |
| `Spawn` / `Delete` / `x` | add or remove an occupant (max 8) |
| `Tab` | cycle the selection |
| `space` / `Re-solve` | force a fresh **cold** solve |
| `p` | save a PNG into `captures/` |
| sliders | city, date, hour, north angle, room W/D/H, and the selected occupant's target lux and footprint |
| `S/N/E/W wall` | which walls are allowed to carry windows |
| `auto-solve` | off = the scene updates but nothing is optimized until you press space |

## Reading the screen

- **Centre** — the floor plan, coloured by *predicted* illuminance on the same 0–3000 lux ramp
  the annual `result.png` uses, so the two are directly comparable. It re-renders as soon as you
  move the sun or the room, without waiting for the optimizer — see below.
- **The rings** — each occupant's footprint. Ring colour answers *is the building serving this
  person*: green ≥ 75% of their footprint inside the comfort band, amber ≥ 40%, red below. The
  coloured dot in the middle is identity; the dashed ring is the current selection. The label
  reads `asked → getting`.
- **Around it** — the four facades unfolded at true W×D×H proportions. White = open panel,
  black = solid. A green border means that wall may carry windows, grey means frozen solid. The
  heading gives the wall's true compass azimuth, which is not its room-local letter once
  `north angle` is non-zero.

## What makes it fast enough to be live

The annual job solves against ~48 representative climate conditions clustered out of 365 days,
which is what makes it a multi-minute run. A live facade only has to answer for the sun that is
in the sky **right now**: one condition. Measured on an RTX 5070 Ti:

| | cost |
|---|---|
| surrogate forward pass (T=1) | 8.4 ms |
| **cold** solve — full coarse-to-fine | **2.6 s** |
| **warm** refine — after someone moves | **1.1 s** |

**Multiple occupants are free.** Everyone is painted into one 32×32 target grid, so five people
with five different target levels cost exactly what one person costs — the batch dimension is
the number of sun conditions, not the number of people.

**Cold vs warm.** Anything that changes the room or the sun (W/D/H, city, date, hour, north
angle, which walls may open) invalidates the search and forces a cold solve. Anything that only
moves or retargets people is a warm refine continued from the previous logits. Warm is not a
quality compromise — measured against a cold solve of the same scene:

| move | warm (1.1 s) | cold (2.4 s) |
|---|---|---|
| realistic 0.3 m step | **94.9%** in band | 89.7% |
| exaggerated 1.5 m jump | 77.9% | 79.4% |

Warm is also the better *picture*: it morphs the facade by a few dozen panels instead of jumping
to an unrelated design, because it continues the same descent rather than restarting it.

**Requests coalesce.** Key-repeat fires a scene every ~30 ms while a solve takes ~1.1 s, so the
worker keeps a single-slot mailbox that overwrites rather than a queue. Measured over a 6-second
held key: 198 scenes submitted, 5 solved — 40× coalescing, and the facade chases where the
person *is* instead of replaying where they were.

**The lag is real.** Hold a movement key and you walk at ~3.6 m/s, which outruns the solver; the
readout will show you standing in whatever light you skidded into while the facade catches up a
second later. That is the honest behaviour of a building with moving parts, and watching the
number recover is most of the point. Tap the keys to walk at a pace it can track.

## The lux map is not gated on the optimizer

Predicting the light for a facade you already have is one forward pass — 8 ms idle, 26 ms while a
solve is saturating the GPU. Designing a new facade is 1.1–2.6 s. So the map is re-predicted
independently: move the sun or the room and the floor re-renders on the next 80 ms UI tick,
against whatever facade is currently on the building, while the optimizer keeps working.

This is what makes the demo argue its own case. Drag the Hour slider and you are watching a
**static** facade meet a moving sun — the same thing every real building does. Measured on one
09:00-designed facade held fixed across a June day in Houston:

| hour | occupant asked | getting |
|---|---|---|
| 08:00 | 500 lx | 257 |
| **09:00** | 500 lx | **482** ← designed for this moment |
| 12:00 | 500 lx | 664 |
| 14:00 | 500 lx | 1182 |
| 16:00 | 500 lx | 596 |

Byte-identical facade throughout. A second later the responsive facade lands and puts them back
in band — which is the entire argument, made visible in one slider drag.

The readout says so outright when the two disagree: *facade designed for 06/21 09:00 — showing it
under 06/21 16:00*.

Three rules keep this honest:

- **The map is a function of `(facade, sun, room)`, never of the occupants.** Walking someone
  around cannot change the light; only the numbers they are standing in change, and those already
  rescore every redraw. Predicts fire on the sun/room controls, not on movement.
- **A landing solve supplies the facade, not the map.** A solve that started at 09:00 and returns
  after you have dragged to 15:00 carries 09:00 light; taking its facade and re-predicting costs
  one extra forward pass and makes that class of staleness unrepresentable.
- **Closing a wall closes those panels at once**, before the re-solve. That is something a real
  facade can actually do immediately, unlike re-deciding where the remaining windows go.

## The facade converges on screen

The optimizer does not sit on the old design until it finishes. Each time it improves on its
best binary it pushes that design to the screen, so you watch the facade assemble itself — and
the floor re-predicts from it, so the occupants' numbers converge with it. One cold solve,
measured through the app's own tick loop:

| t | glazing | occupant asked → getting | ring |
|---|---|---|---|
| 1.2 s | 14.4% | 500 → 574, 200 → 596 | red |
| 1.9 s | 8.1% | 500 → 500, 200 → 247 | amber |
| 2.6 s | 8.3% | 500 → 531, 200 → 210 | green |
| 3.2 s | 6.8% | 500 → 450, 200 → 181 | green |
| 3.9 s | 6.7% | **final** — 500 → 441, 200 → 199 | green |

A cold solve usually opens with one genuinely bad design, straight out of the 2×4 coarse stage —
2000+ lux on someone who asked for 500. That is not a glitch; it is the optimizer starting in the
wrong basin and climbing out, and the readout labels it (*optimizer improving — best so far
L 5.6879*). Watching the rings go red → amber → green is the clearest statement of what the
surrogate is buying.

### When it stops

A badge in the cross's top-left corner says what the optimizer is doing, and the readout spells
it out:

| badge | meaning |
|---|---|
| `● solving…` | still searching |
| `■ settled` | stopped improving, and everyone is in band — nothing left to do |
| `■ settled / local minimum` | stopped improving, but the brief is not met. Press space to re-solve from a new random start |
| `▸ stopped on step budget` | still gaining when the step count ran out |

Any change to the scene retracts the verdict — it settled for the *old* question.

**How "settled" is decided, and why not the obvious way.** The optimizer has a `bin_patience`
early-stop, but it counts 12 flat *evaluations* and at `eval_every=10` against `SHORT_SCHEDULE`'s
30–40 step stages there are only 3–4 evaluations per stage, so it is structurally unreachable —
measured, it fired 0 times in 12 solves. Lowering the patience to make it fire costs quality
(93.4% in-band at patience 2 against 95.2% at the default), which is a bad trade for a
nicer-looking indicator.

The signal used instead needs no change to the search: **did the finest 16×32 stage beat the best
any coarser stage had already found?** If it could not, more of this search will not help. Across
16 solves that fires 44% of the time, and it is *predictive* — settled solves averaged 90.7%
in-band against 95.3% for still-improving ones, so it flags exactly the runs where starting over
is worth it.

Two rules keep the pushes from becoming noise:

- **A push must be an improvement.** The optimizer's callback fires on a step schedule, not on
  progress, so most calls carry the design already on screen. Without this gate, measured, one in
  four cold pushes repainted an identical facade.
- **At most one push every `PREVIEW_COOLDOWN` seconds (0.5).** The rate limit is global, not
  per-solve, so a burst of fast warm refines while you walk cannot exceed it either — measured at
  1.1 pushes/s sustained against a 2.0/s ceiling. Final results ignore the cooldown: they are the
  answer, not a progress report, and a landing result also discards any pending preview so the
  facade can never step backwards.

## Files

| file | what it is |
|---|---|
| `live_solver.py` | the engine — single-moment sun, occupants → target grid, cold/warm solve, background worker. Imports no matplotlib, so it is testable headlessly and reusable. |
| `responsive_facade.py` | the app — layout, input, drawing. |
| `test_live_solver.py` | engine checks. `--gpu` also runs real solves (~25 s). |
| `test_app.py` | drives the UI through a scripted session under Agg — no display, no GPU. |
| `captures/` | PNGs saved with `p`. |

```
python test_live_solver.py --gpu
python test_app.py
```

## Conventions carried over from the rest of the project

These are the same everywhere and are worth not re-deriving:

- **Room-local wall ids.** `S` = the y=0 wall, `N` = y=D, `E` = x=W, `W` = x=0. They are only
  compass directions when `north angle` is 0.
- **The room never rotates — the sun does.** `north angle` is Ladybug's convention (clockwise
  from room +Y to true North) and is applied by rotating the sun vector into the room frame.
- **Floor grid** 32×32, row 0 = the +Y (N) edge, col 0 = the x=0 (W) edge.
- **Facade grid** 4 sides × 16 rows × 32 cols, row 0 = ceiling, `1` = window.
- **Occupant footprints rasterize by cell centre**, the same rule the Grasshopper zone
  rasterizer uses, so a zone drawn here covers the same cells as one drawn there.

Two behaviours that are deliberate and will look like bugs if you do not know them:

- **Below 5° solar altitude nothing is designed.** The surrogate was trained only on hours with
  altitude > 5° — the filter sits in both the dataset generator and the optimizer — so at a
  low sun the app closes the facade and says why rather than feeding an out-of-distribution sun
  vector and drawing whatever comes back. Same limitation the annual runner records in
  `result.json`'s warnings.
- **Where two people overlap, the nearer one owns the cell.** Scheduled zones merge with
  `np.maximum` because that is right for one person doing two things at different hours. It is
  wrong for two people standing together at the same instant, where taking the max would
  silently hold someone who asked for 150 lux at their neighbour's 1000.

## Where this would go next

The gap between this and the real system is the sensing and the hardware, not the optimization:

1. **Replace the keyboard with a camera.** Person detection plus a homography from the camera
   plane to the floor plane gives (x, y); an activity classifier gives the target lux. The scene
   the solver consumes is already exactly that — position, target, footprint — so nothing
   downstream changes.
2. **Actuate.** The output is already a binary panel grid; `gh_read_facade.py` in `rhino bridge`
   already turns one into merged panel geometry.
3. **Add hysteresis.** Real panels have a duty cycle and a travel time. The optimizer currently
   re-decides every panel freely; a term penalising change from the current physical state would
   trade a little illuminance accuracy for far less actuation, and is the obvious next
   experiment.
