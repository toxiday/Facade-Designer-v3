# `facade_job/1` — the Rhino ↔ optimizer interface

One JSON file carries everything the facade optimizer needs. Grasshopper authors it,
`run_facade_job.py` consumes it, and the results come back as CSVs carrying explicit panel
geometry so Grasshopper can build Rhino geometry without re-deriving any index convention.

```
Rhino box + zone curves ──> gh_write_facade_job.py ──> job.json
                                                          │
                                    run_facade_job.py <────┘
                                            │
        facade_wide.csv + facade_panels.csv + result.json + result.png
                                            │
                     gh_read_facade.py <────┘
                                            │
                 merged exterior window surfaces on the room box's walls
```

A whole apartment goes through the same machinery one level up — see
**[Units](#units-many-rooms-at-once)**:

```
7 room boxes + apartment solid ─> gh_write_unit_jobs.py ──> 101.1.json … 101.7.json
                                                            + unit.json  (unit/1)
                                            │
                                    run_unit.py <────┘
                                            │
                        7 result folders + unit_summary.csv
```

---

## The one thing to get straight: frames

**Everything in the job — input and output — is ROOM-LOCAL.**

| | meaning | changes with rotation? |
|---|---|---|
| `S` | the wall at `y = 0` | no |
| `N` | the wall at `y = D` | no |
| `E` | the wall at `x = W` | no |
| `W` | the wall at `x = 0` | no |
| `grid` row 0 | the `+Y` (`"N"`) edge | no |
| `grid` col 0 | the `x = 0` (`"W"`) edge | no |
| panel coordinates | metres in the room's own frame | no |
| `faces_az` / `sides_compass` | true compass azimuth | **yes** |

`north_angle` rotates the **sun** into this frame. `placement` says where the frame sits in
Rhino world space. Because both are derived from the same box, they cannot drift apart.

> **`S` does not mean "south".** At `north_angle = 270` the wall facing true south is `E`.
> Asking for `sides: "S"` there glazes the **west**-facing wall. Read `sides_compass` from
> the Grasshopper component (or `result.json`) before choosing sides.

---

## Job file

```json
{
  "schema": "facade_job/1",
  "units": "m",
  "room":    { "width": 6.0, "depth": 8.0, "height": 3.0, "north_angle": 0.0 },
  "sides":   "SNEW",
  "climate": { "city": "USA_TX_Houston-Bush.Intercontinental.AP.722430" },
  "slots": [
    { "hour": 9,  "grid": [ /* 1024 floats */ ] },
    { "hour": 14, "grid": [ /* 1024 floats */ ] }
  ],
  "optimizer": { "repr_conditions": 48, "mode": "coarse_to_fine", "seed": 0 },
  "placement": { "origin": [0,0,0], "x_axis": [1,0,0], "y_axis": [0,1,0] },
  "output":    { "dir": null, "name": "job_001" }
}
```

Only `room` and `slots` are required; everything else defaults.

### `room`

| field | units | range | notes |
|---|---|---|---|
| `width` | m (X) | 2.5 – 10.0 | |
| `depth` | m (Y) | 2.5 – 12.0 | |
| `height` | m (Z) | 2.5 – 8.0 | |
| `north_angle` | deg | any | Ladybug convention, see below |

Out-of-range dimensions are **clamped by the model, not rejected** — so the runner warns
loudly and the Grasshopper component refuses outright. A 6000 mm box exported without unit
conversion becomes a 10 m room and produces a confident design for the wrong building.

### `north_angle`

Adopted verbatim from Ladybug (`ladybug/compass.py`): the **clockwise angle from the room's
`+Y` axis to true North**.

| `north_angle` | `+Y` (plan "up") | `S` faces | `N` faces | `E` faces | `W` faces |
|---|---|---|---|---|---|
| `0` | North | South | North | East | West |
| `90` | West | East | West | North | South |
| `180` | South | North | South | West | East |
| `270` | East | West | East | **South** | North |

Arbitrary (non-multiple-of-90) angles are fine. The model has no north input at all — it
only ever sees a sun vector in room coordinates, so rotating the building just supplies a
different sun vector. Measured coverage across all 170 climate files (672,603 lit vectors)
shows **36/36 azimuth bins occupied in every altitude band**, so no rotation falls into a
hole. Re-run `check_sun_coverage.py` if the climate library is expanded.

### `sides`

A string of room-local wall ids, e.g. `"SNEW"`, `"SE"`, `"S"`. Order is irrelevant. Walls
not listed are frozen fully solid and can never receive windows.

### `climate.city`

A **name**, resolved by `resolve_city()`: exact filename stem → exact label → unique
case-insensitive substring (`"houston"` works). An ambiguous substring raises and lists the
candidates. An integer index is accepted for back-compatibility but is fragile — it shifts
whenever a `.wea` is added — so job files should always carry the name.

In Grasshopper the name comes from **`gh_pick_city.py`**, a separate component that owns all
climate-library logic. Splitting it out is what lets you *see* the choice before committing
to a run — which city an index actually is, and, via `sun_hours`, whether the sun is even up
at the hours you are about to design for. The writer component takes only the resolved name,
so it needs no library path at all.

### `slots[]`

| field | meaning |
|---|---|
| `hour` | 0–23, local clock hour, evaluated across the whole TMYx year |
| `grid` | 1024 floats, **target lux per cell**, `0` = unoccupied |

`grid` is row-major, `k = row*32 + col`, **row 0 = the `+Y` edge, col 0 = the `x=0` edge**.
Cell centre in room-local metres: `x = (col+0.5)·W/32`, `y = (31−row+0.5)·D/32`, `z = 0.6`.

One design is optimized against **all** slots at once (concatenated on the condition axis),
so multiple slots express "this zone at 09:00 *and* that zone at 14:00".

Because `grid` is per-cell lux rather than a mask + scalar, **multi-level zones already
work**: nest a 600 lux outline inside a 200 lux one and both levels export. Overlapping
zones resolve by `max`.

> ### Cells left at `0` carry no comfort target — but they are no longer ignored
>
> Every term of `zone_penalty_sum` is masked by `target_lux > 0`, glare included. Left at
> that, the optimizer is not merely indifferent to unpainted floor — it **exploits** it:
> dumping surplus daylight where nothing is scored is the cheapest way to hit the in-zone
> target without overshooting. A measured design scored 0% in-zone glare while putting
> 30,276 lux on unpainted floor, with the unoccupied mean at *twice* the occupied mean.
>
> `optimizer.glare_weight` (default `0.25`) adds a separate, deliberately **subordinate**
> penalty on unoccupied cells above `glare_cap`. Set it to `0` for the old behaviour.
>
> `result.json` still reports glare twice — `zone_glare_pct` and `floor_glare_pct`.
>
> To give circulation space a real illuminance target rather than only a cap, paint it at
> a low lux (100–150). That buys the *full* comfort band, so darkness there is penalized
> too.

### `optimizer`

| field | default | notes |
|---|---|---|
| `repr_conditions` | 48 | k-means reduction of the year; `0` = use all days |
| `mode` | `coarse_to_fine` | or `direct` |
| `seed` | *(none)* | set it — the design is otherwise non-reproducible |
| `steps` | 800 | `direct` mode only |
| `glare_weight` | 0.25 | unoccupied-floor glare penalty; `0` = off |
| `glare_cap` | 3000 | lux threshold for that penalty |

**Sizing of `glare_weight`.** It is normalized by the *unoccupied* cell count, so it is
scale-free — a small zone in a big room does not silently attract more glare pressure than
a large one. `0.25` was measured, not guessed: at `glare_cap = 3000` the term's initial
magnitude is ≈0.087 against a ≈0.102 zone loss, putting it near 18% of the objective.
Measured A/B on the same job and seed:

| | glare off | glare on (0.25) |
|---|---|---|
| unoccupied mean lux | 844 | **688** (−18%) |
| unoccupied 99th pct | 16,129 | **12,100** (−25%) |
| unoccupied cell-days > 3000 lux | 2.83% | **1.88%** |
| **in-zone mean lux** | 427 | 420 (−2%) |
| **in-zone 99th pct** | 689 | 670 (−3%) |
| in-zone cell-days > 3000 lux | 0.000% | 0.003% |

The zone distribution moves by 1–3% while unoccupied glare drops by a quarter. Raising the
weight much above ~0.4 starts trading real zone performance — the point is to stop the
free-disposal exploit, not to co-optimize the whole floor.

Note the loss is a **mean** over cell-days, so a handful of extreme cell-days barely
registers: in-zone `>3000 lux` went from 0.000% to 0.003% here (about 3 cell-days out of
73,000, one day of the year). That mean-vs-peak behaviour is pre-existing in
`zone_penalty_sum`, not introduced by this term.

### `placement`

Echoed, never used by the optimizer. It exists so the Grasshopper reader can transform
room-local panel coordinates back into world space.

> **`placement` does NOT locate a room inside its unit.** It is written per room from
> whatever box was on the canvas at the time, and nothing ever checks one room against
> another. Measured across unit 101's seven files, the blocks do not agree: `101.2`×`101.7`
> overlap by 4.88 m², `101.3`×`101.6` by 4.30 m², and `101.4` sits 10 m from every other
> room. Use `unit.json` for room-in-unit position — see below.

---

## Units: many rooms at once

An apartment is a folder of room jobs plus a **`unit/1`** manifest. `gh_write_unit_jobs.py`
writes all of them from one Grasshopper component, and `run_unit.py` optimizes the set.

### Why a separate manifest

Room-in-unit position has to be a **measurement into one shared frame**, not a per-room
record. The apartment outline supplies that frame: each room's origin and axes are projected
onto it, so two rooms cannot disagree about where they are. The component then refuses any
layout where rooms overlap, a room falls outside the outline, or two rooms share a name.

### The apartment is a solid whose footprint is a polygon

`unit_boundary` is a **closed polysurface** — the apartment's floor polygon extruded and
capped — for the same reason a room is a Box rather than a rectangle: one piece of geometry
carries both the plan and the storey height, so the two cannot disagree.

| read from | how |
|---|---|
| footprint | the **bottom** cap's outer loop — rooms sit on the floor, so the floor outline is what their placement is checked against |
| storey height | the separation between the lowest and highest horizontal cap |

The solid is checked for actually being an extrusion:

| case | result |
|---|---|
| open (uncapped) polysurface | **refused**, naming the fix (`Cap`, or Cap Holes in GH) |
| flat — a surface, not a solid | **refused** |
| no horizontal cap in this frame | **refused**, suggesting `unit_plane` if the tilt is deliberate |
| top cap area ≠ base cap area | **flagged**; the base is used, and the log says so |
| base has inner loops (courtyard, shaft) | **flagged**; only the outer boundary is used |

A Box or an Extrusion is accepted and converted. A bare closed planar Curve is accepted as a
convenience but carries no height — `unit_height` or the tallest room then supplies it, and
the log says which. `unit_height`, if wired, overrides the solid and warns when the two
disagree; normally leave it empty.

Flats are L-shaped, notched and stepped far more often than they are rectangular, and every
check that depends on the footprint works on the polygon, never its bounding box:

- **Containment** is point-in-polygon *plus an edge-crossing test*. Across a concave notch,
  all four corners of a room can sit inside the flat while the room still bulges out through
  the gap — corners alone would pass it.
- **Coverage** uses the polygon's true area. On the seven-room test layout that is 95% of a
  137.5 m² flat; against the 157.2 m² bounding box the same rooms read 83%.
- A room **flush against an apartment wall** shares a whole edge with the boundary (two in a
  corner). Shared edges and touching corners are explicitly not crossings, so the normal case
  passes.

A polygon has no intrinsic axes the way a Box does, so the **in-plane orientation** is
**stated, not inferred**: `unit_plane` if wired, otherwise world XY, with the origin moved to
the footprint's bounding-box min corner.

### `unit.json`

```json
{
  "schema": "unit/1",
  "units": "m",
  "name": "101",
  "unit":      { "width": 13.72, "depth": 11.45, "height": 3.0, "area": 137.5 },
  "boundary_uv": [[0,0], [13.74,0], [13.74,7.54], [9.67,7.54],
                  [9.67,10.50], [5.68,10.50], [5.68,11.46], [0,11.46]],
  "climate":   { "city": "USA_TX_Houston-Bush.Intercontinental.AP.722430" },
  "placement": { "origin": [0,0,0], "x_axis": [1,0,0], "y_axis": [0,1,0] },
  "rooms": [
    { "name": "101.1", "width": 5.683, "depth": 3.607, "height": 3.0,
      "north_angle": 15.0, "sides": "E",
      "origin_uv": [0.0, 7.85], "x_axis_uv": [1.0, 0.0], "y_axis_uv": [0.0, 1.0],
      "job_file": "101.1.json", "result_dir": "…/outputs/101.1" }
  ]
}
```

`boundary_uv` is **the apartment's shape**, in unit-local metres, wound as drawn.
`unit.width`/`depth` are only its bounding box and exist to size the composite raster — do
not read them as the size of the flat; `unit.area` is the real floor area.

`origin_uv` / `x_axis_uv` / `y_axis_uv` are the **room-local → unit-local** transform, in
metres. A room-local point `(x, y)` lands at
`origin_uv + x·x_axis_uv + y·y_axis_uv`. Rotated rooms are fine — the axes carry the rotation,
and the overlap check uses true polygon intersection rather than axis-aligned bounds.

| frame | meaning |
|---|---|
| unit-local `(u, v)` | metres in the unit box's plane, origin at its min corner |
| room-local `(x, y)` | unchanged — what every job and every optimizer output speaks |

### Components

| component | bundles | into |
|---|---|---|
| `gh_make_zone.py` | curves + hour + lux | a `Zone` |
| `gh_make_room.py` | room box + name + sides + Zones | a `Room` |
| `gh_write_unit_jobs.py` | unit box + Rooms | all job JSONs + `unit.json` |

```
gh_make_zone ×N ─┐
                 ├─> gh_make_room (101.1) ─┐
room_box 101.1 ──┘                         ├─(Merge)─> rooms ─┐
gh_make_zone ×N ─┐                         │                  ▼
                 ├─> gh_make_room (101.2) ─┘        gh_write_unit_jobs
room_box 101.2 ──┘                             unit_boundary ──┘  (capped extrusion)
```

Room boxes and the apartment solid are the same kind of object one level apart — a room is a
Box, the flat is a capped extrusion of its floor polygon.

`rooms` must have **Access = List and type hint "No type hint"**, for exactly the reason
`zones` must — see the duck-typing note below. Unit-wide `city`, `north`, `seed`,
`repr_conditions`, `glare_weight` and `glare_cap` are set **once** on the unit writer and
applied to every room, which is the point: changing the city for an apartment is one edit,
not seven.

`room_rects` and `unit_outline` preview the measured layout in Rhino **before** `write` is
set, so a bad arrangement — or an outline that discretized differently than you expected — is
visible rather than discovered later.

### A unit is one self-contained folder

Results land **beside the jobs that produced them**, so a unit is one folder holding its
brief, its designs and its summary:

```
All Units/101/
├─ unit.json                         the manifest
├─ 101.1.json … 101.7.json           the briefs
├─ unit_summary.csv                  one row per room
├─ unit_101_zone_matrix.png          the brief, rows = rooms
├─ unit_101_zone_matrix_composite.png   the brief, rows = units
└─ 101.1/ … 101.7/                   one folder per room
     ├─ facade_wide.csv    the design, 2048 columns, 1 = window
     ├─ facade_panels.csv  the design as geometry — wire THIS into gh_read_facade
     ├─ result.json        metrics + sides_open + the job echoed back
     └─ result.png         unfolded facade cross + predicted-lux gallery
```

The redirect is carried by the **job file itself**: `gh_write_unit_jobs` stamps
`output.dir = <out_dir>/<room name>` into each job, and `run_facade_job.py` writes there. So
the destination travels with the brief rather than depending on a runner flag being
remembered. Wire `results_dir` on the unit writer to send results elsewhere.

A job whose `output.dir` is `null` still falls back to `outputs/<name>/`, so anything already
on disk keeps working. To redirect those without re-exporting:

```bash
python run_unit.py "../All Units/101" --out-root "../All Units/101"
```

### Running a unit

```bash
python run_unit.py "../All Units/101"
python run_unit.py "../All Units/101" --only 101.6 101.7 --no-png
```

One room's failure is recorded and the batch continues. `unit_summary.csv` carries one row per
room: glazing %, zone mean lux, underlit %, and both glare figures.

### The occupancy brief as a picture

```bash
python unit_zone_matrix.py "../All Units/101"                 # rows = rooms
python unit_zone_matrix.py "../All Units" --by unit           # rows = units, composited
python unit_zone_matrix.py "../All Units/101" --check-layout  # validate, draw nothing
```

Rooms × time-of-day, or whole units × time-of-day with every room composited into one
apartment plan at its real position. Uses `facade_viz.LUX_CMAP` at **0–3000 lux, identical to
`result.png`**, so the brief and the design can be read against each other. Unoccupied floor
is grey rather than the bottom of the ramp — `0` here means "nobody is here", not "0 lux".

---

## Rasterization: binary centre sampling

A cell is occupied **iff its centre falls inside a zone curve**. A curve crossing mid-cell
does not split it; the boundary snaps to cell centres.

This is the semantically correct rule, not a shortcut. The surrogate's floor output is 1024
**point sensors** at exactly those centres, and the loss scores each point against its own
target — so the question genuinely is yes/no per point.

**Fractional coverage would be a bug.** An edge cell 40% covered would become a
`0.4 × 400 = 160` lux *target*, and the optimizer would deliberately darken the zone edge.
There is no per-cell weight channel to express "partially occupied".

Consequences:

- Zone edges quantize to **± half a cell** (`W/32 × D/32`, anisotropic) — ±9 cm / ±12 cm for
  a 6×8 m room, ±16 cm / ±19 cm at the 10×12 m maximum.
- A zone thinner than one cell can miss every centre. Because that failure is silent, the
  component raises on any curve capturing zero cells, and reports
  rasterized-area / true-area per curve.

Preview `grid_pts` / `grid_vals` in Rhino before exporting — it shows exactly which cells
were captured.

---

## Outputs — `outputs/<name>/`

### `facade_wide.csv`
Header `S_Pix_0..511, N_Pix_0..511, E_Pix_0..511, W_Pix_0..511` plus one data row.
`k = row*32 + col`, row 0 = ceiling. **`1` = window.** Round-trips with the training
dataset's column layout.

### `facade_panels.csv`
One row per panel, all 2048 (not only windows, so solid wall can be built too):

```
side,row,col,pix_index,is_window,x0,y0,z0,x1,y1,z1,nx,ny,nz,faces_az
```

Every panel is a planar axis-aligned rectangle, so the min/max corner pair fully determines
it. Coordinates are **room-local metres**; normals are outward. `faces_az` is the only
compass-referenced column — it stays exact at non-cardinal rotations where an `S/N/E/W`
letter would be meaningless.

> **The CSV holds all 2048 panels; `gh_read_facade.py` does not build all 2048 surfaces.**
> It builds only the **exterior** walls, and merges adjacent window cells into one surface
> per connected region. See below.

| side | plane | col 0 → 31 | row 0 → 15 | outward normal |
|---|---|---|---|---|
| S | `y=0` | `x=0 → W` | ceiling → floor | `−Y` |
| N | `y=D` | `x=0 → W` | ceiling → floor | `+Y` |
| E | `x=W` | `y=0 → D` | ceiling → floor | `+X` |
| W | `x=0` | `y=0 → D` | ceiling → floor | `−X` |

### `result.json`
Verbatim echo of the job, plus resolved city/lat/lon, per-slot day counts, glazing %,
metrics, and **`sides_compass`** — the decoder ring that makes `facade_wide.csv`
unambiguous. Keep the pair together.

> **Every metric in `result.json` is the SURROGATE's account of its own output.** Gradient
> descent drives toward the extremes of the surrogate's prediction surface, so the optimizer
> is adversarial to its own network and accuracy measured on *random* facades does not
> transfer to *optimizer-selected* ones. Run `--verify` before treating these numbers as
> performance rather than intent.

### Verification outputs — only with `--verify`

`verify_facade_radiance.py` re-simulates the finished design in real Radiance and writes:

| file | content |
|---|---|
| `verification_report.json` | the three comparisons, per slot and pooled |
| `verify_radiance.png` | ground-truth lux gallery |
| `verify_surrogate.png` | prediction lux gallery — same layout, colour map and scale, so the pair reads as an A/B |

The report carries three rows, and they answer different questions:

| comparison | question | audience |
|---|---|---|
| Radiance vs target | does the design actually work? | designers |
| **Radiance vs surrogate** | is the model right *where the optimizer landed*? | the method claim |
| surrogate vs target | what the optimizer believed | context for the above |

Read the **in-zone** figures, not the whole-floor RMSE. The surrogate's error is concentrated
in a thin tail of direct-sun cells that usually sit outside the painted zone, so a whole-floor
RMSE can look alarming while in-zone MAE and the decision-level percentages agree closely.
The report prints both for exactly that reason.

**Scope.** Verification rebuilds the *isolated* room the model was trained on — no neighbours,
no slab overhangs, no self-shading. A disagreement here is a real model error. Comparing
against the room in real building context is a different experiment, and reporting a scope
mismatch as a model failure would be wrong.

Radiance's `-aa 0` ambient sampling is stochastic, so repeated verification runs differ
slightly. Use enough conditions (`--verify-conditions`, default 32) that the sampling noise
is small relative to the effect you are reading.

---

## Reading a facade back: exterior only, and merged

`gh_read_facade.py` used to return one Brep per grid cell on all four walls — 2048 surfaces
per room, **14,336 for a seven-room unit**, most of them interior walls that are entirely
solid. It now does neither.

### Exterior walls only

A wall left out of `sides` is frozen fully solid by the optimizer, so in a unit `sides` *is*
the statement of which walls are on the building envelope. The reader honours it, in
precedence order:

| rule | source |
|---|---|
| 1 | the `sides` input, if wired |
| 2 | `resolved.sides_open` in `result.json` — authoritative |
| 3 | inference: sides holding at least one window |

`info` says which rule fired and how many interior panels were skipped. Rule 3 is a fallback,
not the rule: an open side *can* end up with zero windows, so wire `result_json`.

### Adjacent cells merge into one surface

Window cells that touch edge-to-edge become a **single trimmed planar surface** built from the
region's outline, with inner loops for any enclosed solid cell. Measured on unit 101:

| room | `sides_open` | window cells | surfaces | |
|---|---|---|---|---|
| 101.1 | E | 79 | **3** | 26× fewer |
| 101.2 | E | 303 | **2** | 151× fewer |
| 101.3 | E | 155 | **4** | 39× fewer |
| 101.4 | S, E | 136 | **5** | 27× fewer |
| 101.5 | S, E | 155 | **6** | 26× fewer |

Method: each cell contributes its four edges wound counter-clockwise; every edge shared by two
cells of the same region cancels; what survives chains into closed, consistently-oriented
loops. Holes need no special case — an enclosed void simply comes back wound the other way.
Collinear vertices are then dropped, so a 32-cell-wide opening is a rectangle rather than 32
segments.

Two caveats worth knowing:

- Connectivity is **4-way**. Cells touching only at a corner share a point, not an edge;
  fusing them would produce a pinched, non-manifold surface, so they stay separate regions.
- Solid cells on exterior walls are merged the same way, so `walls` is also a handful of
  surfaces rather than hundreds.

Set `merge` to `False` for the old one-surface-per-cell behaviour.

---

## Window encoding — the persistent trap

Three encodings coexist. The authoritative table lives in
`optimize_facade_allsides_climate.py`:

| where | encoding |
|---|---|
| dataset / model input `{side}_Pix_k` | float, **`< 0.5` = window** |
| optimizer internal `P` | window probability, **`1` = window** |
| both output CSVs | **`1` = window** |

---

## Grasshopper wiring — four components, closed loop

```
gh_pick_city.py                          gh_write_facade_job.py
  library_dir ─┐                            room_box, north, sides
  search      ─┼─> city ───────────────────> city
  city_index  ─┘                             repr_conditions, seed
      ├─> matches, n_matches (Panel)         glare_weight, glare_cap (optional)
      ├─> location                           out_path, result_dir
      └─> sun_hours ──┐                      write, run, console
                      │ informs                       ▲
gh_make_zone.py       ▼                               │              │
  curves ─┐                                           │              │ panels_csv
  hour   ─┼─> zone ──────(Merge)───────> zones ───────┘              │ result_json
  lux    ─┘                              (List, NO type hint)        ▼
                                                               gh_read_facade.py
                                                    room_box ───> room_box
                                                                  sides / merge
                                                                  include_walls
                                                                     │
                                            windows / walls / window_curves
                                            window_sides / info
```

Wire **`result_json`** as well as `panels_csv` — that is where the reader gets `sides_open`,
and without it the exterior walls are only inferred.

For a whole apartment, `gh_make_room` + `gh_write_unit_jobs` replace the single writer; see
[Units](#units-many-rooms-at-once).

The writer emits **`panels_csv`**, `result_json` and `result_dir` — the paths the run *will*
produce — so the reader wires up directly instead of having a path typed in. They are
predictions: the files appear only once the optimizer has run, and the reader says so if asked
to read too early.

Wire the **same `room_box`** into both the writer and the reader. That is what puts the facade
back exactly on the box it was designed for, and the reader refuses to build if the CSV's
dimensions disagree with the box it is given.

One `gh_make_zone` component per zone; `Merge` them into `zones`. Each Zone bundles its own
curves, hour and lux, so the correspondence is structural rather than positional — the earlier
three-parallel-list form let a mis-ordered `Merge` silently reassign hours to the wrong curves.

A Panel wired to `zone` reads `Zone 09:00 @ 400 lux (3 curves)`, so the wire is self-describing.

> **`zones` must have Access = List and type hint = "No type hint".** A typed input coerces the
> object and destroys it. The writer duck-types the object rather than using `isinstance`,
> because each component loads its `.py` in its own module instance and therefore holds a
> *different* `Zone` class — `isinstance` would be `False` for every real zone. A plain
> `{"curves": [...], "hour": 9, "lux": 400}` dict works too.

**Multi-level zones**: two Zones at the *same* hour with different lux. `build_slots` merges by
hour taking `max`, so a 600 lux zone nested inside a 200 lux one exports both levels.

### Setting them up

All three are `#! python 3` script components. **Paste each file into its own component**, add
the inputs/outputs listed in its docstring, and you're done — a pasted script receives its
component inputs as module globals, and the trailer at the bottom of each file wires them
through automatically.

Two things follow from pasting rather than importing, both already handled:

- **A pasted script has no `__file__`**, so it cannot locate the project. Only the writer needs
  to (for the optional `run` toggle) — set the **`PROJECT_ROOT` constant at the top of
  `gh_write_facade_job.py`** to your project folder, or wire its `repo_root` input. Everything
  else works without any project path.
- **Each pasted component is its own namespace**, so the `Zone` class in the maker is a
  genuinely different class object from any the writer could see. This is why the writer
  duck-types zones instead of using `isinstance` — with pasting it is not a theoretical concern
  but the normal topology.

The components run inside **Rhino's Python**, which has no numpy and no torch — hence stdlib +
RhinoCommon only. The `run` toggle does not change that: it launches the project's `.venv` as a
**separate process**, so the heavy dependencies never enter Rhino.

The `.gh` file itself can live anywhere; nothing resolves relative to the Grasshopper document
or the working directory.

Alternatively, since each file exposes `gh_main(globals())`, a 5-line loader component can keep
the `.py` on disk as the source of truth so edits are picked up without re-pasting:

```python
#! python 3
import importlib.util
spec = importlib.util.spec_from_file_location("m", r"...\gh_pick_city.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = m.gh_main(globals())
city, matches, n_matches = r["city"], r["matches"], r["n_matches"]
city_list, location, sun_hours, wea_path = (
    r["city_list"], r["location"], r["sun_hours"], r["wea_path"])
```

**Picking a city.** Type into `search` (e.g. `houston`) — `n_matches` tells you the slider
range and `matches` lists them; `city_index` indexes the *filtered* list, so a unique search
means index 0. A search matching nothing raises rather than silently falling back.

**`sun_hours` before you choose `zone_hour`.** It reports, per hour, how many days of the year
the sun clears 5° — the exact filter `annual_conditions()` applies, so these are the numbers
the optimizer will act on. A slot whose hour has no sun is rejected by the runner, so this is
where to catch it. Houston reads `06:00 152 days`, `08:00 365 days`, `no sun at: 00…05, 19…23`.

---

## Running

```bash
cd "allside_Generalised model/model trainer"
python run_facade_job.py ../rhino\ bridge/example_job.json
python run_facade_job.py --list-cities

# optimize, then close the loop against real Radiance
python run_facade_job.py job.json --verify --verify-conditions 32

# or verify a design that was already produced
python verify_facade_radiance.py ../outputs/job_001 --conditions 32 --workers 22
```

Verification cost is roughly `slots x conditions x 1.2 s` on 22 workers (~17.5 s per
simulation single-core), so a two-slot job at 32 conditions is about 80 seconds.

Verification:

```bash
python "../rhino bridge/verify_rotation.py"     # north_angle sign — the load-bearing test
python "../rhino bridge/check_sun_coverage.py"  # rotated sun vectors are in-distribution
```
