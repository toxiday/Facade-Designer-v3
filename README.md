# Differentiable Facade Designer

A neural surrogate of Radiance daylight simulation, run **backwards** as a design tool.

Normally a designer draws a facade and then simulates it. Here the designer paints the
illuminance they want onto a floor plan: *500 lux at this desk at 10:00, 300 lux on the sofa at
15:00*. Gradient descent through a frozen, differentiable model of the room then finds the
facade that delivers it. The result is checked independently in Radiance.

The surrogate is a 31.2M-parameter transformer. It predicts the 32×32 floor illuminance of a
room from its four facades, its size and the sun and sky conditions. It covers any room from
2.5 to 12 m, all four walls, and a 170-city climate library.

> Personal research project. The neural network and much of the code were written with AI
> assistance; the numbers in this README are the ones the repository can reproduce itself (see
> [Reproducible results](#reproducible-results)).

---

## What's here

```
allside_Generalised model/
  model trainer/        the surrogate, the optimizer, the designer GUI, job runners, verification
  responsive facade/    live demo: the facade re-solves as occupants move (+ bench/ timing scripts)
  rhino bridge/         Grasshopper components: rooms, zones, city picker, job writers, reader
  data generator/       Radiance dataset generator (gendaylit Perez sky -> rtrace)
  climate data/         170 TMYx .wea files + the downloader that built them
  datasets/             a 264-row sample CSV (the 8.4 GB training set is not in git)
  models/               trained weights go here (GitHub Release asset, see Setup)
  outputs/              two reference runs, each with its Radiance verification report
  tests/                integration tests + run_all.py
evidence/ablation/      transformer-vs-UNet ablation (script, legacy models, results)
```

## Setup

Tested on Windows 11, Python 3.12, an RTX 5070 Ti, `torch 2.11.0+cu128`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

**Weights.** Download `best_facade_allsides_climate_genroom.pth` from the
[v1.0 release](../../releases/tag/v1.0) into `allside_Generalised model/models/`.
SHA-256: `0be1a5710ffbe7cec4b90ae3fe4e598b42892252d43c081270a8bb3940b30e61`

**Radiance** is only needed to generate data and to run `--verify`. Install it to
`C:\Radiance\bin` (other standard locations are searched too).

Check everything:

```powershell
.venv\Scripts\python "allside_Generalised model\tests\run_all.py"
```

## Use

| | |
|---|---|
| **Designer GUI**: paint zones on a timeline of hours, pick walls and city, optimize over the year | `python "allside_Generalised model/model trainer/facade_designer_allsides_climate.py"` |
| **Responsive facade**: walk occupants around with WASD, and the facade re-solves live | `python "allside_Generalised model/responsive facade/responsive_facade.py"` (see its [README](allside_Generalised%20model/responsive%20facade/README.md)) |
| **A job file**: the format Grasshopper writes | `python "allside_Generalised model/model trainer/run_facade_job.py" "allside_Generalised model/rhino bridge/example_job.json" [--verify]` |
| **A whole apartment unit** (a folder of job files) | `python "allside_Generalised model/model trainer/run_unit.py" <unit folder>` |
| **Grasshopper**: see [FACADE_JOB_SCHEMA.md](allside_Generalised%20model/rhino%20bridge/FACADE_JOB_SCHEMA.md). Set `PROJECT_ROOT` (this folder) and `MODEL_ROOT` (your Rhino project) at the top of the two `gh_write_*` scripts | |
| **Evaluate the surrogate** (needs the training CSV) | `python "allside_Generalised model/model trainer/evaluate32x32.py"` |

## How it works

1. **Data.** Radiance renders rooms with random size (W 2.5–10, D 2.5–12, H 2.5–8 m), random
   facades on 1–4 walls, and a random city and daytime hour (sun altitude > 5°). The sky is
   `gendaylit` Perez; `rtrace -ab 6 -ad 1024` samples 32×32 floor sensors at 0.6 m. Every wall
   is a 16×32 grid of cells, and a cell is glass if its value is < 0.5.
2. **Surrogate.** A transformer: 512 facade-patch tokens → 256 floor queries. It is conditioned
   on an 8-vector (sun direction, DNI, DHI, W/D/H) through FiLM before every layer, and trained
   on sqrt-lux with a glare-aware loss.
3. **Optimizer.** It relaxes the facade to per-cell window probabilities and descends through
   the frozen model, coarse to fine (2×4 → 4×8 → 8×16 → 16×32 logit grids). Every step is
   scored on the **binarised** design, and the best binary design is the one shipped. Model
   inputs stay at the training class means (0.25 / 0.75), because hard 0/1 inputs are out of
   distribution.
4. **Verification.** `verify_facade_radiance.py` rebuilds the shipped design in Radiance with
   the generator's own scene builder and compares Radiance, the surrogate and the target.

"Annual" in this project means one hour slot across every sunny day of the year, reduced by
weighted k-means to 48 representative conditions. **It is not LM-83 sDA.**

## Reproducible results

All measured by the repository's own scripts on 2026-09-22.

| | | reproduce with |
|---|---|---|
| Held-out test split (60k of 400k, seed 23) | SSIM 0.9923, R² 0.9888, glare F1 99.3%, usable-zone MAE 39.6 lux | `evaluate32x32.py` |
| 48 **new** Radiance samples (`--seed 20260922`; the training set used 451) | SSIM 0.992, R² 0.972, usable-zone MAE 40.5 lux | data generator, then `evaluate32x32.py --csv <new.csv> --all` |
| Designed facade re-simulated in Radiance | example job: zone mean 401 lux in Radiance for a 400 lux target; surrogate vs Radiance MAE 21 lux, bias −8 lux | `run_facade_job.py --verify` |
| Transformer vs conv UNet at matched params (~8M, 20×20 single-facade data, one seed) | MAE 460.9 vs 676.3 lux (−31.9%) | `evidence/ablation/` |
| Sun-vector coverage of the climate library | 36/36 azimuth bins in every altitude band | `rhino bridge/check_sun_coverage.py` |
| Live re-solve, one sun condition | forward pass ~8.5 ms, cold solve ~2.9 s, warm refine ~1.3 s | `responsive facade/bench/` |

The ablation supports an empirical claim only: at matched parameter count, the attention model
beat a convolutional encoder-decoder. It does **not** show that attention learns a
light-transport operator. The UNet's own bottleneck attention made no difference.

## Conventions

- Room-local wall ids: `S` = y=0, `N` = y=D, `E` = x=W, `W` = x=0. They are compass directions
  only when `north_angle` = 0. **The room never rotates; the sun does** (Ladybug convention).
- Floor grid 32×32: row 0 = the N edge, col 0 = the W edge. Facade grid 4 × 16 × 32: row 0 =
  ceiling.
- Dataset/model input: `< 0.5` = window. Optimizer and output CSVs: `1` = window.

## Known limits

Point-in-time conditions, not annual metrics. One room per solve. No exterior context or
neighbour shading. Fixed reflectances (wall 0.5, floor 0.2, ceiling 0.8, glass τ 0.698) and fixed
Radiance parameters. Below 5° solar altitude nothing is designed. The model's gradient with
respect to a cell value is not physically meaningful: the training label is a step function of
that value, so the optimizer uses gradients only to *propose* designs and selects them on
accurate binarised forward passes. Many facades satisfy the same target; the optimizer returns
one of them.

## Tests

`allside_Generalised model/tests/run_all.py` runs:
- the integration suite, which checks that the components agree with each other: solar
  position, the city list, wall azimuths, room limits, panel geometry against the Radiance scene,
  and the runner-to-Grasshopper-reader round trip;
- both responsive-facade suites;
- a headless drive of the designer GUI;
- a job run compared with its committed reference;
- the utilities and the ablation report.
