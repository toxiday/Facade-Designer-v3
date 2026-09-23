# Transformer vs UNet ablation

A controlled comparison: five architectures trained with everything else held fixed. That means
the same loss (`FullRangeDaylightLoss`), AdamW at 1.5e-4, the same 70/15/15 split (seed 23),
and the same data.

The run is on the **earlier single-facade 20×20 problem** (`mvp_dataset_20x20_200k.csv`), not on
the production all-sides model. `model20x20_v2.py` and `model_transformer_v2.py` are the legacy
model definitions it imports; they are kept here only for that purpose.

```
python ablation_architecture.py --report         # print the table from ablation_results/*.json
python ablation_architecture.py --params-only    # parameter count of each arm
python ablation_architecture.py                  # re-train all arms (needs the dataset, below)
```

| arm | params | val loss | MAE (lux) | R² | SSIM | glare F1 |
|---|---|---|---|---|---|---|
| transformer, matched | 7.95M | 23.21 | 460.9 | 0.873 | 0.935 | 94.1% |
| transformer, small | 3.44M | 29.11 | 650.2 | 0.824 | 0.913 | 92.7% |
| UNet, matched | 8.04M | 30.22 | 676.3 | 0.843 | 0.909 | 92.0% |
| UNet, small | 3.43M | 32.20 | 698.7 | 0.829 | 0.905 | 91.5% |
| UNet, small, no bottleneck attention | 3.17M | 32.08 | 700.4 | 0.828 | 0.904 | 91.6% |

**What it supports.** At a matched parameter count, the attention model beat the
convolutional encoder-decoder on every metric (MAE −31.9%). It also scaled better: 2.3× the
parameters bought the UNet 3% in MAE and the transformer 29%.

**What it does not support.** It does not show that attention learns a light-transport or
daylight-coefficient operator. Removing the UNet's bottleneck attention changed nothing (32.20
vs 32.08, inside the ~0.12 run-to-run noise).

**Caveats.** One seed, 18k samples of 20×20 data.

**Data.** `mvp_dataset_20x20_200k.csv` (112 MB) is not in git. Re-training needs it placed in
this folder; `--report` and `--params-only` do not.
