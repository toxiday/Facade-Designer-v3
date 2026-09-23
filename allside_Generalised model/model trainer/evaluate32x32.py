"""
evaluate32x32.py  (all-sides + climate + generalized room)
Spatial & structural evaluation for the 4-facade all-sides climate transformer
(model_allsides_climate.py).

Ported from sDA_Generalized Model/model trainer/evaluate32x32.py.  Two changes for the
all-sides model:
  1. Imports the all-sides model + dataset (4 x 16x32 facades, 8-vector conditioning).
  2. The dataset yields (cond, facades, sqrt_lux); the model is called
     model(cond, facades) instead of model(cond, facade).

All metrics (SSIM, MAE/RMSE/R2 on all pixels, per-zone MAE, IoU, glare
precision/recall/F1) operate on the 32x32 floor lux and are identical to the
single-facade script — the floor output convention is unchanged.

The held-out test set is the same 70/15/15 split (seed 23) the trainer uses, so this
scores exactly the samples the model never trained on.

Run (from this folder, so the local model_allsides_climate import resolves):
    cd "allside_Generalised model/model trainer"
    ../../.venv/Scripts/python evaluate32x32.py                 # default 400k dataset
    ../../.venv/Scripts/python evaluate32x32.py --csv <path> --batch 64
    ../../.venv/Scripts/python evaluate32x32.py --csv <fresh.csv> --all   # unseen samples
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from skimage.metrics import structural_similarity as ssim

from model_allsides_climate import (
    AllSidesClimateTransformer, AllSidesClimateDataset, MODEL_CONFIG, BEST_MODEL_PATH)


def evaluate_spatial_accuracy(trained_model, test_loader, device):
    trained_model.eval()

    all_true_lux = []
    all_pred_lux = []

    with torch.no_grad():
        for batch_cond, batch_facades, batch_norm_lux in test_loader:
            batch_cond     = batch_cond.to(device)
            batch_facades  = batch_facades.to(device)
            batch_norm_lux = batch_norm_lux.to(device)

            pred_norm_lux = trained_model(batch_cond, batch_facades)

            # de-normalise sqrt-lux -> lux
            true_lux_batch = torch.square(batch_norm_lux).squeeze().cpu().numpy()
            pred_lux_batch = torch.square(pred_norm_lux).squeeze().cpu().numpy()

            if true_lux_batch.ndim == 2:
                true_lux_batch = np.expand_dims(true_lux_batch, 0)
                pred_lux_batch = np.expand_dims(pred_lux_batch, 0)

            for i in range(true_lux_batch.shape[0]):
                all_true_lux.append(true_lux_batch[i])
                all_pred_lux.append(pred_lux_batch[i])

    all_true_lux = np.array(all_true_lux)  # [N, 32, 32]
    all_pred_lux = np.array(all_pred_lux)

    GLARE_THRESHOLD = 3000.0
    USABLE_LOW      = 300.0

    t_all = all_true_lux.flatten()
    p_all = all_pred_lux.flatten()
    dark_mask   = t_all < USABLE_LOW
    usable_mask = (t_all >= USABLE_LOW) & (t_all <= GLARE_THRESHOLD)
    glare_mask  = t_all > GLARE_THRESHOLD

    def calculate_iou(mask_true, mask_pred):
        intersection = np.logical_and(mask_true, mask_pred).sum()
        union        = np.logical_or(mask_true, mask_pred).sum()
        return (intersection / union * 100) if union > 0 else 100.0

    # --- SSIM (Full Range) ---
    data_range = float(np.max(all_true_lux) - np.min(all_true_lux))
    if data_range == 0: data_range = 1.0

    ssim_scores = [
        ssim(t, p, data_range=data_range, win_size=5)
        for t, p in zip(all_true_lux, all_pred_lux)
    ]
    avg_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    # --- MAE / RMSE / R² on ALL pixels ---
    diff_all  = t_all - p_all
    mae       = np.mean(np.abs(diff_all))
    rmse      = np.sqrt(np.mean(diff_all ** 2))
    ss_res    = np.sum(diff_all ** 2)
    ss_tot    = np.sum((t_all - t_all.mean()) ** 2)
    r2        = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- Per-zone MAE (dark, usable, and full glare intensity) ---
    abs_diff   = np.abs(t_all - p_all)
    mae_dark   = abs_diff[dark_mask].mean()   if dark_mask.any()   else 0.0
    mae_usable = abs_diff[usable_mask].mean() if usable_mask.any() else 0.0
    mae_glare  = abs_diff[glare_mask].mean()  if glare_mask.any()  else 0.0

    # --- Per-sample IoU ---
    iou_usable_scores, iou_glare_scores = [], []
    for t, p in zip(all_true_lux, all_pred_lux):
        t_f, p_f = t.flatten(), p.flatten()
        iou_usable_scores.append(calculate_iou((t_f >= USABLE_LOW) & (t_f <= GLARE_THRESHOLD),
                                               (p_f >= USABLE_LOW) & (p_f <= GLARE_THRESHOLD)))
        iou_glare_scores.append(calculate_iou(t_f > GLARE_THRESHOLD, p_f > GLARE_THRESHOLD))

    # --- Glare detection: precision / recall / F1 (pixel-level binary classification) ---
    true_glare = glare_mask
    pred_glare = p_all > GLARE_THRESHOLD
    tp = np.logical_and(true_glare, pred_glare).sum()
    fp = np.logical_and(~true_glare, pred_glare).sum()
    fn = np.logical_and(true_glare, ~pred_glare).sum()
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n==========================================")
    print("     SPATIAL & STRUCTURAL EVALUATION      ")
    print("        (all-sides climate model)         ")
    print("==========================================")
    print(f"SSIM (mean ± std)      : {avg_ssim:.4f} ± {std_ssim:.4f}  (Full Range)")
    print(f"MAE  (all pixels, lux) : {mae:.1f}")
    print(f"RMSE (all pixels, lux) : {rmse:.1f}")
    print(f"R²   (all pixels)      : {r2:.4f}  (1.0 is perfect)")
    print("------------------------------------------")
    print(f"Zone MAE — Dark        : {mae_dark:.1f} lux  (<300)")
    print(f"Zone MAE — Usable      : {mae_usable:.1f} lux  (300-3000)")
    print(f"Zone MAE — Glare       : {mae_glare:.1f} lux  (>3000)")
    print("------------------------------------------")
    print(f"IoU Usable (mean ± std): {np.mean(iou_usable_scores):.1f}% ± {np.std(iou_usable_scores):.1f}%  (300-3000 lux)")
    print(f"IoU Glare  (mean ± std): {np.mean(iou_glare_scores):.1f}% ± {np.std(iou_glare_scores):.1f}%  (>3000 lux)")
    print(f"Glare Precision        : {precision:.1f}%  (flagged pixels that are truly glare)")
    print(f"Glare Recall           : {recall:.1f}%  (actual glare pixels caught)")
    print(f"Glare F1               : {f1:.1f}%")
    print("==========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spatial evaluation for the all-sides climate model.")
    ap.add_argument("--csv", type=str,
                    default=str(Path(__file__).resolve().parents[1] / "datasets"
                                / "dataset_allsides_climate_genroom_400k.csv"),
                    help="dataset CSV (must match the trained model's dataset)")
    ap.add_argument("--batch", type=int, default=64, help="eval batch size (forward-only)")
    ap.add_argument("--all", action="store_true",
                    help="score EVERY row instead of the seed-23 test split — for a freshly "
                         "generated CSV (new --seed) that the model has never seen")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Evaluation Device: {device.type.upper()}")

    print("Loading dataset and calculating splits...")
    dataset = AllSidesClimateDataset(args.csv)

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    if args.all:
        test_dataset = dataset
        print(f"Test samples: {len(test_dataset)} (every row; --all)")
    else:
        _, _, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(23)
        )
        print(f"Test samples: {len(test_dataset)} (held-out; seed 23 split)")

    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    print("Loading trained model weights...")
    production_model = AllSidesClimateTransformer(**MODEL_CONFIG).to(device)
    # weights_only bare state_dict; strip torch.compile's _orig_mod. prefix if present.
    state = torch.load(BEST_MODEL_PATH, weights_only=True, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    production_model.load_state_dict(state)
    production_model.eval()

    print("Running spatial analysis...")
    evaluate_spatial_accuracy(production_model, test_loader, device)
