"""
model_transformer_v2.py
Patch-token transformer surrogate - v2 (Full Range Lux Prediction)

This version removes the artificial 3000 lux cap and uses a full-range
regression loss to predict the actual physical brightness of glare zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

from model20x20_v2 import FacadeDaylightDataset

GRID_N = 20

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def make_2d_sinusoidal_pos_embed(grid_h, grid_w, d_model):
    assert d_model % 4 == 0
    d_half = d_model // 2
    freq   = 1.0 / (10000.0 ** (2 * torch.arange(d_half // 2).float() / d_half))

    rows   = torch.arange(grid_h).float()
    cols   = torch.arange(grid_w).float()

    row_enc = torch.outer(rows, freq)                              # (H, d_half/2)
    row_enc = torch.cat([row_enc.sin(), row_enc.cos()], dim=-1)   # (H, d_half)

    col_enc = torch.outer(cols, freq)                              # (W, d_half/2)
    col_enc = torch.cat([col_enc.sin(), col_enc.cos()], dim=-1)   # (W, d_half)

    pe = torch.zeros(grid_h, grid_w, d_model)
    pe[:, :, :d_half] = row_enc.unsqueeze(1).expand(grid_h, grid_w, d_half)
    pe[:, :, d_half:] = col_enc.unsqueeze(0).expand(grid_h, grid_w, d_half)

    return pe.reshape(1, grid_h * grid_w, d_model)   # (1, H*W, d_model)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.drop(self.attn(normed, normed, normed)[0])
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FacadeDaylightTransformerV2(nn.Module):
    def __init__(self, grid_n=20, patch_size=2, d_model=256, n_heads=8,
                 n_enc_layers=6, n_dec_layers=3, ffn_dim=1024, dropout=0.1):
        super().__init__()
        assert grid_n % patch_size == 0
        self.grid_n     = grid_n
        self.patch_size = patch_size
        self.n_patches  = (grid_n // patch_size) ** 2     # 100
        self.patch_dim  = patch_size * patch_size         # 4
        self.d_model    = d_model
        gp = grid_n // patch_size                         # 10 patches per side

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.register_buffer('enc_pos', make_2d_sinusoidal_pos_embed(gp, gp, d_model))

        self.enc_film = nn.ModuleList([
            nn.Linear(3, d_model * 2) for _ in range(n_enc_layers)])
        self.enc_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_enc_layers)])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.floor_queries = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.register_buffer('dec_pos', make_2d_sinusoidal_pos_embed(gp, gp, d_model))

        self.dec_film = nn.ModuleList([
            nn.Linear(3, d_model * 2) for _ in range(n_dec_layers)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_dec_layers)])
        self.cross_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_dec_layers)])
        self.dec_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_dec_layers)])

        self.norm_out    = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.patch_dim)

    def _patchify(self, facade_tensor):
        B  = facade_tensor.shape[0]
        ps = self.patch_size
        gp = self.grid_n // ps
        x  = facade_tensor.view(B, 1, gp, ps, gp, ps)
        x  = x.permute(0, 2, 4, 1, 3, 5).reshape(B, gp * gp, ps * ps)
        return x

    def forward(self, sun_tensor, facade_tensor):
        B  = facade_tensor.shape[0]
        gp = self.grid_n // self.patch_size

        # ── Encode facade ─────────────────────────────────────────────────
        enc = self.patch_embed(self._patchify(facade_tensor)) + self.enc_pos

        for film, layer in zip(self.enc_film, self.enc_layers):
            f     = film(sun_tensor)
            scale = f[:, :self.d_model].unsqueeze(1)
            shift = f[:, self.d_model:].unsqueeze(1)
            enc   = enc * (1.0 + scale) + shift
            enc   = layer(enc)

        # ── Decode floor ──────────────────────────────────────────────────
        dec = self.floor_queries.expand(B, -1, -1) + self.dec_pos

        for film, cross, cnorm, self_layer in zip(
                self.dec_film, self.cross_attn, self.cross_norm, self.dec_layers):
            f     = film(sun_tensor)
            scale = f[:, :self.d_model].unsqueeze(1)
            shift = f[:, self.d_model:].unsqueeze(1)
            dec   = dec * (1.0 + scale) + shift

            normed = cnorm(dec)
            dec    = dec + cross(normed, enc, enc)[0]
            dec    = self_layer(dec)

        out = self.output_proj(self.norm_out(dec))
        out = out.permute(0, 2, 1).view(B, self.patch_dim, gp, gp)
        out = F.pixel_shuffle(out, self.patch_size)
        return F.softplus(out)


# ---------------------------------------------------------------------------
# Full Range Loss & Visualization
# ---------------------------------------------------------------------------

class FullRangeDaylightLoss(nn.Module):
    """
    Full-range loss function that does not cap at 3000 lux.
    Uses Huber regression across all pixels to predict actual physical brightness.
    Retains spatial gradient loss to prevent spatial blurring/smoothing.
    """
    def __init__(self, huber_delta=2.0, grad_weight=0.5, tversky_weight=0.5, 
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.grad_weight = grad_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.huber = nn.HuberLoss(reduction='none', delta=huber_delta)
        self.norm_threshold = 3000.0 ** 0.5

    def forward(self, predictions, targets):
        # Raw Huber loss applied over every pixel
        base_huber = self.huber(predictions, targets)
        
        # Boost weight on non-glare pixels (e.g., 5x) so the optimizer doesn't 
        # sacrifice dark-zone precision to perfectly fit the 50,000 lux peaks.
        # To prevent the model from artificially shrinking the sun patches (which hurts Glare Recall),
        # we also apply the 5x boost to "missed glare" (False Negatives) where it underpredicts the boundary.
        weight_mask  = torch.where(targets <= self.norm_threshold, 5.0, 1.0)
        missed_glare = (targets > self.norm_threshold) & (predictions <= self.norm_threshold)
        weight_mask  = torch.where(missed_glare, 5.0, weight_mask)
        huber_loss  = (base_huber * weight_mask).mean()

        # Spatial gradient loss over every pixel
        p_dx = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        t_dx = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        p_dy = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        t_dy = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        
        # Weight gradients: if either pixel in the spatial difference pair is non-glare, boost it.
        w_dx = torch.where((targets[:, :, :, 1:] <= self.norm_threshold) | (targets[:, :, :, :-1] <= self.norm_threshold), 5.0, 1.0)
        w_dy = torch.where((targets[:, :, 1:, :] <= self.norm_threshold) | (targets[:, :, :-1, :] <= self.norm_threshold), 5.0, 1.0)
        
        grad_dx = (F.l1_loss(p_dx, t_dx, reduction='none') * w_dx).mean()
        grad_dy = (F.l1_loss(p_dy, t_dy, reduction='none') * w_dy).mean()
        grad_loss = (grad_dx + grad_dy) * 0.5

        # Tversky loss: recall-biased spatial glare-zone overlap
        pred_glare = torch.sigmoid(predictions - self.norm_threshold)
        target_glare = (targets > self.norm_threshold).float()
        non_glare_mask = 1.0 - target_glare

        tp = (pred_glare * target_glare).sum(dim=[1, 2, 3])
        fp = (pred_glare * non_glare_mask).sum(dim=[1, 2, 3])
        fn = ((1.0 - pred_glare) * target_glare).sum(dim=[1, 2, 3])
        tversky_loss = (1.0 - (tp + 1e-6) /
                        (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6)).mean()

        return huber_loss + self.grad_weight * grad_loss + self.tversky_weight * tversky_loss


def visualize_full_range_predictions(trained_model, eval_dataset, num_samples=10):
    trained_model.eval()
    device = next(trained_model.parameters()).device
    indices = random.sample(range(len(eval_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(4.5 * 3, 4.0 * num_samples))
    if num_samples == 1:
        axes = [axes]

    # Cap display at 3000 lux to better distinguish usable light vs dark zones
    DISPLAY_MAX_LUX = 3000

    def add_plan_labels(ax):
        ax.set_xticks([0, 19]); ax.set_xticklabels(["W", "E"], fontsize=7)
        ax.set_yticks([0, 19]); ax.set_yticklabels(["N", "S"], fontsize=7)
        ax.tick_params(length=2)

    with torch.no_grad():
        for row_idx, data_idx in enumerate(indices):
            sun_tensor, facade_tensor, norm_lux = eval_dataset[data_idx]
            pred_norm = trained_model(sun_tensor.unsqueeze(0).to(device),
                                      facade_tensor.unsqueeze(0).to(device))

            actual_lux    = torch.square(norm_lux).squeeze().numpy()
            predicted_lux = torch.square(pred_norm).squeeze().cpu().numpy()
            facade_grid   = facade_tensor.squeeze().numpy()

            binary_facade = (facade_grid < 0.5).astype(float)
            sun_vec = sun_tensor.squeeze().numpy()
            alt_deg = float(np.degrees(np.arcsin(np.clip(-sun_vec[2], -1.0, 1.0))))

            ax = axes[row_idx][0]
            ax.imshow(binary_facade, cmap='binary_r', vmin=0, vmax=1,
                      interpolation='nearest', aspect='equal')
            ax.set_xticks([0, 19]); ax.set_xticklabels(["W", "E"], fontsize=7)
            ax.set_yticks([0, 19]); ax.set_yticklabels(["ceiling", "floor"], fontsize=7)
            ax.tick_params(length=2)
            ax.set_xlabel(f"Alt={alt_deg:.1f}°", fontsize=8)
            if row_idx == 0: ax.set_title("Facade Elevation", fontsize=9)

            ax = axes[row_idx][1]
            im1 = ax.imshow(actual_lux, cmap='inferno', vmin=0, vmax=DISPLAY_MAX_LUX,
                            interpolation='nearest', aspect='equal')
            add_plan_labels(ax)
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="lux")
            if row_idx == 0: ax.set_title("Ground Truth (Full Range)", fontsize=9)

            ax = axes[row_idx][2]
            im2 = ax.imshow(predicted_lux, cmap='inferno', vmin=0, vmax=DISPLAY_MAX_LUX,
                            interpolation='nearest', aspect='equal')
            add_plan_labels(ax)
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="lux")
            if row_idx == 0: ax.set_title("Prediction (Full Range V2)", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"prediction_gallery_fullrange_{num_samples}_examples.png", dpi=200, bbox_inches='tight')
    print(f"Gallery saved: prediction_gallery_fullrange_{num_samples}_examples.png")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ SYSTEM CHECK ] Active Training Device: {device.type.upper()}\n")

    model = FacadeDaylightTransformerV2(
        grid_n=20, patch_size=2, d_model=256, n_heads=8,
        n_enc_layers=6, n_dec_layers=3, ffn_dim=1024, dropout=0.1
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    dataset = FacadeDaylightDataset(
        r'C:\Computation Project\Facade Designer V2\mvp_dataset_20x20_60k.csv')

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(23)
    )
    print(f"Split: {train_size} train | {val_size} val | {test_size} test")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=True)

    criterion = FullRangeDaylightLoss(huber_delta=2.0, grad_weight=0.5, 
                                      tversky_weight=0.5, tversky_alpha=0.3, tversky_beta=0.7)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    epochs           = 1000
    patience         = 60
    patience_counter = 0
    best_val_loss    = float('inf')
    best_model_path  = "best_facade_transformer_v2.pth"

    print("Starting training…\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_sun, batch_facade, batch_lux in train_loader:
            batch_sun    = batch_sun.to(device, non_blocking=True)
            batch_facade = batch_facade.to(device, non_blocking=True)
            batch_lux    = batch_lux.to(device, non_blocking=True)

            flip_mask = torch.rand(batch_sun.shape[0]) > 0.5
            if flip_mask.any():
                flip_idx = flip_mask.nonzero(as_tuple=False).squeeze(1).to(device)
                batch_facade[flip_idx] = torch.flip(batch_facade[flip_idx], dims=[3])
                batch_lux[flip_idx]    = torch.flip(batch_lux[flip_idx],    dims=[3])
                batch_sun[flip_idx, 0] = -batch_sun[flip_idx, 0]

            optimizer.zero_grad()
            predictions = model(batch_sun, batch_facade)
            loss = criterion(predictions, batch_lux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_sun, batch_facade, batch_lux in val_loader:
                batch_sun    = batch_sun.to(device, non_blocking=True)
                batch_facade = batch_facade.to(device, non_blocking=True)
                batch_lux    = batch_lux.to(device, non_blocking=True)
                val_loss += criterion(
                    model(batch_sun, batch_facade), batch_lux).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(best val loss: {best_val_loss:.4f})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}  →  {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    visualize_full_range_predictions(model, test_dataset, num_samples=10)