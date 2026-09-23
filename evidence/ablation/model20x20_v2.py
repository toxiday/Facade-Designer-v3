import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import random


class FacadeDaylightDataset(Dataset):
    def __init__(self, csv_file):
        print("Pre-loading dataset into RAM...")
        df = pd.read_csv(csv_file)

        sun_array = df[['Sun_X', 'Sun_Y', 'Sun_Z']].values.astype(np.float32)
        self.sun_tensors = torch.tensor(sun_array)

        facade_array = df.iloc[:, 7:407].values.astype(np.float32)
        self.facade_tensors = torch.tensor(facade_array).view(-1, 1, 20, 20).contiguous()

        lux_array = df.iloc[:, 407:807].values.astype(np.float32)
        lux_tensor = torch.tensor(lux_array).view(-1, 1, 20, 20)

        lux_tensor = torch.rot90(lux_tensor, k=1, dims=[2, 3])
        lux_tensor = torch.flip(lux_tensor, dims=[3]).contiguous()

        # sqrt normalization: good dynamic range for lux without saturating the
        # glare zone (3000–50k lux) into a narrow band like log1p+sigmoid does.
        self.lux_tensors = torch.sqrt(lux_tensor)

        print(f"Successfully loaded and transformed {len(self.sun_tensors)} iterations.")

    def __len__(self):
        return len(self.sun_tensors)

    def __getitem__(self, idx):
        return self.sun_tensors[idx], self.facade_tensors[idx], self.lux_tensors[idx]


class HybridDaylightLoss(nn.Module):
    """
    Four-part loss:
      - Non-glare pixels (≤3000 lux): Huber regression for accurate lux values.
          huber_delta is small (default 2.0) so the quadratic regime stays tight,
          forcing fine-grained matching rather than predicting a spatial average.
      - Glare pixels (>3000 lux): one-sided hinge — penalise under-prediction only.
        Predicting any value ≥ threshold is treated as correct; exact intensity
        above 3000 lux does not affect the loss.
      - Tversky loss: recall-biased spatial overlap of the glare mask.
        FN (missed glare) is penalised tversky_beta times, FP (phantom glare)
        tversky_alpha times. Default 0.3/0.7 makes missing a glare pixel 2.3×
        costlier than a false alarm, directly improving glare recall.
        tversky_weight kept low (default 0.5) so Tversky does not drown the
        non-glare Huber signal.
      - Spatial gradient loss on non-glare pixels: penalises smooth predictions
        by matching the spatial differences (dx, dy) of target vs prediction.
        This directly rewards local variation in the <3000 lux region.
    """
    def __init__(self, glare_threshold=3000.0, tversky_weight=0.5,
                 tversky_alpha=0.3, tversky_beta=0.7, huber_delta=2.0,
                 grad_weight=0.5):
        super().__init__()
        self.norm_threshold  = np.sqrt(glare_threshold)
        self.tversky_weight  = tversky_weight
        self.tversky_alpha   = tversky_alpha   # FP penalty weight
        self.tversky_beta    = tversky_beta    # FN penalty weight (recall)
        self.grad_weight     = grad_weight
        self.huber = nn.HuberLoss(reduction='none', delta=huber_delta)

    def forward(self, predictions, targets):
        # Float masks — no .any() / boolean-index syncs, everything stays on GPU
        non_glare_mask = (targets <= self.norm_threshold).float()   # (B,1,H,W)
        glare_mask     = 1.0 - non_glare_mask
        target_glare   = glare_mask                                 # reused below

        # Huber on non-glare pixels: weighted mean, avoids boolean indexing
        huber_map  = self.huber(predictions, targets)               # (B,1,H,W)
        n_ng       = non_glare_mask.sum().clamp(min=1)
        huber_loss = (huber_map * non_glare_mask).sum() / n_ng

        # One-sided hinge on glare pixels: only penalise being below the threshold
        hinge_map   = F.relu(self.norm_threshold - predictions)
        n_gl        = glare_mask.sum().clamp(min=1)
        glare_hinge = (hinge_map * glare_mask).sum() / n_gl

        # Tversky loss: recall-biased spatial glare-zone overlap
        # TP / (TP + alpha*FP + beta*FN)  — beta > alpha favours recall
        pred_glare = torch.sigmoid(predictions - self.norm_threshold)
        tp = (pred_glare * target_glare).sum(dim=[1, 2, 3])
        fp = (pred_glare * non_glare_mask).sum(dim=[1, 2, 3])
        fn = ((1.0 - pred_glare) * target_glare).sum(dim=[1, 2, 3])
        tversky_loss = (1.0 - (tp + 1e-6) /
                        (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6)).mean()

        # Spatial gradient loss on non-glare region: penalise spatial smoothing.
        p_masked = predictions * non_glare_mask
        t_masked = targets     * non_glare_mask
        # Finite differences along H and W
        p_dx = p_masked[:, :, :, 1:] - p_masked[:, :, :, :-1]
        t_dx = t_masked[:, :, :, 1:] - t_masked[:, :, :, :-1]
        p_dy = p_masked[:, :, 1:, :] - p_masked[:, :, :-1, :]
        t_dy = t_masked[:, :, 1:, :] - t_masked[:, :, :-1, :]
        grad_loss = (F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)) * 0.5

        return (huber_loss + glare_hinge
                + self.tversky_weight * tversky_loss
                + self.grad_weight * grad_loss)


def film_apply(x, sun_tensor, film_layer):
    """Apply FiLM conditioning: x * (1 + scale) + shift."""
    film = film_layer(sun_tensor)
    ch = x.shape[1]
    scale = film[:, :ch].view(-1, ch, 1, 1)
    shift = film[:, ch:].view(-1, ch, 1, 1)
    return x * (1.0 + scale) + shift


class BottleneckAttention(nn.Module):
    """
    Self-attention over the spatial positions of the bottleneck feature map.
    At 3x3 this is only 9 tokens — cheap but gives full global context so every
    spatial position can attend to every other before decoding begins.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)        # [B, H*W, C]
        seq = self.norm(seq)
        attn_out, _ = self.attn(seq, seq, seq)
        return x + attn_out.permute(0, 2, 1).view(B, C, H, W)  # residual


class UNet20x20v2(nn.Module):
    """
    Deeper UNet: 20→10→5→3 encoder, wider channels (64/128/256),
    multi-scale FiLM sun conditioning at bottleneck and all three decoder stages,
    self-attention at bottleneck for global context.
    """
    def __init__(self):
        super().__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def downsample(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )

        # Encoder: 20x20 -> 10x10 -> 5x5 -> 3x3
        self.enc1  = double_conv(4, 64)
        self.down1 = downsample(64)
        self.enc2  = double_conv(64, 128)
        self.down2 = downsample(128)
        self.enc3  = double_conv(128, 256)
        self.down3 = downsample(256)   # 5x5 -> 3x3

        # Bottleneck at 3x3: conv + self-attention + FiLM
        self.bottleneck  = double_conv(256, 256)
        self.attn_bottle = BottleneckAttention(256, num_heads=4)
        self.film_bottle = nn.Linear(3, 256 * 2)

        # Decoder: 3x3 -> 5x5 -> 10x10 -> 20x20
        self.upconv3   = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec3      = double_conv(128 + 256, 128)
        self.film_dec3 = nn.Linear(3, 128 * 2)

        self.upconv2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2      = double_conv(64 + 128, 64)
        self.film_dec2 = nn.Linear(3, 64 * 2)

        self.upconv1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1      = double_conv(32 + 64 + 1, 32)  # + facade skip
        self.film_dec1 = nn.Linear(3, 32 * 2)           # FiLM at final stage

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, sun_tensor, facade_tensor):
        sun_spatial = sun_tensor.view(-1, 3, 1, 1).expand(-1, -1, 20, 20)
        x = torch.cat([facade_tensor, sun_spatial], dim=1)

        # Encoder
        e1 = self.enc1(x)                       # [B, 64,  20, 20]
        e2 = self.enc2(self.down1(e1))           # [B, 128, 10, 10]
        e3 = self.enc3(self.down2(e2))           # [B, 256,  5,  5]
        p3 = self.down3(e3)                      # [B, 256,  3,  3]

        # Bottleneck: conv → self-attention → FiLM
        b = self.bottleneck(p3)
        b = self.attn_bottle(b)
        b = film_apply(b, sun_tensor, self.film_bottle)

        # Decoder stage 3: 3x3 -> 5x5
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d3 = film_apply(d3, sun_tensor, self.film_dec3)

        # Decoder stage 2: 5x5 -> 10x10
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d2 = film_apply(d2, sun_tensor, self.film_dec2)

        # Decoder stage 1: 10x10 -> 20x20 (+ facade direct skip + FiLM)
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1, facade_tensor], dim=1))
        d1 = film_apply(d1, sun_tensor, self.film_dec1)

        return F.softplus(self.final_conv(d1))


class UNet20x20v3(nn.Module):
    """
    Reduced-compression UNet: 20→10→5 encoder (2 downsamples, not 3).

    v2 bottleneck: 5→3 stride-2 conv → 3×3 = 9 spatial tokens.
    v3 bottleneck: stays at 5×5 = 25 spatial tokens (2.8× more).
      Dilated convolutions (dilation=1 then dilation=2) give global receptive
      field at 5×5 without further spatial loss — dilation-2 covers the full map.
      Attention over 25 positions instead of 9 preserves the south→north lux
      gradient that v2 smeared at the 5→3 compression step.

    Decoder: 5→10→20 (2 upsamples matching the 2-level encoder).
    Everything else — FiLM conditioning, facade direct skip, Tversky loss — unchanged.
    """
    def __init__(self):
        super().__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        def downsample(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            )

        # Encoder: 20→10→5  (2 downsamples)
        self.enc1  = double_conv(4, 64)
        self.down1 = downsample(64)
        self.enc2  = double_conv(64, 128)
        self.down2 = downsample(128)
        self.enc3  = double_conv(128, 256)       # 5×5 — feeds directly into bottleneck

        # Bottleneck at 5×5: dilated convs → attention → FiLM
        # dilation=2 on a 5×5 map: 3×3 kernel spans 5 pixels → full global coverage
        self.bottle_d1   = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.bottle_d2   = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.attn_bottle = BottleneckAttention(256, num_heads=4)   # 25 tokens
        self.film_bottle = nn.Linear(3, 256 * 2)

        # Decoder: 5→10→20
        self.upconv2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2      = double_conv(128 + 128, 128)   # concat with enc2 skip (128ch)
        self.film_dec2 = nn.Linear(3, 128 * 2)

        self.upconv1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1      = double_conv(64 + 64 + 1, 32)  # concat enc1 skip (64ch) + facade
        self.film_dec1 = nn.Linear(3, 32 * 2)

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, sun_tensor, facade_tensor):
        sun_spatial = sun_tensor.view(-1, 3, 1, 1).expand(-1, -1, 20, 20)
        x = torch.cat([facade_tensor, sun_spatial], dim=1)

        # Encoder
        e1 = self.enc1(x)                       # [B, 64,  20, 20]
        e2 = self.enc2(self.down1(e1))           # [B, 128, 10, 10]
        e3 = self.enc3(self.down2(e2))           # [B, 256,  5,  5]

        # Bottleneck at 5×5: dilated → attention → FiLM
        b = self.bottle_d1(e3)
        b = self.bottle_d2(b)
        b = self.attn_bottle(b)
        b = film_apply(b, sun_tensor, self.film_bottle)

        # Decoder: 5→10, skip enc2
        d2 = self.dec2(torch.cat([self.upconv2(b), e2], dim=1))
        d2 = film_apply(d2, sun_tensor, self.film_dec2)

        # Decoder: 10→20, skip enc1 + facade direct skip
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1, facade_tensor], dim=1))
        d1 = film_apply(d1, sun_tensor, self.film_dec1)

        return F.softplus(self.final_conv(d1))


def visualize_multiple_predictions(trained_model, eval_dataset, num_samples=10):
    trained_model.eval()
    device = next(trained_model.parameters()).device
    indices = random.sample(range(len(eval_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(4.5 * 3, 4.0 * num_samples))
    if num_samples == 1:
        axes = [axes]

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

            # col 0: facade elevation
            ax = axes[row_idx][0]
            ax.imshow(binary_facade, cmap='binary_r', vmin=0, vmax=1,
                      interpolation='nearest', aspect='equal')
            ax.set_xticks([0, 19]); ax.set_xticklabels(["W", "E"], fontsize=7)
            ax.set_yticks([0, 19]); ax.set_yticklabels(["ceiling", "floor"], fontsize=7)
            ax.tick_params(length=2)
            ax.set_xlabel(f"Alt={alt_deg:.1f}°", fontsize=8)
            if row_idx == 0:
                ax.set_title("Facade Elevation", fontsize=9)

            # col 1: ground truth
            ax = axes[row_idx][1]
            im1 = ax.imshow(actual_lux, cmap='inferno', vmin=0, vmax=DISPLAY_MAX_LUX,
                            interpolation='nearest', aspect='equal')
            add_plan_labels(ax)
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="lux")
            if row_idx == 0:
                ax.set_title("Ground Truth (Radiance)", fontsize=9)

            # col 2: prediction
            ax = axes[row_idx][2]
            im2 = ax.imshow(predicted_lux, cmap='inferno', vmin=0, vmax=DISPLAY_MAX_LUX,
                            interpolation='nearest', aspect='equal')
            add_plan_labels(ax)
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="lux")
            if row_idx == 0:
                ax.set_title("Surrogate Prediction (v3)", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"prediction_gallery_{num_samples}_examples_v3.png", dpi=200, bbox_inches='tight')
    print(f"Gallery saved: prediction_gallery_{num_samples}_examples_v3.png")


if __name__ == "__main__":
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ SYSTEM CHECK ] Active Training Device: {device.type.upper()}\n")

    model = UNet20x20v3().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    dataset = FacadeDaylightDataset(r'C:\Computation Project\Facade Designer V2\mvp_dataset_20x20_60k.csv')

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(23)
    )
    print(f"Split: {train_size} train | {val_size} val | {test_size} test")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    criterion = HybridDaylightLoss(glare_threshold=3000.0, tversky_weight=0.5,
                                   tversky_alpha=0.3, tversky_beta=0.7, huber_delta=2.0,
                                   grad_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    epochs          = 1000
    patience        = 60
    patience_counter = 0
    best_val_loss   = float('inf')
    best_model_path = "best_unet_20x20_v3.pth"

    print("Starting training...\n")
    for epoch in range(epochs):

        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch_sun, batch_facade, batch_lux in train_loader:
            batch_sun    = batch_sun.to(device)
            batch_facade = batch_facade.to(device)
            batch_lux    = batch_lux.to(device)

            # Horizontal flip augmentation
            flip_mask = torch.rand(batch_sun.shape[0], device=device) > 0.5
            if flip_mask.any():
                batch_facade[flip_mask] = torch.flip(batch_facade[flip_mask], dims=[3])
                batch_lux[flip_mask]    = torch.flip(batch_lux[flip_mask],    dims=[3])
                batch_sun[flip_mask, 0] = -batch_sun[flip_mask, 0]

            optimizer.zero_grad()
            predictions = model(batch_sun, batch_facade)
            loss = criterion(predictions, batch_lux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_sun, batch_facade, batch_lux in val_loader:
                batch_sun    = batch_sun.to(device)
                batch_facade = batch_facade.to(device)
                batch_lux    = batch_lux.to(device)
                loss = criterion(model(batch_sun, batch_facade), batch_lux)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        is_report = (epoch == 0) or ((epoch + 1) % 10 == 0)
        if is_report:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            if is_report:
                print(f"   -> Best model saved (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if is_report:
                print(f"   -> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[ EARLY STOPPING ] Epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("\nTraining complete. Best model loaded.")

    

    print("\nGenerating prediction gallery...")
    visualize_multiple_predictions(model, test_dataset, num_samples=10)
