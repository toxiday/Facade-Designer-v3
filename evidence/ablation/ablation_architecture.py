"""
ablation_architecture.py
Controlled architecture ablation: does the patch-token transformer actually
beat a conv UNet, or was the historical win just parameter count / loss change?

The original v2-era comparison was confounded four ways (architecture, loss
function, optimizer, and a 2.31x parameter gap), and neither the UNet
checkpoint nor the 60k dataset survived. This script re-runs the comparison
with everything except the architecture held fixed:

    data          mvp_dataset_20x20_200k.csv, 70/15/15 split, seed 23
    loss          FullRangeDaylightLoss (identical for every arm)
    optimizer     AdamW lr=1.5e-4 wd=0.01 + ReduceLROnPlateau(0.5, 10)
    schedule      batch 128, grad clip 1.0, h-flip augmentation, patience 60
    metrics       same test split, same physical-lux metrics

Arms are chosen so parameter count is crossed against architecture in BOTH
directions -- a param-matched transformer and a param-matched UNet -- so
"transformer" and "bigger" can be told apart:

    unet_small          UNet v3 as originally written            ~3.4M
    unet_small_noattn   same, bottleneck self-attention removed  ~3.4M
    unet_matched        UNet widened to the transformer's size   ~7.9M
    xf_small            transformer narrowed to the UNet's size  ~3.4M
    xf_matched          FacadeDaylightTransformerV2 as written   ~7.9M

Usage:
    python ablation_architecture.py --arms all --seeds 0
    python ablation_architecture.py --arms unet_small,xf_matched --seeds 0,1,2
    python ablation_architecture.py --arms unet_small,xf_matched --frac 0.25,0.5,1.0
    python ablation_architecture.py --report          # aggregate existing results
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from model20x20_v2 import FacadeDaylightDataset, BottleneckAttention, film_apply
from model_transformer_v2 import FacadeDaylightTransformerV2, FullRangeDaylightLoss

ROOT     = Path(__file__).parent
DATA_CSV = ROOT / "mvp_dataset_20x20_200k.csv"
OUT_DIR  = ROOT / "ablation_results"

SPLIT_SEED = 23          # same split seed both original models used
BATCH_SIZE = 128
MAX_EPOCHS = 1000
PATIENCE   = 60


# ---------------------------------------------------------------------------
# Width-parameterised UNet (structurally identical to UNet20x20v3)
# ---------------------------------------------------------------------------

class UNetAblation(nn.Module):
    """
    UNet20x20v3 with two knobs: channel width and whether the bottleneck
    self-attention block is present. base_ch=64 reproduces UNet20x20v3 exactly.
    """
    def __init__(self, base_ch=64, use_attn=True, n_heads=4):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        c_out = max(32, base_ch // 2)
        self.use_attn = use_attn

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

        # Encoder: 20 -> 10 -> 5
        self.enc1  = double_conv(4, c1)
        self.down1 = downsample(c1)
        self.enc2  = double_conv(c1, c2)
        self.down2 = downsample(c2)
        self.enc3  = double_conv(c2, c3)

        # Bottleneck at 5x5: dilated convs -> (attention) -> FiLM
        self.bottle_d1 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1, dilation=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.bottle_d2 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=2, dilation=2),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.attn_bottle = BottleneckAttention(c3, num_heads=n_heads) if use_attn else None
        self.film_bottle = nn.Linear(3, c3 * 2)

        # Decoder: 5 -> 10 -> 20
        self.upconv2   = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2      = double_conv(c2 + c2, c2)
        self.film_dec2 = nn.Linear(3, c2 * 2)

        self.upconv1   = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1      = double_conv(c1 + c1 + 1, c_out)
        self.film_dec1 = nn.Linear(3, c_out * 2)

        self.final_conv = nn.Conv2d(c_out, 1, 1)

    def forward(self, sun_tensor, facade_tensor):
        sun_spatial = sun_tensor.view(-1, 3, 1, 1).expand(-1, -1, 20, 20)
        x = torch.cat([facade_tensor, sun_spatial], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottle_d2(self.bottle_d1(e3))
        if self.attn_bottle is not None:
            b = self.attn_bottle(b)
        b = film_apply(b, sun_tensor, self.film_bottle)

        d2 = self.dec2(torch.cat([self.upconv2(b), e2], dim=1))
        d2 = film_apply(d2, sun_tensor, self.film_dec2)

        d1 = self.dec1(torch.cat([self.upconv1(d2), e1, facade_tensor], dim=1))
        d1 = film_apply(d1, sun_tensor, self.film_dec1)

        return F.softplus(self.final_conv(d1))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ARMS = {
    # conv baseline, exactly UNet20x20v3
    "unet_small":        lambda: UNetAblation(base_ch=64, use_attn=True),
    # same, minus the bottleneck attention -- isolates attention from convs
    "unet_small_noattn": lambda: UNetAblation(base_ch=64, use_attn=False),
    # UNet widened to the transformer's parameter budget
    "unet_matched":      lambda: UNetAblation(base_ch=98, use_attn=True),
    # transformer narrowed to the small UNet's parameter budget
    "xf_small":          lambda: FacadeDaylightTransformerV2(
                             d_model=168, n_heads=8, n_enc_layers=6,
                             n_dec_layers=3, ffn_dim=672, dropout=0.1),
    # transformer as originally trained
    "xf_matched":        lambda: FacadeDaylightTransformerV2(
                             d_model=256, n_heads=8, n_enc_layers=6,
                             n_dec_layers=3, ffn_dim=1024, dropout=0.1),
}


# ---------------------------------------------------------------------------
# Metrics (physical lux, identical to evaluate_transformer_v2.py)
# ---------------------------------------------------------------------------

def compute_metrics(model, loader, device):
    from skimage.metrics import structural_similarity as ssim

    model.eval()
    true_all, pred_all = [], []
    with torch.no_grad():
        for sun, facade, norm_lux in loader:
            pred = model(sun.to(device), facade.to(device))
            true_all.append(torch.square(norm_lux).squeeze(1).cpu().numpy())
            pred_all.append(torch.square(pred).squeeze(1).cpu().numpy())

    true_all = np.concatenate(true_all, axis=0)   # [N, 20, 20]
    pred_all = np.concatenate(pred_all, axis=0)

    GLARE, USABLE_LOW = 3000.0, 300.0
    t, p = true_all.ravel(), pred_all.ravel()
    diff = t - p

    data_range = float(true_all.max() - true_all.min()) or 1.0
    ssim_scores = [ssim(a, b, data_range=data_range, win_size=5)
                   for a, b in zip(true_all, pred_all)]

    dark   = t < USABLE_LOW
    usable = (t >= USABLE_LOW) & (t <= GLARE)
    glare  = t > GLARE
    absd   = np.abs(diff)

    pred_glare = p > GLARE
    tp = np.logical_and(glare, pred_glare).sum()
    fp = np.logical_and(~glare, pred_glare).sum()
    fn = np.logical_and(glare, ~pred_glare).sum()
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) else 0.0

    def iou(a, b):
        u = np.logical_or(a, b).sum()
        return np.logical_and(a, b).sum() / u * 100 if u else 100.0

    iou_glare = [iou(a.ravel() > GLARE, b.ravel() > GLARE)
                 for a, b in zip(true_all, pred_all)]

    return {
        "ssim":        float(np.mean(ssim_scores)),
        "mae":         float(absd.mean()),
        "rmse":        float(np.sqrt((diff ** 2).mean())),
        "r2":          float(1.0 - (diff ** 2).sum() / ((t - t.mean()) ** 2).sum()),
        "mae_dark":    float(absd[dark].mean())   if dark.any()   else 0.0,
        "mae_usable":  float(absd[usable].mean()) if usable.any() else 0.0,
        "mae_glare":   float(absd[glare].mean())  if glare.any()  else 0.0,
        "iou_glare":   float(np.mean(iou_glare)),
        "glare_prec":  float(prec),
        "glare_rec":   float(rec),
        "glare_f1":    float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
    }


# ---------------------------------------------------------------------------
# Training (identical for every arm)
# ---------------------------------------------------------------------------

def run_arm(arm, seed, frac, dataset, device, max_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n - n_train - n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED))

    # Data-scale sweep subsamples TRAIN only; val/test stay fixed so the
    # metric is comparable across fractions.
    if frac < 1.0:
        keep = int(len(train_ds) * frac)
        idx = torch.randperm(len(train_ds),
                             generator=torch.Generator().manual_seed(SPLIT_SEED))[:keep]
        train_ds = Subset(train_ds, idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model  = ARMS[arm]().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = FullRangeDaylightLoss(huber_delta=2.0, grad_weight=0.5,
                                      tversky_weight=0.5, tversky_alpha=0.3,
                                      tversky_beta=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)

    tag = f"{arm}_s{seed}_f{frac:g}"
    print(f"\n=== {tag} | {params:,} params | {len(train_ds)} train ===", flush=True)

    best_val, patience_ctr, best_epoch = float('inf'), 0, 0
    ckpt = OUT_DIR / f"ckpt_{tag}.pth"
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for sun, facade, lux in train_loader:
            sun    = sun.to(device, non_blocking=True)
            facade = facade.to(device, non_blocking=True)
            lux    = lux.to(device, non_blocking=True)

            flip = torch.rand(sun.shape[0], device=device) > 0.5
            if flip.any():
                facade[flip] = torch.flip(facade[flip], dims=[3])
                lux[flip]    = torch.flip(lux[flip],    dims=[3])
                sun[flip, 0] = -sun[flip, 0]

            optimizer.zero_grad()
            loss = criterion(model(sun, facade), lux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sun, facade, lux in val_loader:
                val_loss += criterion(model(sun.to(device), facade.to(device)),
                                      lux.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_ctr, best_epoch = val_loss, 0, epoch + 1
            torch.save(model.state_dict(), ckpt)
        else:
            patience_ctr += 1

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"  ep {epoch+1:04d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| best {best_val:.4f} @{best_epoch} | pat {patience_ctr}", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"  early stop @ epoch {epoch+1}", flush=True)
            break

    minutes = (time.time() - t0) / 60
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    metrics = compute_metrics(model, test_loader, device)

    result = {
        "arm": arm, "seed": seed, "frac": frac, "params": params,
        "n_train": len(train_ds), "best_val_loss": best_val,
        "best_epoch": best_epoch, "epochs_run": epoch + 1,
        "minutes": round(minutes, 1), **metrics,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"  -> val {best_val:.4f} | MAE {metrics['mae']:.1f} | "
          f"SSIM {metrics['ssim']:.4f} | R2 {metrics['r2']:.4f} | "
          f"glareF1 {metrics['glare_f1']:.1f}% | {minutes:.1f} min", flush=True)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report():
    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("*.json"))]
    if not rows:
        print("No results yet.")
        return
    rows.sort(key=lambda r: (r["frac"], r["mae"]))
    hdr = (f"{'arm':<19}{'seed':>5}{'frac':>6}{'params':>11}{'val':>9}"
           f"{'MAE':>9}{'RMSE':>9}{'R2':>9}{'SSIM':>8}{'glF1':>7}{'min':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<19}{r['seed']:>5}{r['frac']:>6g}{r['params']:>11,}"
              f"{r['best_val_loss']:>9.4f}{r['mae']:>9.1f}{r['rmse']:>9.1f}"
              f"{r['r2']:>9.4f}{r['ssim']:>8.4f}{r['glare_f1']:>7.1f}{r['minutes']:>7.1f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",   default="all")
    ap.add_argument("--seeds",  default="0")
    ap.add_argument("--frac",   default="1.0")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--params-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if args.report:
        report()
        raise SystemExit

    if args.params_only:
        for name, ctor in ARMS.items():
            m = ctor()
            print(f"{name:<20}{sum(p.numel() for p in m.parameters()):>12,}")
        raise SystemExit

    arms  = list(ARMS) if args.arms == "all" else args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    fracs = [float(f) for f in args.frac.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")
    dataset = FacadeDaylightDataset(str(DATA_CSV))

    for frac in fracs:
        for seed in seeds:
            for arm in arms:
                run_arm(arm, seed, frac, dataset, device, args.epochs)

    report()
