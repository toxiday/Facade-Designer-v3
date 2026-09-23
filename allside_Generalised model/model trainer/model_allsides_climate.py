"""
model_allsides_climate.py  (all-sides + climate + generalized room)
Climate- AND geometry-conditioned 4-facade patch-token transformer surrogate
(4 x 16x32 facades -> 32x32 floor lux, point-in-time).

Fuses two proven surrogates:
  * model_allsides.py (repo root) — the 4-facade encoder (per-side patchify +
    side-type embedding + shared positional encoding -> concatenated tokens) and the
    floor cross-attention decoder.  Original: fixed 12x12x3 m room, sun-only (3-vec).
  * sDA_Generalized Model/model trainer/model_transformer_climate.py — the 8-vector
    ClimateConditionEncoder + FiLM-at-every-layer, the min-max geometry normalisation,
    the self-contained building blocks + FullRangeDaylightLoss, and checkpoint/resume.

Conditioning (8) = [Sun_X, Sun_Y, Sun_Z, DNI/scale, DHI/scale, W_n, D_n, H_n] — the
same vector as the generalized single-facade model.  The new data structure differs
from model_allsides only in the facade grid (8->16 rows) and this conditioning; the
4-facade token layout, the floor decoder, the sqrt-lux target, and the rot90+fliplr
plan-view transform are unchanged, so the horizontal-flip augmentation (flip S/N cols,
swap E<->W, flip floor cols, negate Sun_X) stays valid — DNI/DHI and W/D/H are
flip-invariant.

Architecture is a lean adaptation of the proven climate config: the wider d_model=512
gives channel width to encode the 4 walls, but the FFN is kept lean (1024) and the
encoder at 8 layers (~30M params) — sequence length (512 tokens from 4 facades) drives
compute/VRAM, not parameter count, so depth need not grow. The floor decoder task
(256 queries -> 32x32) is unchanged, so it stays 4 layers.  Grow ffn_dim / n_enc_layers
later if it underperforms.

Facade convention (matches the generator / model_allsides): index = row*FACADE_COLS+col,
row 0 = top (high Z).

S/N/E/W are ROOM-LOCAL wall ids, NOT compass directions:
    S = wall at y=0    N = wall at y=D    E = wall at x=W    W = wall at x=0
    col 0 = the x=0 end for S/N;  col 0 = the y=0 end for E/W.
They coincide with true compass directions only when north_angle = 0.  The model has no
concept of north at all — it only ever sees a sun vector expressed in this room frame — so
a rotated building is handled by rotating the SUN, not by relabelling these walls.  See
optimize_facade_allsides_climate.annual_conditions() for the north_angle frame.

Window encoding here (and in the dataset) is <0.5 = window.  Three encodings coexist across
the project; the single authoritative mapping lives next to WIN_VAL/WALL_VAL in
optimize_facade_allsides_climate.py — do not restate it, point at it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path

# ── Dataset / grid constants (mirror generate_dataset_allsides_climate_pit.py) ──
N_FACADES   = 4
FACADE_ROWS = 16
FACADE_COLS = 32
N_CELLS     = FACADE_ROWS * FACADE_COLS     # 512 per facade
GRID_N      = 32                            # floor sensor grid side
N_SENSORS   = GRID_N * GRID_N               # 1024 floor sensors

DNI_SCALE = 1000.0
DHI_SCALE = 1000.0

# Room-dimension normalisation — MUST match the generator's W/D/H ranges.
W_MIN, W_MAX = 2.5, 10.0    # width  (X)
D_MIN, D_MAX = 2.5, 12.0    # depth  (Y)
H_MIN, H_MAX = 2.5,  8.0    # height (Z)

COND_DIM = 8                # [Sun_X, Sun_Y, Sun_Z, DNI/s, DHI/s, W_n, D_n, H_n]

# ── Architecture config — proven climate model, scaled up for 4 facades ─────────
MODEL_CONFIG = dict(
    patch_size=2, d_model=512, n_heads=8,
    n_enc_layers=8, n_dec_layers=4, ffn_dim=1024,
    dropout=0.1, cond_emb_dim=128,
)

MODELS_DIR      = Path(__file__).resolve().parents[1] / "models"
BEST_MODEL_PATH = str(MODELS_DIR / "best_facade_allsides_climate_genroom.pth")
CHECKPOINT_PATH = str(MODELS_DIR / "checkpoint_allsides_climate_genroom.pth")


# ---------------------------------------------------------------------------
# Building blocks  (inlined so this trainer is self-contained)
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
# Dataset
# ---------------------------------------------------------------------------

class AllSidesClimateDataset(Dataset):
    """
    Loads the all-sides climate point-in-time CSV into RAM.
    Returns (cond[8], facades[4,16,32], sqrt_lux[1,32,32]).
      cond    : [Sun_X, Sun_Y, Sun_Z, DNI/s, DHI/s, W_n, D_n, H_n]
      facades : S/N/E/W stacked on dim 0 (elevation; row 0 = ceiling), <0.5 = window
      lux     : plan orientation (N up, W left) via rot90+flip, then sqrt-normalised
    """
    def __init__(self, csv_file):
        print("Pre-loading all-sides climate dataset into RAM...")
        df = pd.read_csv(csv_file)

        cond = np.stack([
            df['Sun_X'].values, df['Sun_Y'].values, df['Sun_Z'].values,
            df['DNI'].values / DNI_SCALE, df['DHI'].values / DHI_SCALE,
            (df['Width'].values  - W_MIN) / (W_MAX - W_MIN),
            (df['Depth'].values  - D_MIN) / (D_MAX - D_MIN),
            (df['Height'].values - H_MIN) / (H_MAX - H_MIN),
        ], axis=1).astype(np.float32)
        self.cond_tensors = torch.tensor(cond)

        # 4 facades stacked -> [N, 4, 16, 32]
        fac_list = []
        for side in ('S', 'N', 'E', 'W'):
            cols = [f'{side}_Pix_{i}' for i in range(N_CELLS)]
            fac_list.append(torch.tensor(df[cols].values.astype(np.float32)))   # [N,512]
        self.facade_tensors = (torch.stack(fac_list, dim=1)
                                    .view(-1, N_FACADES, FACADE_ROWS, FACADE_COLS)
                                    .contiguous())                              # [N,4,16,32]

        lcols = [f'Lux_{i}' for i in range(N_SENSORS)]
        lux   = torch.tensor(df[lcols].values.astype(np.float32)).view(-1, 1, GRID_N, GRID_N)
        # same plan-view transform as both parents: rot90 then flip W -> N up, W left
        lux   = torch.rot90(lux, k=1, dims=[2, 3])
        lux   = torch.flip(lux, dims=[3]).contiguous()
        self.lux_tensors = torch.sqrt(lux)

        print(f"Loaded {len(self.cond_tensors)} samples  "
              f"(facades {N_FACADES}x{FACADE_ROWS}x{FACADE_COLS}, cond {COND_DIM}, floor {GRID_N}x{GRID_N}).")

    def __len__(self):
        return len(self.cond_tensors)

    def __getitem__(self, idx):
        return self.cond_tensors[idx], self.facade_tensors[idx], self.lux_tensors[idx]


# ---------------------------------------------------------------------------
# Climate + geometry conditioning encoder
# ---------------------------------------------------------------------------

class ClimateConditionEncoder(nn.Module):
    """[Sun_X, Sun_Y, Sun_Z, DNI_n, DHI_n, W_n, D_n, H_n] -> embedding for FiLM."""
    def __init__(self, in_dim=COND_DIM, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
        )

    def forward(self, cond):
        return self.net(cond)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AllSidesClimateTransformer(nn.Module):
    """
    Encoder-decoder patch transformer: 4 facades (climate + geometry conditioned)
    -> floor lux.  model_allsides token layout with the generalized model's 8-vector
    FiLM conditioning at every layer.
    """
    def __init__(self, patch_size=2, d_model=512, n_heads=8,
                 n_enc_layers=12, n_dec_layers=4, ffn_dim=2048, dropout=0.1,
                 cond_emb_dim=128):
        super().__init__()
        assert FACADE_ROWS % patch_size == 0 and FACADE_COLS % patch_size == 0
        assert GRID_N % patch_size == 0
        ps = patch_size
        self.patch_size = ps
        self.patch_dim  = ps * ps                         # 4
        self.d_model    = d_model

        fac_ph = FACADE_ROWS // ps                        # 8  patch rows per facade
        fac_pw = FACADE_COLS // ps                        # 16 patch cols per facade
        self.n_fac_pat   = fac_ph * fac_pw                # 128 patches per facade
        self.n_enc_tok   = N_FACADES * self.n_fac_pat     # 512 encoder tokens
        floor_p          = GRID_N // ps                   # 16
        self.n_floor_tok = floor_p * floor_p              # 256 floor query tokens

        # Shared climate + geometry conditioning embedding
        self.cond_encoder = ClimateConditionEncoder(COND_DIM, cond_emb_dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.side_embed  = nn.Embedding(N_FACADES, d_model)          # S/N/E/W token type
        self.register_buffer('enc_pos',
            make_2d_sinusoidal_pos_embed(fac_ph, fac_pw, d_model))   # (1, 128, d_model)
        self.enc_film = nn.ModuleList([
            nn.Linear(cond_emb_dim, d_model * 2) for _ in range(n_enc_layers)])
        self.enc_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_enc_layers)])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.floor_queries = nn.Parameter(torch.zeros(1, self.n_floor_tok, d_model))
        self.register_buffer('dec_pos',
            make_2d_sinusoidal_pos_embed(floor_p, floor_p, d_model)) # (1, 256, d_model)
        self.dec_film = nn.ModuleList([
            nn.Linear(cond_emb_dim, d_model * 2) for _ in range(n_dec_layers)])
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

    def _patchify(self, facades):
        """(B, 4, 16, 32) -> (B, 4, 128, 4)  non-overlapping 2x2 patches per facade."""
        B  = facades.shape[0]
        ps = self.patch_size
        rh = FACADE_ROWS // ps
        rc = FACADE_COLS // ps
        x  = facades.view(B, N_FACADES, rh, ps, rc, ps)
        x  = x.permute(0, 1, 2, 4, 3, 5).reshape(B, N_FACADES, rh * rc, ps * ps)
        return x                                          # (B, 4, 128, 4)

    def _film(self, layer, cond_emb):
        f = layer(cond_emb)
        return f[:, :self.d_model].unsqueeze(1), f[:, self.d_model:].unsqueeze(1)

    def forward(self, cond, facades):
        """cond (B,8), facades (B,4,16,32) [S=0,N=1,E=2,W=3] -> (B,1,32,32) sqrt-lux."""
        B = facades.shape[0]
        cond_emb = self.cond_encoder(cond)                          # (B, cond_emb_dim)

        # ── Encode 4 facades ──────────────────────────────────────────────
        tokens = self.patch_embed(self._patchify(facades))          # (B,4,128,d_model)
        tokens = tokens + self.enc_pos.unsqueeze(1)                 # shared within-facade pos
        side_emb = self.side_embed(torch.arange(N_FACADES, device=facades.device))
        tokens = tokens + side_emb[None, :, None, :]                # S/N/E/W type
        enc = tokens.reshape(B, self.n_enc_tok, self.d_model)       # (B,512,d_model)
        for film, layer in zip(self.enc_film, self.enc_layers):
            scale, shift = self._film(film, cond_emb)
            enc = enc * (1.0 + scale) + shift
            enc = layer(enc)

        # ── Decode floor ──────────────────────────────────────────────────
        dec = self.floor_queries.expand(B, -1, -1) + self.dec_pos   # (B,256,d_model)
        for film, cross, cnorm, self_layer in zip(
                self.dec_film, self.cross_attn, self.cross_norm, self.dec_layers):
            scale, shift = self._film(film, cond_emb)
            dec    = dec * (1.0 + scale) + shift
            normed = cnorm(dec)
            dec    = dec + cross(normed, enc, enc)[0]
            dec    = self_layer(dec)

        ps  = self.patch_size
        gp  = GRID_N // ps                                          # 16
        out = self.output_proj(self.norm_out(dec))                 # (B,256,4)
        out = out.permute(0, 2, 1).view(B, self.patch_dim, gp, gp) # (B,4,16,16)
        out = F.pixel_shuffle(out, ps)                             # (B,1,32,32)
        return F.softplus(out)


# ---------------------------------------------------------------------------
# Loss  (inlined from model_transformer_climate — self-contained)
# ---------------------------------------------------------------------------

class FullRangeDaylightLoss(nn.Module):
    """
    Full-range loss (no 3000-lux cap).  Weighted Huber over all pixels (5x boost on
    non-glare pixels and missed-glare false-negatives), plus a spatial gradient term
    and a recall-biased Tversky glare-overlap term.
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
        base_huber = self.huber(predictions, targets)
        weight_mask  = torch.where(targets <= self.norm_threshold, 5.0, 1.0)
        missed_glare = (targets > self.norm_threshold) & (predictions <= self.norm_threshold)
        weight_mask  = torch.where(missed_glare, 5.0, weight_mask)
        huber_loss  = (base_huber * weight_mask).mean()

        p_dx = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        t_dx = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        p_dy = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        t_dy = targets[:, :, 1:, :] - targets[:, :, :-1, :]

        w_dx = torch.where((targets[:, :, :, 1:] <= self.norm_threshold) | (targets[:, :, :, :-1] <= self.norm_threshold), 5.0, 1.0)
        w_dy = torch.where((targets[:, :, 1:, :] <= self.norm_threshold) | (targets[:, :, :-1, :] <= self.norm_threshold), 5.0, 1.0)

        grad_dx = (F.l1_loss(p_dx, t_dx, reduction='none') * w_dx).mean()
        grad_dy = (F.l1_loss(p_dy, t_dy, reduction='none') * w_dy).mean()
        grad_loss = (grad_dx + grad_dy) * 0.5

        pred_glare = torch.sigmoid(predictions - self.norm_threshold)
        target_glare = (targets > self.norm_threshold).float()
        non_glare_mask = 1.0 - target_glare

        tp = (pred_glare * target_glare).sum(dim=[1, 2, 3])
        fp = (pred_glare * non_glare_mask).sum(dim=[1, 2, 3])
        fn = ((1.0 - pred_glare) * target_glare).sum(dim=[1, 2, 3])
        tversky_loss = (1.0 - (tp + 1e-6) /
                        (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6)).mean()

        return huber_loss + self.grad_weight * grad_loss + self.tversky_weight * tversky_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ SYSTEM CHECK ] Active Training Device: {device.type.upper()}\n")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = AllSidesClimateTransformer(**MODEL_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model config    : {MODEL_CONFIG}")
    print(f"Model parameters: {total_params:,}")
    print(f"Encoder tokens  : {model.n_enc_tok}  ({N_FACADES} facades x {model.n_fac_pat} patches)")
    print(f"Decoder tokens  : {model.n_floor_tok}  (floor queries -> {GRID_N}x{GRID_N} via pixel_shuffle)\n")

    # torch.compile probe (graceful fallback if no compatible mode).
    _pc = torch.zeros(1, COND_DIM, device=device)
    _pf = torch.zeros(1, N_FACADES, FACADE_ROWS, FACADE_COLS, device=device)
    for _mode in ['max-autotune', 'reduce-overhead', 'default']:
        try:
            _cand = torch.compile(model, mode=_mode)
            with torch.no_grad():
                _cand(_pc, _pf)
            model = _cand
            print(f"torch.compile : enabled ({_mode})\n")
            break
        except Exception:
            pass
    else:
        print("torch.compile : skipped (no compatible mode found)\n")
    del _pc, _pf

    dataset = AllSidesClimateDataset(
        str(Path(__file__).resolve().parents[1] / "datasets" / "dataset_allsides_climate_genroom_400k.csv"))

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(23))
    print(f"Split: {train_size} train | {val_size} val | {test_size} test\n")

    BATCH_SIZE = 32          # fits 16 GB VRAM (48 ran ~5 GB over); drop to 24 if still tight
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    criterion = FullRangeDaylightLoss(huber_delta=2.0, grad_weight=0.5,
                                      tversky_weight=0.5, tversky_alpha=0.3, tversky_beta=0.7)
    # LR lowered from 2.5e-4 (that value was for model_allsides' batch 96): batch 32 +
    # the brighter all-sides data + the deeper 12-layer model all need a gentler LR.
    BASE_LR      = 1.5e-4
    WARMUP_STEPS = 1000      # ~1 epoch linear warmup — the deep FiLM stack diverges without it
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    use_amp   = (device.type == 'cuda')
    amp_dtype = torch.bfloat16

    VAL_EVERY        = 5
    epochs           = 1000
    patience         = 12    # val rounds without improvement (= 60 epochs)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float('inf')
    patience_counter = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if ckpt.get('model_config', {}) and ckpt['model_config'] != MODEL_CONFIG:
            print(f"  Architecture changed — checkpoint {ckpt['model_config']}")
            print(f"  Current            {MODEL_CONFIG}\n  Starting from scratch.\n")
        else:
            try:
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                scheduler.load_state_dict(ckpt['scheduler_state'])
                start_epoch      = ckpt['epoch'] + 1
                best_val_loss    = ckpt['best_val_loss']
                patience_counter = ckpt['patience_counter']
                print(f"  Resumed at epoch {start_epoch+1}  |  best val {best_val_loss:.4f}  "
                      f"|  patience {patience_counter}/{patience}\n")
            except (KeyError, RuntimeError) as e:
                print(f"  Checkpoint incompatible ({e}), starting from scratch.\n")
    else:
        print("No checkpoint found, starting from scratch.\n")

    print("Starting training...\n")
    global_step = start_epoch * len(train_loader)   # resume past warmup skips it
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        for batch_cond, batch_facades, batch_lux in train_loader:
            # Linear LR warmup for the first WARMUP_STEPS (stabilises the deep FiLM stack).
            if global_step < WARMUP_STEPS:
                for g in optimizer.param_groups:
                    g['lr'] = BASE_LR * (global_step + 1) / WARMUP_STEPS
            global_step += 1

            batch_cond    = batch_cond.to(device, non_blocking=True)
            batch_facades = batch_facades.to(device, non_blocking=True)
            batch_lux     = batch_lux.to(device, non_blocking=True)

            # Horizontal flip augmentation (W<->E mirror): flip S/N cols, swap E<->W,
            # flip floor cols, negate Sun_X.  DNI/DHI and W/D/H (cond[:,3:]) are
            # flip-invariant -> unchanged.
            flip_mask = torch.rand(batch_cond.shape[0]) > 0.5
            if flip_mask.any():
                idx = flip_mask.nonzero(as_tuple=False).squeeze(1).to(device)
                f = batch_facades[idx].clone()
                f[:, 0] = torch.flip(f[:, 0], dims=[2])     # S cols
                f[:, 1] = torch.flip(f[:, 1], dims=[2])     # N cols
                f[:, [2, 3]] = f[:, [3, 2]]                 # swap E <-> W
                batch_facades[idx] = f
                batch_lux[idx]     = torch.flip(batch_lux[idx], dims=[3])
                batch_cond[idx, 0] = -batch_cond[idx, 0]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                predictions = model(batch_cond, batch_facades)
                loss        = criterion(predictions, batch_lux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.detach()

        avg_train_loss = train_loss.item() / len(train_loader)
        do_val = (epoch == 0) or ((epoch + 1) % VAL_EVERY == 0)

        if do_val:
            model.eval()
            val_loss = torch.tensor(0.0, device=device)
            with torch.no_grad():
                for batch_cond, batch_facades, batch_lux in val_loader:
                    batch_cond    = batch_cond.to(device, non_blocking=True)
                    batch_facades = batch_facades.to(device, non_blocking=True)
                    batch_lux     = batch_lux.to(device, non_blocking=True)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        val_loss += criterion(model(batch_cond, batch_facades), batch_lux).detach()

            avg_val_loss = val_loss.item() / len(val_loader)
            scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:04d}/{epochs} | Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | LR: {lr:.2e}")

            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break

            torch.save({
                'epoch':            epoch,
                'model_config':     MODEL_CONFIG,
                'model_state':      model.state_dict(),
                'optimizer_state':  optimizer.state_dict(),
                'scheduler_state':  scheduler.state_dict(),
                'best_val_loss':    best_val_loss,
                'patience_counter': patience_counter,
            }, CHECKPOINT_PATH)
        elif (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:04d}/{epochs} | Train: {avg_train_loss:.4f} | LR: {lr:.2e}")

    print(f"\nBest val loss: {best_val_loss:.4f}  ->  {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()
