#!/usr/bin/env python
"""Standalone Darcy flow experiment — SVGP + WNO mean + WNO kernel embedding.

    a(x,y) -> u(x,y)

Usage:
    python experiments/darcy/standalone_svgp_wno_darcy.py
"""

import sys, os, time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.mlls import VariationalELBO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sgpo.models.svgp import SVGP
from sgpo.wno.wno_models import WNO2d
from sgpo.wno.latent_mean import LatentWNOMean
from sgpo.wno.embedding import WNOEmbedding
from sgpo.data.readers import MatReader
from sgpo.data.normalizers import UnitGaussianNormalizer
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params

# ── Hyperparameters ──────────────────────────────────────────────────

TRAIN_PATH = "/DATA/Sawan_projects/research/DATA/piececonst_r421_N1024_smooth1.mat"
TEST_PATH = "/DATA/Sawan_projects/research/DATA/piececonst_r421_N1024_smooth2.mat"
X_FIELD, Y_FIELD = "coeff", "sol"
NTRAIN, NTEST, SUB = 1000, 100, 5
RESOLUTION = [85, 85]
NORMALIZE_X, NORMALIZE_Y = True, True
BOUNDARY_ZERO = True

NUM_LATENTS = 120
NUM_INDUCING = 200
KERNEL = "matern"
KERNEL_NU = 2.5
USE_ARD = True

WNO_WIDTH = 32
WNO_LEVEL = 4
WNO_LAYERS = 4
WNO_WAVELET = "db4"
EMBED_DIM = 32

LR = 4e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

SEED = 0
DEVICE = "cuda:1"
SAVE_DIR = "results/darcy"


def main():
    set_seed(SEED)
    device = resolve_device(DEVICE)
    s = RESOLUTION[0]

    print("=" * 70)
    print("  Darcy Flow — SVGP + WNO mean + WNO kernel embedding")
    print(f"  Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  WNO width={WNO_WIDTH}, layers={WNO_LAYERS}, embed_dim={EMBED_DIM}")
    print("=" * 70)

    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field(X_FIELD)[:NTRAIN, ::SUB, ::SUB][:, :s, :s]
    y_train = reader.read_field(Y_FIELD)[:NTRAIN, ::SUB, ::SUB][:, :s, :s]

    if BOUNDARY_ZERO:
        y_train[:, 0, :] = 0; y_train[:, -1, :] = 0
        y_train[:, :, 0] = 0; y_train[:, :, -1] = 0

    reader.load_file(TEST_PATH)
    x_test = reader.read_field(X_FIELD)[:NTEST, ::SUB, ::SUB][:, :s, :s]
    y_test = reader.read_field(Y_FIELD)[:NTEST, ::SUB, ::SUB][:, :s, :s]

    if BOUNDARY_ZERO:
        y_test[:, 0, :] = 0; y_test[:, -1, :] = 0
        y_test[:, :, 0] = 0; y_test[:, :, -1] = 0

    print(f"  x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")

    x_normalizer = y_normalizer = None
    if NORMALIZE_X:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
    if NORMALIZE_Y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)
    if y_normalizer: y_normalizer.to(device)
    num_tasks = x_tr.shape[-1]

    wno_mean_net = WNO2d(width=WNO_WIDTH, level=WNO_LEVEL, layers=WNO_LAYERS,
                         size=RESOLUTION, wavelet=WNO_WAVELET).to(device)
    wno_embed_net = WNO2d(width=WNO_WIDTH, level=WNO_LEVEL, layers=WNO_LAYERS,
                          size=RESOLUTION, wavelet=WNO_WAVELET).to(device)
    wno_mean = LatentWNOMean(wno_mean_net)
    wno_embed = WNOEmbedding(wno_embed_net, embed_dim=EMBED_DIM)

    model = SVGP(
        x_train=x_tr, num_tasks=num_tasks, num_latents=NUM_LATENTS,
        num_inducing=NUM_INDUCING,
        kernel_type=KERNEL, kernel_nu=KERNEL_NU,
        use_ard=USE_ARD,
        mean_module=wno_mean, wno_embedding=wno_embed,
    ).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks).to(device)

    print(f"  Model params: {count_params(model):,} | "
          f"Likelihood params: {count_params(likelihood):,}")

    model.train(); likelihood.train()
    mll = VariationalELBO(likelihood, model, num_data=y_tr.size(0))
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    loader = DataLoader(TensorDataset(x_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True)
    elbo_hist, mse_hist = [], []

    print(f"\nTraining for {NUM_EPOCHS} epochs ...\n")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        te = time.time(); ep_elbo = ep_mse = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = -mll(out, by)
            loss_mse = mse_fn(out.mean, by)
            loss.backward(); optimizer.step()
            ep_elbo += loss.item(); ep_mse += loss_mse.item()
        scheduler.step()
        nb = len(loader)
        elbo_hist.append(ep_elbo / nb); mse_hist.append(ep_mse / nb)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>4d}/{NUM_EPOCHS} | "
                  f"elbo: {elbo_hist[-1]:.4f} | mse: {mse_hist[-1]:.6f} | "
                  f"time: {time.time()-te:.1f}s")
    print(f"\nTraining done in {time.time()-t0:.1f}s")

    model.eval(); likelihood.eval()
    all_m, all_v = [], []
    with torch.no_grad():
        for bx, _ in DataLoader(TensorDataset(x_t, x_t),
                                batch_size=BATCH_SIZE, shuffle=False):
            out = model(bx)
            m, v = out.mean, out.variance
            if m.ndim == 3 and m.shape[1] == 1:
                m, v = m.squeeze(1), v.squeeze(1)
            all_m.append(m.cpu()); all_v.append(v.cpu())
    mean_pred = torch.cat(all_m, 0); var_pred = torch.cat(all_v, 0)

    if y_normalizer is not None:
        std = y_normalizer.std + y_normalizer.eps
        mean_pred = mean_pred.to(device).reshape(-1, *RESOLUTION)
        mean_pred = y_normalizer.decode(mean_pred).reshape(-1, num_tasks).cpu()
        var_pred = var_pred.to(device).reshape(-1, *RESOLUTION)
        var_pred = (var_pred * std**2).reshape(-1, num_tasks).cpu()

    y_t_cpu = y_t.cpu()
    mse = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
    rel = relative_l2_error(mean_pred, y_t_cpu).item()
    print(f"\n  MSE: {mse:.6f} | Relative L2: {100*rel:.2f}%")

    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(SAVE_DIR, f"svgp_wno_{ts}.npz"),
             mean_pred=mean_pred.numpy(), var_pred=var_pred.numpy(),
             y_test=y_t_cpu.numpy(),
             elbo_history=np.array(elbo_hist), mse_history=np.array(mse_hist))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"svgp_wno_{ts}.pt"))
    print(f"  Saved to {SAVE_DIR}/svgp_wno_{ts}.*")


if __name__ == "__main__":
    main()
