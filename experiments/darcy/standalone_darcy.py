#!/usr/bin/env python
"""Standalone Darcy flow experiment.

Self-contained script that runs the full VNNGP pipeline for 2-D Darcy flow:
    a(x,y) -> u(x,y)

All hyperparameters are defined as constants at the top of the file.
No YAML config needed -- just run:
    python experiments/darcy/standalone_darcy.py

"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.mlls import VariationalELBO

from sgpo.models.vnngp import VNNGP
from sgpo.data.readers import MatReader
from sgpo.data.normalizers import UnitGaussianNormalizer
from sgpo.utils.metrics import relative_l2_error
from sgpo.utils.misc import set_seed, resolve_device, count_params


#  Hyperparameters

# --- Data ---
TRAIN_PATH = "/DATA/Sawan_projects/research/DATA/piececonst_r421_N1024_smooth1.mat"
TEST_PATH = "/DATA/Sawan_projects/research/DATA/piececonst_r421_N1024_smooth2.mat"
X_FIELD = "coeff"
Y_FIELD = "sol"
NTRAIN = 1000
NTEST = 100
SUB = 5
RESOLUTION = [85, 85]
NORMALIZE_X = True
NORMALIZE_Y = True
BOUNDARY_ZERO = True

# --- Model ---
NUM_LATENTS = 120
NUM_NN = 100
NUM_INDUCING = 200
KERNEL = "matern"
KERNEL_NU = 2.5

# --- Training ---
LR = 4e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.75

# --- Misc ---
SEED = 0
DEVICE = "cuda:1"
SAVE_DIR = "results/darcy"


#  Main

def main():
    # 1. Setup
    set_seed(SEED)
    device = resolve_device(DEVICE)
    s = RESOLUTION[0]  # 85

    print("=" * 70)
    print("  Standalone Darcy Flow Experiment")
    print(f"  Device     : {device}")
    print(f"  Resolution : {RESOLUTION}")
    print(f"  Train/Test : {NTRAIN} / {NTEST}")
    print(f"  Epochs     : {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  LR         : {LR} | WD: {WEIGHT_DECAY}")
    print(f"  Kernel     : {KERNEL} (nu={KERNEL_NU})")
    print(f"  Latents    : {NUM_LATENTS} | NN: {NUM_NN} | Inducing: {NUM_INDUCING}")
    print("=" * 70)

    # 2. Load data
    print("\nLoading training data ...")
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field(X_FIELD)[:NTRAIN, ::SUB, ::SUB][:, :s, :s]
    y_train = reader.read_field(Y_FIELD)[:NTRAIN, ::SUB, ::SUB][:, :s, :s]

    if BOUNDARY_ZERO:
        y_train[:, 0, :] = 0
        y_train[:, -1, :] = 0
        y_train[:, :, 0] = 0
        y_train[:, :, -1] = 0

    print("Loading test data ...")
    reader.load_file(TEST_PATH)
    x_test = reader.read_field(X_FIELD)[:NTEST, ::SUB, ::SUB][:, :s, :s]
    y_test = reader.read_field(Y_FIELD)[:NTEST, ::SUB, ::SUB][:, :s, :s]

    if BOUNDARY_ZERO:
        y_test[:, 0, :] = 0
        y_test[:, -1, :] = 0
        y_test[:, :, 0] = 0
        y_test[:, :, -1] = 0

    print(f"  x_train: {list(x_train.shape)}, y_train: {list(y_train.shape)}")
    print(f"  x_test:  {list(x_test.shape)}, y_test:  {list(y_test.shape)}")

    # 3. Normalize BEFORE flattening
    x_normalizer, y_normalizer = None, None

    if NORMALIZE_X:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

    if NORMALIZE_Y:
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        # NOTE: y_test is NOT encoded -- it stays in original space for evaluation

    # 4. Flatten to (n, s*s) for GP and move to device
    x_tr = x_train.reshape(NTRAIN, -1).to(device)
    y_tr = y_train.reshape(NTRAIN, -1).to(device)
    x_t = x_test.reshape(NTEST, -1).to(device)
    y_t = y_test.reshape(NTEST, -1).to(device)

    print(f"  Flattened: x_tr={list(x_tr.shape)}, y_tr={list(y_tr.shape)}")
    print(f"             x_t ={list(x_t.shape)},  y_t ={list(y_t.shape)}")

    # 5. Build model + likelihood
    num_tasks = x_tr.shape[-1]  # s*s = 7225

    model = VNNGP(
        x_train=x_tr,
        num_tasks=num_tasks,
        num_latents=NUM_LATENTS,
        num_nn=NUM_NN,
        num_inducing=NUM_INDUCING,
        training_batch_size=BATCH_SIZE,
        kernel_type=KERNEL,
        kernel_nu=KERNEL_NU,
    ).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
    ).to(device)

    n_params = count_params(model)
    n_params_lik = count_params(likelihood)
    print(f"\n  Model params    : {n_params:,}")
    print(f"  Likelihood params: {n_params_lik:,}")

    # 6. Training loop
    model.train()
    likelihood.train()

    mll = VariationalELBO(likelihood, model, num_data=y_tr.size(0))
    mse_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA,
    )

    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    elbo_history = []
    mse_history = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs ...\n")
    train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_elbo = 0.0
        epoch_mse = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss_mse = mse_fn(output.mean, batch_y)
            loss.backward()
            optimizer.step()

            epoch_elbo += loss.item()
            epoch_mse += loss_mse.item()

        scheduler.step()

        n_batches = len(train_loader)
        avg_elbo = epoch_elbo / n_batches
        avg_mse = epoch_mse / n_batches
        elbo_history.append(avg_elbo)
        mse_history.append(avg_mse)
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>4d}/{NUM_EPOCHS} | "
                f"elbo: {avg_elbo:.4f} | mse: {avg_mse:.4f} | "
                f"time: {epoch_time:.1f}s"
            )

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time / 60:.1f}min)")

    # 7. Prediction
    model.eval()
    likelihood.eval()

    test_loader = DataLoader(
        TensorDataset(x_t, x_t), batch_size=BATCH_SIZE, shuffle=False,
    )

    all_means = []
    all_vars = []

    print("\nRunning predictions ...")
    with torch.no_grad():
        for batch_x, _ in test_loader:
            output = model(batch_x)
            mean_batch = output.mean
            var_batch = output.variance

            if mean_batch.ndim == 3 and mean_batch.shape[1] == 1:
                mean_batch = mean_batch.squeeze(1)
                var_batch = var_batch.squeeze(1)

            all_means.append(mean_batch.cpu())
            all_vars.append(var_batch.cpu())

    mean_pred = torch.cat(all_means, dim=0)
    var_pred = torch.cat(all_vars, dim=0)

    # 8. Decode predictions back to original space
    if y_normalizer is not None:
        std = y_normalizer.std + y_normalizer.eps

        # Reshape to (n, 85, 85) for decode, then flatten back
        mean_pred = mean_pred.reshape(mean_pred.shape[0], RESOLUTION[0], RESOLUTION[1])
        mean_pred = y_normalizer.decode(mean_pred)
        mean_pred = mean_pred.reshape(mean_pred.shape[0], -1)

        # Variance: var_y = std^2 * var_normalized
        var_pred = var_pred.reshape(var_pred.shape[0], RESOLUTION[0], RESOLUTION[1])
        var_pred = var_pred * (std ** 2)
        var_pred = var_pred.reshape(var_pred.shape[0], -1)

    # 9. Evaluate
    y_t_cpu = y_t.detach().cpu()
    mse = torch.nn.functional.mse_loss(mean_pred, y_t_cpu).item()
    rel_err = relative_l2_error(mean_pred, y_t_cpu).item()

    print("-" * 50)
    print("  Evaluation Results:")
    print(f"    MSE              : {mse:.6f}")
    print(f"    Relative L2 error: {rel_err:.6f}")
    print(f"    Relative L2 (%)  : {100.0 * rel_err:.2f}%")
    print("-" * 50)

    # 10. Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predictions + losses as .npz
    npz_path = os.path.join(SAVE_DIR, f"results_{timestamp}.npz")
    np.savez(
        npz_path,
        mean_pred=mean_pred.detach().cpu().numpy(),
        var_pred=var_pred.detach().cpu().numpy(),
        y_test=y_t_cpu.numpy(),
        elbo_history=np.array(elbo_history),
        mse_history=np.array(mse_history),
    )
    print(f"\nResults saved to {npz_path}")

    # Save model checkpoint as .pt
    pt_path = os.path.join(SAVE_DIR, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Model saved to {pt_path}")

    print(f"\nDone. MSE={mse:.6f} | Relative L2={100.0 * rel_err:.2f}%")


if __name__ == "__main__":
    main()
