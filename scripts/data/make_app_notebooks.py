"""Builds and executes the dataset notebooks of the two applications:
    notebooks/data_ns1024.ipynb   Navier-Stokes Re=500 vorticity, 1024 x 1024 (neuraloperator Zenodo 12825163)
    notebooks/data_co2.ipynb      CO2 storage gas saturation, U-FNO (Wen et al. 2022)
Each asks: what is in the data, how hard is the map, and what does that imply for training.

    python scripts/data/make_app_notebooks.py [--no-exec]
"""
import argparse, os
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = '''import sys; sys.path.insert(0, "..")
import math, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rel = lambda p, t: ((p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1))'''

NS = [
    md("""# Navier–Stokes, Re = 500, 1024 × 1024 (application)

**Source:** neuraloperator Zenodo record 12825163 (`nsforcing_1024_{train,test}.pt`, CC BY 4.0), the benchmark of
Kossaifi et al. 2023 (MG-TFNO, arXiv:2310.00120). 2D incompressible Navier–Stokes on the torus at Reynolds number 500
with zero initial vorticity; **the input is the forcing f** (a Gaussian random field, covariance 27(−Δ+9I)⁻⁴) and the
output the vorticity at T = 5. The map is exactly translation-equivariant. 1000 train + 1000 test fields at 1024²;
a separate 128² release has 10000 train + 2000 test. Published test rL2 (trained at 128², 10k samples):
FNO 1.34%, TFNO 0.22–0.47%, TFNO tested zero-shot at 1024² 0.3%.

Loader: `sgpo.data.benchmarks.ns1024` (`ns1024@sub=k` keeps every k-th point). Split used here: train 900, val 100
from the train file, test 200 from the official test file. The sections below look at how hard the map is,
independently of any model."""),
    code(SETUP + '''
from sgpo.data.benchmarks import RAW
RAW = f"{RAW}/zenodo/ns1024"
tr = torch.load(f"{RAW}/nsforcing_1024_train.pt", mmap=True, weights_only=True)
te = torch.load(f"{RAW}/nsforcing_1024_test.pt", mmap=True, weights_only=True)
NTR, NTE = 300, 100                                  # subsets keep the notebook light (full fields, 1024^2)
x, y = tr["x"][:NTR].clone(), tr["y"][:NTR].clone()
xt, yt = te["x"][:NTE].clone(), te["y"][:NTE].clone()
pd.DataFrame({k: {"shape": tuple(v.shape), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
              for k, v in {"x (train)": x, "y (train)": y, "x (test)": xt, "y (test)": yt}.items()}).T'''),
    md("## What the fields look like\nFull field and a 128 × 128 zoom: the fine-scale filaments are what a 1024² model must get right."),
    code('''fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
for j in range(2):
    for c, (v, t) in enumerate([(x[j], "input forcing"), (y[j], "output vorticity")]):
        l = v.abs().max()
        ax[j, 2 * c].imshow(v, cmap="RdBu_r", vmin=-l, vmax=l); ax[j, 2 * c].set_title(f"{t}, sample {j}")
        ax[j, 2 * c + 1].imshow(v[:128, :128], cmap="RdBu_r", vmin=-l, vmax=l); ax[j, 2 * c + 1].set_title("zoom 128²")
for a in ax.flat: a.axis("off")
plt.tight_layout()'''),
    md("## Energy spectra\nTurbulence puts energy at many scales. The share of output energy at high wavenumbers bounds what any smooth, low-mode model can capture."),
    code('''def spectrum(t):
    F = torch.fft.fft2(t.double()).abs().pow(2).mean(0); n = t.shape[-1]
    k = torch.fft.fftfreq(n, 1 / n); r = (k[None] ** 2 + k[:, None] ** 2).sqrt().round().long()
    return torch.zeros(r.max() + 1, dtype=torch.float64).scatter_add_(0, r.flatten(), F.flatten())[: n // 2]
Ex, Ey = spectrum(x[:50]), spectrum(y[:50])
kk = torch.arange(1, len(Ex))
plt.figure(figsize=(5.6, 3.4)); plt.loglog(kk, Ex[1:], label="input (forcing)"); plt.loglog(kk, Ey[1:], label="output (vorticity)")
plt.loglog(kk, Ey[8] * (kk / 8.0) ** -3, "k:", lw=0.8, label="k^-3 (enstrophy cascade)")
plt.xlabel("wavenumber k"); plt.ylabel("E(k)"); plt.legend(); plt.title("vorticity spectra")
cum = Ey.cumsum(0) / Ey.sum()
pd.Series({f"output energy above k={k}": float(1 - cum[k]) for k in (8, 16, 32, 64, 128, 256)})'''),
    md("## How much of the output is the input?\nWith zero initial vorticity, the response to a forcing is linear at early times and becomes nonlinear by T = 5; `y ≈ a·x` and a per-mode linear map measure how much is linear."),
    code('''a_ = ((x * y).sum() / (x * x).sum()).item()
corr = torch.stack([torch.corrcoef(torch.stack([x[i].flatten(), y[i].flatten()]))[0, 1] for i in range(50)])
def lowpass(t, kmax):
    F = torch.fft.rfft2(t); n = t.shape[-1]; ky = torch.fft.fftfreq(n, 1 / n).abs()[:, None]; kx = torch.arange(F.shape[-1])[None]
    return torch.fft.irfft2(F * ((ky <= kmax) & (kx <= kmax)), s=t.shape[-2:])
rows = {"predict zero (= 100%)": rel(0 * yt, yt).mean(), "persistence y = a x": rel(a_ * xt, yt).mean(),
        "train-mean field": rel(y.mean(0).expand_as(yt), yt).mean()}
for k in (16, 32, 64):
    rows[f"truth low-passed to k<={k} (resolution floor)"] = rel(lowpass(yt, k), yt).mean()
print(f"pointwise corr(x, y) per sample: median {corr.median():.3f}; persistence gain a = {a_:.3f}")
pd.Series({k: f"{100 * float(v):.1f}%" for k, v in rows.items()}, name="test rL2")'''),
    md("## Is the map learnable from the data we have? Nearest-neighbour test\nFor each test input, the training input closest to it (at 64², after removing the mean): if close inputs give very different outputs, the map is chaotic at this horizon and errors stay high whatever the model."),
    code('''s = 16                                                    # compare at 64^2
f = lambda t: t[:, ::s, ::s].flatten(1)
D = torch.cdist(f(xt), f(x)); j = D.argmin(1)
dx = (D.min(1).values / f(xt).norm(dim=1))
dy = rel(y[j], yt)
print(f"nearest training input: median relative input distance {dx.median():.3f}; output rL2 of copying its output {100 * dy.median():.1f}%")
plt.figure(figsize=(4.6, 3.4)); plt.scatter(dx, dy, s=10); plt.xlabel("relative distance to nearest training input"); plt.ylabel("rL2 of its output")
plt.title("sensitivity of the map")'''),
    md("## A linear operator baseline\nA per-Fourier-mode linear map (the operator a single linear spectral layer can represent), fitted by least squares at 128² on the training subset. Its error is a floor for what *linear* structure explains."),
    code('''s = 8; n = 1024 // s
X, Y = torch.fft.rfft2(x[:, ::s, ::s].double()), torch.fft.rfft2(y[:, ::s, ::s].double())
G = (X.conj() * Y).sum(0) / ((X.abs() ** 2).sum(0) + 1e-12)       # per-mode least squares
Xt = torch.fft.rfft2(xt[:, ::s, ::s].double())
pred = torch.fft.irfft2(G * Xt, s=(n, n))
print(f"diagonal Fourier linear map at {n}²: test rL2 {100 * rel(pred, yt[:, ::s, ::s].double()).mean():.1f}%")'''),
    md("## Train vs test distribution"),
    code('''Et = spectrum(xt[:50]); plt.figure(figsize=(5.2, 3.2))
plt.loglog(kk, Ex[1:], label="train input"); plt.loglog(kk, Et[1:], "--", label="test input"); plt.legend(); plt.xlabel("k")
pd.Series({"train x std": float(x.std()), "test x std": float(xt.std()), "train y std": float(y.std()), "test y std": float(yt.std())})'''),
    md("""## Takeaways for training
Read the numbers above together:
* **Resolution floor:** the low-pass rows give the error of the *truth itself* truncated at k; a model that only resolves
  low modes cannot beat them.
* **Nearest neighbour:** independent forcings are far apart (relative distance ~1), so copying a neighbour's output
  fails; this says the input space is not densely sampled, not that the map is chaotic (FNO reaches 1.34%).
* **Symmetry:** the map is translation-equivariant; a model with location-specific weights (the WNO's coarse wavelet
  kernel) should be trained with random shifts (`--aug shift2d`), otherwise it can fit positions instead of the map.
* **Linear baseline:** a stage-1 operator should beat the diagonal Fourier map clearly; if it does not, training (not the
  data) is the problem."""),
]

CO2 = [
    md("""# CO2 geological storage: gas saturation (application)

**Source:** Wen et al. 2022, *U-FNO — an enhanced Fourier neural operator-based deep-learning model for multiphase flow*,
Advances in Water Resources 163:104180 (data and code: github.com/gegewen/ufno). Radial (r, z) reservoir models of CO2
injection through one well, simulated with ECLIPSE; 4500 train / 500 val / 500 test.

**Input** `(96, 200, 24, 12)` per sample: channels `[kr, kz, porosity, injection location, injection rate, pressure,
temperature, Swi, lambda, grid_r, grid_z, grid_t]`; only the time channel varies over the 24 snapshots, so the loader
keeps the 11 static channels at t=0. **Output** `(96, 200, 24)`: gas saturation over 24 log-spaced times (1 day to 30 years).
Reservoirs differ in thickness: cells below the reservoir are zero-padded, and channel 0 (kr) is the active-cell mask.

Loader: `sgpo.data.benchmarks.co2_sg`, `load_split("co2_sg", 4500, 500, 500)`."""),
    code(SETUP + '''
from sgpo.data.benchmarks import load_split
(xtr, ytr), (xva, yva), (xte, yte) = load_split("co2_sg", 4500, 500, 500)
NAMES = ["kr", "kz", "porosity", "inj_loc", "inj_rate", "pressure", "temperature", "Swi", "lambda", "grid_r", "grid_z"]
act = lambda x: (x[..., 0] != 0)
pd.DataFrame({n: {"min": float(xtr[..., i].min()), "max": float(xtr[..., i].max()), "per-sample constant": bool((xtr[:50, ..., i].flatten(1).std(1) == 0).all())}
              for i, n in enumerate(NAMES)}).T'''),
    md("## Reservoirs and plumes"),
    code('''T = [0, 8, 14, 18, 21, 23]
fig, ax = plt.subplots(3, 1 + len(T), figsize=(15, 5.4))
for r, i in enumerate((0, 1, 2)):
    m = act(xtr[i]); th = int(m[:, 0].sum())
    ax[r, 0].imshow(xtr[i, :th, :, 0], aspect="auto", cmap="viridis"); ax[r, 0].set_title(f"log kr, thickness {th} cells")
    for c, t in enumerate(T):
        ax[r, c + 1].imshow(ytr[i, :th, :, t], aspect="auto", cmap="inferno", vmin=0, vmax=1); ax[r, c + 1].set_title(f"saturation t{t}")
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
plt.tight_layout()'''),
    md("## Sparsity over time\nThe plume grows from a few cells to a large region; early snapshots are almost empty. Relative errors averaged over all 24 snapshots weight the late, large plumes."),
    code('''A = act(xtr[:500])[..., None].expand(-1, -1, -1, 24)
plume = (ytr[:500] > 0) & A
frac = plume.float().sum((1, 2)) / A.float().sum((1, 2))
norm_t = (ytr[:500] * A).flatten(1, 2).norm(dim=1)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].plot(frac.mean(0)); ax[0].set_title("plume fraction of active cells"); ax[0].set_xlabel("snapshot")
ax[1].semilogy(norm_t.mean(0)); ax[1].set_title("output norm per snapshot"); ax[1].set_xlabel("snapshot")
th = act(xtr)[:, :, 0].sum(1); ax[2].hist(th.numpy(), 40); ax[2].set_title("reservoir thickness (cells)")
pd.Series({"output cells that are zero": float((ytr[:500] == 0).float().mean()), "active cells": float(A.float().mean()),
           "plume cells among active (all t)": float(plume.float().sum() / A.float().sum())})'''),
    md("## Why per-cell normalisation fails here\nThe per-cell standard deviation over training samples is ~0 wherever plumes rarely reach; dividing by it inflates those cells' targets by orders of magnitude."),
    code('''sd = ytr.std(0)                                            # (96, 200, 24)
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].imshow(sd[..., 23], aspect="auto", cmap="magma"); ax[0].set_title("per-cell std of saturation, last snapshot")
ax[1].hist(torch.log10(sd.flatten() + 1e-12).numpy(), 60); ax[1].set_title("log10 per-cell std (all cells, all t)")
print(f"cells with std < 1e-4: {100 * float((sd < 1e-4).float().mean()):.1f}%")'''),
    md("## How hard is the map? Baselines\nMasked relative L2 over active cells and all snapshots (the stage-1 training metric), and the plume MAE of Wen et al. (plume = cells non-zero in data or prediction)."),
    code('''def masked_rel(p, x, y):
    m = act(x)[..., None].float(); return rel(p * m, y * m)
def plume_mae(p, x, y, thr=0.0):
    out = []
    for i in range(len(y)):
        m = act(x[i])[..., None].expand_as(y[i]); pl = m & ((y[i] > 0) | (p[i] > thr))
        out.append((p[i] - y[i])[pl].abs().mean())
    return torch.stack(out)
pool = lambda x: torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), 8).flatten(1)
D = torch.cdist(pool(xte[:, :, :, :9]), pool(xtr[:, :, :, :9]))          # reservoir fields + scalars
nn_pred = ytr[D.argmin(1)] * act(xte)[..., None]
mean_pred = ytr.mean(0, keepdim=True).expand_as(yte) * act(xte)[..., None]
pd.DataFrame({"masked rL2 %": {"train mean": 100 * float(masked_rel(mean_pred, xte, yte).mean()),
                               "nearest training sample": 100 * float(masked_rel(nn_pred, xte, yte).mean())},
              "plume MAE": {"train mean": float(plume_mae(mean_pred, xte, yte).mean()),
                            "nearest training sample": float(plume_mae(nn_pred, xte, yte).mean())}})'''),
    md("## What drives the plume?\nPlume size at 30 years against the scalar controls and the reservoir's mean permeability."),
    code('''size = ((ytr[..., 23] > 0) & act(xtr)).float().sum((1, 2))
feat = {"injection rate": xtr[:, 0, 0, 4], "pressure": xtr[:, 0, 0, 5], "temperature": xtr[:, 0, 0, 6], "Swi": xtr[:, 0, 0, 7],
        "lambda": xtr[:, 0, 0, 8], "thickness": act(xtr)[:, :, 0].sum(1).float(),
        "mean log kr (active)": (xtr[..., 0] * act(xtr)).sum((1, 2)) / act(xtr).sum((1, 2))}
pd.Series({k: float(torch.corrcoef(torch.stack([v.float(), size]))[0, 1]) for k, v in feat.items()}, name="corr with plume size").round(3)'''),
    md("""## Takeaways for training
* **No per-cell normalisation:** many cells have ~0 spread (see above); use the data as given (`--norm none`), as U-FNO does.
* **Mask everything to active cells** (loss, metrics, GP likelihood, calibration), per sample.
* **Time is a dimension, not just channels:** the plume at t is the plume at t−1 grown; a model that shares structure
  across snapshots (3D operator over r, z, t, or an autoregressive-in-time head) uses that; 24 independent output channels do not.
* **Metric:** the plume MAE depends strongly on how a "non-zero prediction" is thresholded; compare methods under one definition."""),
]


def build(cells, path, execute):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if execute:
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    nbf.write(nb, path); print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--no-exec", action="store_true"); p.add_argument("--which", default="ns,co2")
    a = p.parse_args()
    if "ns" in a.which:
        build(NS, os.path.join(ROOT, "notebooks/data_ns1024.ipynb"), not a.no_exec)
    if "co2" in a.which:
        build(CO2, os.path.join(ROOT, "notebooks/data_co2.ipynb"), not a.no_exec)
