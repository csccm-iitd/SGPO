"""Figures for Navier-Stokes 1024^2 from ns1024_eval.py --save output.

ns1024_field.png: a typical test field (median error) at full resolution, forcing, truth, mean and |error|, with a
zoomed 128 x 128 patch below. ns1024_cases.png: field-level predicted sigma against actual error for every test field.

    python experiments/ns1024_figs.py results/apps/ns1024_m32_gp_const/fig_data_group.pt --out project_page/static/images
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
from scipy.stats import spearmanr

INK, MUTED, GRID, BLUE, ORANGE = "#1f2430", "#6b7280", "#e5e7eb", "#3b5bdb", "#e8590c"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.edgecolor": GRID,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
p = argparse.ArgumentParser(); p.add_argument("data"); p.add_argument("--out", required=True)
p.add_argument("--patch", type=int, default=128, help="size of the zoomed patch, centred on the largest local error")
a = p.parse_args()
D = torch.load(a.data, weights_only=True)
os.makedirs(a.out, exist_ok=True)

# ---- field figure: the median-error case of the three saved ones
i = 1
x, y, mu, sd = (D[k][i].numpy() for k in ("x", "y", "mu", "sd"))
err = np.abs(mu - y)
w = a.patch                                                   # zoom where the error is largest (box-averaged)
box = torch.nn.functional.avg_pool2d(torch.tensor(err)[None, None], w, stride=w // 4)[0, 0].numpy()
br, bc = np.unravel_index(box.argmax(), box.shape); r0, c0 = br * (w // 4), bc * (w // 4)
lim = np.percentile(np.abs(y), 99.5); xl = np.percentile(np.abs(x), 99.5)
emax = np.percentile(np.concatenate([sd.ravel(), err.ravel()]), 99.5)
emin = max(np.percentile(sd, 1) / 2, emax / 1e3)
panels = [("Forcing", x, dict(cmap="PuOr_r", vmin=-xl, vmax=xl)), ("Truth", y, dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
          ("LoGoS-GPO mean", mu, dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
          ("|error|", np.maximum(err, emin), dict(cmap="magma", norm=LogNorm(emin, emax, clip=True)))]
fig, ax = plt.subplots(2, 4, figsize=(14, 7.2), gridspec_kw={"wspace": 0.05, "hspace": 0.12})
fig.subplots_adjust(right=0.9)
for j, (t, f, kw) in enumerate(panels):
    ax[0, j].imshow(f, origin="lower", interpolation="antialiased", **kw)
    ax[0, j].add_patch(Rectangle((c0, r0), w, w, fill=False, ec="#22d3ee", lw=1.4))
    ax[0, j].set_title(t)
    im = ax[1, j].imshow(f[r0:r0 + w, c0:c0 + w], origin="lower", interpolation="nearest", **kw)
    for aa in ax[:, j]:
        aa.set_xticks([]); aa.set_yticks([])
        for sp in aa.spines.values():
            sp.set_visible(False)
    for sp in ax[1, j].spines.values():
        sp.set_visible(True); sp.set_edgecolor("#22d3ee"); sp.set_linewidth(1.4)
cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.3, 0.01, 0.4])); cb.outline.set_visible(False)
ax[0, 0].set_ylabel("1024 × 1024", color=MUTED); ax[1, 0].set_ylabel(f"zoom, {w} × {w}", color=MUTED)
fig.savefig(f"{a.out}/ns1024_field.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- case-level figure: does sigma know which fields are hard?
fe, fs = D["rms_err"].numpy(), D["rms_sd"].numpy()
rho = spearmanr(fs, fe)[0]
fig, a2 = plt.subplots(figsize=(4.8, 4.5))
lo, hi = min(fe.min(), fs.min()) * 0.9, max(fe.max(), fs.max()) * 1.1
a2.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=MUTED)
a2.scatter(fs, fe, s=16, color=BLUE, alpha=0.75, edgecolors="none")
a2.scatter(fs[D["pick"].numpy()][1], fe[D["pick"].numpy()][1], s=60, facecolors="none", edgecolors=ORANGE, lw=1.6)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
a2.set_xlabel("predicted σ (field RMS)"); a2.set_ylabel("actual error (field RMS)")
a2.set_title(f"200 test fields, Spearman ρ = {rho:.2f}"); a2.grid(color=GRID, lw=0.6)
for sp in ("top", "right"):
    a2.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{a.out}/ns1024_cases.png", dpi=170); plt.close(fig)
print("wrote ns1024_field.png, ns1024_cases.png; rho", round(rho, 3))
