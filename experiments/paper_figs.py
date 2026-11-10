"""Figures and Table 3 of the paper (arXiv 2506.15906) from trained runs.

Prediction figures: 1D (Burgers, wave advection) 3x3 grid of input / truth + mean + 95% CI / signed error for three
test samples (25th, 50th, 75th error percentile); 2D (Darcy notch, NS) 1x5 input / truth / prediction / std / |error|.
Runs are read from --root in the layout written by scripts/reproduce.sh:
    <root>/<case>/s<seed>/{op,logos,gpo,svgp}   stage 1, LoGoS-GPO, GPO and SVGP baselines
    <root>/sweeps/<bench>/<point>/{op,logos,gpo} grid-size and training-size sweeps (optional <point>/timing)
    <root>/flops/<bench>.json                    FLOPs study (experiments/flops.py)
Output: <out>/*.pdf and .png.

    python experiments/paper_figs.py --which all --root runs --out figures
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Polygon
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from experiments.vnngp_io import load_run

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "axes.labelsize": 13, "axes.titlesize": 15, "xtick.labelsize": 10, "ytick.labelsize": 10})
RUNS = {"burgers": "burgers/s0/logos", "wave_advection": "advection/s1/logos",
        "darcy_notch": "darcy_notch_l2/s0/logos", "navier_stokes": "navier_stokes_l2/s0/logos"}
S3 = 3 ** 0.5 / 2
NOTCH = [(0, 0), (0.49, 0), (0.49, 0.4), (0.51, 0.4), (0.51, 0), (1, 0), (0.5, S3)]


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{out}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote", f"{out}/{name}.pdf")


def picks(R, q=(0.25, 0.5, 0.75)):
    d = R["dom"]
    e = (R["mu"][:, d] - R["y"][:, d]).norm(dim=1) / R["y"][:, d].norm(dim=1)
    o = e.argsort()
    return [o[int(t * (len(o) - 1))].item() for t in q]


def pred_1d(R, name, out, in_label="u", t_out="1"):
    x = R["coords"][:, 0].numpy()
    fig, ax = plt.subplots(3, 3, figsize=(13, 7.2), sharex=True)
    for c, i in enumerate(picks(R)):
        k = c + 1
        ax[0, c].plot(x, R["x"][i, :, 0], color="#3b3b98", lw=1.2, label="Test Input")
        mu, sd, y = R["mu"][i], R["sd"][i], R["y"][i]
        ax[1, c].fill_between(x, mu - 1.96 * sd, mu + 1.96 * sd, color="#f9d9a6", alpha=0.8, label="95% CI")
        ax[1, c].plot(x, mu, color="#5dade2", lw=1.4, label="Prediction")
        ax[1, c].plot(x, y, "k--", lw=1.6, label="Truth")
        ax[2, c].plot(x, mu - y, color="#ff7f50", lw=1.2, label="Error")
        ax[0, c].set_ylabel(rf"${in_label}^{{({k})}}(x, 0)$"); ax[1, c].set_ylabel(rf"${in_label}^{{({k})}}(x, {t_out})$")
        ax[2, c].set_ylabel(rf"$\varepsilon^{{({k})}}(x)$"); ax[2, c].set_xlabel(r"$x$")
        if c == 0:
            for r in range(3):
                ax[r, c].legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout(); save(fig, out, name)


def fill_outside(f, dom2):
    if dom2.all():
        return f
    _, near = distance_transform_edt(~dom2, return_indices=True)
    return f[near[0], near[1]]


def pred_2d(R, name, out, idx, cmap, cmap_err, in_title="Input"):
    g = R["grid"]; dom2 = R["dom"].reshape(g).numpy()
    y, mu, sd = R["y"][idx], R["mu"][idx], R["sd"][idx]
    fields = [(R["x"][idx][..., 0].numpy(), in_title, cmap, None), (y.reshape(g).numpy(), "Truth", cmap, None),
              (mu.reshape(g).numpy(), "Prediction", cmap, None), (sd.reshape(g).numpy(), "Std. Deviation", cmap, None),
              ((mu - y).abs().reshape(g).numpy(), "Error", cmap_err, None)]
    lo, hi = float(y[R["dom"]].min()), float(y[R["dom"]].max())
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
    for c, (f, title, cm, _) in enumerate(fields):
        vr = (lo, hi) if title in ("Truth", "Prediction") else (None, None)
        f = fill_outside(f, dom2)
        dm = f[dom2] if not dom2.all() else f
        h = ax[c].imshow(f, origin="lower", extent=(0, 1, 0, 1), cmap=cm, interpolation="bilinear",
                         vmin=vr[0] if vr[0] is not None else dm.min(), vmax=vr[1] if vr[1] is not None else dm.max())
        if not dom2.all():
            poly = Polygon(NOTCH, closed=True, fc="none", ec="none"); ax[c].add_patch(poly); h.set_clip_path(poly)
            ax[c].set_ylim(0, 1)
        ax[c].set_title(title); ax[c].set_xlabel("x")
        if not dom2.all():
            ax[c].set_xticks([0, 0.5, 1]); ax[c].set_yticks([0, 0.5, 1] if c == 0 else [])
        else:
            ax[c].set_xticks([]); ax[c].set_yticks([])
        fig.colorbar(h, ax=ax[c], fraction=0.046, pad=0.04)
    ax[0].set_ylabel("y")
    fig.tight_layout(); save(fig, out, name)


import glob, json
from statistics import mean, stdev

# Case directory of each row of Table 3 (seeds 0-2); rows marked "level 2" use wavelet level 2, the largest level of
# db6 at 51^2 and 64^2, instead of the paper's wavelet depth.
SEEDS = {"Burger": "burgers", "Wave Advection": "advection", "Darcy (notch)": "darcy_notch", "Navier Stokes": "navier_stokes",
         "Darcy (notch), level 2": "darcy_notch_l2", "Navier Stokes, level 2": "navier_stokes_l2"}
NSEED = 3
SVGP_FROM = {"Darcy (notch), level 2": "Darcy (notch)", "Navier Stokes, level 2": "Navier Stokes"}   # zero mean: operator-free


def _ms(v):
    v = [100 * x for x in v]
    return (f"{mean(v):.2f} \\pm {stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}") + (f" ({len(v)})" if len(v) < 3 else "") if v else "-"


def _read(case, kind, key):
    """Metric `key` of <case>/s<k>/<kind> for every seed that exists."""
    fs = (f"{ROOT}/{case}/s{k}/{kind}/metrics.json" for k in range(NSEED))
    return [json.load(open(f))[key] for f in fs if os.path.exists(f)]


def table(out):
    """Table 3 of the paper (relative L2 %, mean +- std over seeds): LoGoS-GPO, GPO, SVGP."""
    rows = []
    for name, case in SEEDS.items():
        lg, gp = (_read(case, kind, key) for kind, key in (("logos", "logos"), ("gpo", "gpo")))
        sv = _read(SEEDS[SVGP_FROM.get(name, name)], "svgp", "logos")
        rows.append((name, _ms(lg), _ms(gp), _ms(sv)))
    tex = ["\\begin{tabular}{lccc}", "\\hline",
           "\\textbf{Case} & \\textbf{LoGoS-GPO} & \\textbf{GPO} & \\textbf{SVGP} \\\\", "\\hline"]
    md = ["| Case | LoGoS-GPO | GPO | SVGP |", "|---|---|---|---|"]
    for name, lg, gp, sv in rows:
        tex.append(f"{name} & $\\mathbf{{{lg}}}$ & ${gp}$ & ${sv}$ \\\\")
        md.append("| " + " | ".join([name, lg, gp, sv]).replace("\\pm", "+-") + " |")
    tex += ["\\hline", "\\end{tabular}"]
    open(f"{out}/table3.tex", "w").write("\n".join(tex) + "\n"); open(f"{out}/table3.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


SWEEP_NAMES = {"burgers": "burger", "advection": "wave_adv", "notch": "darcy", "ns": "navier"}


def scaling(out):
    """Scaling figures: L2 error, time per epoch and memory vs grid size (case1) and training samples (case2)."""
    for bench, stem in SWEEP_NAMES.items():
        for kind, fname, xl in (("grid_sub", "case1_grid", "Grid Size"), ("n", "case2_samples", "Number of Samples")):
            pts = []
            for d in glob.glob(f"{ROOT}/sweeps/{bench}/{kind}*"):
                lf, gf = f"{d}/logos/metrics.json", f"{d}/gpo/metrics.json"
                if not os.path.exists(lf):
                    continue
                L = json.load(open(lf)); G = json.load(open(gf)) if os.path.exists(gf) else None
                for M, tf in ((L, f"{d}/timing/logos/metrics.json"), (G, f"{d}/timing/gpo/metrics.json")):
                    if M is not None and os.path.exists(tf):        # optional re-timing of the point on a dedicated GPU
                        T = json.load(open(tf)); M.update(time_per_epoch=T["time_per_epoch"], peak_mem_gb=T["peak_mem_gb"])
                pts.append((L, G))
            if len(pts) < 2:
                continue
            key = (lambda P: P[0]["init"]["ntrain"]) if kind == "n" else (lambda P: _grid_n(P[0]))
            pts.sort(key=key)
            xs = [key(P) for P in pts]
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
            series = [("LoGoS-GPO", "o", "#4c72b0", [P[0] for P in pts], "logos", "time_per_epoch"),
                      ("GPO", "s", "#dd8452", [P[1] for P in pts], "gpo", "time_per_epoch")]
            for lab, mk, col, Ms, ek, tk in series:
                ok = [i for i, m in enumerate(Ms) if m is not None]
                X = [xs[i] for i in ok]
                ax[0].plot(X, [100 * Ms[i][ek] for i in ok], ":", marker=mk, color=col, label=lab, ms=5)
                ax[1].plot(X, [Ms[i].get(tk, Ms[i].get("time_per_step", 0)) / 60 for i in ok], ":", marker=mk, color=col, ms=5)
                ax[2].plot(X, [Ms[i]["peak_mem_gb"] for i in ok], ":", marker=mk, color=col, ms=5)
            for k, yl in enumerate(("L2 Error (%)", "Time per Epoch (min)", "Memory (GB)")):
                ax[k].set_xlabel(xl); ax[k].set_ylabel(yl); ax[k].grid(ls="--", alpha=0.5)
                if kind != "n" and bench in ("notch", "ns"):
                    ax[k].set_xticks(xs); ax[k].set_xticklabels([f"{v}x{v}" for v in xs])
            ax[0].legend(frameon=True)
            fig.tight_layout(); save(fig, out, f"{stem}_{fname}")


def _grid_n(L):
    data = L["init"]["data"]
    base, _, sub = data.partition("@sub=")
    native = {"burgers": 8192, "advection_paper": 200, "darcy_notch": 101, "ns_forcing": 64}[base]
    sub = int(sub) if sub else {"burgers": 8, "darcy_notch": 2}.get(base, 1)
    return -(-native // sub)


def flops_fig(out):
    """FLOPs study: GP-layer training FLOPs per epoch vs grid size and vs training samples, prediction FLOPs per field
    vs grid size. The operator (identical in both models) is shown for reference."""
    for bench, stem in SWEEP_NAMES.items():
        f = f"{ROOT}/flops/{bench}.json"
        if not os.path.exists(f):
            continue
        R = json.load(open(f))
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        two_d = bench in ("notch", "ns")
        for k, (kind, xk, xl, keys) in enumerate((("grid", "grid", "Grid Size", ("logos_gp_train", "gpo_gp_train", "op_train")),
                                                   ("samples", "ntrain", "Number of Samples", ("logos_gp_train", "gpo_gp_train", "op_train")),
                                                   ("grid", "grid", "Grid Size", ("logos_predict", "gpo_predict", None)))):
            P = sorted(R[kind], key=lambda r: r[xk])
            X = [r[xk] for r in P]
            ax[k].plot(X, [r[keys[0]] / 1e9 for r in P], ":o", color="#4c72b0", ms=5, label="LoGoS-GPO")
            ax[k].plot(X, [r[keys[1]] / 1e9 for r in P], ":s", color="#dd8452", ms=5, label="GPO")
            if keys[2]:
                ax[k].plot(X, [r[keys[2]] / 1e9 for r in P], "--", color="#9aa5b1", lw=1, label="operator (both)")
            ax[k].set_yscale("log"); ax[k].set_xscale("log", base=2); ax[k].set_xticks(X)
            ax[k].set_xticklabels([f"{v}x{v}" if two_d and kind == "grid" else str(v) for v in X], rotation=30 if two_d or kind == "samples" else 0)
            ax[k].minorticks_off(); ax[k].set_xlabel(xl); ax[k].grid(ls="--", alpha=0.5)
        ax[0].set_ylabel("GP training GFLOPs / epoch"); ax[1].set_ylabel("GP training GFLOPs / epoch")
        ax[2].set_ylabel("Prediction GFLOPs / field"); ax[0].legend(frameon=True, fontsize=9)
        fig.tight_layout(); save(fig, out, f"{stem}_flops")


p = argparse.ArgumentParser()
p.add_argument("--which", default="predictions", choices=["predictions", "table", "scaling", "flops", "all"])
p.add_argument("--root", default="runs", help="directory holding the trained runs (layout above)")
p.add_argument("--out", default="figures")
a = p.parse_args()
ROOT = a.root
os.makedirs(a.out, exist_ok=True)
if a.which in ("predictions", "all"):
    pred_1d(load_run(ROOT + "/" + RUNS["burgers"]), "burger_predictions", a.out)
    pred_1d(load_run(ROOT + "/" + RUNS["wave_advection"]), "wa_predictions", a.out, t_out="0.5")
    R = load_run(ROOT + "/" + RUNS["darcy_notch"])
    pred_2d(R, "darcy_notch_mean_pred", a.out, picks(R)[1], "seismic", "jet")
    R = load_run(ROOT + "/" + RUNS["navier_stokes"])
    pred_2d(R, "ns_prediction_80", a.out, min(80, len(R["y"]) - 1), "inferno", "jet")
if a.which in ("table", "all"):
    table(a.out)
if a.which in ("scaling", "all"):
    scaling(a.out)
if a.which in ("flops", "all"):
    flops_fig(a.out)
