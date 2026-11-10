"""Evaluate a LoGoS-GPO run on the 1024^2 Navier-Stokes release, optionally averaged over the symmetry group.

The forcing -> vorticity map on the torus commutes with rotations, and with reflections up to a sign (vorticity is a
pseudo-scalar). With --group the prediction is the mixture over the 8 group elements g: mean = average of
s_g g^-1 mu(s_g g x), variance = average variance + variance of the means. sigma is rescaled by one factor fitted on the
second half of the validation set (as in train_logos_vnngp.py). Reports rL2, NLL, 95% coverage and the rank
correlation of sigma with |error| within fields and across test cases.

    python experiments/ns1024_eval.py results/apps/ns1024_m32_gp --group
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from scipy.stats import spearmanr
from sgpo.data.benchmarks import load_split
from sgpo.models.logos_vnngp import LoGoSVNNGP
from sgpo.operators import FNO
from sgpo.operators.resample import Resample
from sgpo.operators.symmetry import d4, d4_inv

p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--group", action="store_true")
p.add_argument("--batch", type=int, default=2)
a = p.parse_args()
dev = "cuda"
M = json.load(open(f"{a.run}/metrics.json")); S = M["args"]
C = torch.load(f"{S['init']}/ckpt.pt", weights_only=False, map_location="cpu"); A = C["args"]
_, (xva, yva), (xte, yte) = load_split(S["data"], *S["split"], True)
n = xte.shape[1]
op = FNO(A["modes"], A["width"], A["layers"], in_ch=1, out_ch=1, pad=A["pad"]); op.load_state_dict(C["state"])
op = Resample(op, S["resample"])
ax = torch.linspace(0, 1, n)
coords = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
model = LoGoSVNNGP(op, coords, A["width"], n_inducing=S["inducing"], embed_dim=S["embed"], k_x=S["kx"], noise=S["noise"],
                   n_out=1, head_out=1, kernel_x=S.get("kernel_x", "matern52"), kernel_a=S.get("kernel_a", "matern52"),
                   noise_head=S.get("noise_head", "linear"))
model.load_state_dict(torch.load(f"{a.run}/model.pt", map_location="cpu")); model = model.to(dev).eval()
xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
G = [(k, -1.0 if k & 4 else 1.0) for k in range(8)] if a.group else [(0, 1.0)]


@torch.no_grad()
def predict(x):
    mus, vs = [], []
    for i in range(0, len(x), a.batch):
        xb = (x[i:i + a.batch].to(dev) - xm) / xs                                  # (b, n, n, 1)
        m_k, v_k = [], []
        for k, sg in G:
            _, mu, var = model(sg * d4(xb, k))
            v = (var[:, None] + model.noise_var()).expand_as(mu)
            m_k.append(sg * d4_inv(mu.reshape(-1, n, n, 1), k)); v_k.append(d4_inv(v.reshape(-1, n, n, 1), k))
        m_k, v_k = torch.stack(m_k), torch.stack(v_k)
        mus.append((m_k.mean(0) * ys + ym).flatten(1).cpu()); vs.append(((v_k.mean(0) + m_k.var(0, unbiased=False)) * ys ** 2).flatten(1).cpu())
    return torch.cat(mus).double(), torch.cat(vs).double()


mv, vv = predict(xva); mt, vt = predict(xte)
yv, yt = yva.flatten(1).double(), yte.flatten(1).double()
h = len(yv) // 2
tau = (((yv[h:] - mv[h:]) ** 2 / vv[h:]).mean()).sqrt().item()
res = {"group": a.group, "tau": tau}
for name, sc in (("raw", 1.0), ("calibrated", tau)):
    s = vt.sqrt() * sc; e = mt - yt
    res[name] = {"rl2_pct": 100 * (e.norm(dim=1) / yt.norm(dim=1)).mean().item(),
                 "nll": (0.5 * (e / s) ** 2 + s.log() + 0.5 * math.log(2 * math.pi)).mean().item(),
                 "cov95": (e.abs() <= 1.96 * s).double().mean().item()}
e, s = (mt - yt).abs(), vt.sqrt()
idx = torch.randperm(e.shape[1], generator=torch.Generator().manual_seed(0))[:200_000]      # subsample per field
res["rho_point"] = float(np.nanmean([spearmanr(s[i, idx], e[i, idx])[0] for i in range(len(e))]))
res["rho_case"] = float(spearmanr(s.pow(2).mean(1).sqrt(), e.pow(2).mean(1).sqrt())[0])
res["outputs_per_field"] = e.shape[1]
print(json.dumps(res, indent=1))
json.dump(res, open(f"{a.run}/eval{'_group' if a.group else ''}.json", "w"), indent=1)
