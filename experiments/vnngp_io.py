"""Builds the stage-1 operator from its saved arguments and loads a trained LoGoS-GPO run (train_logos_vnngp.py output)."""
import json, os
import torch
from sgpo.data.benchmarks import domain_mask, load_split
from sgpo.models.logos_vnngp import LoGoSVNNGP
from sgpo.operators import EWNO, FNO, WNO
from sgpo.operators.spacetime import SpaceTime


def build_operator(A, grid, cin):
    arch = A["arch"]
    if A.get("spacetime") and arch in ("wno", "wnof"):
        times = torch.load(A["spacetime"], weights_only=True)
        return SpaceTime(WNO((*grid, len(times)), A["width"], A["level"], A["layers"], A["wavelet"], in_ch=cin + 1, out_ch=1,
                             pad=A["pad"], modes=A["modes"] if arch == "wnof" else None, fine=not A.get("no_fine", False),
                             wave_dims=A.get("wave_dims")), times)
    if arch in ("wno", "wnof"):
        return WNO(grid, A["width"], A["level"], A["layers"], A["wavelet"], in_ch=cin, out_ch=A.get("out_ch", 1), pad=A["pad"],
                   modes=A["modes"] if arch == "wnof" else None, fine=not A.get("no_fine", False))
    if arch == "ewno":
        return EWNO(grid, A["width"], A["level"], A["layers"], modes=A["modes"][0], pad=A["pad"])
    if arch == "fno":
        return FNO(A["modes"], A["width"], A["layers"], in_ch=cin, out_ch=A.get("out_ch", 1), pad=A["pad"])
    if arch == "ufno":                                  # pre-trained U-FNO mean (CO2), see experiments/ufno_mean.py
        from experiments.ufno_mean import UFNOMean
        return UFNOMean(A["modes"], A["ufno_width"])
    raise ValueError(arch)


@torch.no_grad()
def load_run(run, dev="cuda", batch=25):
    """Returns a dict with test inputs x, truth y, mean mu, calibrated std sd (all (n, d) in physical units, float64),
    grid, coords (d, D), domain mask dom (d,), the metrics, and what posterior draws need: the calibrated latent std
    lat_sd (n,) in normalised units, the output scale ysf and the spatial Vecchia kernel kx."""
    M = json.load(open(f"{run}/metrics.json")); S = M["args"]; A = M["init"]
    init = S["init"] if os.path.exists(f"{S['init']}/ckpt.pt") else os.path.join(os.path.dirname(run), "op")   # moved run trees
    C = torch.load(f"{init}/ckpt.pt", weights_only=False, map_location="cpu")
    _, _, (xte, yte) = load_split(A["data"], A["ntrain"], A["nval"], A["ntest"], A["tail_test"])
    grid, cin = tuple(xte.shape[1:-1]), xte.shape[-1]
    nout = A.get("out_ch", 1)
    gout = (*grid, nout) if nout > 1 else grid
    axes = [torch.linspace(0, 1, n) for n in gout]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(gout))
    model = LoGoSVNNGP(build_operator(A, grid, cin), coords, A["width"], n_inducing=S["inducing"], embed_dim=S["embed"], k_x=S["kx"],
                       noise=S.get("noise", "pixel"), n_out=nout, head_out=1 if A.get("spacetime") or A["arch"] == "ufno" else nout,
                       kernel_x=S.get("kernel_x", "matern52"), kernel_a=S.get("kernel_a", "matern52"),
                       noise_head=S.get("noise_head", "linear"))
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location="cpu")); model = model.to(dev).eval()
    xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
    ysf = ys.reshape(1, -1) if ys.numel() > 1 else ys
    mus, sds, lats = [], [], []
    for i in range(0, len(xte), batch):
        _, mu, var = model((xte[i:i + batch].to(dev) - xm) / xs)
        mus.append((mu.reshape(-1, *grid, nout) * ys + ym).flatten(1))
        sds.append((var[:, None] + model.noise_var()).sqrt() * ysf * M["tau"])
        lats.append(var.sqrt() * M["tau"])
    mu = torch.cat(mus).double()
    sd = torch.cat(sds).double().expand_as(mu).contiguous()
    dom = domain_mask(A["data"])
    dom = torch.ones(mu.shape[1], dtype=torch.bool) if dom is None else dom
    return {"x": xte.double(), "y": yte.flatten(1).double(), "mu": mu.cpu(), "sd": sd.cpu(), "grid": grid,
            "coords": coords, "dom": dom, "metrics": M, "data": A["data"],
            # posterior function draws for test field i: mu[i] + lat_sd[i] * ysf * kx.sample(n)
            "lat_sd": torch.cat(lats).double().cpu(), "ysf": ysf.flatten().double().cpu(), "kx": model.kx}
