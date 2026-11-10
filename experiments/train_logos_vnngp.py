"""LoGoS-GPO training: joint ELBO over the neural-operator mean, input embedding and K_a (x) K_x GP.

Stage 1 is a train_operator.py checkpoint on the same split (the operator pretrained with the relative L2 loss);
stage 2 trains everything jointly on the ELBO of sgpo/models/logos_vnngp.py. The report compares the operator
alone (before and after joint training) with the GP predictive mean, plus raw and calibrated uncertainty.

    python experiments/train_logos_vnngp.py --init runs/operator/burgers --out runs/logos/burgers
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from sgpo.data.benchmarks import domain_mask, load_split
from sgpo.models.logos_vnngp import LoGoSVNNGP, SVGP
from sgpo.operators import EWNO, FNO, WNO
from sgpo.operators.resample import Resample
from sgpo.uq import scorecard
from experiments.vnngp_io import build_operator

p = argparse.ArgumentParser()
p.add_argument("--init", required=True, help="train_operator.py output dir (ckpt.pt), defines data split and operator")
p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr-op", type=float, default=3e-5)
p.add_argument("--freeze-frac", type=float, default=0.3, help="fraction of epochs with the operator frozen (GP warm-up)")
p.add_argument("--inducing", type=int, default=256); p.add_argument("--embed", type=int, default=32); p.add_argument("--kx", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--noise", default="pixel", choices=["pixel", "scalar", "feature"],
               help="noise variance per grid point, single, or heteroscedastic from operator features (see LoGoSVNNGP)")
p.add_argument("--noise-head", default="linear", choices=["linear", "mlp"], help="map from operator features to log noise")
p.add_argument("--noise-init", default="const", choices=["const", "residual"],
               help="initial noise variance: 1e-2, or the pretrained operator's training residual variance")
p.add_argument("--kernel-x", default="matern52", help="spatial kernel (matern12/32/52, rbf, rq)")
p.add_argument("--kernel-a", default="matern52", help="input-function kernel")
p.add_argument("--ckpt-every", type=int, default=1, help="epochs between resumable checkpoints (out/resume.pt)")
p.add_argument("--select", default="rl2", choices=["rl2", "nll"],
               help="epoch selection on validation rL2 (default) or on validation NLL at the optimal scale")
p.add_argument("--mask-domain", action="store_true", help="likelihood and tau on in-domain grid points only (irregular domains)")
p.add_argument("--data", default=None, help="fit the GP on another dataset than the operator's (e.g. its fine-grid release)")
p.add_argument("--split", type=int, nargs=3, default=None, help="ntrain nval ntest for --data (tail test)")
p.add_argument("--resample", type=int, default=1,
               help="the operator runs on every k-th grid point and its features are Fourier-upsampled (band-limited data)")
p.add_argument("--save-pred", action="store_true", help="also save the test-set predictive mean (physical units) to out/test_pred.pt")
p.add_argument("--svgp", action="store_true", help="SVGP baseline: zero mean, K_a on the raw input (no operator)")
a = p.parse_args()
torch.manual_seed(a.seed)
dev = torch.device("cuda")
C = torch.load(f"{a.init}/ckpt.pt", weights_only=False, map_location="cpu"); A = C["args"]
if a.data:
    (xtr, ytr), (xva, yva), (xte, yte) = load_split(a.data, *a.split, True)
else:
    (xtr, ytr), (xva, yva), (xte, yte) = load_split(A["data"], A["ntrain"], A["nval"], A["ntest"], A["tail_test"])
grid, cin = xtr.shape[1:-1], xtr.shape[-1]
grid_op = tuple(n // a.resample for n in grid)                         # the operator's own grid
a.inducing = min(a.inducing, len(xtr))                                # M <= N (small-data sweep points)
arch = A["arch"]
if A.get("spacetime"):
    op = build_operator(A, grid, cin)                           # 3D operator over (space, time)
elif arch == "fno":
    op = FNO(A["modes"], A["width"], A["layers"], in_ch=cin, out_ch=A.get("out_ch", 1), pad=A["pad"])
elif arch in ("wno", "wnof"):
    op = WNO(grid_op, A["width"], A["level"], A["layers"], A["wavelet"], in_ch=cin, out_ch=A.get("out_ch", 1), pad=A["pad"],
             modes=A["modes"] if arch == "wnof" else None, fine=not A.get("no_fine", False))
elif arch == "ewno":
    op = EWNO(grid, A["width"], A["level"], A["layers"], modes=A["modes"][0], pad=A["pad"])
elif arch == "ufno":                                    # pre-trained U-FNO mean (CO2), see experiments/ufno_mean.py
    from experiments.ufno_mean import UFNOMean
    op = UFNOMean(A["modes"], A["ufno_width"])
else:
    raise ValueError(arch)
op.load_state_dict(C["state"])
if a.resample > 1:
    assert C["xm"].numel() == 1 and C["ym"].numel() == 1, "resampling needs grid-free (scalar) normalisation"
    op = Resample(op, a.resample)
xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
nx = lambda x: (x.to(dev) - xm) / xs
ny = lambda y: ((y.to(dev) - ym) / ys).flatten(1)
nout = ytr.shape[-1]                                                  # >1: space-time output, one channel per snapshot
dn = lambda f: f.reshape(-1, *grid, nout) * ys + ym                   # back to physical units

gout = (*grid, nout) if nout > 1 else tuple(grid)
axes = [torch.linspace(0, 1, n) for n in gout]
coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(gout))
if a.svgp:
    model = SVGP(coords, grid, cin, n_inducing=a.inducing, embed_dim=a.embed, k_x=a.kx, noise=a.noise).to(dev)
else:
    model = LoGoSVNNGP(op, coords, A["width"], n_inducing=a.inducing, embed_dim=a.embed, k_x=a.kx, noise=a.noise,
                       n_out=nout, head_out=1 if A.get("spacetime") or arch == "ufno" else nout, kernel_x=a.kernel_x, kernel_a=a.kernel_a,
                       noise_head=a.noise_head).to(dev)
Xtr, Ytr = nx(xtr), ny(ytr)
dmask = domain_mask(A["data"]) if a.mask_domain else None
dmask = dmask.to(dev) if dmask is not None else None
# per-sample active cells (stage 1 trained with --mask-ch): outputs are known zeros elsewhere
amask = (lambda x: (x[..., A["mask_ch"]] != 0)[..., None].expand(*x.shape[:-1], nout).flatten(1).to(dev)) \
    if A.get("mask_ch", -1) >= 0 else None
Mtr = amask(xtr) if amask else None
EB = int(max(1, min(50, 2 ** 28 // (len(coords) * 128))))   # evaluation chunk: projection activations stay ~1 GB
model.init_inducing(Xtr, batch=EB)
if a.noise_init == "residual":          # start from the operator's own residual level (strong means: residuals << 0.1)
    with torch.no_grad():
        r2 = torch.cat([(model(Xtr[i:i + EB])[0] - Ytr[i:i + EB]) ** 2 for i in range(0, len(Xtr), EB)])
        w = torch.ones_like(r2) if Mtr is None else Mtr.float()
        if dmask is not None:
            w = w * dmask
        v = (r2 * w).sum(0) / w.sum(0).clamp_min(1)
        v = v.mean().expand_as(v) if model.log_noise.numel() == 1 else torch.where(w.sum(0) > 0, v, v[w.sum(0) > 0].mean())
        model.log_noise.copy_(v.clamp_min(1e-12).log().reshape(model.log_noise.shape) if model.log_noise.numel() > 1
                              else v.mean().clamp_min(1e-12).log().reshape(1))
rel = lambda p_, t: (p_ - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1)


@torch.no_grad()
def evaluate(x, y):
    model.eval()
    m0s, mus, vs = [], [], []
    for i in range(0, len(x), EB):
        m0, mu, var = model(nx(x[i:i + EB]))
        m0s.append(m0); mus.append(mu); vs.append(var[:, None] + model.noise_var())
    m0, mu, v = torch.cat(m0s), torch.cat(mus), torch.cat(vs)
    yd = y.to(dev)
    s = v.sqrt() * (ys.reshape(1, -1) if ys.numel() > 1 else ys)         # std in physical units
    mf, sf, yf, m0f = dn(mu).flatten(1), s.expand_as(dn(mu).flatten(1)), yd.flatten(1), dn(m0).flatten(1)
    w = torch.ones_like(mf)
    if amask is not None:                                  # inactive cells: known zeros, excluded from every score
        w = amask(x).float(); mf, m0f = mf * w, m0f * w
    elif dmask is not None:
        w = w * dmask
    z2, ls = ((yf - mf) / sf) ** 2, sf.log()
    k = max(1, len(x) // 2)                                # validation half 1 only (half 2 calibrates tau)
    wk = w[:k]
    nll = (0.5 * ((z2[:k] * wk).sum() / wk.sum()).log() + (ls[:k] * wk).sum() / wk.sum()).item()   # NLL at the optimal scale
    return {"op": rel(m0f, yf).mean().item(), "gp": rel(mf, yf).mean().item(), "nll": nll}, mf, sf


before, _, _ = evaluate(xte, yte)
print(f"pretrained operator test rL2 {100*before['op']:.4f}%", flush=True)
gp_params = [q for n_, q in model.named_parameters() if not n_.startswith("mean_op.")]
opt = torch.optim.AdamW([{"params": model.mean_op.parameters(), "lr": a.lr_op}, {"params": gp_params, "lr": a.lr, "weight_decay": 0.0}], weight_decay=1e-4)
steps = a.epochs * (len(Xtr) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_op, a.lr], total_steps=steps, pct_start=0.05)
shift = A.get("aug") == "shift"
r0, _, _ = evaluate(xva, yva)                                          # the warm start is a selection candidate
crit = lambda r: r["nll"] if a.select == "nll" else r["gp"]
best, state, t0 = crit(r0), copy.deepcopy(model.state_dict()), time.time()
print(f"ep 0 val op {100*r0['op']:.4f}% gp {100*r0['gp']:.4f}%", flush=True)
n_freeze = int(a.freeze_frac * a.epochs)
torch.cuda.reset_peak_memory_stats(); t_train = 0.0
ckpt_path, start = f"{a.out}/resume.pt", 0
if os.path.exists(ckpt_path):        # continue an interrupted run exactly where its last checkpoint left it
    R = torch.load(ckpt_path, weights_only=False, map_location=dev)
    model.load_state_dict(R["model"]); opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"])
    start, best, state, t_train = R["ep"], R["best"], R["state"], R["t_train"]
    torch.set_rng_state(R["rng"].cpu()); torch.cuda.set_rng_state(R["cuda_rng"].cpu(), dev)
    print(f"resumed at epoch {start}", flush=True)
for ep in range(start, a.epochs):
    model.mean_op.requires_grad_(ep >= n_freeze)                          # AdamW skips params without grads
    model.train(); te = time.time()
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(len(Xtr) // a.batch):
        b = perm[i * a.batch:(i + 1) * a.batch]
        xb, yb = Xtr[b], Ytr[b]
        if shift:                                                         # same augmentation as stage 1 (periodic 1D)
            sh = int(torch.randint(xb.shape[1], (1,)))
            xb, yb = xb.roll(sh, 1), yb.roll(sh, 1)
        loss = model.elbo(xb, yb, len(Xtr), mask=dmask if Mtr is None else Mtr[b])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    torch.cuda.synchronize(); t_train += time.time() - te
    if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
        r, _, _ = evaluate(xva, yva)
        if crit(r) < best:
            best, state = crit(r), copy.deepcopy(model.state_dict())
        print(f"ep {ep+1} -elbo/pt {loss.item():.4f} val op {100*r['op']:.4f}% gp {100*r['gp']:.4f}% "
              f"ell_x {model.kx.log_ell.exp().mean().item():.3f} s_a {model.log_sa.exp().item():.3f} t {time.time()-t0:.0f}s", flush=True)
    if (ep + 1) % a.ckpt_every == 0 and ep < a.epochs - 1:
        os.makedirs(a.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "ep": ep + 1,
                    "best": best, "state": state, "t_train": t_train, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(dev)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)          # atomic: a kill mid-write never corrupts the checkpoint

model.load_state_dict(state)
after, mt, st = evaluate(xte, yte)
_, mv, sv = evaluate(xva, yva)
yt_f, yv_f = yte.to(dev).flatten(1).double(), yva.to(dev).flatten(1).double()
mt, st, mv, sv = mt.double(), st.double(), mv.double(), sv.double()
half = len(xva) // 2
z2 = (yv_f[half:] - mv[half:]) ** 2 / sv[half:] ** 2
if amask is not None:                                       # active cells only
    wv = amask(xva[half:]).double(); tau = ((z2 * wv).sum() / wv.sum()).sqrt()
else:
    tau = (z2 if dmask is None else z2[:, dmask]).mean().sqrt()
res = {"op_pretrained": before["op"], "op_joint": after["op"], "logos": after["gp"],
       "uq_raw": scorecard(mt, st, yt_f), "uq_calibrated": scorecard(mt, st * tau, yt_f), "tau": tau.item(), "time_per_epoch": t_train / a.epochs,
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "args": vars(a), "init": A}
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=1)
if a.save_pred:
    torch.save(mt.float().reshape(len(mt), *gout).cpu(), f"{a.out}/test_pred.pt")
torch.save(model.state_dict(), f"{a.out}/model.pt")
print(f"FINAL test rL2: operator pretrained {100*before['op']:.4f}%  operator joint {100*after['op']:.4f}%  LoGoS {100*after['gp']:.4f}%")
for k in ("uq_raw", "uq_calibrated"):
    r = res[k]; print(f"  {k:14s} NLL {r['nll']:.4f} CRPS {r['crps']:.3e} cov95 {r['cov95']:.3f}")
print(f"  tau {tau.item():.3f}")
