"""GPO baseline: exact Kronecker GP (sgpo/models/gpo.py) on the residuals of a frozen stage-1 operator.

Same split, operator and normalisation as the LoGoS-GPO run that starts from the same --init. Full-batch Adam on the
exact marginal likelihood; the step with the best validation rL2 is kept; tau calibrated on validation half 2.

    python experiments/train_gpo.py --init runs/operator/burgers --out runs/gpo/burgers
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from sgpo.data.benchmarks import load_split
from sgpo.models.gpo import GPO
from sgpo.uq import scorecard
from experiments.vnngp_io import build_operator

p = argparse.ArgumentParser()
p.add_argument("--init", required=True); p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=300); p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--embed", type=int, default=32); p.add_argument("--seed", type=int, default=0)
p.add_argument("--ntrain-use", type=int, default=0, help="use only the first n training samples (0: all)")
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
C = torch.load(f"{a.init}/ckpt.pt", weights_only=False, map_location="cpu"); A = C["args"]
(xtr, ytr), (xva, yva), (xte, yte) = load_split(A["data"], A["ntrain"], A["nval"], A["ntest"], A["tail_test"])
if a.ntrain_use:
    xtr, ytr = xtr[:a.ntrain_use], ytr[:a.ntrain_use]
grid, cin = tuple(xtr.shape[1:-1]), xtr.shape[-1]
op = build_operator(A, grid, cin); op.load_state_dict(C["state"]); op = op.to(dev).eval()
xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
ysf = ys.reshape(1, -1) if ys.numel() > 1 else ys


@torch.no_grad()
def feats(x, y):
    """Operator features (spatial mean and std of the last hidden layer), operator mean and normalised residual."""
    fs, m0s = [], []
    for i in range(0, len(x), 25):
        h = op.features((x[i:i + 25].to(dev) - xm) / xs)
        f = h.flatten(1, -2)
        fs.append(torch.cat([f.mean(1), f.std(1)], -1)); m0s.append(op.proj(h).flatten(1))
    m0 = torch.cat(m0s)
    return torch.cat(fs), m0, ((y.to(dev) - ym) / ys).flatten(1) - m0


axes = [torch.linspace(0, 1, n) for n in grid]
coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(grid)).to(dev)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tf = time.time()
ftr, _, rtr = feats(xtr, ytr)
torch.cuda.synchronize(); t_feat = time.time() - tf                  # one operator pass over the training set
fva, m0v, _ = feats(xva, yva)
fte, m0t, _ = feats(xte, yte)
model = GPO(coords, ftr.shape[1], embed_dim=a.embed).to(dev)
rel = lambda p_, t: (p_ - t).norm(dim=1) / t.norm(dim=1)
phys = lambda z: (z.reshape(-1, *grid, 1) * ys + ym).flatten(1)


@torch.no_grad()
def predict(f, m0):
    model.fit_cache(ftr, rtr)
    mus, vs = [], []
    for i in range(0, len(f), 50):
        mu, v = model.predict(f[i:i + 50]); mus.append(mu); vs.append(v)
    mu, v = torch.cat(mus).float(), torch.cat(vs).float()
    return phys(m0 + mu).double(), (v.sqrt() * ysf).double()


yv, yt = yva.to(dev).flatten(1).double(), yte.to(dev).flatten(1).double()
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
best, state = float("inf"), None
t_train, t0 = 0.0, time.time()
for step in range(a.steps):
    te = time.time()
    loss = model.nll(ftr, rtr)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); t_train += time.time() - te
    if (step + 1) % 20 == 0 or step == a.steps - 1:
        mv, _ = predict(fva, m0v)
        r = rel(mv, yv).mean().item()
        if r < best:
            best, state = r, copy.deepcopy(model.state_dict())
        print(f"step {step+1} nll/pt {loss.item():.4f} val {100*r:.4f}% ell_x {model.log_ell_x.exp().item():.3f} "
              f"noise {model.log_noise.exp().item():.2e} t {time.time()-t0:.0f}s", flush=True)
model.load_state_dict(state)
mt, st = predict(fte, m0t)
mv, sv = predict(fva, m0v)
half = len(xva) // 2
tau = ((yv[half:] - mv[half:]) ** 2 / sv[half:] ** 2).mean().sqrt()
op_only = rel(phys(m0t).double(), yt).mean().item()
res = {"op": op_only, "gpo": rel(mt, yt).mean().item(), "uq_raw": scorecard(mt, st, yt), "uq_calibrated": scorecard(mt, st * tau, yt),
       "tau": tau.item(), "time_per_step": t_train / a.steps, "time_feat": t_feat,
       "time_per_epoch": t_feat + t_train / a.steps, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
       "ntrain": len(xtr), "grid": list(grid), "args": vars(a), "init": A}
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=1)
torch.save(model.state_dict(), f"{a.out}/model.pt")
print(f"FINAL test rL2: operator {100*op_only:.4f}%  GPO {100*res['gpo']:.4f}%  cov95 cal {res['uq_calibrated']['cov95']:.3f} "
      f"time/step {res['time_per_step']:.2f}s mem {res['peak_mem_gb']:.2f} GB")
