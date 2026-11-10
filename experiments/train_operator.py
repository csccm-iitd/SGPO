"""Train a deterministic operator with relative-L2 loss, one-cycle LR and EMA."""
import argparse, copy, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from sgpo.data.benchmarks import load_split
from sgpo.operators.symmetry import d4, d4_inv
from sgpo.operators.spacetime import SpaceTime
from sgpo.operators import EWNO, FNO, WNO
from sgpo.utils.misc import set_seed

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--ntrain", type=int, required=True)
p.add_argument("--nval", type=int, default=200); p.add_argument("--ntest", type=int, default=200)
p.add_argument("--tail-test", action="store_true")
p.add_argument("--modes", type=int, nargs="+", default=[16, 16]); p.add_argument("--width", type=int, default=64)
p.add_argument("--layers", type=int, default=4); p.add_argument("--epochs", type=int, default=500)
p.add_argument("--batch", type=int, default=32); p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda:0"); p.add_argument("--out", required=True)
p.add_argument("--arch", default="fno", choices=["ewno", "fno", "wno", "wnof"], help="wnof: WNO with a parallel Fourier path")
p.add_argument("--level", type=int, default=4); p.add_argument("--wavelet", default="db4"); p.add_argument("--no-fine", action="store_true"); p.add_argument("--pad", type=int, default=0); p.add_argument("--h1", type=float, default=0.0)
p.add_argument("--aug", default="none", choices=["none", "d4", "shift", "shift_ad", "shift2d", "shift2d_d4p"],
               help="exact symmetry: d4 (square, isotropic), shift (periodic along axis 1), shift2d (periodic 2D), "
                    "shift2d_d4p (periodic 2D shifts and D4 with a sign flip under reflections, for pseudo-scalar fields), "
                    "shift_ad (periodic 2D shift by (s, -s), for forcing of the form f(x + y))")
p.add_argument("--spacetime", default=None,
               help="file with the T snapshot times: a 3D operator over (space, time) maps static inputs to (*grid, T)")
p.add_argument("--wave-dims", type=int, default=None, help="with --spacetime: 2 = wavelet path on space only")
p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip (0: none)")
p.add_argument("--no-ema", dest="ema", action="store_false", help="evaluate the raw weights instead of their EMA")
p.add_argument("--log-every", type=int, default=10, help="epochs between validation / model selection")
p.add_argument("--init-from", default=None, help="start from the weights of another run's ckpt.pt (fine-tuning)")
p.add_argument("--sched", default="onecycle", choices=["onecycle", "step"], help="step: lr x0.9 every 2 epochs (U-FNO recipe)")
p.add_argument("--norm", default="pixel", choices=["pixel", "none"],
               help="pixel: per-grid-point mean/std; none: data used as given (U-FNO CO2 fields are pre-scaled to [0, 1])")
p.add_argument("--mask-ch", type=int, default=-1,
               help="active-cell mask from input channel C (cells where it is non-zero); outputs are zero elsewhere (U-FNO CO2 data)")
p.add_argument("--ckpt-every", type=int, default=50, help="epochs between resumable checkpoints")
a = p.parse_args()

set_seed(a.seed)
dev = torch.device(a.device)
(xtr, ytr), (xva, yva), (xte, yte) = load_split(a.data, a.ntrain, a.nval, a.ntest, a.tail_test)
xm, xs = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-8
ym, ys = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True) + 1e-8
if a.aug == "d4":   # invariant normalisation
    xm, ym = [sum(d4(t, k) for k in range(8)) / 8 for t in (xm, ym)]
    xs, ys = [sum(d4(t, k) for k in range(8)) / 8 for t in (xs, ys)]
elif a.aug in ("shift", "shift_ad", "shift2d", "shift2d_d4p"):   # spatially constant statistics are invariant under any shift
    xm, ym = xtr.mean(), ytr.mean()
    xs, ys = xtr.std() + 1e-8, ytr.std() + 1e-8
if a.norm == "none":   # sparse fields (93% zeros): per-point std is ~0 where the plume rarely reaches
    xm, ym, xs, ys = (torch.zeros(()), torch.zeros(()), torch.ones(()), torch.ones(()))
to = lambda t: t.to(dev)
xtr_n, ytr_n, xva_n, xte_n = map(to, ((xtr - xm) / xs, ytr, (xva - xm) / xs, (xte - xm) / xs))
ym, ys, yva, yte = map(to, (ym, ys, yva, yte))
act = lambda x: (x[..., a.mask_ch] != 0)[..., None].float().to(dev) if a.mask_ch >= 0 else None
mtr, mva, mte = act(xtr), act(xva), act(xte)
msk = lambda o, m: o if m is None else o * m             # inactive cells are known from the input and always zero

grid, cin = xtr.shape[1:-1], xtr.shape[-1]
a.out_ch = ytr.shape[-1]                                  # >1 for space-time outputs (one channel per snapshot)
if a.arch == "fno":
    model = FNO(a.modes, a.width, a.layers, in_ch=cin, out_ch=a.out_ch, pad=a.pad)
elif a.arch == "ewno":
    model = EWNO(grid, a.width, a.level, a.layers, modes=a.modes[0], pad=a.pad)
elif a.arch in ("wno", "wnof") and a.spacetime:
    times = torch.load(a.spacetime, weights_only=True)
    model = SpaceTime(WNO((*grid, len(times)), a.width, a.level, a.layers, a.wavelet, in_ch=cin + 1, out_ch=1, pad=a.pad,
                          modes=a.modes if a.arch == "wnof" else None, fine=not a.no_fine, wave_dims=a.wave_dims), times)
else:
    model = WNO(grid, a.width, a.level, a.layers, a.wavelet, in_ch=cin, out_ch=a.out_ch, pad=a.pad,
                modes=a.modes if a.arch == "wnof" else None, fine=not a.no_fine)
if a.init_from:
    model.load_state_dict(torch.load(f"{a.init_from}/ckpt.pt", weights_only=False, map_location="cpu")["state"])
model = model.to(dev)
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps = a.epochs * (len(xtr_n) // a.batch)
if a.sched == "step":     # U-FNO recipe: lr x gamma every 2 epochs (stepped per batch)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2 * max(1, len(xtr_n) // a.batch), gamma=0.9)
else:
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
rel = lambda p, t: ((p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1))
# evaluation chunk: keep every activation (batch x 4 width x grid, wavelet sub-bands included) under 2^31 elements
EB = int(max(1, min(64, 2 ** 31 // (8 * a.width * math.prod(grid)))))

def forward(x):
    """EMA prediction, averaged over the symmetry group when one applies."""
    if a.aug == "shift2d_d4p":                              # pseudo-scalar fields: reflections flip the sign
        return sum(sg * d4_inv(ema(sg * d4(x, k)), k) for k in range(8) for sg in [-1 if k & 4 else 1]) / 8 * ys + ym
    if a.aug != "d4":
        return ema(x) * ys + ym
    return sum(d4_inv(ema(d4(x, k)) * ys + ym, k) for k in range(8)) / 8


@torch.no_grad()
def score(x, y, m=None):
    ema.eval()
    return torch.cat([rel(msk(forward(x[i:i + EB]), None if m is None else m[i:i + EB]), y[i:i + EB])
                      for i in range(0, len(x), EB)]).mean().item()

best, t0, start, state = 1e9, time.time(), 0, None
ckpt_path = f"{a.out}/resume.pt"
if os.path.exists(ckpt_path):   # continue an interrupted run exactly where its last checkpoint left it
    R = torch.load(ckpt_path, weights_only=False, map_location=dev)
    model.load_state_dict(R["model"]); ema.load_state_dict(R["ema"]); opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"])
    start, best, state = R["ep"], R["best"], R["state"]
    torch.set_rng_state(R["rng"].cpu()); torch.cuda.set_rng_state(R["cuda_rng"].cpu(), dev)
    print(f"resumed at epoch {start}", flush=True)
for ep in range(start, a.epochs):
    model.train()
    perm = torch.randperm(len(xtr_n), device=dev)
    for i in range(len(xtr_n) // a.batch):
        b = perm[i * a.batch:(i + 1) * a.batch]
        xb, yb = xtr_n[b], ytr_n[b]
        if a.aug == "d4":
            k = int(torch.randint(8, (1,)))
            xb, yb = d4(xb, k), d4(yb, k)
        elif a.aug == "shift":
            sh = int(torch.randint(xb.shape[1], (1,)))
            xb, yb = xb.roll(sh, 1), yb.roll(sh, 1)
        elif a.aug == "shift2d":                             # periodic 2D, translation-equivariant map
            s1, s2 = (int(torch.randint(xb.shape[k], (1,))) for k in (1, 2))
            xb, yb = xb.roll((s1, s2), (1, 2)), yb.roll((s1, s2), (1, 2))
        elif a.aug == "shift2d_d4p":    # 2D NS forcing -> vorticity on the torus: shifts, rotations, and reflections
            s1, s2 = (int(torch.randint(xb.shape[k], (1,))) for k in (1, 2))     # with a sign flip (pseudo-scalars)
            k = int(torch.randint(8, (1,))); sg = -1.0 if k & 4 else 1.0
            xb, yb = sg * d4(xb.roll((s1, s2), (1, 2)), k), sg * d4(yb.roll((s1, s2), (1, 2)), k)
        elif a.aug == "shift_ad":
            sh = int(torch.randint(xb.shape[1], (1,)))
            xb, yb = xb.roll((sh, -sh), (1, 2)), yb.roll((sh, -sh), (1, 2))
        out = msk(model(xb) * ys + ym, None if mtr is None else mtr[b])
        loss = rel(out, yb).mean()
        if a.h1 > 0:
            for dim in range(1, out.ndim - 1):
                loss = loss + a.h1 * rel(torch.diff(out, dim=dim), torch.diff(yb, dim=dim)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step()
        with torch.no_grad():
            d = min(0.999, (sched.last_epoch + 1) / (sched.last_epoch + 10))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d if a.ema else 1.0)        # --no-ema: the "EMA" copy is the raw model
    if (ep + 1) % a.log_every == 0 or ep == a.epochs - 1:
        v = score(xva_n, yva, mva)
        if v < best:
            best, state = v, copy.deepcopy(ema.state_dict())
        print(f"ep {ep+1} train {loss.item():.5f} val {100*v:.3f}% t {time.time()-t0:.0f}s", flush=True)
    if (ep + 1) % a.ckpt_every == 0 and ep < a.epochs - 1:
        os.makedirs(a.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ep": ep + 1, "best": best, "state": state, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev)},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

ema.load_state_dict(state)
test = score(xte_n, yte, mte)
ema.eval()
with torch.no_grad():
    test_plain = torch.cat([rel(ema(xte_n[i:i + EB]) * ys + ym, yte[i:i + EB]) for i in range(0, len(xte_n), EB)]).mean().item()
print(f"test without group averaging {100*test_plain:.3f}%")
print(f"FINAL val {100*best:.3f}% test {100*test:.3f}% params {sum(p.numel() for p in model.parameters())}")
os.makedirs(a.out, exist_ok=True)
torch.save({"state": state, "args": vars(a), "xm": xm, "xs": xs, "ym": ym.cpu(), "ys": ys.cpu()}, f"{a.out}/ckpt.pt")


@torch.no_grad()
def predict(x):
    ema.eval()
    xn = ((x - xm) / xs).to(dev)
    return torch.cat([forward(xn[i:i + EB]).cpu() for i in range(0, len(xn), EB)])

pred = {"val": predict(xva), "test": predict(xte)}
torch.save(pred, f"{a.out}/pred.pt")
json.dump({"val": best, "test": test, "test_plain": test_plain, "time": time.time() - t0}, open(f"{a.out}/metrics.json", "w"))
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
