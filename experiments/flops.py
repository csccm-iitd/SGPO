"""FLOPs of LoGoS-GPO vs GPO as functions of grid size and training-set size (hardware-independent cost study).

Counts one training step (forward + backward) and one prediction with torch's FlopCounterMode (matmul, conv) plus
analytic counts for the ops it skips:
    Cholesky n^3/3, symmetric eigendecomposition with vectors 9 n^3 (Golub & Van Loan), triangular solve n^2 k,
    LU solve 2/3 n^3 + 2 n^2 k, FFT 5 N log2 N (complex) and 2.5 N log2 N (real input/output) per transform.
Reported per configuration:
    gp_train   GP-layer FLOPs per training epoch (LoGoS-GPO: N/B minibatch ELBO steps minus the operator's share;
               GPO: one exact marginal-likelihood step on all N fields), operator excluded for both
    op_train   operator FLOPs per epoch (forward + backward over N fields; the same network in both models)
    predict    FLOPs per test field (operator forward + GP predictive mean and variance)
Inputs are random tensors of the right shapes: FLOPs do not depend on values.

The operator architecture is taken from a stage-1 run (--init, its ckpt.pt arguments); the wavelet level and Fourier
modes are capped for each grid. Output: <out>/<bench>.json, read by paper_figs.py --which flops.

    python experiments/flops.py --bench burgers --init runs/burgers/s0/op --out runs/flops
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pywt
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode
from sgpo.models.gpo import GPO
from sgpo.models.logos_vnngp import LoGoSVNNGP
from experiments.vnngp_io import build_operator


class LinalgFlops(TorchDispatchMode):
    """Analytic FLOPs for the linear-algebra and FFT ops that FlopCounterMode does not count."""

    def __init__(self):
        super().__init__(); self.flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in ("linalg_cholesky_ex", "_linalg_eigh", "linalg_solve_triangular", "_linalg_solve_ex", "cholesky_solve"):
            A = args[0]; n = A.shape[-1]; b = math.prod(A.shape[:-2])
            if name == "linalg_cholesky_ex":
                self.flops += b * n ** 3 / 3
            elif name == "_linalg_eigh":
                self.flops += b * 9 * n ** 3
            else:
                B = args[1]; k = B.shape[-1] if B.ndim == A.ndim else 1
                self.flops += b * (n * n * k if name != "_linalg_solve_ex" else 2 / 3 * n ** 3 + 2 * n * n * k) * (2 if name == "cholesky_solve" else 1)
        elif name in ("_fft_r2c", "_fft_c2r", "_fft_c2c"):
            x, dims = args[0], args[1]
            shape = out.shape if name == "_fft_c2r" else x.shape
            N = math.prod(shape[d] for d in dims)
            self.flops += (5 if name == "_fft_c2c" else 2.5) * math.prod(shape) * math.log2(max(N, 2))
        return out


def count(fn):
    lin = LinalgFlops()
    with FlopCounterMode(display=False) as fc, lin:
        fn()
    return fc.get_total_flops() + lin.flops


GRIDS = {"burgers": [64, 128, 256, 512, 1024, 2048, 4096, 8192], "advection": [25, 50, 100, 200, 400],
         "notch": [26, 34, 51, 101, 151], "ns": [16, 32, 64, 128]}
SAMPLES = {"burgers": [100, 250, 500, 1000, 1900], "advection": [100, 500, 1000, 2000, 2700],
           "notch": [100, 200, 500, 1000, 1500], "ns": [100, 500, 1000, 2000, 5000]}
SAMPLE_GRID = {"burgers": 512, "advection": 200, "notch": 51, "ns": 64}
BATCH2 = {"burgers": 20, "advection": 32, "notch": 16, "ns": 16}


def config(A, bench, n, ntrain, dev):
    grid = (n,) if bench in ("burgers", "advection") else (n, n)
    B = dict(A)
    B["level"] = max(1, min(A["level"], pywt.dwt_max_level(n, pywt.Wavelet(A.get("wavelet", "db4")).dec_len)))
    B["modes"] = [min(m, n // 2) for m in A["modes"]]
    op = build_operator(B, grid, 1).to(dev)
    axes = [torch.linspace(0, 1, g) for g in grid]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(grid)).to(dev)
    d, bs = len(coords), BATCH2[bench]
    res = {"grid": n, "d": d, "ntrain": ntrain}
    # operator: forward + backward per field
    x = torch.randn(bs, *grid, 1, device=dev)
    res["op_train"] = count(lambda: op(x).sum().backward()) / bs * ntrain
    # LoGoS-GPO: minibatch ELBO step (operator + GP), minus the operator's share
    lg = LoGoSVNNGP(op, coords.cpu(), A["width"], n_inducing=min(256, ntrain)).to(dev)
    y = torch.randn(bs, d, device=dev)
    step = count(lambda: lg.elbo(x, y, ntrain).backward())
    res["logos_gp_train"] = max(step - res["op_train"] / ntrain * bs, 0) * (ntrain / bs)
    lg.eval()
    with torch.no_grad():
        res["logos_predict"] = count(lambda: lg(x[:1]))
    # GPO: exact Kronecker likelihood on all N fields (features precomputed by the operator)
    fdim = 2 * A["width"]
    gpo = GPO(coords, fdim).to(dev)
    f, r = torch.randn(ntrain, fdim, device=dev), torch.randn(ntrain, d, device=dev)
    res["gpo_gp_train"] = count(lambda: gpo.nll(f, r).backward())
    gpo.fit_cache(f, r)
    with torch.no_grad():
        op_fwd = count(lambda: op(x[:1]))
        res["gpo_predict"] = op_fwd + count(lambda: gpo.predict(f[:1]))
        res["gpo_fit_cache"] = count(lambda: gpo.fit_cache(f, r))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=list(GRIDS))
    p.add_argument("--init", required=True, help="stage-1 run (train_operator.py output dir) defining the operator")
    p.add_argument("--out", default="runs/flops")
    a = p.parse_args()
    dev = "cuda"
    A = torch.load(f"{a.init}/ckpt.pt", weights_only=False, map_location="cpu")["args"]
    out = {"grid": [], "samples": []}
    jobs = [("grid", n, A["ntrain"]) for n in GRIDS[a.bench]] + [("samples", SAMPLE_GRID[a.bench], nt) for nt in SAMPLES[a.bench]]
    for kind, n, nt in jobs:
        try:
            out[kind].append(config(A, a.bench, n, nt, dev)); print(kind, out[kind][-1], flush=True)
        except torch.cuda.OutOfMemoryError:
            print("OOM", kind, n, nt, flush=True)
        torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/{a.bench}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
