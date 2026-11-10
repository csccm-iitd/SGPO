"""Advection with periodic boundary (Lu et al. 2022, lu-group/deeponet-fno data/advection), any resolution.

Exact solution u(x, t) = u0(x - t) on [0, 1). IC2 is a square wave plus a parabolic bump,
parameters drawn from the same ranges as `main_IC2`. Saves u at nt x nx points.

    python scripts/data/gen_advection.py --n 2000 --nx 200 --nt 101 --out DIR
"""
import argparse, json, os
import numpy as np


def ic2(x, c1, w, h1, c2, a, h2):
    x = np.mod(x, 1.0)
    u1 = h1 * np.heaviside(x - (c1 - w / 2), 0.5) * np.heaviside(c1 + w / 2 - x, 0.5)
    return u1 + np.sqrt(np.maximum(h2 ** 2 - a ** 2 * (x - c2) ** 2, 0))


p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=2000); p.add_argument("--nx", type=int, default=200)
p.add_argument("--nt", type=int, default=101); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
x = np.linspace(0, 1, a.nx, endpoint=False)
t = np.linspace(0, 1, a.nt)
P = np.stack([0.1 * rng.random(a.n) + 0.2, 0.1 * rng.random(a.n) + 0.1, 1.5 * rng.random(a.n) + 0.5,
              0.1 * rng.random(a.n) + 0.7, 5 * rng.random(a.n) + 5, 1.5 * rng.random(a.n) + 0.5], 1)
u = np.stack([ic2(x[None] - t[:, None], *q) for q in P]).astype(np.float32)   # (n, nt, nx)
os.makedirs(a.out, exist_ok=True)
np.savez(f"{a.out}/advection_ic2.npz", x=x, t=t, u=u, params=P)
json.dump(vars(a), open(f"{a.out}/meta.json", "w"), indent=1)
print(u.shape)
