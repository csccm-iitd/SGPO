"""CO2 storage (U-FNO gas saturation) metrics, as defined by Wen et al. 2022 (Section 4, Table I.14).

Per test sample, over all 24 snapshots and active cells (input channel 0 non-zero):
    plume      cells where the data or the prediction is non-zero (prediction clipped to the physical range [0, 1];
               --thr adds a variant that counts a prediction as non-zero only above a threshold)
    MPE        mean absolute error within the plume
    R2_plume   1 - sum (pred - y)^2 / sum (y - mean_plume(y))^2 within the plume
Reported: mean and std over test samples. The plume definition above is the one used here (no reference evaluation
code is available).

    python experiments/co2_eval.py --init runs/operator/co2
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from sgpo.data.benchmarks import load_split
from experiments.vnngp_io import build_operator

p = argparse.ArgumentParser()
p.add_argument("--init", default=None, help="stage-1 operator run (predicts the test set)")
p.add_argument("--pred", default=None, help="or: saved test predictions (500, 96, 200, 24) of another model")
p.add_argument("--thr", type=float, default=0.01)
p.add_argument("--out", default=None)
a = p.parse_args()
dev = "cuda"
_, _, (xte, yte) = load_split("co2_sg", 4500, 500, 500)
if a.pred:
    pred = torch.load(a.pred, weights_only=True)
else:
    C = torch.load(f"{a.init}/ckpt.pt", weights_only=False, map_location="cpu"); A = C["args"]
    op = build_operator(A, tuple(xte.shape[1:-1]), xte.shape[-1]); op.load_state_dict(C["state"]); op = op.to(dev).eval()
    xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
    with torch.no_grad():
        pred = torch.cat([(op((xte[i:i + 8].to(dev) - xm) / xs) * ys + ym).cpu() for i in range(0, len(xte), 8)])
act = (xte[..., 0] != 0)[..., None].expand_as(yte)                          # active cells (U-FNO's mask)
pred = (pred * act).clamp(0, 1)


def scores(thr):
    mpe, r2 = [], []
    for i in range(len(yte)):
        y, q, m = yte[i], pred[i], act[i]
        plume = m & ((y > 0) | (q > thr))
        e = (q - y)[plume]
        mpe.append(e.abs().mean().item())
        yp = y[plume]
        r2.append(1 - (e ** 2).sum().item() / ((yp - yp.mean()) ** 2).sum().item())
    t = lambda v: (float(torch.tensor(v).mean()), float(torch.tensor(v).std()))
    return {"MPE": t(mpe), "R2_plume": t(r2)}


rel = ((pred - yte) * act).flatten(1).norm(dim=1) / (yte * act).flatten(1).norm(dim=1)
res = {"rl2_active_pct": 100 * rel.mean().item(), "strict": scores(0.0), f"thr{a.thr}": scores(a.thr)}
print(json.dumps(res, indent=1))
json.dump(res, open(a.out or os.path.join(a.init or os.path.dirname(a.pred), "co2_metrics.json"), "w"), indent=1)
