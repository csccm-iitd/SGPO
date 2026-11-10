"""Proper scores and calibration diagnostics for field-valued predictions (b, d)."""
import math
import torch


def rel_l2(m, y):
    return ((m - y).norm(dim=1) / y.norm(dim=1)).mean().item()


def gaussian_nll(m, s, y):
    return (0.5 * ((y - m) / s) ** 2 + s.log() + 0.5 * math.log(2 * math.pi)).mean().item()


def gaussian_crps(m, s, y):
    z = (y - m) / s
    n = torch.distributions.Normal(0.0, 1.0)
    return (s * (z * (2 * n.cdf(z) - 1) + 2 * n.log_prob(z).exp() - 1 / math.sqrt(math.pi))).mean().item()


def coverage(m, s, y, level):
    q = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + level / 2, dtype=m.dtype))
    return ((y - m).abs() <= q * s).float().mean().item()


def _rank(v):
    return v.argsort().argsort().float()


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def sample_error_corr(m, s, y):
    """Rank correlation, across samples, of predicted vs realised field error."""
    return spearman(s.pow(2).mean(1), (y - m).pow(2).mean(1))


def pointwise_error_corr(m, s, y, nmax=2_000_000):
    a, b = s.flatten(), (y - m).abs().flatten()
    if len(a) > nmax:
        i = torch.randperm(len(a), device=a.device)[:nmax]
        a, b = a[i], b[i]
    return spearman(a, b)


def energy_score(samples, y):
    """samples: (S, b, d). Normalised by sqrt(d) so grids are comparable."""
    d = y.shape[-1] ** 0.5
    t1 = (samples - y).norm(dim=-1).mean(0)
    t2 = (samples[:-1] - samples[1:]).norm(dim=-1).mean(0)
    return ((t1 - 0.5 * t2) / d).mean().item()


def ood_pct(m, s, y, level=0.95):
    """Fraction of samples whose per-field coverage misses the nominal level by more than 5 points."""
    q = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + level / 2, dtype=m.dtype))
    per = ((y - m).abs() <= q * s).float().mean(1)
    return (per < level - 0.05).float().mean().item()


def scorecard(m, s, y, samples=None):
    out = {
        "rl2_pct": 100 * rel_l2(m, y),
        "nll": gaussian_nll(m, s, y),
        "crps": gaussian_crps(m, s, y),
        "cov50": coverage(m, s, y, 0.5),
        "cov90": coverage(m, s, y, 0.9),
        "cov95": coverage(m, s, y, 0.95),
        "sharp": s.mean().item(),
        "rho_point": pointwise_error_corr(m, s, y),
        "rho_sample": sample_error_corr(m, s, y),
        "fields_undercovered": ood_pct(m, s, y),
    }
    if samples is not None:
        out["energy"] = energy_score(samples, y)
    return out
