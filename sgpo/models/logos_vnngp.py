"""LoGoS-GPO (arXiv 2506.15906, Eqs. 11-23): a variational Gaussian process operator with a neural-operator mean.

Prior over output fields f(a, .) in R^d:
    f ~ GP( M[a],  K_a(Phi(a), Phi(a')) (x) K_x )                         Eqs. 11-12
M is a wavelet neural operator, Phi a learned embedding of the input function, K_a a Matern-5/2 ARD kernel, and
K_x a spatial Matern kernel in nearest-neighbour sparse form (Eq. 13), written as a Vecchia precision
K_x^{-1} ~ B^T D^{-1} B with k neighbours per grid point.

Inducing variables at M locations Z in embedding space, whitened: V - M[Z] = L_a W, q(W) = N(W_mu, R R^T (x) K_x).
Then (Eq. 23)
    mean(a) = M[a] + A^T W_mu,          A = L_a^{-1} K_a(Z, Phi(a))
    var_a(a) = k_a(a, a) - |A|^2 + |R^T A|^2        (latent variance, times K_x spatially)
and the ELBO (Eq. 21) with a per-point Gaussian noise sigma^2(x) is available in closed form:
    E log p(y|f) = -1/2 sum_x [ (y - mean)^2 + var_a ] / sigma_x^2 + log 2 pi sigma_x^2
    KL = 1/2 [ d tr(RR^T) - M d - d logdet(RR^T) + sum_m w_m^T K_x^{-1} w_m ]
The operator mean M[a] enters the predictive mean directly, so it is trained jointly with the GP on the ELBO.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgpo.uq.kernels import KERNELS
from sgpo.uq.vecchia import predecessor_neighbours


def _chol(K, jitter=1e-5):
    """Cholesky with jitter 1e-5, raised tenfold only if the factorisation fails (rare, near-duplicate inducing points)."""
    eye = torch.eye(len(K), device=K.device, dtype=K.dtype)
    for _ in range(4):
        L, info = torch.linalg.cholesky_ex(K + jitter * eye)
        if not info.any():
            return L
        jitter *= 10
    return torch.linalg.cholesky(K + jitter * eye)


class VecchiaKx(nn.Module):
    """Unit-variance Matern-5/2 spatial kernel in Vecchia form, differentiable in its length scale(s).
    ard=True learns one length scale per coordinate axis (e.g. space and time of a space-time output)."""

    def __init__(self, coords, k=16, ell=0.1, seed=0, ard=False, kernel="matern52"):
        super().__init__()
        d = len(coords)
        order = torch.randperm(d, generator=torch.Generator().manual_seed(seed))
        c = coords[order]
        self.register_buffer("order", order)
        self.register_buffer("c", c)
        self.register_buffer("nbr", torch.from_numpy(predecessor_neighbours(c.double().numpy(), k)))
        self.log_ell = nn.Parameter(torch.full((coords.shape[1],) if ard else (), math.log(ell)))
        self.kern = KERNELS[kernel]                               # unit-variance stationary kernel on scaled coords

    def factor(self, nugget=1e-3):
        ell = self.log_ell.exp()
        mask = self.nbr >= 0
        P = self.c[self.nbr.clamp(min=0)] / ell                               # (d, k, D), scaled per axis
        Knn = self.kern(P, P, 1.0) * (mask[:, :, None] & mask[:, None, :]) + torch.diag_embed((~mask).float() + nugget * mask)
        kin = self.kern(P, self.c[:, None] / ell, 1.0)[..., 0] * mask
        b = torch.linalg.solve(Knn, kin.unsqueeze(-1)).squeeze(-1)
        cond = (1 + nugget - (b * kin).sum(-1)).clamp_min(1e-6)
        return b * mask, cond

    def quad(self, W):
        """sum over rows w of w^T K_x^{-1} w, W (m, d) in grid order."""
        b, cond = self.factor()
        nb = self.nbr.clamp(min=0)
        rows = max(1, int(2 ** 28 // (nb.numel() or 1)))          # gather (rows, d, k) stays ~1 GB on fine grids
        out = 0.0
        for i in range(0, len(W), rows):
            z = W[i:i + rows, self.order]
            e = z - (z[:, nb] * b).sum(-1)
            out = out + (e ** 2 / cond).sum()
        return out

    @torch.no_grad()
    def sample(self, n):
        """Draws from N(0, K_x^Vecchia), (n, d) in grid order, by forward substitution in Vecchia order."""
        b, cond = self.factor()
        d = len(cond)
        z = torch.zeros(n, d, device=b.device)
        eps = torch.randn(n, d, device=b.device) * cond.sqrt()
        nb = self.nbr.clamp(min=0)
        for i in range(d):                                     # sequential; only used for evaluation samples
            z[:, i] = eps[:, i] + (z[:, nb[i]] * b[i]).sum(-1)
        out = torch.empty_like(z)
        out[:, self.order] = z
        return out


class LoGoSVNNGP(nn.Module):
    def __init__(self, mean_op, coords, feat_width, n_inducing=256, embed_dim=32, k_x=16, ell_x=0.1, noise="pixel", n_out=1, head_out=None,
                 kernel_x="matern52", kernel_a="matern52", noise_head="linear"):
        """noise: "pixel" learns one noise variance per grid point, "scalar" a single sigma_n^2 (likelihood N(f, sigma_n^2 I)),
        "feature" a heteroscedastic sigma_n^2(a, x) = exp(base(x) + w^T h_a(x)) with h_a(x) the operator's last hidden
        features at x (the per-point base plus a linear head, zero-initialised so training starts from "pixel")."""
        super().__init__()
        self.mean_op = mean_op
        self.embed = nn.Sequential(nn.Linear(2 * feat_width, 128), nn.GELU(), nn.Linear(128, embed_dim))   # Phi
        self.d, self.M = len(coords), n_inducing
        self.Z = nn.Parameter(torch.randn(n_inducing, embed_dim))
        self.log_ell_a = nn.Parameter(torch.zeros(embed_dim))
        self.log_sa = nn.Parameter(torch.tensor(math.log(0.1)))
        self.W = nn.Parameter(torch.zeros(n_inducing, self.d))
        self.R = nn.Parameter(torch.eye(n_inducing) * 1e-2)
        self.log_noise = nn.Parameter(torch.full((1 if noise == "scalar" else self.d,), math.log(1e-2)))
        # one noise output per output channel; head_out=1 when the operator's features already span (space, time)
        ho = n_out if head_out is None else head_out
        if noise != "feature":
            self.noise_head = None
        elif noise_head == "mlp":                                 # small per-point MLP on the operator features
            self.noise_head = nn.Sequential(nn.Linear(feat_width, 64), nn.GELU(), nn.Linear(64, ho))
        else:
            self.noise_head = nn.Linear(feat_width, ho)
        if self.noise_head is not None:                           # zero output at init: training starts from "pixel"
            last = self.noise_head[-1] if isinstance(self.noise_head, nn.Sequential) else self.noise_head
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self._s2 = None
        self.kx = VecchiaKx(coords, k=k_x, ell=ell_x, ard=n_out > 1, kernel=kernel_x)
        self.kern_a = KERNELS[kernel_a]                           # kernel on the input embedding

    def phi(self, h):
        """Embedding from the operator's features h (b, *grid, w): spatial mean and std, then an MLP."""
        f = h.flatten(1, -2)
        return self.embed(torch.cat([f.mean(1), f.std(1)], -1))

    def ka(self, a, b):
        ell = self.log_ell_a.exp()
        return self.log_sa.exp() ** 2 * self.kern_a(a / ell, b / ell, 1.0)

    def forward(self, x):
        """Returns operator mean (b, d), GP mean (b, d), latent variance (b,); the noise variance of this batch is then
        available from noise_var()."""
        h = self.mean_op.features(x)
        m0 = self.mean_op.proj(h).flatten(1)
        self._s2 = ((self.log_noise + self.noise_head(h).flatten(1)).clamp(-25, 5).exp() if self.noise_head is not None
                    else self.log_noise.exp())
        p = self.phi(h)
        L = _chol(self.ka(self.Z, self.Z))
        A = torch.linalg.solve_triangular(L, self.ka(self.Z, p), upper=False)          # (M, b)
        Rl = torch.tril(self.R)
        var = (self.log_sa.exp() ** 2 - A.pow(2).sum(0) + (Rl.T @ A).pow(2).sum(0)).clamp_min(1e-10)
        return m0, m0 + A.T @ self.W, var

    def noise_var(self):
        """Noise variance of the last forward pass: (b, d) for "feature", else (d,) or (1,)."""
        return self._s2

    def kl(self):
        Rl = torch.tril(self.R)
        S = Rl @ Rl.T
        logdet = 2 * Rl.diagonal().abs().clamp_min(1e-10).log().sum()
        return 0.5 * (self.d * (S.trace() - self.M - logdet) + self.kx.quad(self.W))

    def elbo(self, x, y, n_data, mask=None):
        """Per-point negative ELBO for a minibatch; y (b, d) in the same normalised units as the operator output.
        mask (d,) bool restricts the likelihood to grid points inside the physical domain (irregular domains);
        a (b, d) mask does so per sample (e.g. active reservoir cells that differ between samples)."""
        _, mu, var = self(x)
        s2 = self.noise_var()
        ll = -0.5 * (((y - mu) ** 2 + var[:, None]) / s2 + (2 * math.pi * s2).log())          # K_x has unit diagonal
        ll = ll.expand(len(y), -1)
        if mask is None:
            d_eff, ell = self.d, ll.sum(1)
        elif mask.ndim == 1:
            d_eff, ell = int(mask.sum()), ll[:, mask].sum(1)
        else:
            d_eff, ell = mask.float().sum(1).mean(), (ll * mask).sum(1)
        return -(ell.mean() * n_data - self.kl()) / (n_data * d_eff)

    @torch.no_grad()
    def predict(self, x):
        """Predictive mean and marginal variance (b, d) of y, Eq. 23 plus noise."""
        _, mu, var = self(x)
        return mu, var[:, None] + self.noise_var()

    @torch.no_grad()
    def init_inducing(self, x_all, batch=64):
        """Place Z on embeddings of training inputs (after the operator is pretrained)."""
        ps = torch.cat([self.phi(self.mean_op.features(x_all[i:i + batch])) for i in range(0, len(x_all), batch)])
        idx = torch.randperm(len(ps), device=ps.device)[: self.M]
        self.Z.copy_(ps[idx])
        self.log_ell_a.copy_((ps.std(0).clamp_min(1e-3) * ps.shape[1] ** 0.5).log())    # typical scaled distance ~ 1


class _ZeroMean(nn.Module):
    """No operator: features are the raw input, the prior mean is zero (data are normalised)."""

    def features(self, x):
        return x

    def proj(self, h):
        return torch.zeros_like(h[..., :1])


class SVGP(LoGoSVNNGP):
    """SVGP baseline (Table 3 of the paper): the same K_a (x) K_x variational GP with M inducing points, but zero mean
    and K_a on the raw input function (average-pooled to `pool` points per axis) instead of operator features."""

    def __init__(self, coords, grid, in_ch, pool=16, **kw):
        super().__init__(_ZeroMean(), coords, 1, **kw)
        self.pool = [min(pool, g) for g in grid]
        self.embed = nn.Sequential(nn.Linear(in_ch * math.prod(self.pool), 128), nn.GELU(), nn.Linear(128, self.Z.shape[1]))
        with torch.no_grad():
            self.log_sa.zero_()                                           # zero mean: the GP carries the whole (unit-variance) field

    def phi(self, h):
        x = h.movedim(-1, 1)
        x = F.adaptive_avg_pool1d(x, self.pool[0]) if x.ndim == 3 else F.adaptive_avg_pool2d(x, self.pool)
        return self.embed(x.flatten(1))
