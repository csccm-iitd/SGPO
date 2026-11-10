"""GPO baseline (Kumar & Chakraborty): exact GP over all training fields with an operator mean.

    vec(Y) ~ N( vec(M[A]),  K_a(Phi(A), Phi(A)) (x) K_x + s^2 I )
M is a pretrained (frozen) operator, Phi a learned embedding of its features, K_a Matern-5/2 ARD on Phi, K_x a dense
Matern-5/2 kernel on the grid. The Kronecker structure gives the exact marginal likelihood and predictive
distribution from two eigendecompositions, K_a = Qa La Qa^T (n x n) and K_x = Qx Lx Qx^T (d x d): memory O(n^2 + d^2 + nd)
and time O(n^3 + d^3) per step, versus LoGoS-GPO's minibatch ELBO with M inducing points and a Vecchia K_x.
"""
import math
import torch
import torch.nn as nn

from sgpo.uq.kernels import matern52


def _eigh(K):
    """Symmetric eigendecomposition; falls back to the CPU solver, then to tenfold jitter, when it fails to converge."""
    try:
        return torch.linalg.eigh(K)
    except torch.linalg.LinAlgError:
        pass
    eye = torch.eye(len(K), dtype=K.dtype)
    for jitter in (0, 1e-6, 1e-5, 1e-4):
        try:
            la, Q = torch.linalg.eigh(K.cpu() + jitter * K.diagonal().mean().item() * eye)
            return la.to(K.device), Q.to(K.device)
        except torch.linalg.LinAlgError:
            continue
    raise torch.linalg.LinAlgError("eigh failed with jitter up to 1e-4")


class _KronNLL(torch.autograd.Function):
    """Sum of 0.5 r^T S^-1 r + 0.5 log|S| for S = Ka (x) Kx + s2 I, with closed-form gradients.

    Differentiating through eigh multiplies by 1 / (la_i - la_j), which overflows when Ka has (near-)repeated
    eigenvalues; the exact gradients below need no eigenvalue gaps. With A = S^-1 r reshaped to (n, d):
        dL/dKa = 0.5 (Qa diag(sum_j lx_j / lam_ij) Qa^T - A Kx A^T),  dL/dKx likewise,
        dL/ds2 = 0.5 (sum 1 / lam - |A|^2),                         dL/dr = A.
    """

    @staticmethod
    def forward(ctx, Ka, Kx, s2, r):
        la, Qa = _eigh(Ka); lx, Qx = _eigh(Kx)
        la, lx = la.clamp_min(0), lx.clamp_min(0)
        lam = la[:, None] * lx[None] + s2
        rt = Qa.T @ r @ Qx
        A = Qa @ (rt / lam) @ Qx.T
        ctx.save_for_backward(Qa, Qx, la, lx, lam, A, Ka, Kx)
        return 0.5 * (rt ** 2 / lam).sum() + 0.5 * lam.log().sum()

    @staticmethod
    def backward(ctx, g):
        Qa, Qx, la, lx, lam, A, Ka, Kx = ctx.saved_tensors
        dKa = 0.5 * ((Qa * (lx[None] / lam).sum(1)[None]) @ Qa.T - A @ Kx @ A.T)
        dKx = 0.5 * ((Qx * (la[:, None] / lam).sum(0)[None]) @ Qx.T - A.T @ Ka @ A)
        ds2 = 0.5 * ((1 / lam).sum() - (A ** 2).sum())
        return g * dKa, g * dKx, g * ds2, g * A


class GPO(nn.Module):
    def __init__(self, coords, feat_dim, embed_dim=32, ell_x=0.1, jitter=1e-6):
        super().__init__()
        self.register_buffer("coords", coords)
        self.embed = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.log_ell_a = nn.Parameter(torch.zeros(embed_dim))
        self.log_sa = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_ell_x = nn.Parameter(torch.tensor(math.log(ell_x)))
        self.log_noise = nn.Parameter(torch.tensor(math.log(1e-2)))
        self.jitter = jitter

    def ka(self, p, q):
        ell = self.log_ell_a.exp()
        return self.log_sa.exp() ** 2 * matern52(p / ell, q / ell, 1.0)

    def kx(self):
        return matern52(self.coords, self.coords, self.log_ell_x.exp())

    def _eig(self, f):
        p = self.embed(f)
        n, d = len(p), len(self.coords)
        la, Qa = _eigh(self.ka(p, p).double() + self.jitter * torch.eye(n, device=p.device, dtype=torch.float64))
        lx, Qx = _eigh(self.kx().double() + self.jitter * torch.eye(d, device=p.device, dtype=torch.float64))
        return p, la.clamp_min(0), Qa, lx.clamp_min(0), Qx

    def nll(self, f, r):
        """Per-point negative log marginal likelihood. f (n, feat_dim) features, r (n, d) residuals y - M[a]."""
        p = self.embed(f)
        n, d = r.shape
        eye = lambda k: self.jitter * torch.eye(k, device=p.device, dtype=torch.float64)
        L = _KronNLL.apply(self.ka(p, p).double() + eye(n), self.kx().double() + eye(d), self.log_noise.exp().double(), r.double())
        return (L + 0.5 * n * d * math.log(2 * math.pi)) / (n * d)

    @torch.no_grad()
    def fit_cache(self, f, r):
        p, la, Qa, lx, Qx = self._eig(f)
        lam = la[:, None] * lx[None] + self.log_noise.exp().double()
        self._cache = (p.double(), Qa, Qx, lx, lam, (Qa.T @ r.double() @ Qx) / lam)
        return self

    @torch.no_grad()
    def predict(self, f_star):
        """Predictive mean of the residual and marginal variance of y, both (b, d)."""
        p, Qa, Qx, lx, lam, alpha_t = self._cache
        ks = self.ka(self.embed(f_star).double(), p) @ Qa                   # (b, n) in the K_a eigenbasis
        mean = ks @ alpha_t @ (lx[:, None] * Qx.T)                          # K_*a (x) K_x alpha
        C = (Qx * lx[None]) ** 2                                            # (d, d): Qx[s, j]^2 lx_j^2
        var = self.log_sa.exp().double() ** 2 - (ks ** 2) @ (1 / lam) @ C.T
        return mean, var.clamp_min(0) + self.log_noise.exp().double()
