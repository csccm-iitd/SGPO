"""Correctness tests for the operator and uncertainty building blocks. Run: python -m pytest tests -q"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgpo.operators import EWNO, FNO, WNO
from sgpo.operators.fno import SpectralConv
from sgpo.operators.symmetry import d4, d4_inv, group_average
from sgpo.operators.wno import WaveConv
from sgpo.uq import metrics
from sgpo.uq.kernels import matern52

torch.manual_seed(0)


# ---------------------------------------------------------------- operators
def _identity_waveconv(width, grid, level):
    wc = WaveConv(width, grid, level)
    eye = torch.eye(width)
    with torch.no_grad():
        wc.w_approx.copy_(eye.view(width, width, *([1] * len(grid))).expand_as(wc.w_approx))
        wc.w_detail.copy_(eye.view(1, width, width, *([1] * len(grid))).expand_as(wc.w_detail))
        wc.w_fine.copy_(eye.expand_as(wc.w_fine))
    return wc


@pytest.mark.parametrize("grid,level", [((64,), 3), ((101,), 4), ((48, 48), 2), ((53, 61), 3)])
def test_waveconv_identity_weights_reconstruct(grid, level):
    """With identity mixing on every band the layer must be DWT followed by IDWT, i.e. the identity."""
    wc = _identity_waveconv(4, grid, level).double()
    x = torch.randn(2, 4, *grid, dtype=torch.float64)
    assert torch.allclose(wc(x), x, atol=1e-5)      # db4 taps are stored to ~1e-8, round trip ~1e-7


def test_waveconv_fine_bands_matter():
    """The fine-level weights must reach the output."""
    wc = _identity_waveconv(3, (64,), 3).double()
    x = torch.randn(1, 3, 64, dtype=torch.float64)
    with torch.no_grad():
        wc.w_fine.zero_()
    assert (wc(x) - x).abs().max() > 1e-3


@pytest.mark.parametrize("cls", ["fno", "wno", "wnof", "ewno"])
@pytest.mark.parametrize("pad", [0, 5])
def test_operator_shapes_and_padding(cls, pad):
    grid = (29, 29)
    m = {"fno": lambda: FNO((6, 6), 8, 2, pad=pad), "ewno": lambda: EWNO(grid, 8, 2, 2, modes=6, pad=pad),
         "wno": lambda: WNO(grid, 8, 2, 2, pad=pad), "wnof": lambda: WNO(grid, 8, 2, 2, pad=pad, modes=(6, 6))}[cls]()
    x = torch.randn(3, *grid, 1)
    y = m(x)
    assert y.shape == (3, *grid, 1) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_wno_1d_shape():
    m = WNO((100,), 8, 3, 2)
    assert m(torch.randn(2, 100, 1)).shape == (2, 100, 1)


def test_spectral_conv_shift_equivariant():
    sc = SpectralConv(4, (5, 5)).to(torch.complex128)   # Module.double() skips complex parameters
    x = torch.randn(2, 4, 32, 32, dtype=torch.float64)
    assert torch.allclose(sc(torch.roll(x, (3, 7), (2, 3))), torch.roll(sc(x), (3, 7), (2, 3)), atol=1e-10)


# ---------------------------------------------------------------- symmetry
def test_d4_inverse_and_group():
    x = torch.randn(2, 7, 7, 3)
    outs = []
    for k in range(8):
        assert torch.equal(d4_inv(d4(x, k), k), x)
        outs.append(d4(x, k))
    assert len({tuple(o.flatten().tolist()) for o in outs}) == 8      # 8 distinct elements


def test_group_average_is_equivariant():
    f = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(25, 25), torch.nn.Unflatten(1, (5, 5, 1)))
    x = torch.randn(4, 5, 5, 1)
    ga = lambda t: group_average(f, t)
    with torch.no_grad():
        for k in range(8):
            assert torch.allclose(ga(d4(x, k)), d4(ga(x), k), atol=1e-5)


# ---------------------------------------------------------------- metrics
def test_gaussian_scores_match_closed_form_and_mc():
    m, s = torch.zeros(1, 200000, dtype=torch.float64), torch.full((1, 200000), 2.0, dtype=torch.float64)
    y = torch.randn(1, 200000, dtype=torch.float64) * 2.0
    assert metrics.gaussian_nll(m, s, y) == pytest.approx(math.log(2.0) + 0.5 * math.log(2 * math.pi) + 0.5, abs=1e-2)
    assert metrics.gaussian_crps(m, s, y) == pytest.approx(2.0 / math.sqrt(math.pi), rel=1e-2)   # E CRPS = s/sqrt(pi)
    for lvl in (0.5, 0.9, 0.95):
        assert metrics.coverage(m, s, y, lvl) == pytest.approx(lvl, abs=5e-3)
    # sample CRPS identity at a single point
    yy, mm, ss = torch.tensor([[0.7]], dtype=torch.float64), torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([[1.3]], dtype=torch.float64)
    xs = (mm + ss * torch.randn(400000, dtype=torch.float64)).flatten()
    mc = (xs - yy).abs().mean() - 0.5 * (xs[:-1] - xs[1:]).abs().mean()
    assert metrics.gaussian_crps(mm, ss, yy) == pytest.approx(mc.item(), rel=2e-2)


def test_energy_score_prefers_truth():
    y = torch.randn(50, 30)
    good = y.unsqueeze(0) + 0.1 * torch.randn(64, 50, 30)
    bad = y.unsqueeze(0) + 0.5 + 0.1 * torch.randn(64, 50, 30)
    assert metrics.energy_score(good, y) < metrics.energy_score(bad, y)


# ---------------------------------------------------------------- Vecchia
def test_vecchia_kx_full_conditioning_matches_dense_kernel():
    """With every predecessor as a neighbour, the Vecchia quadratic form is exact: sum_m w_m^T (K + nugget I)^-1 w_m."""
    from sgpo.models.logos_vnngp import VecchiaKx
    g = torch.linspace(0, 1, 6)
    coords = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    d = len(coords)
    kx = VecchiaKx(coords, k=d - 1, ell=0.3).double()
    W = torch.randn(4, d, dtype=torch.float64)
    c = coords.double()
    K = matern52(c, c, 0.3) + 1e-3 * torch.eye(d, dtype=torch.float64)
    assert torch.allclose(kx.quad(W), (W * torch.linalg.solve(K, W.T).T).sum(), rtol=1e-6)


def test_vecchia_neighbours_match_brute_force():
    from sgpo.uq.vecchia import predecessor_neighbours
    c = np.random.default_rng(1).random((700, 2))
    nbr = predecessor_neighbours(c, 10)
    for i in range(1, 700):
        dist = ((c[:i] - c[i]) ** 2).sum(-1)
        want = set(np.argsort(dist)[: min(10, i)].tolist())
        assert set(nbr[i][nbr[i] >= 0].tolist()) == want


# ---------------------------------------------------------------- models
def test_gpo_kronecker_matches_dense_gp():
    from sgpo.models.gpo import GPO
    torch.manual_seed(0)
    n, d, D = 7, 9, 5
    coords = torch.rand(d, 1)
    m = GPO(coords, D, embed_dim=3).double()
    m.jitter = 0.0                                                        # compare the exact algebra
    f, r, fs = torch.randn(n, D, dtype=torch.float64), torch.randn(n, d, dtype=torch.float64), torch.randn(2, D, dtype=torch.float64)
    p, ps = m.embed(f), m.embed(fs)
    s2 = m.log_noise.exp()
    K = torch.kron(m.ka(p, p) + m.jitter * torch.eye(n, dtype=torch.float64), m.kx() + m.jitter * torch.eye(d, dtype=torch.float64))
    K = K + s2 * torch.eye(n * d, dtype=torch.float64)
    dense = -torch.distributions.MultivariateNormal(torch.zeros(n * d, dtype=torch.float64), K).log_prob(r.flatten()) / (n * d)
    assert torch.allclose(m.nll(f, r), dense, atol=1e-8)
    m.fit_cache(f, r)
    mean, var = m.predict(fs)
    Ks = torch.kron(m.ka(ps, p), m.kx())                                   # (2d, nd)
    mean_d = (Ks @ torch.linalg.solve(K, r.flatten())).reshape(2, d)
    var_d = m.log_sa.exp() ** 2 - (Ks * torch.linalg.solve(K, Ks.T).T).sum(1).reshape(2, d) + s2
    assert torch.allclose(mean, mean_d, atol=1e-7) and torch.allclose(var, var_d, atol=1e-7)


def test_gpo_closed_form_gradients_match_dense_and_survive_repeated_eigenvalues():
    from sgpo.models.gpo import GPO
    torch.manual_seed(0)
    n, d, D = 7, 9, 5
    m = GPO(torch.rand(d, 1), D, embed_dim=3).double()
    f, r = torch.randn(n, D, dtype=torch.float64), torch.randn(n, d, dtype=torch.float64)
    m.nll(f, r).backward()
    g = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    p = m.embed(f); eye = lambda k: m.jitter * torch.eye(k, dtype=torch.float64)
    K = torch.kron(m.ka(p, p) + eye(n), m.kx() + eye(d)) + m.log_noise.exp() * torch.eye(n * d, dtype=torch.float64)
    (-torch.distributions.MultivariateNormal(torch.zeros(n * d, dtype=torch.float64), K).log_prob(r.flatten()) / (n * d)).backward()
    assert all(torch.allclose(a, p_.grad, atol=1e-8) for a, p_ in zip(g, m.parameters()))
    m.zero_grad()
    f[3:] = f[2]                                                          # duplicated inputs: K_a has repeated eigenvalues
    m.nll(f, r).backward()
    assert all(torch.isfinite(p_.grad).all() for p_ in m.parameters())


def test_logos_space_time_output_with_ard_and_feature_noise():
    """Multi-channel (space-time) output: coords over (*grid, t) with t fastest, one noise-head output per channel,
    per-axis Vecchia length scales; ELBO is finite and differentiable, predictions have one value per (x, t)."""
    from sgpo.models.logos_vnngp import LoGoSVNNGP
    from sgpo.operators.wno import WNO
    torch.manual_seed(0)
    grid, nt, cin, w = (16, 12), 5, 3, 8
    op = WNO(grid, width=w, level=1, layers=1, wavelet="db4", in_ch=cin, out_ch=nt)
    axes = [torch.linspace(0, 1, n) for n in (*grid, nt)]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    m = LoGoSVNNGP(op, coords, w, n_inducing=4, embed_dim=3, k_x=6, noise="feature", n_out=nt)
    assert m.kx.log_ell.shape == (3,)
    x, y = torch.randn(4, *grid, cin), torch.randn(4, grid[0] * grid[1] * nt)
    m.init_inducing(x)
    with torch.no_grad():
        m.W.normal_()                                     # K_x enters the ELBO through the KL quadratic in W
    loss = m.elbo(x, y, 40)
    loss.backward()
    assert torch.isfinite(loss) and m.kx.log_ell.grad.abs().sum() > 0 and m.noise_head.weight.grad is not None
    mu, var = m.predict(x)
    assert mu.shape == var.shape == (4, grid[0] * grid[1] * nt) and (var > 0).all()
    # operator output order matches the coordinate order: (r, z, t) with t fastest
    assert torch.allclose(m.mean_op.proj(m.mean_op.features(x)).flatten(1), m(x)[0])


def test_wno_3d_shapes_gradients_and_identity_reconstruction():
    """3D WNO+F (ptwt transforms): output shape, finite gradients, and an identity-weighted wavelet layer that
    reconstructs its input (so the 3D transform pair and the coefficient ordering are consistent)."""
    from sgpo.operators.wno import WNO, WaveConv
    torch.manual_seed(0)
    grid = (20, 24, 12)
    m = WNO(grid, width=6, level=2, layers=2, wavelet="db4", in_ch=3, out_ch=1, modes=[4, 4, 4])
    x = torch.randn(2, *grid, 3)
    out = m(x)
    assert out.shape == (2, *grid, 1)
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    w = WaveConv(3, grid, 2, "db4", fine=True)
    with torch.no_grad():
        eye = torch.eye(3)
        w.w_approx.copy_(eye[..., None, None, None].expand_as(w.w_approx))
        w.w_detail.copy_(eye[None, ..., None, None, None].expand_as(w.w_detail))
        w.w_fine.copy_(eye[None, None].expand_as(w.w_fine))
        z = torch.randn(2, 3, *grid)
        assert torch.allclose(w(z), z, atol=1e-4)


def test_wno_wave_dims_2_on_3d_grid_equals_per_slice_2d_wavelet():
    """wave_dims=2 on a 3D grid: the wavelet path acts on each trailing-axis slice independently (folded into the
    batch) and the layout is restored exactly."""
    from sgpo.operators.wno import WNO
    torch.manual_seed(0)
    m = WNO((20, 24, 5), width=4, level=2, layers=1, wavelet="db4", in_ch=2, modes=[4, 4, 3], wave_dims=2)
    h = torch.randn(3, 4, 20, 24, 5)
    out = m._wave(0, h)
    ref = torch.stack([m.wave[0](h[..., t]) for t in range(5)], -1)
    assert out.shape == h.shape and torch.allclose(out, ref, atol=1e-6)


def test_fourier_upsample_is_exact_on_band_limited_periodic_fields():
    from sgpo.operators.resample import fourier_upsample
    torch.manual_seed(0)
    N, f = 64, 4
    g = torch.arange(N, dtype=torch.float64) / N
    X, Y = torch.meshgrid(g, g, indexing="ij")
    field = sum(torch.randn(()) * torch.cos(2 * torch.pi * (kx * X + ky * Y) + torch.rand(()) * 6)
                for kx in range(-5, 6) for ky in range(-5, 6))[None, None].double()
    coarse = field[..., ::f, ::f]
    assert torch.allclose(fourier_upsample(coarse, f), field, atol=1e-9)


# ---------------------------------------------------------------- FLOPs and adapters
def test_flop_counter_linalg_and_fft():
    from experiments.flops import count
    n, k = 32, 5
    A = torch.randn(n, n, dtype=torch.float64); K = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    B = torch.randn(n, k, dtype=torch.float64)
    assert count(lambda: torch.linalg.cholesky(K)) == pytest.approx(n ** 3 / 3)
    assert count(lambda: torch.linalg.eigh(K)) == pytest.approx(9 * n ** 3)
    L = torch.linalg.cholesky(K)
    assert count(lambda: torch.linalg.solve_triangular(L, B, upper=False)) == pytest.approx(n * n * k)
    assert count(lambda: K @ B) == 2 * n * n * k                            # matmul from FlopCounterMode
    assert count(lambda: torch.fft.fft(torch.randn(3, 64, dtype=torch.complex64))) == pytest.approx(5 * 3 * 64 * 6)


def test_ufno_adapter_imports_without_ufno_and_explains_how_to_get_it(monkeypatch, tmp_path):
    """The U-FNO adapter is importable without U-FNO; building it without the code raises an actionable error."""
    import importlib
    import experiments.ufno_mean as um
    monkeypatch.delenv("UFNO_DIR", raising=False)
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if k != "ufno"})
    real = importlib.import_module

    def no_ufno(name, *args):
        if name == "ufno":
            raise ImportError(name)
        return real(name, *args)
    monkeypatch.setattr(importlib, "import_module", no_ufno)
    with pytest.raises(ImportError, match="UFNO_DIR"):
        um.import_ufno(str(tmp_path))
