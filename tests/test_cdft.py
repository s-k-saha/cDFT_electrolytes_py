"""Run with:  python -m pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cdft.io import _format_table                      # noqa: E402
from cdft.utils import poisson_1D, poisson_factorize   # noqa: E402
from cdft import System                                # noqa: E402


def test_fast_formatter_matches_printf():
    rng = np.random.default_rng(1)
    M = 50000
    v = np.concatenate([
        rng.standard_normal(M) * 10.0 ** rng.integers(-14, 4, M),
        (rng.integers(-10**15, 10**15, M) + 0.5) / 1e12,   # exact/near ties
        [0.0, -0.0, 5e-13, -5e-13, 1.5e-12, 2.5e-12, -1e-300],
    ])
    v = v[np.abs(v) < 4500]
    vals = v[: len(v) // 3 * 3].reshape(-1, 3)
    out, ok = _format_table(vals, np.array([6, 12, 12], dtype=np.int64))
    ref = "".join("%d %f %.12f %.12f \n" % ((i,) + tuple(r)) for i, r in enumerate(vals))
    assert ok and out.tobytes().decode() == ref


def test_poisson_against_dense_solve():
    h, Nt = 0.01, 400
    x = h * np.arange(Nt)
    phi = np.sin(3 * x) * np.exp(-x)
    P = np.cos(2 * x)
    dummy = np.zeros(Nt)
    for bc, (L, R) in {0: (0.3, -0.1), 1: (0.3, 0.05), 2: (0.05, -0.1)}.items():
        th = poisson_factorize(h, Nt, bc)
        psi = np.zeros(Nt)
        poisson_1D(h, Nt, L, R, phi, psi, bc, th, False, dummy, 0.0, dummy, 1.0)
        assert np.allclose(np.diff(psi, 2)[1:-1] / h**2, phi[2:-2], atol=1e-8)
        if bc in (0, 1):
            assert psi[0] == L
        if bc in (0, 2):
            assert psi[-1] == R
        # with polarisation: psi'' = phi + Pf * P', and E = -psi'/sigma
        Pf, sig = 0.7, 2.0
        E = np.zeros(Nt)
        poisson_1D(h, Nt, L, R, phi, psi, bc, th, True, P, Pf, E, sig)
        dP = (P[2:] - P[:-2]) / (2 * h)
        assert np.allclose(np.diff(psi, 2) / h**2, (phi + 0 * P)[1:-1] + Pf * dP, atol=1e-8)
        assert np.allclose(E[1:-1], -(psi[2:] - psi[:-2]) / (2 * h * sig))


def test_bulk_fluid_is_a_fixed_point(tmp_path):
    """Without a wall potential and far from the wall, the bulk density is stationary."""
    (tmp_path / "params_geometry.txt").write_text(
        "Nspecies=2\nN=4000\nR=0.5\ndx=0.01\nalpha=0.05\nNbatch=5\nIs_polar=0\n")
    (tmp_path / "params_system.txt").write_text(
        "T=300\new=[0.0,0.0]\nrhob=[0.2,0.1]\neps=[1.0,0.5,0.5,0.8]\n")
    s = System(str(tmp_path), verbose=False)
    s.run(max_batches=2, verbose=False)
    mid = slice(1500, 2500)
    assert np.allclose(s.rho[0, mid], 0.2, rtol=1e-5)  # grid vs analytic bulk mu
    assert np.allclose(s.rho[1, mid], 0.1, rtol=1e-5)  # grid vs analytic bulk mu
    assert os.path.exists(s.outfname)


def test_free_energy_terms(tmp_path):
    """Bulk Rosenfeld = PY free energy; id/LJ/ES/ext derivatives match the solver's c1."""
    from cdft import free_energy as fe
    from cdft.fmt import fmt_omegas, fmt_dphidn, weight_products
    from cdft.solver import truncate_rho, rhocpy, c1_hat
    from cdft.utils import poisson_1D

    W = fmt_omegas(0.5, 10.0, 4)
    rho = 0.7
    eta = np.pi / 6 * rho
    n = np.repeat((W[:, 0].real * rho)[:, None], 10, axis=1)
    ref = rho * (-np.log(1 - eta) + 3 * eta * (2 - eta) / (2 * (1 - eta) ** 2))
    assert np.isclose(fe.fmt_phi(n, 10, 0.1), ref, rtol=1e-6)

    (tmp_path / "params_geometry.txt").write_text(
        "Nspecies=3\nN=2000\nR=0.5\ndx=0.01\nalpha=0.01\nNbatch=10\nIs_polar=0\n")
    (tmp_path / "params_system.txt").write_text(
        "T=300\new=[1.2,0.0,0.0]\nrhob=[0.6,0.01,0.01]\neps=[1.,0.8,0.8,0.8,0,0,0.8,0,0]\n"
        "alpha_ES=0.003\nq=[0.,1.,-1.]\nBC=DD\nVq_L=0.05\nVq_R=0.0\n")
    s = System(str(tmp_path), verbose=False)
    s.run(max_batches=5, verbose=False)
    P = s.P
    assert set(s.F) == {"id", "hs", "LJ", "ES", "ext", "tot", "Omega"}
    footer = open(s.outfname).read().splitlines()
    assert "F_hs :" in footer[-1] and "f_hs :" in footer[-2]

    # psi = exact Poisson solution of the current rho -> F_ES is stationary in psi
    phi = (P.q @ s.rho[:, P.NiR:P.iend]) * s.phi_factor
    poisson_1D(P.dx, P.N_ES, P.Vq_L, P.Vq_R, phi, s.psi, s.bc, s.thomas,
               False, s.Pol, 0.0, s.E_tmp, P.sigma_nm)
    truncate_rho(s.rho, P.NiLJ, s.rhotrunc)
    s._rfft(s.rhotrunc, s.RH)
    c1_lj = s._irfft(-s.Uhat * (P.eps @ s.RH), np.zeros((3, P.N)))
    # FMT part of the solver's c1 for the current rho
    rhocpy(s.rho, P.NiR, s.rhocopy)
    s._rfft(s.rhocopy, s.Rh)
    weight_products(s.Rh, s.W, s.nhat)
    s._irfft(s.nhat, s.n)
    fmt_dphidn(s.n, s.dphi)
    s._rfft(s.dphi, s.D)
    c1_hat(s.D, s.W, s.Uhat, P.eps, s.RH, False, s.c1h)
    c1_hs = s._irfft(s.c1h, np.zeros((3, P.N)))
    for j, i in [(0, 100), (0, 300), (1, 700), (2, 1000)]:
        r0 = s.rho[j, i]
        h = 1e-6
        s.rho[j, i] = r0 + h
        Fp = fe.compute(s)
        s.rho[j, i] = r0 - h
        Fm = fe.compute(s)
        s.rho[j, i] = r0
        d = {k: (Fp[k] - Fm[k]) / (2 * h * P.dx) for k in Fp}
        assert np.isclose(d["hs"], -c1_hs[j, i], rtol=1e-6)
        assert np.isclose(d["id"], np.log(r0), atol=1e-6)
        assert np.isclose(d["ext"], s.Vext[j, i], atol=1e-6)
        assert np.isclose(d["LJ"], -c1_lj[j, i], rtol=1e-6)
        assert np.isclose(d["ES"], P.q[j] * s.psi[i - P.NiR] * s.e_kBT, rtol=1e-5, atol=1e-9)
    F0 = fe.compute(s)["ES"]
    for k in (100, 600):
        s.psi[k] += 1e-4
        Fp = fe.compute(s)["ES"]
        s.psi[k] -= 2e-4
        Fm = fe.compute(s)["ES"]
        s.psi[k] += 1e-4
        assert abs(Fp - Fm) < 1e-8 * abs(F0)


def _polar_case(path, bc="DD", VqL=0.05, VqR=0.0):
    (path / "params_geometry.txt").write_text(
        "Nspecies=3\nN=2000\nR=0.5\ndx=0.01\nalpha=0.01\nNbatch=10\nIs_polar=1\n")
    (path / "params_system.txt").write_text(
        "T=300\new=[1.2,0.0,0.0]\nrhob=[0.6,0.01,0.01]\neps=[1.,0.8,0.8,0.8,0,0,0.8,0,0]\n"
        "alpha_ES=0.003\nq=[0.,1.,-1.]\nBC=%s\nVq_L=%r\nVq_R=%r\np=17.31\n" % (bc, VqL, VqR))


def test_polar_restart(tmp_path):
    """rho_0 near the wall is written as 0: restarting must not divide by it, and P, E carry over."""
    _polar_case(tmp_path)
    s = System(str(tmp_path), verbose=False)
    s.run(max_batches=3, verbose=False)
    s2 = System(str(tmp_path), verbose=False)
    assert s2.count_iter == s.count_iter
    assert np.allclose(s2.Pol, s.Pol, atol=1e-12)
    assert np.allclose(s2.E_exec, s.E_exec, atol=1e-6 * np.abs(s.E_exec).max())
    s2.run(max_batches=1, verbose=False)
    assert np.isfinite(s2.dev).all()
    assert s2.dev.max() < 2 * s.dev.max()  # continues instead of restarting from P = E = 0


def test_neumann_boundary_is_kept(tmp_path):
    """With DN/ND the Neumann end of psi obeys psi' = Vq (V/sigma) instead of being set to Vq."""
    for bc, VqL, VqR in (("DN", 0.05, 0.001), ("ND", 0.002, 0.0)):
        d = tmp_path / bc
        d.mkdir()
        _polar_case(d, bc, VqL, VqR)
        s = System(str(d), verbose=False)
        s.run(max_batches=2, verbose=False)
        P = s.P
        if bc == "DN":
            assert s.psi[0] == VqL
            assert np.isclose((s.psi[-1] - s.psi[-2]) / P.dx, VqR, rtol=1e-9)
            assert np.isclose(s.E_tmp[-1], -VqR / P.sigma_nm, rtol=1e-6)
        else:
            assert s.psi[-1] == VqR
            assert np.isclose((s.psi[1] - s.psi[0]) / P.dx, VqL, rtol=1e-9)
            assert np.isclose(s.E_tmp[0], -VqL / P.sigma_nm, rtol=1e-6)


def test_bulk_free_energy():
    """d f/d rhob_j = mu_j of the solver, and omega = -p (Percus-Yevick + mean-field LJ)."""
    from cdft.free_energy import bulk_free_energy
    from cdft.fmt import mu_HS
    from cdft.lj import add_mu_LJ
    from cdft.constants import ALPHA_LJ
    rhob = np.array([0.6, 0.01, 0.02])
    eps = np.array([[1., 0.8, 0.8], [0.8, 0., 0.], [0.8, 0., 0.]])
    mu = mu_HS(rhob)
    add_mu_LJ(mu, eps, rhob)
    h = 1e-7
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        d = (bulk_free_energy(rhob + e, eps)["tot"] - bulk_free_energy(rhob - e, eps)["tot"]) / (2 * h)
        assert np.isclose(d, mu[j], rtol=1e-7)
    rho = rhob.sum()
    eta = np.pi / 6 * rho
    p = rho * (1 + eta + eta ** 2) / (1 - eta) ** 3 - 0.5 * ALPHA_LJ * rhob @ eps @ rhob
    f = bulk_free_energy(rhob, eps)
    assert np.isclose(f["omega"], -p, rtol=1e-6)
    assert "LJ" not in bulk_free_energy(rhob)
