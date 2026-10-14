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
    out, ok = _format_table(vals, True)
    ref = "".join("%d %f %.12f %.12f \n" % ((i,) + tuple(r)) for i, r in enumerate(vals))
    assert ok and out.tobytes().decode() == ref


def test_poisson_against_dense_solve():
    h, Nt = 0.01, 400
    x = h * np.arange(Nt)
    phi = np.sin(3 * x) * np.exp(-x)
    for bc, (L, R) in {0: (0.3, -0.1), 1: (0.3, 0.05), 2: (0.05, -0.1)}.items():
        psi = np.zeros(Nt)
        poisson_1D(h, Nt, L, R, phi, psi, bc, poisson_factorize(h, Nt, bc))
        assert np.allclose(np.diff(psi, 2)[1:-1] / h**2, -phi[2:-2], atol=1e-8)
        if bc in (0, 1):
            assert psi[0] == L
        if bc in (0, 2):
            assert psi[-1] == R


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
