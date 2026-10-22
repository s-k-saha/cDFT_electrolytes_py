"""Free-energy contributions of the current profiles.

All terms are beta*F/A in units of kBT/sigma^2, integrated over the grid
points 0 .. iend-1 (the region the Picard iteration updates; the points
beyond iend only serve as a bulk reservoir). For a functional F[rho], the
Euler-Lagrange equation solved in solver.py reads

    ln rho_j + beta*Vext_j - c1_j = beta*mu_j,   c1_j = -delta beta*F_ex / delta rho_j,

and every term below is the functional whose derivative the solver uses:

  id      sum_j  int rho_j (ln rho_j - 1)
  orient  int rho_0 [t L(t) - ln(sinh t / t)],  t = L^-1(P / (p rho_0))
          (orientational entropy of the Langevin dipoles, polar solvent only)
  hs      int Phi_Rosenfeld(n_a)
  LJ      1/2 sum_ij eps_ij int int rho_i(x) U(x-x') rho_j(x')
  ES      max_psi { int rho_eff psi - eps0/2 int psi'^2 } (+ Neumann terms),
          rho_eff = e sum_j q_j rho_j - P'. At the Poisson solution this is the
          electrostatic energy at fixed electrode potentials/surface fields;
          for psi = 0 at both ends it equals 1/2 int rho_eff psi.
  ext     sum_j int rho_j Vext_j

plus the total F and the grand potential Omega = F - sum_j mu_j N_j.

bulk_free_energy gives the free-energy density (kBT/sigma^3) of the uniform
bulk fluid at the densities rhob; the electrostatic and orientational terms
vanish there (neutral bulk, psi' = 0).
"""
import math

import numpy as np
from numba import njit

from .constants import PI, EPSILON_0, ALPHA_LJ
from .utils import langevin, inv_langevin, smooth_3pt
from . import fmt, lj


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def ideal(rho, iend, dx):
    Ns = rho.shape[0]
    f = 0.
    for j in range(Ns):
        for i in range(iend):
            r = rho[j, i]
            if r > 0.:
                f += r * (math.log(r) - 1.)
    return f * dx


@njit(cache=True, nogil=True)
def external(rho, Vext, iend, dx):
    Ns = rho.shape[0]
    f = 0.
    for j in range(Ns):
        for i in range(iend):
            if rho[j, i] != 0.:
                f += rho[j, i] * Vext[j, i]
    return f * dx


@njit(cache=True, nogil=True)
def fmt_phi(n, iend, dx):
    """int Phi dx of the Rosenfeld free-energy density (consistent with fmt.fmt_dphidn)."""
    f = 0.
    for i in range(iend):
        n0 = n[0, i]; n1 = n[1, i]; n2 = n[2, i]; n3 = n[3, i]
        n1v = n[4, i]; n2v = n[5, i]
        om = 1 - n3
        f += (-n0 * math.log(om) + (n1 * n2 - n1v * n2v) / om
              + (n2 * n2 * n2 - 3 * n2 * (n2v * n2v)) / (24 * PI * om * om))
    return f * dx


@njit(cache=True, nogil=True)
def lj_energy(rhotrunc, conv, iend, dx):
    """1/2 sum_s int rho_s (U (*) sum_j eps_sj rho_j) dx."""
    Ns = rhotrunc.shape[0]
    f = 0.
    for s in range(Ns):
        for i in range(iend):
            f += rhotrunc[s, i] * conv[s, i]
    return 0.5 * f * dx


@njit(cache=True, nogil=True)
def orientational(rho, P, p, NiR, iend, dx):
    """int rho_0 [t L(t) - ln(sinh t / t)] dx; its rho_0-derivative is the c1 term of es.add_c1_ES."""
    f = 0.
    for j in range(NiR, iend):
        r = rho[0, j]
        if r <= 0.:
            continue
        t = inv_langevin(P[j - NiR] / (p * r))
        if math.fabs(t) <= 1.0e-8:
            g = t * t / 6.
        else:
            g = t * langevin(t) - math.log(math.sinh(t) / t)
        f += r * g
    return f * dx


@njit(cache=True, nogil=True)
def es_energy(rho, q, psi, P, is_polar, N_ES, NiR, dx, bc, Vq_L, Vq_R,
              phi_factor, P_factor, pref):
    """beta*F_ES/A for the current psi (V) on the ES grid.

    rhs = psi'' of Poisson's equation (as in utils.poisson_1D), so that
    rho_eff = -eps0 * rhs / sigma^2 and pref = eps0 * sigma / kBT converts
    to kBT/sigma^2. Charges sit on the interior nodes, fields on the links.
    """
    Ns = rho.shape[0]
    f_q = 0.
    for k in range(1, N_ES - 1):
        rhs = 0.
        for j in range(Ns):
            rhs += q[j] * rho[j, k + NiR] * phi_factor
        if is_polar:
            rhs += (P[k + 1] - P[k - 1]) * P_factor / (2.0 * dx)
        f_q -= rhs * psi[k]
    f_q *= dx
    f_E = 0.
    for k in range(N_ES - 1):
        g = (psi[k + 1] - psi[k]) / dx
        f_E += g * g
    f_E *= 0.5 * dx
    f_b = 0.
    if bc == 1:      # DN: psi'(right) = Vq_R
        f_b = Vq_R * psi[N_ES - 1]
    elif bc == 2:    # ND: psi'(left) = Vq_L
        f_b = -Vq_L * psi[0]
    return pref * (f_q - f_E + f_b)


# ---------------------------------------------------------------------------
def bulk_free_energy(rhob, eps=None):
    """Bulk free-energy density in kBT/sigma^3, consistent with fmt.mu_HS and lj.add_mu_LJ.

    id   sum_j rhob_j (ln rhob_j - 1)
    hs   rho [-ln(1-eta) + 3 eta (2-eta) / (2 (1-eta)^2)]   (Rosenfeld bulk = Percus-Yevick)
    LJ   -alphaLJ/2 sum_ij eps_ij rhob_i rhob_j             (only if eps is given)
    omega = f_tot - sum_j mu_j rhob_j = -p
    """
    rhob = np.asarray(rhob, dtype=float)
    eta = float(fmt.get_eta(rhob))
    f = {"id": float(np.sum(rhob * (np.log(rhob) - 1.))),
         "hs": float(np.sum(rhob)) * (-math.log(1. - eta)
                                      + 3. * eta * (2. - eta) / (2. * (1. - eta) ** 2))}
    mu = fmt.mu_HS(rhob)
    if eps is not None:
        f["LJ"] = -0.5 * ALPHA_LJ * float(rhob @ eps @ rhob)
        lj.add_mu_LJ(mu, eps, rhob)
    f["tot"] = sum(f.values())
    f["omega"] = f["tot"] - float(np.dot(mu, rhob))
    return f


def compute(sys):
    """Return a dict of the free-energy terms of ``sys`` (a solver.System), in kBT/sigma^2."""
    P = sys.P
    rho = sys.rho
    iend, dx = P.iend, P.dx
    F = {"id": ideal(rho, iend, dx)}

    if P.Is_polar:
        F["orient"] = orientational(rho, sys.Pol, P.p, P.NiR, iend, dx)

    # FMT weighted densities of the current rho (same path as System.iterate)
    from .solver import rhocpy, truncate_rho
    rhocopy = np.zeros(P.N)
    Rh = np.zeros(P.K, dtype=np.complex128)
    nhat = np.zeros((6, P.K), dtype=np.complex128)
    n = np.zeros((6, P.N))
    rhocpy(rho, P.NiR, rhocopy)
    sys._rfft(rhocopy, Rh)
    fmt.weight_products(Rh, sys.W, nhat)
    sys._irfft(nhat, n)
    F["hs"] = fmt_phi(n, iend, dx)

    if P.LJ_exists:
        rhotrunc = np.zeros((P.Nspecies, P.N))
        RH = np.zeros((P.Nspecies, P.K), dtype=np.complex128)
        conv = np.zeros((P.Nspecies, P.N))
        truncate_rho(rho, P.NiLJ, rhotrunc)
        sys._rfft(rhotrunc, RH)
        sys._irfft(sys.Uhat[None, :] * (P.eps @ RH), conv)
        F["LJ"] = lj_energy(rhotrunc, conv, iend, dx)

    if P.ES_exists:
        Pol = sys.Pol
        if P.Is_polar and sys.smooth_P:
            Pol = np.empty_like(sys.Pol)
            smooth_3pt(sys.Pol, Pol)
        F["ES"] = es_energy(rho, sys.q, sys.psi, Pol, P.Is_polar, P.N_ES, P.NiR, dx,
                            sys.bc, P.Vq_L, P.Vq_R, sys.phi_factor, sys.P_factor,
                            EPSILON_0 * P.sigma_nm / P.kBT)

    F["ext"] = external(rho, sys.Vext, iend, dx)

    F["tot"] = sum(F.values())
    Nj = rho[:, :iend].sum(axis=1) * dx
    F["Omega"] = F["tot"] - float(np.dot(sys.mu, Nj))
    return F
