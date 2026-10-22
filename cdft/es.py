"""Mean-field electrostatics and polar-solvent coupling (functional_ES.c).

The potential psi (in volts) lives on the ES grid, i.e. grid points
NiR .. iend-1 (psi[k] <-> x = (k + NiR) dx).  Every iteration Poisson's
equation is solved for the current charge (and polarisation) density, and
the result is mixed into psi with alpha_ES.
"""
import math

import numpy as np
from numba import njit

from .constants import ELECTRON_CHARGE, EPSILON_0
from .utils import poisson_1D, smooth_3pt, inv_langevin


def debye_length(P):
    """Debye length in metres of the salt formed by species 1 (C_2 convention)."""
    return (1. / (abs(P.q[1]) * ELECTRON_CHARGE)
            * math.sqrt(EPSILON_0 * P.kBT / 2 / (P.rhob[1] * P.sigma_nm ** -3)))


def mixing_gain(P):
    """Stability of the Picard scheme for the slowest electrostatic mode.

    Linearising the two-step update (psi mixed with alpha_ES, then rho with
    alpha) for the lowest Fourier mode of the box gives the stability
    condition  alpha * alpha_ES * G < ~4,  with G = (L / (pi * lambda_D))^2
    and lambda_D the *bare* Debye length (the solvent polarisation starts
    at zero, so it cannot be counted on for screening).
    Returns (alpha*alpha_ES*G, largest stable alpha_ES).
    """
    kappa2 = (ELECTRON_CHARGE ** 2 * np.sum(P.q ** 2 * P.rhob) / P.sigma_nm ** 3
              / (EPSILON_0 * P.kBT))          # 1/m^2
    L = P.N_ES * P.dx * P.sigma_nm            # m
    G = kappa2 * (L / math.pi) ** 2
    return P.alpha * P.alpha_ES * G, 4.0 / (P.alpha * G)


def init_psi(P, psi):
    """Debye-Hueckel initial guess, consistent with the initial ion profiles."""
    l_deb = debye_length(P)
    k = np.arange(len(psi))
    psi[:] = P.Vq_L * np.exp(-P.dx * k * P.sigma_nm / l_deb)


@njit(cache=True, nogil=True)
def add_c1_ES(rho, c1, q, dx, N_ES, Vq_L, Vq_R, bc, thomas, is_polar, smooth_P,
              P, P_smooth, E_exec, E_tmp, phi, psi, psi_tmp, alpha_ES,
              phi_factor, P_factor, sigma_nm, e_kBT, p, NiR, iend):
    Ns = rho.shape[0]
    for i in range(N_ES):
        phi[i] = 0.
        psi_tmp[i] = 0.
    if is_polar:
        for i in range(N_ES):
            E_tmp[i] = 0.

    # phi = -e/(sigma eps0) * sum_j q_j rho_j  (so that psi'' = phi gives psi in V)
    for i in range(NiR, iend):
        for j in range(Ns):
            phi[i - NiR] += q[j] * rho[j, i] * phi_factor

    if is_polar:
        if smooth_P:
            smooth_3pt(P, P_smooth)
            poisson_1D(dx, N_ES, Vq_L, Vq_R, phi, psi_tmp, bc, thomas,
                       True, P_smooth, P_factor, E_tmp, sigma_nm)
        else:
            poisson_1D(dx, N_ES, Vq_L, Vq_R, phi, psi_tmp, bc, thomas,
                       True, P, P_factor, E_tmp, sigma_nm)
    else:
        poisson_1D(dx, N_ES, Vq_L, Vq_R, phi, psi_tmp, bc, thomas,
                   False, P, P_factor, E_tmp, sigma_nm)

    # potential (and field) mixing
    for i in range(N_ES):
        psi[i] = alpha_ES * psi_tmp[i] + (1 - alpha_ES) * psi[i]
    if bc == 2:      # ND: psi'(left) = Vq_L (V/sigma)
        psi[0] = psi[1] - dx * Vq_L
    else:
        psi[0] = Vq_L
    if bc == 1:      # DN: psi'(right) = Vq_R (V/sigma)
        psi[N_ES - 1] = psi[N_ES - 2] + dx * Vq_R
    else:
        psi[N_ES - 1] = Vq_R
    if is_polar:
        for i in range(N_ES):
            E_exec[i] = alpha_ES * E_tmp[i] + (1 - alpha_ES) * E_exec[i]

    for i in range(Ns):
        for j in range(NiR, iend):
            c1[i, j] -= q[i] * psi[j - NiR] * e_kBT

    # ideal (orientational) contribution of the dipolar solvent, species 0
    if is_polar:
        for j in range(NiR, iend):
            if rho[0, j] <= 0.:
                continue  # no solvent (e.g. rho_0 read back as 0 on restart): no entropy term
            temp = inv_langevin(P[j - NiR] / (p * rho[0, j]))
            if math.fabs(temp) <= 1.0e-8:
                c1[0, j] -= -math.log(1. + temp * temp / 6.)
            else:
                c1[0, j] -= math.log(temp / math.sinh(temp))
