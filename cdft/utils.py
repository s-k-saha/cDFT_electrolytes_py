"""Numerical helpers (utils.c): Poisson solvers and Langevin functions.

All routines are numba-compiled; they are called once per Picard iteration.
"""
import math

import numpy as np
from numba import njit

BC_CODES = {"DD": 0, "DN": 1, "ND": 2, "NN": 3}


def poisson_factorize(h, Nt, bc):
    """LU factors of the (constant) tridiagonal Poisson matrix.

    Returns an (4, M) array with rows a, c, m, bprime of the Thomas algorithm,
    computed with exactly the same arithmetic as the C code so results agree
    bit-for-bit; only the right-hand side has to be swept every iteration.
    """
    M = Nt - 2
    hh = h * h
    a = np.full(M, 1.0 / hh)
    b = np.full(M, -2.0 / hh)
    c = np.full(M, 1.0 / hh)
    if bc == 1:      # DN
        b[M - 1] = -1.0 / hh
        c[M - 1] = 0.0
    elif bc == 2:    # ND
        a[0] = 0.0
        b[0] = -1.0 / hh
    m = np.zeros(M)
    bprime = np.empty(M)
    bprime[0] = b[0]
    for i in range(1, M):
        m[i] = a[i] / bprime[i - 1]
        bprime[i] = b[i] - m[i] * c[i - 1]
    return np.vstack((a, c, m, bprime))


@njit(cache=True, nogil=True)
def poisson_1D(h, Nt, psi_left, psi_right, phi, psi, bc, thomas):
    """Finite-difference psi'' = phi with DD/DN/ND boundary conditions (Thomas algorithm)."""
    M = Nt - 2
    if M <= 0:
        psi[0] = psi_left
        psi[Nt - 1] = psi_right
        return
    a = thomas[0]
    c = thomas[1]
    m = thomas[2]
    bprime = thomas[3]
    dprime = np.empty(M)
    for i in range(M):
        dprime[i] = phi[i + 1]

    if bc == 0:      # DD
        psi[0] = psi_left
        psi[Nt - 1] = psi_right
        dprime[0] -= a[0] * psi_left
        dprime[M - 1] -= c[M - 1] * psi_right
    elif bc == 1:    # DN: psi'(right) = psi_right
        psi[0] = psi_left
        dprime[0] -= a[0] * psi_left
        dprime[M - 1] = phi[Nt - 2] - psi_right / h
    elif bc == 2:    # ND: psi'(left) = psi_left
        psi[Nt - 1] = psi_right
        dprime[0] = phi[1] + psi_left / h
        dprime[M - 1] -= c[M - 1] * psi_right
    else:
        return

    for i in range(1, M):
        dprime[i] -= m[i] * dprime[i - 1]

    psi[Nt - 2] = dprime[M - 1] / bprime[M - 1]
    for i in range(M - 2, -1, -1):
        psi[i + 1] = (dprime[i] - c[i] * psi[i + 2]) / bprime[i]

    if bc == 1:
        psi[Nt - 1] = psi[Nt - 2] + h * psi_right
    if bc == 2:
        psi[0] = psi[1] - h * psi_left


@njit(cache=True, nogil=True)
def psi_calculator(h, Nt, psi_left, psi_right, phi, psi):
    """Backward-sweep Poisson solver (NN, net-neutral systems)."""
    M_end = 0.0
    Q_end = 0.0
    for i in range(Nt - 2, -1, -1):
        x = h * i
        Q_end += phi[i] * h
        M_end += x * phi[i] * h
        psi[i] = psi_right - (M_end - x * Q_end)
    psi[Nt - 1] = psi_right


@njit(cache=True, nogil=True)
def psi_calculator_P(h, Nt, psi_left, psi_right, phi, psi, Eexec, P, lambdaB, P_factor):
    """Backward-sweep Poisson solver with polarisation (trapezoidal quadrature)."""
    M_end = 0.0
    Q_end = 0.0
    P_end = 0.0
    for i in range(Nt - 2, -1, -1):
        x_i = h * i
        x_ip = h * (i + 1)
        Q_end += 0.5 * h * (phi[i] + phi[i + 1])
        M_end += 0.5 * h * (x_i * phi[i] + x_ip * phi[i + 1])
        P_end += 0.5 * h * (P[i] + P[i + 1])
        psi[i] = psi_right - (M_end - x_i * Q_end) - P_factor * P_end
        Eexec[i] = -Q_end
    for i in range(Nt - 2, -1, -1):
        Eexec[i] = Eexec[i] - P[i] * P_factor
    psi[Nt - 1] = psi_right
    Eexec[Nt - 1] = 0.


@njit(cache=True, nogil=True)
def langevin(x):
    if math.fabs(x) < 1e-8:
        return x / 3.0 - x * x * x / 45.0
    return math.cosh(x) / math.sinh(x) - 1.0 / x


@njit(cache=True, nogil=True)
def inv_langevin(y):
    """Fast rational approximation of the inverse Langevin function."""
    if math.isnan(y):
        return math.nan
    if y >= 1.0:
        return math.inf
    if y <= -1.0:
        return -math.inf
    ay = math.fabs(y)
    y2 = y * y
    y4 = y2 * y2
    y6 = y4 * y2
    y8 = y4 * y4
    num = 3.0 - 1.00651 * y2 - 0.962251 * y4 + 1.47353 * y6 - 0.48953 * y8
    den = (1.0 - ay) * (1.0 + 1.01524 * ay)
    return y * num / den
