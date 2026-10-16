"""Numerical helpers (utils.c): Poisson solver, smoothing, Langevin functions."""
import math

import numpy as np
from numba import njit

BC_CODES = {"DD": 0, "DN": 1, "ND": 2}


def poisson_factorize(h, Nt, bc):
    """LU factors of the (constant) tridiagonal Poisson matrix.

    Returns a (4, M) array with rows a, c, m, bprime of the Thomas algorithm,
    computed with the same arithmetic as the C code; only the right-hand side
    has to be swept every iteration.
    """
    M = max(Nt - 2, 1)
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
def poisson_1D(h, Nt, psi_left, psi_right, phi, psi, bc, thomas,
               with_P, P, P_factor, Exec, sigma_nm):
    """Solve psi'' = phi (+ P_factor * P') on Nt points (poisson_1D / poisson_1D_P).

    bc: 0 = DD, 1 = DN (psi'(right) = psi_right), 2 = ND (psi'(left) = psi_left).
    phi already carries the -e/(sigma*eps0) factor, so psi is in volts.
    If with_P, P' uses central differences and the field E = -psi'/sigma_nm
    (V/m) is written to Exec.
    """
    M = Nt - 2
    if M <= 0:
        psi[0] = psi_left
        psi[Nt - 1] = psi_right
        if with_P and Nt == 2:
            E = -(psi[1] - psi[0]) / h / sigma_nm
            Exec[0] = E
            Exec[1] = E
        return
    a = thomas[0]
    c = thomas[1]
    m = thomas[2]
    bprime = thomas[3]
    dprime = np.empty(M)
    if with_P:
        for i in range(M):
            j = i + 1
            dprime[i] = phi[j] + (P[j + 1] - P[j - 1]) * P_factor / (2.0 * h)
    else:
        for i in range(M):
            dprime[i] = phi[i + 1]

    if bc == 0:      # DD
        psi[0] = psi_left
        psi[Nt - 1] = psi_right
        dprime[0] -= a[0] * psi_left
        dprime[M - 1] -= c[M - 1] * psi_right
    elif bc == 1:    # DN
        psi[0] = psi_left
        dprime[0] -= a[0] * psi_left
        dprime[M - 1] -= psi_right / h
    else:            # ND
        psi[Nt - 1] = psi_right
        dprime[0] += psi_left / h
        dprime[M - 1] -= c[M - 1] * psi_right

    for i in range(1, M):
        dprime[i] -= m[i] * dprime[i - 1]
    psi[Nt - 2] = dprime[M - 1] / bprime[M - 1]
    for i in range(M - 2, -1, -1):
        psi[i + 1] = (dprime[i] - c[i] * psi[i + 2]) / bprime[i]

    if bc == 1:
        psi[Nt - 1] = psi[Nt - 2] + h * psi_right
    if bc == 2:
        psi[0] = psi[1] - h * psi_left

    if with_P:
        # electric field E = -dpsi/dx, second order everywhere
        for j in range(1, Nt - 1):
            Exec[j] = -(psi[j + 1] - psi[j - 1]) / (2.0 * h * sigma_nm)
        if bc == 2:
            Exec[0] = -psi_left
        else:
            Exec[0] = -(-3.0 * psi[0] + 4.0 * psi[1] - psi[2]) / (2.0 * h * sigma_nm)
        if bc == 1:
            Exec[Nt - 1] = -psi_right
        else:
            Exec[Nt - 1] = -(3.0 * psi[Nt - 1] - 4.0 * psi[Nt - 2] + psi[Nt - 3]) / (2.0 * h * sigma_nm)


@njit(cache=True, nogil=True)
def smooth_3pt(inp, out):
    """3-point binomial (0.25, 0.5, 0.25) smoothing; removes grid-scale noise in P."""
    Nt = inp.shape[0]
    if Nt <= 2:
        for i in range(Nt):
            out[i] = inp[i]
        return
    out[0] = inp[0]
    out[Nt - 1] = inp[Nt - 1]
    for i in range(1, Nt - 1):
        out[i] = 0.25 * inp[i - 1] + 0.5 * inp[i] + 0.25 * inp[i + 1]


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
