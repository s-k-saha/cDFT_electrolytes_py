"""Mean-field (RPA) Lennard-Jones attraction (functional_LJ.c)."""
import numpy as np
import scipy.fft as sfft

from .constants import PI, ALPHA_LJ


def getU(z, rc, rmin):
    if z >= rc:
        return 0.
    if z >= rmin:
        return .4 * z ** -10 - z ** -4 - .4 * rc ** -10 + rc ** -4
    return .5 * (z * z - rmin * rmin) + .4 * rmin ** -10 - rmin ** -4 - .4 * rc ** -10 + rc ** -4


def U_filter_fft(P):
    """FFT of the planar-integrated WCA-split LJ potential, scaled by dx."""
    U = np.zeros(P.N)
    for i in range(1, P.NiLJ + 1):
        u = getU(P.dx * i, P.rc, P.rmin) * 2. * PI
        U[i] = u
        U[P.N - i] = u
    U[0] = getU(0., P.rc, P.rmin) * 2. * PI
    return sfft.rfft(U) * P.dx


def add_mu_LJ(mu, eps, rhob):
    """mu_i += -alphaLJ * sum_j eps_ij rhob_j  (in place, same summation order as C)."""
    for i in range(len(rhob)):
        for j in range(len(rhob)):
            mu[i] += -ALPHA_LJ * eps[i, j] * rhob[j]
