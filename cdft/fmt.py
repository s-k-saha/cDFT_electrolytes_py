"""Rosenfeld FMT for hard spheres (functional_fmt.c).

Weight functions are stored as one (6, K) complex array in the order
n0, n1, n2, n3, n1v, n2v so that all six convolutions run as one batched FFT.
"""
import math

import numpy as np
from numba import njit

from .constants import PI


def get_eta(rhob):
    eta = 0.0
    for r in rhob:
        eta += (PI / 6.) * r
    return eta


def mu_HS(rhob):
    """log(rhob_i) + excess HS chemical potential (Percus-Yevick/Rosenfeld)."""
    eta = get_eta(rhob)
    contribution = ((14. * eta - 13. * eta * eta + 5. * eta * eta * eta)
                    / (2. * (1. - eta) * (1. - eta) * (1. - eta)) - math.log(1. - eta))
    return np.log(rhob) + contribution


def fmt_omegas(R, Lx, K):
    dk = 2. * PI / Lx
    k = dk * np.arange(1, K)
    W = np.zeros((6, K), dtype=np.complex128)
    w2 = np.empty(K)
    w3 = np.empty(K)
    w2v = np.empty(K)
    w2[0] = 4. * PI * R * R
    w3[0] = 4. / 3 * PI * R * R * R
    w2v[0] = 0.0
    w2[1:] = 4. * PI * R / k * np.sin(k * R)
    w3[1:] = 4. * PI / (k * k * k) * (np.sin(k * R) - k * R * np.cos(k * R))
    w2v[1:] = (-4. * PI) * (-k * R * np.cos(k * R) + np.sin(k * R)) / (k * k)
    W[0].real = w2 / (4. * PI * R * R)
    W[1].real = w2 / (4. * PI * R)
    W[2].real = w2
    W[3].real = w3
    W[4].imag = w2v / (4. * PI * R)
    W[5].imag = w2v
    return W


@njit(cache=True, nogil=True)
def fmt_dphidn(n, d):
    """Partial derivatives of the Rosenfeld free-energy density, (6,N) -> (6,N)."""
    N = n.shape[1]
    for i in range(N):
        n0 = n[0, i]; n1 = n[1, i]; n2 = n[2, i]; n3 = n[3, i]
        n1v = n[4, i]; n2v = n[5, i]
        om = 1 - n3
        io = 1. / om
        io2 = io * io
        d[0, i] = -math.log(om)
        d[1, i] = n2 * io
        d[2, i] = n1 * io + (3 * n2 * n2 - 3 * (n2v * n2v)) * io2 * (1. / (24 * PI))
        d[3, i] = (n0 * io + (n1 * n2 - n1v * n2v) * io2
                   + (n2 * n2 * n2 - 3 * n2 * (n2v * n2v)) * io2 * io * (1. / (12 * PI)))
        d[4, i] = -n2v * io
        d[5, i] = -n1v * io - 3 * n2 * n2v * io2 * (1. / (12 * PI))


@njit(cache=True, nogil=True)
def weight_products(Rh, W, out):
    """out[a,k] = Rh[k] * W[a,k]   (Fourier-space convolution for the n_a)."""
    for a in range(W.shape[0]):
        for k in range(W.shape[1]):
            out[a, k] = Rh[k] * W[a, k]
