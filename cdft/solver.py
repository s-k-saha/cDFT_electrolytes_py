"""Picard solver for the Euler-Lagrange equations (init.c + iterator.c).

Performance notes (relative to the C code, results are identical up to
floating-point round-off):

* All convolutions are linear, so they are combined in Fourier space:
  the six FMT weighted densities need 1 forward + 6 inverse FFTs, and the
  FMT + LJ one-body direct correlation functions of *all* species need
  6 + Nspecies forward and Nspecies inverse FFTs.  The C code performs
  24 + 2*Nspecies**2 FFTs (and creates a new FFTW plan for each one).
* FFTs are batched (one call per group) and write into preallocated
  buffers (numpy.fft ``out=``); with ``workers > 1`` scipy.fft is used to
  run them multithreaded (worthwhile for N >~ 50000).
* Every point-wise loop is a numba-compiled kernel (cached on disk).
"""
import math
import os
import time

import numpy as np
import scipy.fft as sfft
from numba import njit

from .constants import ELECTRON_CHARGE, EPSILON_0, P_D
from .params import read_params
from .utils import BC_CODES, langevin, poisson_factorize
from . import fmt, lj, es, io


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def rhocpy(rho, NiR, out):
    """Total density, zeroed on the last NiR-1 points (input of the FMT convolutions)."""
    Ns, N = rho.shape
    for i in range(N):
        out[i] = 0.
    for j in range(Ns):
        for i in range(N - NiR + 1):
            out[i] += rho[j, i]


@njit(cache=True, nogil=True)
def truncate_rho(rho, NiLJ, out):
    """Species densities zeroed on the last NiLJ-1 points (input of the LJ convolutions)."""
    Ns, N = rho.shape
    for j in range(Ns):
        for i in range(N - NiLJ + 1):
            out[j, i] = rho[j, i]
        for i in range(N - NiLJ + 1, N):
            out[j, i] = 0.


@njit(cache=True, nogil=True)
def c1_hat(D, W, Uhat, eps, RH, use_lj, out):
    """Fourier transform of c1 for every species.

    out[s,k] = -sum_a D[a,k] W[a,k] - Uhat[k] * sum_j eps[s,j] RH[j,k]
    """
    Ns = out.shape[0]
    K = out.shape[1]
    for k in range(K):
        f = 0j
        for a in range(W.shape[0]):
            f -= D[a, k] * W[a, k]
        for s in range(Ns):
            v = f
            if use_lj:
                g = 0j
                for j in range(Ns):
                    g += eps[s, j] * RH[j, k]
                v -= Uhat[k] * g
            out[s, k] = v


@njit(cache=True, nogil=True)
def picard_update(rho, rhonew, c1, Vext, rhob, c1_bulk, mu, iend, NiR, alpha, dx,
                  LG, rhobL, rhobG, h_target, is_polar, P, Pnew, E_exec, p, L_factor, dev):
    Ns = rho.shape[0]
    if not LG:
        for j in range(Ns):
            for i in range(iend):
                rhonew[j, i] = rhob[j] * math.exp(-Vext[j, i] + c1[j, i] - c1_bulk[j])
    else:
        for j in range(Ns):
            for i in range(iend):
                rhonew[j, i] = math.exp(-Vext[j, i] + c1[j, i] + mu[j])

    if is_polar:
        for i in range(iend):
            Pnew[i] = p * rho[0, i] * langevin(p * E_exec[i] * L_factor)

    if not LG:
        for j in range(Ns):
            d = 0.
            for i in range(iend):
                d += math.fabs(rhonew[j, i] - rho[j, i])
            dev[j] = d * dx
        for i in range(iend):
            for j in range(Ns):
                rho[j, i] = (1 - alpha) * rho[j, i] + alpha * rhonew[j, i]
    else:
        # fixed-adsorption (Gamma = h_target*(rhobL-rhobG)) liquid-gas iteration
        for j in range(Ns):
            Gamma_new = 0.
            for i in range(NiR, iend):
                rhonew[j, i] = (1 - alpha) * rho[j, i] + alpha * rhonew[j, i]
                Gamma_new += rhonew[j, i] - rhobG[j]
            Gamma_new *= dx
            factor = (h_target * (rhobL[j] - rhobG[j])) / Gamma_new
            d = 0.
            for i in range(NiR, iend):
                rhonew[j, i] = factor * (rhonew[j, i] - rhobG[j]) + rhobG[j]
                d += math.fabs(rhonew[j, i] - rho[j, i])
                rho[j, i] = rhonew[j, i]
            dev[j] = d * dx

    if is_polar:
        for i in range(NiR, iend):
            P[i] = (1 - alpha) * P[i] + alpha * Pnew[i]


# ---------------------------------------------------------------------------
class System:
    def __init__(self, data_dir, workers=1, verbose=True):
        self.data_dir = data_dir
        self.workers = workers
        P = self.P = read_params(data_dir)
        Ns, N, K = P.Nspecies, P.N, P.K

        # state
        self.rho = np.zeros((Ns, N))
        self.rhonew = np.zeros((Ns, N))
        self.c1 = np.zeros((Ns, N))
        self.dev = np.zeros(Ns)
        self.count_iter = 0

        # work buffers
        self.rhocopy = np.zeros(N)
        self.rhotrunc = np.zeros((Ns, N))
        self.nhat = np.zeros((6, K), dtype=np.complex128)
        self.dphi = np.zeros((6, N))
        self.c1h = np.zeros((Ns, K), dtype=np.complex128)
        self.Rh = np.zeros(K, dtype=np.complex128)
        self.n = np.zeros((6, N))
        self.D = np.zeros((6, K), dtype=np.complex128)
        self.RH = np.zeros((Ns, K), dtype=np.complex128)

        # FMT weights, LJ kernel
        self.W = fmt.fmt_omegas(P.R, P.Lx, K)
        self.Uhat = lj.U_filter_fft(P) if P.LJ_exists else np.zeros(K, dtype=np.complex128)

        # electrostatics / polarisation
        nES = max(P.N_ES, 1)
        self.phi = np.zeros(nES)
        self.psi = np.zeros(nES)
        self.Pol = np.zeros(N)
        self.Pnew = np.zeros(N)
        self.E_exec = np.zeros(N)
        self.q = P.q if P.ES_exists else np.zeros(Ns)
        self.bc = BC_CODES.get(P.BC, 0)
        self.thomas = poisson_factorize(P.dx, max(P.N_ES, 3), self.bc)
        self.c1_factor = (ELECTRON_CHARGE * ELECTRON_CHARGE) * (1. / (P.sigma_nm * EPSILON_0 * P.kBT))
        self.P_factor = P_D / (P.sigma_nm * ELECTRON_CHARGE)
        self.L_factor = (P_D * ELECTRON_CHARGE) / (P.sigma_nm * P.sigma_nm * P.kBT * EPSILON_0)

        self.rhobL = P.rhobL if P.LG_exists else np.zeros(Ns)
        self.rhobG = P.rhobG if P.LG_exists else np.zeros(Ns)

        self._init_Vext()
        self._init_rho()

        # bulk chemical potentials
        self.mu = fmt.mu_HS(P.rhob)
        if P.LJ_exists:
            lj.add_mu_LJ(self.mu, P.eps, P.rhob)
        self.eps_r = self._dielectric_constant()
        self.c1_bulk = -self.mu + np.log(P.rhob)

        self.outfname = os.path.join(data_dir, io.create_outfname(P))
        found, cnt = io.read_restart(self.outfname, self.rho, P.N)
        if verbose:
            print("ouput file found,reading from it" if found else "no ouput file found")
        if cnt is not None:
            self.count_iter = cnt * P.Nbatch

    # -- initialisation ----------------------------------------------------
    def _init_Vext(self):
        P = self.P
        V = np.empty((P.Nspecies, P.N))
        x = P.dx * np.arange(P.NiR, P.N)
        for j in range(P.Nspecies):
            V[j, P.NiR:] = P.ew[j] * (2. / 15 * x ** -9 - x ** -3)
        V[:, :P.NiR] = 1000.
        self.Vext = V

    def _init_rho(self):
        P = self.P
        rho = self.rho
        rho[:, :P.NiR] = 0.
        if not P.LG_exists:
            rho[:, P.NiR:] = P.rhob[:, None] * np.exp(-self.Vext[:, P.NiR:])
        else:
            hN = int(P.h_target / P.dx)
            rho[:, P.NiR:hN + P.NiR] = P.rhobL[:, None]
            rho[:, hN:] = P.rhobG[:, None]

    def _dielectric_constant(self):
        P = self.P
        if P.Is_polar:
            s3 = P.sigma_nm ** 3
            return 1. + (P.rhob[0] * P.p * P.p) / 3. * ((P_D * P_D) / (s3 * EPSILON_0 * P.kBT))
        return 1.

    # -- one Picard step ---------------------------------------------------
    def _rfft(self, x, out):
        if self.workers > 1:
            out[...] = sfft.rfft(x, axis=-1, workers=self.workers)
            return out
        return np.fft.rfft(x, axis=-1, out=out)

    def _irfft(self, X, out):
        if self.workers > 1:
            out[...] = sfft.irfft(X, n=self.P.N, axis=-1, workers=self.workers)
            return out
        return np.fft.irfft(X, n=self.P.N, axis=-1, out=out)

    def iterate(self):
        P = self.P
        rho = self.rho
        c1 = self.c1

        # FMT weighted densities n_a = rho_tot (*) omega_a
        rhocpy(rho, P.NiR, self.rhocopy)
        self._rfft(self.rhocopy, self.Rh)
        fmt.weight_products(self.Rh, self.W, self.nhat)
        self._irfft(self.nhat, self.n)
        fmt.fmt_dphidn(self.n, self.dphi)
        self._rfft(self.dphi, self.D)

        # FMT + LJ contributions to c1, assembled in Fourier space
        if P.LJ_exists:
            truncate_rho(rho, P.NiLJ, self.rhotrunc)
            self._rfft(self.rhotrunc, self.RH)
        c1_hat(self.D, self.W, self.Uhat, P.eps, self.RH, P.LJ_exists, self.c1h)
        self._irfft(self.c1h, c1)

        if P.ES_exists:
            es.add_c1_ES(rho, c1, self.q, P.lambdaB, P.dx, P.N_ES, P.Vq_L, P.Vq_R,
                         self.bc, P.Is_polar, self.Pol, self.E_exec, self.phi, self.psi,
                         self.c1_factor, self.P_factor, P.p, P.NiR, P.iend, self.thomas)

        picard_update(rho, self.rhonew, c1, self.Vext, P.rhob, self.c1_bulk, self.mu,
                      P.iend, P.NiR, P.alpha, P.dx, P.LG_exists, self.rhobL, self.rhobG,
                      P.h_target, P.Is_polar, self.Pol, self.Pnew, self.E_exec, P.p,
                      self.L_factor, self.dev)
        self.count_iter += 1

    # -- driver ------------------------------------------------------------
    def write(self, elapsed):
        io.write_rho(self, elapsed)

    def run(self, max_batches=None, tol=None, verbose=True):
        """Iterate in batches of Nbatch, writing the profile after every batch.

        Runs forever (like the C code) unless ``max_batches`` or ``tol``
        (stop when max(dev) < tol) is given.
        """
        P = self.P
        start = time.perf_counter()
        b = 0
        while max_batches is None or b < max_batches:
            for _ in range(P.Nbatch):
                self.iterate()
            elapsed = time.perf_counter() - start
            self.write(elapsed)
            maxdev = self.dev.max()
            if verbose:
                print("%d x %d batches; time: %f s; max dev: %f"
                      % (self.count_iter // P.Nbatch, P.Nbatch, elapsed, maxdev), flush=True)
            b += 1
            if tol is not None and maxdev < tol:
                break
