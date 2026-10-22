"""Output file naming, writing and restart reading (init.c / iterator.c).

The file format is byte-for-byte the one of the C code, so profiles produced
by either implementation can be used to restart the other.
"""
import math
import os
import re

import numpy as np
from numba import njit

from .es import debye_length

# ---------------------------------------------------------------------------
# Fast, exactly-rounded "%.<d>f" formatting (identical output to printf).
#
# x*10^d is evaluated exactly as p + e with an error-free (Dekker) product,
# then rounded half-to-even, which is what glibc's printf does with the exact
# binary value.  Valid while |x*10^d| < 2^52; otherwise the caller falls back
# to Python's formatter.
# ---------------------------------------------------------------------------
_LIMIT = 2.0 ** 52


@njit(cache=True, inline="always")
def _two_prod(a, b):
    p = a * b
    c = 134217729.0 * a
    ah = c - (c - a)
    al = a - ah
    c = 134217729.0 * b
    bh = c - (c - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@njit(cache=True)
def _put_uint(buf, pos, v, width):
    """Write non-negative int v, zero-padded to at least `width` digits."""
    n = 1
    t = v // 10
    while t > 0:
        n += 1
        t //= 10
    if n < width:
        n = width
    for k in range(n - 1, -1, -1):
        buf[pos + k] = 48 + v % 10
        v //= 10
    return pos + n


@njit(cache=True)
def _put_fixed(buf, pos, x, ndec, scale, ipow):
    """Append "%.<ndec>f" % x. Returns new pos, or -1 if x is out of range."""
    if not math.isfinite(x):
        return -1
    neg = x < 0.0 or (x == 0.0 and math.copysign(1.0, x) < 0.0)
    ax = math.fabs(x)
    p, e = _two_prod(ax, scale)
    if p >= _LIMIT:
        return -1
    r = math.floor(p)
    g = (p - r) - 0.5
    t = g + e
    ri = np.int64(r)
    if t > 0.0 or (t == 0.0 and ri % 2 == 1):
        ri += 1
    if neg:
        buf[pos] = 45  # '-'
        pos += 1
    pos = _put_uint(buf, pos, ri // ipow, 1)
    buf[pos] = 46  # '.'
    pos += 1
    return _put_uint(buf, pos, ri % ipow, ndec)


@njit(cache=True)
def _format_table(vals, decs):
    """Rows "<i> <v0> <v1> ... \n" with v_c printed as "%.<decs[c]>f"; returns (bytes, ok)."""
    N, ncol = vals.shape
    buf = np.empty(N * (ncol * 26 + 32), np.uint8)
    pos = 0
    for i in range(N):
        pos = _put_uint(buf, pos, i, 1)
        for c in range(ncol):
            buf[pos] = 32
            pos += 1
            d = decs[c]
            pos = _put_fixed(buf, pos, vals[i, c], d, 10.0 ** d, np.int64(10) ** d)
            if pos < 0:
                return buf[:0], False
        buf[pos] = 32
        buf[pos + 1] = 10
        pos += 2
    return buf[:pos], True


def create_outfname(P):
    s = "rho1Ddx%fL%f" % (P.dx, P.Lx)
    for i in range(P.Nspecies):
        s += "rhob%d_%.12f" % (i + 1, P.rhob[i])
    for i in range(P.Nspecies):
        s += "ew%d_%f" % (i + 1, P.ew[i])
    if P.ES_exists:
        q1 = P.q[1] if P.Nspecies > 1 else P.q[0]
        s += "lambdaB%fVq_L%fVq_R%fBC%sq%f" % (P.lambdaB, P.Vq_L, P.Vq_R, P.BC, q1)
    if P.LG_exists:
        s += "h_target%f" % P.h_target
    if P.Is_polar:
        s += "p%f" % P.p
    if P.LJ_exists:
        s += "_LJ"
    if P.ES_exists:
        s += "_ES"
    if P.Is_polar:
        s += "_polar"
    if P.LG_exists:
        s += "_LG"
    return s + ".dat"


_CYCLES = re.compile(r"(-?\d+) x (-?\d+) cycles time:\s*\S+ s")


def read_restart(fname, rho, N, psi=None, NiR=0, iend=0):
    """Load rho (and psi, if given) from an existing output file.

    Returns (found, ncycles or None).
    """
    if not os.path.exists(fname):
        return False, None
    Ns = rho.shape[0]
    ncycles = None
    with open(fname) as f:
        lines = iter(f)
        i = 0
        for line in lines:
            if i >= N - 1:
                break
            if not line[:1].isdigit():
                continue
            tok = line.split()
            i = int(tok[0])
            for s in range(Ns):
                rho[s, i] = float(tok[2 + 2 * s])
            if psi is not None and NiR <= i < iend and len(tok) > 2 + 2 * Ns:
                psi[i - NiR] = float(tok[2 + 2 * Ns])
        for line in lines:
            m = _CYCLES.match(line)
            if m:
                ncycles = int(m.group(2))
    return True, ncycles


def write_rho(sys, elapsed):
    P = sys.P
    N, Ns, NiR, iend = P.N, P.Nspecies, P.NiR, P.iend
    cols = [P.dx * np.arange(N)]
    decs = [6]
    for j in range(Ns):
        cols += [sys.rho[j], sys.c1[j]]
        decs += [12, 6]
    if P.ES_exists:
        psi = np.full(N, P.Vq_R)
        psi[:NiR] = P.Vq_L
        psi[NiR:iend] = sys.psi[:iend - NiR]
        rhoq = np.zeros(N)  # charge density sum_j q_j rho_j
        rhoq[NiR:iend] = sys.phi[:iend - NiR] / sys.phi_factor
        cols += [psi, rhoq]
        decs += [12, 12]
    if P.Is_polar:
        Pc = np.zeros(N)
        Ec = np.zeros(N)
        Pc[NiR:iend] = sys.Pol[:iend - NiR]
        Ec[NiR:iend] = sys.E_exec[:iend - NiR]
        cols += [Pc, Ec]
        decs += [12, 12]

    flat = np.empty((N, len(cols)))
    for c, col in enumerate(cols):
        flat[:, c] = col
    data, ok = _format_table(flat, np.array(decs, dtype=np.int64))
    if ok:
        body = data.tobytes()
    else:  # values too large / non-finite for the fast path
        fmt = "%d " + "".join("%%.%df " % d for d in decs) + "\n"
        vals = np.column_stack([np.arange(N), flat]).ravel().tolist()
        for r in range(N):
            vals[r * (len(cols) + 1)] = r
        body = ((fmt * N) % tuple(vals)).encode()

    fact = 1. / (6.022e23 * P.sigma_nm ** 3 * 1000)
    footer = ["------------------\n%d x %d cycles time: %f s\n"
              % (P.Nbatch, sys.count_iter // P.Nbatch, elapsed)]
    footer.append("".join("conc. of species %d :%.10f M; " % (i + 1, P.rhob[i] * fact)
                          for i in range(Ns)) + "\n")
    if P.ES_exists:
        footer.append("Debye length:%.10f nm\n" % (debye_length(P) * 1e9))
    footer.append("".join("mu[%d] :%f; " % (i, sys.mu[i]) for i in range(Ns)) + "\n")
    if P.LG_exists:
        footer.append("h_target :%f; \n" % P.h_target)
    if P.LJ_exists:
        footer.append("".join("eps[%d][%d] :%f; " % (i, j, P.eps[i, j])
                              for i in range(Ns) for j in range(i, Ns)) + "\n")
    footer.append("eps_r:%f\n" % sys.eps_r)
    footer.append("".join("dev[%d] :%.12f; " % (i, sys.dev[i]) for i in range(len(sys.dev))) + "\n")
    if sys.F is not None:
        footer.append("free energy (kBT/sigma^2): "
                      + "".join("F_%s :%.12f; " % kv for kv in sys.F.items()) + "\n")

    tmp = sys.outfname + ".tmp"
    with open(tmp, "wb") as f:
        f.write(body)
        f.write("".join(footer).encode())
    os.replace(tmp, sys.outfname)  # atomic: the old file survives a crash
