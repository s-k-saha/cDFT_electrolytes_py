"""Parsing of ``params_geometry.txt`` and ``params_system.txt``.

Mirrors ``read_params_geometry`` / ``read_params_system`` of the C code
(including its quirks: only the first ``[...]`` group of a list is read, the
order in which lines are tested, and ``rhobG`` overriding ``rhob``).
"""
import re
from dataclasses import dataclass

import numpy as np

from .constants import SIGMA_D, ELECTRON_CHARGE, KB, EPSILON_0

_INT = re.compile(r"\s*[+-]?\d+")
_FLOAT = re.compile(r"\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _scan(line, prefix, conv):
    """Emulate ``sscanf(line, "<prefix>%d|%lf", &x) == 1``."""
    if not line.startswith(prefix):
        return None
    m = (_INT if conv is int else _FLOAT).match(line, len(prefix))
    return conv(m.group()) if m else None


def _scan_list(line, n):
    """Read ``n`` numbers following the first ``[`` (strtod semantics)."""
    out = np.zeros(n)
    i = line.find("[")
    if i < 0:
        return out
    pos = i + 1
    for k in range(n):
        m = _FLOAT.match(line, pos)
        if not m:
            break  # strtod fails -> stays at 0 and does not advance
        out[k] = float(m.group())
        pos = m.end()
        while pos < len(line) and line[pos] in " ,":
            pos += 1
    return out


@dataclass
class Params:
    # geometry
    Nspecies: int = 0
    N: int = 0
    R: float = 0.0
    dx: float = 0.0
    alpha: float = 0.0
    Nbatch: int = 0
    Is_polar: int = 0
    # derived geometry
    sigma: float = 0.0
    sigma_nm: float = 1.0
    NiR: int = 0
    K: int = 0
    Lx: float = 0.0
    iend: int = 0
    psi_factor: float = 0.0
    E_factor: float = 0.0
    # system
    T: float = 300.0
    kBT: float = KB * 300.0
    ew: np.ndarray = None
    rhob: np.ndarray = None
    rhobL: np.ndarray = None
    rhobG: np.ndarray = None
    eps: np.ndarray = None      # (Nspecies, Nspecies)
    q: np.ndarray = None
    BC: str = ""
    lambdaB: float = 0.0
    Vq_L: float = 0.0           # already divided by psi_factor
    Vq_R: float = 0.0           # already divided by psi_factor
    p: float = 0.0
    h_target: float = 0.0
    LJ_exists: bool = False
    ES_exists: bool = False
    LG_exists: bool = False
    # LJ geometry
    rc: float = 0.0
    rmin: float = 0.0
    NiLJ: int = 0
    # ES grid size (iend+1 at the moment the q line was read)
    N_ES: int = 0


def read_params_geometry(path, P=None):
    P = P or Params()
    with open(path) as f:
        for line in f:
            for name, conv in (("Nspecies", int), ("N", int), ("R", float),
                               ("dx", float), ("alpha", float),
                               ("Nbatch", int), ("Is_polar", int)):
                v = _scan(line, name + "=", conv)
                if v is not None:
                    setattr(P, name, v)
                    break
    P.sigma = 2 * P.R
    P.sigma_nm = P.sigma * SIGMA_D
    P.NiR = int(P.R / P.dx)
    P.K = P.N // 2 + 1
    P.Lx = P.dx * P.N
    P.iend = P.N - 3 * P.NiR
    P.psi_factor = ELECTRON_CHARGE / (EPSILON_0 * P.sigma_nm)
    P.E_factor = P.psi_factor / P.sigma_nm
    return P


def read_params_system(path, P):
    ns = P.Nspecies
    P.ew = np.zeros(ns)
    P.rhob = np.zeros(ns)
    with open(path) as f:
        for line in f:
            v = _scan(line, "T=", float)
            if v is not None:
                P.T = v
                P.kBT = KB * v
                continue
            if line.startswith("eps"):
                P.LJ_exists = True
                P.rc = 2.5 * P.sigma
                P.rmin = 2.0 ** (1.0 / 6) * P.sigma
                P.NiLJ = int(P.rc / P.dx)
                P.iend = P.N - 3 * P.NiLJ
                P.eps = _scan_list(line, ns * ns).reshape(ns, ns)
                continue
            v = _scan(line, "h_target=", float)
            if v is not None:
                P.h_target = v
                continue
            if line.startswith("rhobL"):
                P.LG_exists = True
                P.rhobL = _scan_list(line, ns)
                P.rhobG = np.zeros(ns)
                continue
            if line.startswith("rhobG"):
                P.rhobG = _scan_list(line, ns)
                P.rhob = P.rhobG.copy()  # right end fixed at bulk gas density
                continue
            if line.startswith("q"):
                P.ES_exists = True
                P.lambdaB = 0.0
                P.N_ES = P.iend + 1
                P.q = _scan_list(line, ns)
                continue
            if line.startswith("BC="):
                tok = line[3:].split()
                if tok:
                    P.BC = tok[0]
                    print("BC=%s" % P.BC)
                    continue
            v = _scan(line, "lambdaB=", float)
            if v is not None:
                P.lambdaB = v
                continue
            v = _scan(line, "Vq_L=", float)
            if v is not None:
                P.Vq_L = v / P.psi_factor
                continue
            v = _scan(line, "Vq_R=", float)
            if v is not None:
                P.Vq_R = v / P.psi_factor
                continue
            v = _scan(line, "p=", float)
            if v is not None:
                P.p = v
                continue
            if line.startswith("ew"):
                P.ew = _scan_list(line, ns)
                continue
            if line.startswith("rhob"):
                P.rhob = _scan_list(line, ns)
                continue
    if P.LJ_exists is False:
        P.eps = np.zeros((ns, ns))
    if P.Is_polar and not P.ES_exists:
        raise ValueError("Is_polar=1 requires charges (a 'q=' line) in params_system.txt")
    if P.ES_exists and P.BC not in ("DD", "DN", "ND", "NN"):
        raise ValueError("unknown BC type %r" % P.BC)
    return P


def read_params(data_dir):
    import os
    P = read_params_geometry(os.path.join(data_dir, "params_geometry.txt"))
    read_params_system(os.path.join(data_dir, "params_system.txt"), P)
    return P
