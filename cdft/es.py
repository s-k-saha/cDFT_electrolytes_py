"""Mean-field electrostatics and polar-solvent coupling (functional_ES.c)."""
import math

from numba import njit

from .utils import poisson_1D, psi_calculator, psi_calculator_P, inv_langevin


@njit(cache=True, nogil=True)
def add_c1_ES(rho, c1, q, lambdaB, dx, N_ES, Vq_L, Vq_R, bc, is_polar,
              P, E_exec, phi, psi, c1_factor, P_factor, p, NiR, iend, thomas):
    """Solve Poisson's equation for the current densities and add -q_i psi to c1."""
    Ns = rho.shape[0]
    for i in range(N_ES):
        phi[i] = 0.
        psi[i] = 0.
    if is_polar:
        for i in range(E_exec.shape[0]):
            E_exec[i] = 0.

    for i in range(N_ES):
        for j in range(Ns):
            phi[i] += q[j] * lambdaB * rho[j, i]   # free charge density

    if bc != 3:
        poisson_1D(dx, N_ES, Vq_L, Vq_R, phi, psi, bc, thomas)
    elif is_polar:
        psi_calculator_P(dx, N_ES, Vq_L, Vq_R, phi, psi, E_exec, P, 1.0, P_factor)
    else:
        psi_calculator(dx, N_ES, Vq_L, Vq_R, phi, psi)

    for i in range(Ns):
        for j in range(NiR, iend + 1):
            c1[i, j] -= q[i] * psi[j] * c1_factor

    # ideal contribution to the solvent density due to polarisation coupling
    if is_polar:
        for j in range(NiR, iend + 1):
            temp = inv_langevin(P[j] / (p * rho[0, j]))
            if math.fabs(temp) <= 1.0e-8:
                c1[0, j] -= -math.log(1. + temp * temp / 6.)
            else:
                c1[0, j] -= math.log(temp / math.sinh(temp))
