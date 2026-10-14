"""Physical and numerical constants (identical to globals.c)."""

PI = 3.14159265358979  # the C code uses this truncated value; kept for parity

# SI constants
SIGMA_D = 0.275e-9          # water average diameter
ELECTRON_CHARGE = 1.6e-19
KB = 1.380649e-23
EPSILON_0 = 8.854e-12
P_D = 1.e-30                # dipole moment unit

# LJ mean-field bulk coefficient (alphaLJ in functional_LJ.c)
ALPHA_LJ = 1.171861897 * (4. * PI)
