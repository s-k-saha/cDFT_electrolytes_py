# Classical DFT solver (Python)

Python/numba port of the C solver (`../codebase_c`, with the electrostatics
and polar-solvent part taken from `../codebase_c_2`). It computes
equilibrium density profiles of **n interacting species** in a planar (1D)
geometry with:

1. Rosenfeld FMT for hard spheres
2. Lennard-Jones attraction (WCA split, `r_c = 2.5σ`, mean-field/RPA)
3. Mean-field electrostatics: Poisson equation psi'' = -(rho_q - P')/eps0 with
   DD/DN/ND boundary conditions, psi in volts, optionally with a polar
   (Langevin-dipole) solvent as species 0
4. Optional liquid–gas coexistence at fixed adsorption (`h_target`)

See `../codebase_c/theory.pdf` for the theory, and `../codebase_c/README.md`
for a description of all parameters. The parameter files have the same format.

## Requirements

```bash
pip install -r requirements.txt     # numpy>=2.0, scipy, numba
```

## Running

```bash
python run.py                                # uses ./data/params_*.txt, runs forever (like the C code)
python run.py --data-dir /path/to/case       # params read from and output written to that dir
python run.py --max-batches 500              # stop after 500 x Nbatch iterations
python run.py --tol 1e-8                     # stop when max(dev) < 1e-8
python run.py --workers 4                    # multithreaded FFTs (only helps for N >~ 50000)
python run.py --smooth-P                     # 3-point smoothing of P before it enters Poisson (smooth_P_flag in C)
```

The first run compiles the numba kernels, which takes a few seconds. They are
cached in `cdft/__pycache__`, so later runs start in about 1 s.

From Python:

```python
from cdft import System
s = System("data")
s.run(max_batches=100)
s.rho, s.c1, s.mu        # (Nspecies, N) profiles, bulk chemical potentials
```

## Electrostatics (as in `codebase_c_2`)

* Extra parameter in `params_system.txt`: `alpha_ES` (default 0.01), the
  mixing parameter of the potential psi (and of the field E when polar).
* Stability: the Picard iteration for the slowest electrostatic mode of the box
  is stable only if `alpha * alpha_ES * (L / (pi*lambda_D))^2 < ~4`, where
  lambda_D is the bare (vacuum) Debye length. This limit gets stricter as L^2,
  so a 10x longer box needs a ~100x smaller `alpha * alpha_ES`. The code checks
  this at startup and prints the largest safe `alpha_ES`.
* `Vq_L`, `Vq_R` are in volts (Dirichlet) or V/sigma (Neumann). `NN` is not
  supported because the Poisson problem is singular.
* psi, phi, P and E live on the grid points `NiR .. iend-1`.
* Coupling: `c1_i -= q_i e psi / kBT`. With a polar solvent, P = p rho_0 L(p p_d E/kBT)
  and the orientational ideal term is added to species 0.
* Initial guess: linearised Debye-Hueckel profiles for the ions (species >= 1)
  and for psi. The solvent (species 0) starts from Boltzmann in the wall
  potential.

## Output

The output file name and format are the same as `codebase_c_2`'s
(`rho1Ddx...dat` with `rhob` printed to 12 decimals; columns
`i x rho_1 c1_1 ... [psi rho_q] [P E]`; footer with concentrations, Debye
length, mu, eps, eps_r and dev). Restart files are interchangeable with
`codebase_c_2`: if the output file exists, rho (and psi, P and E) are read from it and
the iteration continues.

The footer has two extra last lines. The first is the free-energy density of
the bulk fluid at `rhob`, in kBT/sigma^3, computed once at startup
(`free_energy.bulk_free_energy`):

```text
bulk free energy density (kBT/sigma^3): f_id :...; f_hs :...; [f_LJ :...;] f_tot :...; f_omega :...;
```

with `f_hs` the Percus-Yevick (= bulk Rosenfeld) excess, `f_LJ` =
-alphaLJ/2 sum_ij eps_ij rhob_i rhob_j and `f_omega` = f_tot - sum_j mu_j rhob_j
= -pressure. Electrostatics and dipole orientation do not contribute in the
neutral, field-free bulk.

The second line has the free-energy contributions of the
current profiles (`cdft/free_energy.py`), recomputed at every write (every
`Nbatch` iterations), as beta F/A in kBT/sigma^2 integrated over grid points
`0 .. iend-1`:

```text
free energy (kBT/sigma^2): F_id :...; [F_orient :...;] F_hs :...; [F_LJ :...;] [F_ES :...;] F_ext :...; F_tot :...; F_Omega :...;
```

`id` = sum_j int rho_j(ln rho_j - 1), `orient` = orientational entropy of the
Langevin dipoles (polar only), `hs` = Rosenfeld FMT, `LJ` = mean-field
attraction, `ES` = max_psi [int rho_eff psi - eps0/2 int psi'^2] (+ Neumann
boundary terms) with rho_eff = sum_j q_j e rho_j - P', `ext` = sum_j int rho_j Vext_j,
`tot` = their sum and `Omega` = F_tot - sum_j mu_j N_j. Each term is the
functional whose derivative the solver uses in c1.

## Layout

```text
codebase_py/
├── run.py              # command-line entry point (replaces main.c)
├── plot.py             # plot two columns of an output file (python plot.py --list)
├── cdft/
│   ├── constants.py    # globals.c constants
│   ├── params.py       # read_params_geometry / read_params_system
│   ├── fmt.py          # functional_fmt.c
│   ├── lj.py           # functional_LJ.c
│   ├── es.py           # functional_ES.c
│   ├── utils.py        # utils.c (Poisson solvers, Langevin functions)
│   ├── solver.py       # init.c + iterator.c (System class, Picard iteration)
│   ├── free_energy.py  # free-energy contributions written in the footer
│   └── io.py           # output file name, writer, restart reader
├── data/               # params_geometry.txt, params_system.txt, outputs
└── tests/              # python -m pytest tests
```

## Performance

All point-wise loops are numba kernels. The convolutions are linear, so they
are combined in Fourier space. One iteration needs `13 + 2*Nspecies` FFTs,
where the C code needs `24 + 2*Nspecies²` and creates an FFTW plan for each.
FFTs are batched and write into preallocated buffers. Profiles are written by
a numba formatter that produces exactly the same digits as `printf`.

Measured on the same machine (C compiled with `-O3 -march=native`, FFTW3):

| case | C | Python |
|---|---|---|
| shipped params (3 species, LJ, LG, N=10⁴), 1000 iterations + 50 writes | 2.81 s | 0.92 s |
| 2-species electrolyte, DD, N=10⁵, 400 iterations + 20 writes | 11.97 s | 3.81 s (3.39 s with `--workers 8`) |

The Python times include about 0.2 s of one-time loading of the cached kernels.

## Validation against the C code

The FMT, LJ and liquid-gas parts were checked against `codebase_c`, and the
electrostatics against `codebase_c_2`, by running the codes side by side:

* non-polar ES with DD, DN and ND, polar with ND, and a restart from a
  C_2-written file: agree within 1 in the last printed (12th) decimal;
* polar with DD and DN (with and without `--smooth-P`): densities agree to
  1e-12. P and E differ at the ~1e-8 to 1e-6 level of their scale. C_2
  differs from itself by the same amount when only its floating-point
  rounding changes (`-mno-fma`). E is a numerical derivative of an
  ill-conditioned Poisson solution (condition number ~(N/pi)^2), so
  round-off is amplified. This also sets the floor (~1e-10) at which the P
  residual `dev[Nspecies]` stops decreasing.

Speed: on the polar 3-species case (N=10^4, 2000 iterations) C_2 takes 5.9 s
and Python 2.5 s.

## Differences from the C code

* Input and output live in `--data-dir` (default `./data`) instead of the path
  `../data` relative to the executable.
* `--max-batches` / `--tol` stopping criteria were added. Without them the run
  is endless, as in C. The run also stops if dev becomes NaN.
* The reported `time` is wall-clock time. C used CPU time (`clock()`).
* A missing `T=` line means T = 300 K. In C it left `kBT = 0`.
* `init_psi` in C_2 runs while the `q=` line is being parsed, before `Vq_L`
  is read, and mixes units (x in sigma, Debye length in m). So C_2
  effectively starts from psi = 0. Here psi starts from the intended
  Debye-Hueckel profile, which changes only the initial guess.
* The Debye length uses |q_1|; C_2 uses q_1, which breaks the initial guess
  if species 1 is an anion.
* Polar + liquid-gas: C_2 updates `P[i]` instead of `P[i-NiR]` in that branch
  (out-of-bounds write); fixed here.
* FMT c1: the back-convolution with the vector weights n1v, n2v uses the
  mirrored weights (conj(W) in Fourier space). The C code uses W, which flips
  the sign of the vector terms and makes c1 inconsistent with the Rosenfeld
  functional near walls (bulk results are unaffected). Profiles near a wall
  therefore differ from the C code.
* With DN/ND, the Neumann end of psi is kept at psi' = Vq after mixing; C_2
  overwrote it with the field value Vq. The boundary field E there is
  -Vq/sigma (V/m); C_2 wrote -Vq (V/sigma).
* Polar restart: rho_0 at the wall is printed as 0, which made C_2's
  orientational term divide 0 by 0. It is skipped where rho_0 = 0, and P and E
  are read back from the file as well.
* Without charges, C_2 crashes when writing the Debye length. Here that footer
  line is simply omitted. `Is_polar=1` without a `q=` line, or BC = NN or an
  unknown BC, stops with an error.
