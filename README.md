# Classical DFT solver (Python)

Python/numba port of the C solver in `../codebase_c`. It computes
equilibrium density profiles of **n interacting species** in a planar (1D)
geometry with:

1. Rosenfeld FMT for hard spheres
2. Lennard-Jones attraction (WCA split, `r_c = 2.5σ`, mean-field/RPA)
3. Mean-field electrostatics (Poisson equation, DD/DN/ND/NN boundary conditions),
   optionally with a polar (Langevin-dipole) solvent
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

## Output

The output file name and format are **the same as the C code's**
(`rho1Ddx...dat`, columns `i x rho_1 c1_1 ... [psi phi] [P E]`, same footer).
Restart files are interchangeable: if the output file exists, the profiles are
read from it and the iteration continues. This works for files written by
either implementation.

## Layout

```text
codebase_py/
├── run.py              # command-line entry point (replaces main.c)
├── cdft/
│   ├── constants.py    # globals.c constants
│   ├── params.py       # read_params_geometry / read_params_system
│   ├── fmt.py          # functional_fmt.c
│   ├── lj.py           # functional_LJ.c
│   ├── es.py           # functional_ES.c
│   ├── utils.py        # utils.c (Poisson solvers, Langevin functions)
│   ├── solver.py       # init.c + iterator.c (System class, Picard iteration)
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

The two codes were run side by side on these cases: the shipped parameters
(LJ + LG), ES with DD/DN/ND/NN boundary conditions, a `q` line read before the
`eps` line, a polar solvent with NN and with DD, a single hard-sphere species,
and a restart from a C-written file. In every case the output files agree to
within 1 in the last printed (12th) decimal. The remaining differences are
floating-point round-off from summing the convolutions in a different order.

## Differences from the C code

* Input and output live in `--data-dir` (default `./data`) instead of the path
  `../data` relative to the executable.
* `--max-batches` / `--tol` stopping criteria were added. Without them the run
  is endless, as in C.
* The reported `time` is wall-clock time. C used CPU time (`clock()`).
* A missing `T=` line means T = 300 K. In C it left `kBT = 0`, which gives
  division by zero.
* `Is_polar=1` without a `q=` line, or an unknown `BC`, now stops with an
  error. In C these crashed or silently produced garbage.
