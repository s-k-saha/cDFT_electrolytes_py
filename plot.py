#!/usr/bin/env python3
"""Plot one column of an output file (rho1Ddx...dat) against another.

Set DATAFILE, XCOL and YCOL below (or override them on the command line:
python plot.py [datafile] [xcol] [ycol] [--out fig.png]). A column is given
by its index or its name; run with --list to print the names of a file:

    0 i   1 x   2 rho_1  3 c1_1  ...  rho_n c1_n  [psi rho_q]  [P E]
"""
import argparse
import itertools
import re

import numpy as np
import matplotlib.pyplot as plt

DATAFILE = "data/rho1Ddx0.010000L100.000000rhob1_0.614000148080rhob2_0.000025000000rhob3_0.000025000000ew1_1.200000ew2_0.000000ew3_0.000000lambdaB1.000000Vq_L0.100000Vq_R0.000000BCDDq1.000000_LJ_ES.dat"
XCOL = "x"
YCOL = "rho_1"
OUTFILE = None          # e.g. "profile.png"; None opens a window

LABELS = {"x": r"$x\ [\sigma]$", "psi": r"$\psi\ [\mathrm{V}]$", "rho_q": r"$\rho_q\ [e/\sigma^3]$",
          "P": r"$P$", "E": r"$E\ [\mathrm{V/m}]$"}


def read_datafile(fname):
    """Return (data, column names); the footer starting at '------' is skipped."""
    with open(fname) as f:
        lines = f.readlines()
    body = itertools.takewhile(lambda l: not l.startswith("-"), lines)
    data = np.loadtxt(body, ndmin=2)
    ns = len(re.findall(r"conc\. of species", "".join(lines)))
    names = ["i", "x"]
    for j in range(1, ns + 1):
        names += ["rho_%d" % j, "c1_%d" % j]
    extra = data.shape[1] - len(names)
    names += ["psi", "rho_q", "P", "E"][:extra]
    names += ["col%d" % k for k in range(len(names), data.shape[1])]
    return data, names


def column(spec, names):
    if isinstance(spec, int) or spec.lstrip("-").isdigit():
        return int(spec)
    if spec not in names:
        raise SystemExit("unknown column %r; available: %s" % (spec, " ".join(names)))
    return names.index(spec)


def label(name):
    if name in LABELS:
        return LABELS[name]
    m = re.fullmatch(r"(rho|c1)_(\d+)", name)
    if m:
        return (r"$\rho_{%s}\ [\sigma^{-3}]$" if m.group(1) == "rho" else r"$c^{(1)}_{%s}$") % m.group(2)
    return name


def main():
    ap = argparse.ArgumentParser(description="plot two columns of a cDFT output file")
    ap.add_argument("datafile", nargs="?", default=DATAFILE)
    ap.add_argument("xcol", nargs="?", default=XCOL, help="column index or name (default: %(default)s)")
    ap.add_argument("ycol", nargs="?", default=YCOL, help="column index or name (default: %(default)s)")
    ap.add_argument("--out", default=OUTFILE, help="save the figure instead of showing it")
    ap.add_argument("--list", action="store_true", help="print the column names and exit")
    args = ap.parse_args()

    data, names = read_datafile(args.datafile)
    if args.list:
        print("".join("%d: %s\n" % kv for kv in enumerate(names)))
        return
    ix, iy = column(args.xcol, names), column(args.ycol, names)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data[:, ix], data[:, iy], lw=1.2)
    ax.set_xlabel(label(names[ix]))
    ax.set_ylabel(label(names[iy]))
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=200)
        print("saved", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
