#!/usr/bin/env python3
"""Entry point: python run.py [--data-dir data] [--max-batches B] [--tol T]"""
import argparse
import os

from cdft import System


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="1D classical DFT solver (FMT + LJ + ES)")
    ap.add_argument("--data-dir", default=os.path.join(here, "data"),
                    help="directory with params_geometry.txt / params_system.txt; "
                         "output is written here too (default: ./data)")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="stop after this many batches of Nbatch iterations (default: run forever)")
    ap.add_argument("--tol", type=float, default=None,
                    help="stop once max(dev) drops below this value")
    ap.add_argument("--workers", type=int, default=1, help="threads used by each FFT call")
    ap.add_argument("--smooth-P", action="store_true",
                    help="3-point smooth the polarisation before it enters Poisson's equation")
    args = ap.parse_args()

    System(args.data_dir, workers=args.workers, smooth_P=args.smooth_P).run(max_batches=args.max_batches, tol=args.tol)


if __name__ == "__main__":
    main()
