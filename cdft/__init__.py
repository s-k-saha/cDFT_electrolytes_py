"""Classical DFT solver for planar (1D) multi-component fluids and electrolytes."""
from .solver import System
from .params import read_params

__all__ = ["System", "read_params"]
