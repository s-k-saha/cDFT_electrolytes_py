"""Classical DFT solver for planar (1D) multi-component fluids and electrolytes."""
import glob as _glob
import hashlib as _hashlib
import os as _os


def _invalidate_stale_numba_cache():
    """numba's cache only tracks the file of the cached function itself, so a
    kernel calling an edited function from another module would silently keep
    the old machine code. Drop all cached kernels whenever any source changes."""
    here = _os.path.dirname(_os.path.abspath(__file__))
    h = _hashlib.sha1()
    for f in sorted(_glob.glob(_os.path.join(here, "*.py"))):
        with open(f, "rb") as fh:
            h.update(fh.read())
    digest = h.hexdigest()
    cache = _os.path.join(here, "__pycache__")
    stamp = _os.path.join(cache, "numba_src.sha1")
    try:
        with open(stamp) as fh:
            if fh.read() == digest:
                return
    except OSError:
        pass
    try:
        for f in _glob.glob(_os.path.join(cache, "*.nbi")) + _glob.glob(_os.path.join(cache, "*.nbc")):
            _os.remove(f)
        _os.makedirs(cache, exist_ok=True)
        with open(stamp, "w") as fh:
            fh.write(digest)
    except OSError:
        pass  # read-only install: numba will simply not cache


_invalidate_stale_numba_cache()

from .solver import System          # noqa: E402
from .params import read_params     # noqa: E402

__all__ = ["System", "read_params"]
