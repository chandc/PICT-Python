"""Import the HydroGym core contract, preferring the real package.

The genuine `hydrogym.core` is used when importable (python >= 3.10 with the
git-main package). Otherwise the vendored copy of the same file serves as the
development target -- the class objects differ but the CONTRACT is identical,
and everything here subclasses whichever is loaded. `HYDROGYM_SOURCE` records
which one won, so gates can report it.
"""
try:
    from hydrogym.core import (ActuatorBase, CallbackBase, FlowEnv, PDEBase,
                               TransientSolver)
    HYDROGYM_SOURCE = "hydrogym (installed package)"
except Exception:                        # ImportError, or the stale PyPI
    from hydrogym_pict._vendored_core import (ActuatorBase, CallbackBase,     # noqa: F401
                                              FlowEnv, PDEBase,
                                              TransientSolver)
    HYDROGYM_SOURCE = "vendored core.py (dynamicslab/hydrogym main, 2026-09-14)"
