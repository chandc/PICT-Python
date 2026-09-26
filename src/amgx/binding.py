"""
ctypes binding to NVIDIA AmgX, used as a COMPLETE solver for the pressure Poisson system.

NOT a preconditioner handed to SciPy's CG. The measured 28 ms/step comes from AmgX running its
own AMG-preconditioned CG entirely on the device; feeding one V-cycle at a time back into a host
Krylov loop would pay the round trip 53 times per solve and discard AmgX's own iteration control.
So this replaces the solve, and `linear_solve()` in src/linsolve.py is the dispatch point.

REUSE IS THE WHOLE POINT. The pressure matrix keeps identical sparsity every step -- the grid
does not move -- and only its values drift, ~1e-3 relative per step, as Gamma = J/rowsum(A)
follows the flow. So the hierarchy is built ONCE and thereafter only the coefficients are
replaced: measured 53 iterations per step either way, with setup falling from 43 ms to 0.5 ms.
Drift accumulates, so `drift_tol` triggers a rebuild -- the same policy the spilu cache in
piso_numpy_3d already uses.

Requires libamgxsh.so; import fails cleanly on any machine without it, and src/precond.py falls
back to Jacobi rather than refusing to run.
"""
import ctypes
import os

import numpy as np

AMGX_MODE_dDDI = 8193          # device, double vec, double mat, int index
AMGX_RC_OK = 0

_LIB_ENV = "AMGX_LIB"          # explicit path wins
_LIB_CANDIDATES = ("libamgxsh.so", "/opt/amgx/lib/libamgxsh.so",
                   "/tmp/AMGX/build/libamgxsh.so")
_CFG_ENV = "AMGX_CONFIG"


def _load():
    path = os.environ.get(_LIB_ENV)
    errs = []
    for cand in ([path] if path else []) + list(_LIB_CANDIDATES):
        if not cand:
            continue
        try:
            return ctypes.CDLL(cand)
        except OSError as e:
            # Report the REAL dlopen error. "not found" is usually wrong: the file is
            # normally present and it is a missing CUDA runtime dependency that fails,
            # which a bare "not found" hides and sends you looking in the wrong place.
            errs.append(f"{cand}: {e}")
    raise ImportError(
        "could not load libamgxsh.so. Set AMGX_LIB, and note the library needs the CUDA "
        "runtime present in the same container. Tried:\n  " + "\n  ".join(errs))


_lib = _load()


def _chk(rc, what):
    if rc != AMGX_RC_OK:
        raise RuntimeError(f"AmgX {what} failed with code {rc}")


# Declare argtypes explicitly. Without them ctypes infers from the Python value, which
# rejects a numpy scalar outright ("Don't know how to convert parameter 2") and, worse, would
# silently truncate a 64-bit pointer passed as an int on some platforms.
_P = ctypes.c_void_p
_I = ctypes.c_int
for _f, _a in (
    ("AMGX_initialize", []),
    ("AMGX_finalize", []),
    ("AMGX_config_create_from_file", [ctypes.POINTER(_P), ctypes.c_char_p]),
    ("AMGX_config_add_parameters", [ctypes.POINTER(_P), ctypes.c_char_p]),
    ("AMGX_config_destroy", [_P]),
    ("AMGX_resources_create_simple", [ctypes.POINTER(_P), _P]),
    ("AMGX_resources_destroy", [_P]),
    ("AMGX_matrix_create", [ctypes.POINTER(_P), _P, _I]),
    ("AMGX_matrix_destroy", [_P]),
    ("AMGX_vector_create", [ctypes.POINTER(_P), _P, _I]),
    ("AMGX_vector_destroy", [_P]),
    ("AMGX_solver_create", [ctypes.POINTER(_P), _P, _I, _P]),
    ("AMGX_solver_destroy", [_P]),
    ("AMGX_matrix_upload_all", [_P, _I, _I, _I, _I, _P, _P, _P, _P]),
    ("AMGX_matrix_replace_coefficients", [_P, _I, _I, _P, _P]),
    ("AMGX_vector_upload", [_P, _I, _I, _P]),
    ("AMGX_vector_download", [_P, _P]),
    ("AMGX_solver_setup", [_P, _P]),
    ("AMGX_solver_solve", [_P, _P, _P]),
    ("AMGX_solver_get_iterations_number", [_P, ctypes.POINTER(_I)]),
):
    _fn = getattr(_lib, _f)
    _fn.argtypes = _a
    _fn.restype = ctypes.c_int

_initialised = False


# AmgX prints per-iteration convergence stats to stdout by default. Inside a 3,000-step run
# that is tens of thousands of lines, it drowns the solver's own output, and the formatting
# itself costs time. Register a no-op sink; AMGX_solver_get_iterations_number still reports
# what we need.
_PRINT_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)
_silent = _PRINT_CB(lambda msg, n: None)


_cfg_shared = None            # kept for the resources bootstrap
_cfg_by_rtol = {}             # one config PER TOLERANCE; resources stay shared
_rsrc_shared = None
_cfg_rtol = None


def reset():
    """Tear the WHOLE AmgX context down so the next solver starts clean.

    Escalation of last resort: on the R11 butterfly grid, momentum solves
    started failing at a deterministic step and stayed failed through fresh
    solver objects -- but the same matrix solved instantly in a fresh
    PROCESS. The durable state is here: the shared resources and config
    handles. Every AmgXSolver must be close()d BEFORE calling this.
    """
    global _initialised, _cfg_shared, _rsrc_shared, _cfg_by_rtol
    for cfg, _path in list(_cfg_by_rtol.values()):
        try:
            _lib.AMGX_config_destroy(cfg)
        except Exception:
            pass
    _cfg_by_rtol = {}
    if _rsrc_shared is not None:
        try:
            _lib.AMGX_resources_destroy(_rsrc_shared)
        except Exception:
            pass
    _cfg_shared = None
    _rsrc_shared = None
    try:
        _lib.AMGX_finalize()
    except Exception:
        pass
    _initialised = False


def _config_with_tolerance(cfg_path, rtol):
    """Return a config path whose outer-solver tolerance is `rtol`. Substitution, not patching.

    The first attempt used `AMGX_config_add_parameters(&cfg, "main:tolerance=...")` on the
    already-created handle. AmgX answered `Caught amgx exception: Invalid/null C wrapper` and
    the run died at step 1 -- the call does not augment a handle in place the way the name
    suggests. Rewriting the JSON and letting `AMGX_config_create_from_file` do the only thing it
    is known to do correctly is duller and works.

    Written beside the template so a failed run leaves the exact config behind to inspect.
    """
    if rtol is None:
        return cfg_path
    import json
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    solver = cfg.get("solver", cfg)
    solver["tolerance"] = float(rtol)
    out = os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                       f"_generated_tol_{rtol:.3e}.json")
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=4)
    return out


def _init_once(cfg_path, rtol=None):
    """Initialise AmgX and create the config + resources ONCE for the process.

    AmgX expects a single resources object per device. Creating one per solver -- which the
    first version of this file did -- makes the SECOND solver throw
    `Cuda failure: 'invalid argument'`, and in the spanwise study that showed up as nz=16 and
    nz=32 "diverging at step 1" while nz=8 ran fine. The config is shared for the same reason.
    """
    global _initialised, _cfg_shared, _rsrc_shared
    if not _initialised:
        _chk(_lib.AMGX_initialize(), "initialize")
        try:
            _lib.AMGX_register_print_callback(_silent)
        except Exception:
            pass          # cosmetic only; never fail a run over logging
        _initialised = True
    # ONE CONFIG PER TOLERANCE, ONE RESOURCES PER PROCESS. The resources object
    # must be a singleton (a second one throws Cuda 'invalid argument' -- see
    # the spanwise-study note above), but AMGX_solver_create takes its OWN
    # config handle, so different tolerances can coexist against the shared
    # resources. The previous process-wide-config rule made the momentum solver
    # (rtol 1e-9) and the pressure solver (1e-6) mutually exclusive, and the
    # loser fell back to scipy -- silently, before the fallback learned to
    # print.
    key = None if rtol is None else float(f"{rtol:.6e}")
    cfg, used_path = _cfg_by_rtol.get(key, (None, None))
    if cfg is None:
        # AMGX_CONFIG_TIGHT: a different template for tight-tolerance systems
        # (the momentum solves at 1e-9). Measured on the real cylinder
        # operators: momentum under PCG+Jacobi converges in ONE iteration
        # (0.014 s); under the pressure-tuned aggregation AMG it never
        # converges (20,000 iters, 56 s) -- and the reverse holds for the
        # pressure system (AMG 0.062 s vs Jacobi 0.599 s). One template per
        # tolerance class, not one per process.
        tight = os.environ.get("AMGX_CONFIG_TIGHT")
        if tight and rtol is not None and rtol <= 1e-8:
            cfg_path = tight
        cfg_path = _config_with_tolerance(cfg_path, rtol)
        cfg = ctypes.c_void_p()
        _chk(_lib.AMGX_config_create_from_file(ctypes.byref(cfg),
                                               cfg_path.encode()), "config_create")
        used_path = cfg_path
        _cfg_by_rtol[key] = (cfg, used_path)
    if _rsrc_shared is None:
        _cfg_shared = cfg
        _rsrc_shared = ctypes.c_void_p()
        _chk(_lib.AMGX_resources_create_simple(ctypes.byref(_rsrc_shared), cfg),
             "resources_create")
    return cfg, _rsrc_shared, used_path


class NeedsRebuild(Exception):
    """Raised by solve() when the matrix drifted past drift_tol; the caller
    rebuilds by constructing a fresh AmgXSolver (in-place setup crashes)."""


class AmgXSolver:
    """One AmgX solver bound to one sparsity pattern.

    Call `solve(values, b)` per step with the current matrix VALUES; the structure passed at
    construction is reused. The hierarchy is rebuilt only when the values have drifted past
    `drift_tol` relative to those it was built from.
    """

    def __init__(self, A, config=None, drift_tol=0.05, rtol=None):
        A = A.tocsr()
        A.sort_indices()
        self.n = int(A.shape[0])
        self.nnz = int(A.nnz)
        self._ptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        self._col = np.ascontiguousarray(A.indices, dtype=np.int32)
        self.drift_tol = drift_tol

        cfg_path = config or os.environ.get(_CFG_ENV)
        if not cfg_path or not os.path.exists(cfg_path):
            raise RuntimeError(
                "no AmgX config; set AMGX_CONFIG to e.g. PCG_CLASSICAL_V_JACOBI.json. "
                "Config choice dominates performance -- AGGREGATION_JACOBI run as a standalone "
                "solver hit its iteration cap at convergence rate 0.90 on this operator.")

        self.rtol = rtol
        self.config_path = cfg_path
        # config_path is the TEMPLATE the caller named; config_used is what the
        # solver actually loaded after tight-tolerance routing + substitution.
        self._cfg, self._rsrc, self.config_used = _init_once(cfg_path, rtol)
        self._A = ctypes.c_void_p(); self._b = ctypes.c_void_p(); self._x = ctypes.c_void_p()
        self._slv = ctypes.c_void_p()
        m = ctypes.c_int(AMGX_MODE_dDDI)
        _chk(_lib.AMGX_matrix_create(ctypes.byref(self._A), self._rsrc, m), "matrix_create")
        _chk(_lib.AMGX_vector_create(ctypes.byref(self._b), self._rsrc, m), "vector_create b")
        _chk(_lib.AMGX_vector_create(ctypes.byref(self._x), self._rsrc, m), "vector_create x")
        _chk(_lib.AMGX_solver_create(ctypes.byref(self._slv), self._rsrc, m, self._cfg),
             "solver_create")

        vals = np.ascontiguousarray(A.data, dtype=np.float64)
        _chk(_lib.AMGX_matrix_upload_all(
            self._A, self.n, self.nnz, 1, 1,
            self._ptr.ctypes.data_as(ctypes.c_void_p),
            self._col.ctypes.data_as(ctypes.c_void_p),
            vals.ctypes.data_as(ctypes.c_void_p), None), "matrix_upload_all")
        _chk(_lib.AMGX_solver_setup(self._slv, self._A), "solver_setup")
        self._ref = vals.copy()          # values the hierarchy was built from
        self.rebuilds = 1
        self.iterations = 0

    def solve(self, values, b, x0=None):
        vals = np.ascontiguousarray(values, dtype=np.float64)
        # SKIP THE UPLOAD WHEN THE MATRIX IS BIT-IDENTICAL to the last one
        # uploaded. replace_coefficients makes AmgX refresh its Galerkin coarse
        # operators even when nothing changed -- measured at ~0.4 s per solve
        # on the 160k cylinder pressure system, which was most of the gap
        # between the 62 ms shootout solve and the 449 ms in-run solve. The
        # correctors within a step share one matrix, so this is exact, not an
        # approximation. self._last tracks the upload; self._ref still tracks
        # the hierarchy build for the drift rebuild below.
        _last = getattr(self, "_last", None)
        if _last is not None and vals.shape == _last.shape and \
                np.array_equal(vals, _last):
            drift = 0.0
        else:
            drift = (np.abs(vals - self._ref).max()
                     / max(np.abs(self._ref).max(), 1e-300))
            _chk(_lib.AMGX_matrix_replace_coefficients(
                self._A, self.n, self.nnz,
                vals.ctypes.data_as(ctypes.c_void_p), None), "replace_coefficients")
            self._last = vals.copy()
        if drift > self.drift_tol:
            # The hierarchy is stale. The in-place AMGX_solver_setup rebuild
            # crashes with a CUDA failure (rc 5) on the 2026-09 builds, for
            # PBICGSTAB and for PCG+aggregation alike, so the caller must tear
            # this solver down and construct a fresh one -- the construction
            # path is the one that demonstrably works.
            raise NeedsRebuild(f"drift {drift:.3e} > {self.drift_tol}")

        rhs = np.ascontiguousarray(b, dtype=np.float64)
        # x MUST BE A COPY, never the caller's x0: vector_download writes the
        # result into this buffer, and np.ascontiguousarray does NOT copy an
        # already-contiguous array. With the caller's buffer aliased, a FAILED
        # solve overwrote the warm start with divergence garbage in place --
        # so every retry, fresh solver, and even full-context reset then
        # "failed" too, because each inherited the poisoned x0 (R11, solve
        # #115: x0 went 2.29 -> 2.2e5 across one rejected solve).
        x = (np.zeros_like(rhs) if x0 is None
             else np.array(x0, dtype=np.float64, copy=True))
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        one = 1
        _chk(_lib.AMGX_vector_upload(self._b, self.n, one,
                                     rhs.ctypes.data_as(ctypes.c_void_p)), "vector_upload b")
        _chk(_lib.AMGX_vector_upload(self._x, self.n, one,
                                     x.ctypes.data_as(ctypes.c_void_p)), "vector_upload x")
        _chk(_lib.AMGX_solver_solve(self._slv, self._b, self._x), "solver_solve")
        _chk(_lib.AMGX_vector_download(self._x,
                                       x.ctypes.data_as(ctypes.c_void_p)), "vector_download")
        it = ctypes.c_int(0)
        _lib.AMGX_solver_get_iterations_number(self._slv, ctypes.byref(it))
        self.iterations = it.value
        # AMGX_solver_solve returns rc 0 even when the ITERATION diverged --
        # the rc reports API health only. Divergence lives in the status
        # handle, and a diverged solve hands back NaN with no error (R11:
        # one silent NaN solve poisoned every field within a few dozen steps).
        st = ctypes.c_int(0)
        _lib.AMGX_solver_get_status(self._slv, ctypes.byref(st))
        if st.value != 0 or np.isnan(x).any():
            err = FloatingPointError(
                f"AmgX solve unhealthy: status={st.value} (0=ok 1=failed "
                f"2=diverged 3=not converged) after {it.value} iters, NaNs in "
                f"x: {int(np.isnan(x).sum())}/{x.size} (n={self.n}, "
                f"rtol={self.rtol}, config={self.config_used})")
            # The caller holds A and can judge the TRUE residual -- AmgX's
            # "not converged" means it missed ITS criterion (relative to the
            # initial residual), which a good warm start makes unattainable.
            err.x = x
            err.status = st.value
            raise err
        return x

    def close(self):
        # Destroy only what THIS solver owns. The config and resources are process-wide;
        # tearing them down here would break every other live solver.
        for fn, h in ((_lib.AMGX_solver_destroy, self._slv),
                      (_lib.AMGX_vector_destroy, self._x),
                      (_lib.AMGX_vector_destroy, self._b),
                      (_lib.AMGX_matrix_destroy, self._A)):
            try:
                fn(h)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
