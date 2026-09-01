"""
Preconditioners for the Krylov solves, and the measurements behind the default.

The solver used to pass no preconditioner at all. On the five-domain BFS pressure operator that
cost 1,577 CG iterations for 12,576 cells -- and since ~94% of a step is spent inside Krylov
iterations, iteration count IS the runtime.

Measured on that operator, CG to rtol=1e-10:

    preconditioner        iterations     total
    none (the old default)     1,577     0.115 s
    jacobi                       785     0.059 s     <- 2x, and free to build
    ILU drop_tol=1e-3         20,000     11.9 s      <- FAILED (hit maxiter)
    amg (pyamg V-cycle)           78     0.157 s

WHY JACOBI IS THE DEFAULT rather than AMG. AMG cuts iterations ~20x and its count barely moves
with problem size -- 70 / 78 / 78 / 76 across 3,888 -> 52,780 cells, which is the O(1)
convergence multigrid is for. But each AMG iteration is a full V-cycle, far dearer than one
sparse mat-vec, so on wall time it still LOSES at every size tested:

    cells      none      jacobi     amg
     3,888    0.035 s    0.020 s   0.035 s
    12,576    0.115 s    0.060 s   0.160 s
    26,720    0.248 s    0.130 s   0.303 s
    52,780    0.590 s    0.290 s   0.550 s

## AMG AS A PRECONDITIONER vs AMG AS A SOLVER

Multigrid can either drive the solve itself or precondition a Krylov method. Measured on the
82,096-unknown square-cylinder pressure operator at the production tolerance 1e-6:

    method                      iters   setup s   solve s   total s
    CG + Jacobi  (the default)    442     0.000     0.160     0.160
    CG + AMG (preconditioner)      29     0.073     0.217     0.290
    AMG standalone (V-cycles)     196     0.000     1.290     1.290
    AMG + CG accel (pyamg's own)   29     0.000     0.215     0.215

**Preconditioning beats standalone by 4.4x.** Pure V-cycles need 196 iterations where
AMG-preconditioned CG needs 29 on the SAME hierarchy -- a convergence rate near 0.93 per cycle,
which is poor for multigrid and says this hierarchy is a mediocre fit for a curvilinear
multi-block Laplacian. The Krylov acceleration is doing most of the work. Rows 2 and 4 agreeing
to 1% is the consistency check: SciPy's CG and pyamg's own accelerator land in the same place.

So if AMG is used at all here, use it as a preconditioner. Never standalone.

## AND THE SETUP COST IS NOT WHAT SINKS IT

The obvious rescue is hierarchy reuse -- build once, reuse across steps, which is exactly what
makes AmgX fast. It does not help:

    AMG built once            0.074 s
    then per solve, AMG       0.208 s
    per solve, Jacobi         0.160 s

**Jacobi still wins per-solve by 1.30x**, so there is no number of steps after which reuse pays
back. The 442 -> 29 iteration collapse is real, but a V-cycle costs about 15x a
Jacobi-preconditioned mat-vec and 442/29 = 15.2 -- almost exactly a wash, decided by overhead.

## THE SAME LIBRARY GIVES THE OPPOSITE VERDICT ON A TOY PROBLEM

On a 20,000-unknown 1-D Laplacian, pyamg was **15x FASTER** than Jacobi (52.7 ms against
806.2 ms). Same code, same version, opposite conclusion. A 1-D Laplacian is the ideal case for
multigrid; a 3-D curvilinear multi-block operator with stretched cells is not. Any AMG benchmark
run on anything other than the operator you actually intend to solve is worthless here.

## WHY THE GPU VERDICT DIFFERS

AmgX wins on the GB10 while pyamg loses on the M3 Max, and the reason is arithmetic, not
implementation quality: a V-cycle is bandwidth-bound and highly parallel, so the GPU does one far
more cheaply RELATIVE TO A MAT-VEC than a single CPU core does. That is also why AmgX replaces
the whole solve rather than preconditioning SciPy's CG -- see src/linsolve.py.

The AMG/Jacobi gap narrows with size across the range below (2.7x -> 1.9x), which suggested AMG
would overtake somewhere above it. IT DOES NOT. Measured on the square-cylinder pressure
operator, same solver settings, 8 steps after a warm-up step:

    cells      jacobi     amg      ratio
    63,280     1.631 s   3.733 s   2.29x
   126,560     3.037 s   7.361 s   2.42x

The gap REOPENS -- 1.9x at 53k, 2.3x at 63k, 2.4x at 127k -- so the trend in the table above
does not extrapolate, and the crossover this note used to predict is not there for this
operator. AMG stays an option because its iteration count really is O(1); what costs is the
V-cycle, and on these problem sizes that never pays back. Re-measure before trusting it on a
new problem; do not assume the crossover, in either direction.

WHY NOT ILU. `spilu` produces a NON-SYMMETRIC preconditioner, which breaks CG's assumptions
outright -- it does not merely converge slowly, it fails to converge at all (20,000 iterations,
100x slower than no preconditioner). It remains useful for the non-symmetric implicit-cross
operator, which is solved with BiCGStab; that path builds its own and is untouched here.
"""
import numpy as np
import scipy.sparse.linalg as spla

KINDS = ("none", "jacobi", "amg")     # amgx is a BACKEND, not a preconditioner


def make(A, kind="jacobi"):
    """A LinearOperator to pass as `M=` to cg/bicgstab, or None.

    Unknown or unavailable kinds fall back to Jacobi rather than raising: a preconditioner is
    an acceleration, and refusing to run because pyamg is missing would be the wrong trade.
    """
    if kind in (None, "none"):
        return None

    if kind == "amg":
        try:
            import pyamg
        except ImportError:
            kind = "jacobi"          # fall through, below
        else:
            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=200)
            return ml.aspreconditioner(cycle="V")

    if kind == "amgx":
        # AmgX is NOT a preconditioner in this codebase -- it REPLACES the solve, and is
        # selected with `linear_backend="amgx"` on MultiBlockPISO, not with this argument.
        # This branch used to import `make_amgx_preconditioner`, WHICH DOES NOT EXIST, so it
        # silently fell through to Jacobi: a plausible-looking option that quietly did something
        # else. Raising beats that.
        raise ValueError(
            "'amgx' is not a preconditioner kind. AmgX replaces the whole solve; pass "
            "linear_backend='amgx' to MultiBlockPISO instead. See src/linsolve.py.")

    if kind != "jacobi":
        raise ValueError(f"unknown preconditioner {kind!r}; expected one of {KINDS}")

    d = A.diagonal().astype(float)
    # A zero diagonal would make this a division by zero. It should not happen for the
    # operators here -- the conservative diffusion matrix carries a positive diagonal by
    # construction -- but a silent inf would corrupt the solve rather than fail it.
    bad = d == 0.0
    if bad.any():
        d = d.copy()
        d[bad] = 1.0
    dinv = 1.0 / d
    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=lambda v: dinv * v, dtype=float)
