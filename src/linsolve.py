"""
One dispatch point for every sparse solve, so a backend swap is a single argument.

Three backends:

  scipy  SciPy's CG / BiCGStab with a preconditioner from src.precond (default: jacobi).
  amgx   NVIDIA AmgX solving the whole system on the GPU. Requires libamgxsh.so; falls back
         to scipy when it is unavailable, because a missing GPU should not stop a run.

THE CALLER'S `rtol` IS NOW HONOURED ON BOTH PATHS. It previously reached SciPy and was discarded
by `_amgx_solve`, so on the GPU path the AmgX JSON file was the real tolerance and tightening
`tol` did nothing. Any CPU-vs-GPU timing taken then compared two different problems.

WHY AMGX REPLACES THE SOLVE RATHER THAN PRECONDITIONING SciPy's. Handing AmgX one V-cycle at a
time to a host Krylov loop would pay the round trip per iteration and discard AmgX's own
iteration control. Measured whole-system on the 86k-unknown pressure operator: 28 ms/step at 53
iterations against 784 ms for SciPy CG+Jacobi -- about 25x.

CORROBORATED ON THE CPU. The same question -- multigrid as a solver or as a preconditioner --
is measurable without a GPU, and pyamg answers it the same way on our 82k pressure operator at
rtol 1e-6: standalone V-cycles take 196 iterations and 1.290 s, AMG-preconditioned CG takes 29
and 0.290 s. Multigrid alone is a poor solver on this operator (convergence ~0.93/cycle); it is
the Krylov acceleration that does the work. AmgX therefore runs its OWN PCG internally rather
than being handed to ours -- see src/amgx/pcg_amg_1e6.json.

That same measurement is why Jacobi, not AMG, is the CPU default: AMG-preconditioned CG loses to
CG+Jacobi 0.290 s vs 0.160 s, and still loses 0.208 vs 0.160 with the hierarchy reused. The GPU
verdict flips only because a V-cycle is bandwidth-bound and parallel, so it costs far less there
relative to a mat-vec. See src/precond.py for the full table.

HIERARCHY REUSE IS WHAT MAKES IT FAST, and it is legal here because the pressure matrix keeps
IDENTICAL sparsity every step. Verified on the Dong path specifically, where the solve is on a
REDUCED system with Dirichlet outlet nodes eliminated: the Dirichlet set is geometric (outlet
faces), so it does not move -- 296 nodes, M_ff nnz 182,792, both identical across 8 steps. The
cache is therefore keyed on the sparsity pattern and invalidated if it ever changes.
"""
from time import perf_counter as _perf

import numpy as np
import scipy.sparse.linalg as spla

from src.precond import make as make_precond


def maxiter_default():
    return 20000


def _is_singular(A, tol=1e-9):
    """Does A actually have a constant nullspace? Cheap test: is A @ 1 zero?

    A pure all-Neumann Laplacian annihilates the constant vector exactly. The matrix this
    solver produces does not -- a handful of rows carry boundary contributions that make it
    definite -- so the constant is not a nullspace and must not be projected away.
    """
    import numpy as _np
    one = _np.ones(A.shape[1])
    r = A @ one
    return float(_np.abs(r).max()) <= tol * float(abs(A).max())


def _has_pc(name, _cache={}):
    """Is this PETSc preconditioner available in the build? Cached; falls back when not."""
    if name not in _cache:
        try:
            from petsc4py import PETSc
            pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
            pc.setType(name)
            pc.destroy()
            _cache[name] = True
        except Exception:
            _cache[name] = False
    return _cache[name]


class SolveCache:
    """Holds a backend solver bound to one sparsity pattern, across steps.

    Lives on the PISO solver instance, because the whole point is to survive from one step to
    the next. `key()` is the pattern; if it changes the cached solver is discarded rather than
    silently reused on a matrix it was not built for.
    """

    def __init__(self, backend="scipy", precond="jacobi", drift_tol=0.05, config=None):
        self.backend = backend
        self.precond = precond
        self.drift_tol = drift_tol
        self.config = config
        self._key = None
        self._amgx = None
        self.iterations = 0
        self.fell_back = False
        # TIME SPENT INSIDE THE LINEAR SOLVES, accumulated across every call. This is not
        # curiosity: Gate 6 of the PETSc plan has to decide whether to continue, and that
        # decision turns on what fraction of the runtime is even ADDRESSABLE by distributing
        # the solve. If assembly dominates, no amount of parallel solve delivers the projected
        # speed-up. Before this counter existed the split was unmeasured and the plan's
        # projections rested on an assumption.
        self.t_solve = 0.0
        self.n_solve = 0

    @staticmethod
    def key(A):
        # cheap fingerprint of the sparsity: shape, nnz, and the two index arrays
        return (A.shape, A.nnz,
                hash(A.indptr.tobytes()), hash(A.indices.tobytes()))

    def solve(self, A, b, x0=None, symmetric=True, rtol=1e-12, maxiter=20000,
              singular=False):
        _t0 = _perf()
        try:
            return self._solve_timed(A, b, x0, symmetric, rtol, maxiter, singular)
        finally:
            self.t_solve += _perf() - _t0
            self.n_solve += 1

    def _solve_timed(self, A, b, x0=None, symmetric=True, rtol=1e-12, maxiter=20000,
                     singular=False):
        A = A.tocsr()
        A.sort_indices()
        if self.backend == "petsc":
            # THE NORMALISATION IS DECIDED INSIDE, where the measured singularity is known.
            # `singular` is a HINT the SciPy path ignores entirely, and this is the THIRD place
            # trusting it went wrong: the preconditioner choice, the nullspace attachment, and
            # here. Subtracting the mean is right for a genuinely all-Neumann operator, whose
            # solution is fixed only up to a constant; applied to the DEFINITE matrix this
            # solver actually produces it corrupts a correct answer by exactly that mean --
            # measured as max|x_scipy - x_petsc| = 292 against |x|max = 6123, with both solvers
            # reporting success.
            x = self._petsc_solve(A, b, x0, rtol, symmetric, singular)
            if x is not None:
                return x
            self.fell_back = True

        if self.backend == "amgx":
            x = self._amgx_solve(A, b, x0, rtol)
            if x is not None:
                if singular:
                    # An all-Neumann operator fixes the solution only up to a constant, and
                    # AmgX may land on a different member than SciPy would. Removing the mean
                    # picks the same one, which keeps a backend swap from shifting p by a
                    # constant. It does not change grad(p), so the flow is unaffected either way.
                    x = x - x.mean()
                return x
            self.fell_back = True

        M = make_precond(A, self.precond)
        solver = spla.cg if symmetric else spla.bicgstab
        x, info = solver(A, b, x0=x0, M=M, rtol=rtol, maxiter=maxiter)
        return x

    def _petsc_solve(self, A, b, x0=None, rtol=None, symmetric=True, singular=False):
        """Gate 3: the pressure system through a PETSc KSP.

        SERIAL FIRST, ON PURPOSE. This runs on PETSC_COMM_SELF and assembles the whole matrix
        on one rank -- no distribution at all. That separates two questions the plan folds into
        one gate: does PETSc reproduce SciPy's answer, and does distributing the matrix
        reproduce the serial one. Testing them together means a discrepancy has two possible
        causes and no way to tell them apart. The distributed Mat comes next, on top of a
        backend already known to agree.

        THE SINGULAR CASE NEEDS A NULLSPACE, not just a mean subtraction afterwards. With every
        boundary Neumann the operator has a constant nullspace, and KSP will happily converge to
        a solution containing an arbitrary amount of it -- or stall, since the residual never
        drops in that direction. Attaching the nullspace tells the Krylov method to project it
        out each iteration, which is what makes the iteration count comparable to SciPy's rather
        than an artefact of how much constant crept in.

        Returns None on any failure, so the caller falls back to SciPy exactly as it does for
        AmgX, and `fell_back` records it rather than the run silently changing solver.
        """
        try:
            from petsc4py import PETSc
        except Exception:
            import os
            if os.environ.get("PICT_PETSC_STRICT"):
                raise
            return None
        try:
            n = A.shape[0]
            comm = PETSc.COMM_SELF
            # ONE FACT, USED FOR BOTH DECISIONS. Making the nullspace conditional on
            # `_is_singular` while leaving the PRECONDITIONER keyed on the `singular` HINT left
            # the two disagreeing: this matrix is not singular, so no nullspace was attached,
            # but it still got the semi-definite preconditioner (GAMG), which converges poorly
            # on it -- true residual 2.6e-1 against SciPy's 1.0e-6, while PETSc reported
            # success. A fresh KSP with ILU reached 9.9e-7 in 330 iterations, which is what
            # showed the fault was the choice and not the solve.
            really_singular = singular and _is_singular(A)
            key = self.key(A)
            if getattr(self, "_petsc_key", None) != key:
                self._petsc_mat = PETSc.Mat().createAIJ(
                    size=(n, n), csr=(A.indptr, A.indices, A.data), comm=comm)
                self._petsc_mat.assemble()
                ksp = PETSc.KSP().create(comm=comm)
                ksp.setOperators(self._petsc_mat)
                ksp.setType("cg" if symmetric else "bcgs")
                # THE SINGULAR SYSTEM NEEDS AN SPD PRECONDITIONER, and ILU is not one on it.
                # With every boundary Neumann the pressure Laplacian is positive SEMI-definite
                # -- a constant nullspace -- and an incomplete factorisation of it comes back
                # indefinite. CG then fails with DIVERGED_INDEFINITE_PC (reason -8) rather than
                # producing a wrong answer, which is the good outcome, but it silently fell
                # back to SciPy so the backend under test was never exercised. Measured on the
                # cylinder: 197 iterations, then -8, every step.
                if really_singular and symmetric:
                    ksp.getPC().setType("gamg" if _has_pc("gamg") else "jacobi")
                else:
                    ksp.getPC().setType("hypre" if _has_pc("hypre") else "ilu")
                self._petsc_ksp = ksp
                self._petsc_key = key
            else:
                # same sparsity, new values -- overwrite in place rather than rebuild, which is
                # the whole point of caching on the pattern
                self._petsc_mat.setValuesCSR(A.indptr, A.indices, A.data)
                self._petsc_mat.assemble()
                # TELL THE KSP THE OPERATOR CHANGED. Modifying a Mat in place leaves the
                # preconditioner PETSc already built from the old values in place; without
                # this, every solve after the first reuses a stale factorisation.
                self._petsc_ksp.setOperators(self._petsc_mat)
            ksp = self._petsc_ksp
            ns = None
            # `singular` IS A HINT, NOT A FACT, and the SciPy path ignores it entirely -- it is
            # consulted only in the AmgX branch, to pick a consistent member of the nullspace.
            # Taking it literally here was wrong and cost a long detour: the matrix the solver
            # actually hands over is exactly symmetric, has a strictly positive diagonal, and
            # a smallest eigenvalue of +2.0e-3. It is positive DEFINITE. Attaching a constant
            # nullspace to a non-singular operator, and projecting that direction out of the
            # right-hand side, removes a component the solution genuinely needs -- which
            # presented as DIVERGED_INDEFINITE_PC under every preconditioner tried, including
            # Jacobi, whose positive diagonal scaling cannot make an SPD operator indefinite.
            # That contradiction was the clue: the fault was never in the preconditioner.
            #
            # So the nullspace is attached only when the operator really is singular, which is
            # checked rather than trusted.
            if really_singular:
                ns = PETSc.NullSpace().create(constant=True, comm=comm)
                self._petsc_mat.setNullSpace(ns)
            # "CONVERGED" MUST MEAN THE SAME THING IN BOTH BACKENDS. PETSc's default CG
            # convergence test measures the PRECONDITIONED residual; SciPy measures the true
            # one. On this pressure matrix -- condition number about 1.8e6 -- the difference is
            # not academic: PETSc declared convergence after 22 iterations at a TRUE relative
            # residual of 2.6e-1 while SciPy reached 1.0e-6, and the trajectory then diverged
            # by 1180% of |u|max while every diagnostic reported success. A backend swap that
            # silently redefines the convergence criterion is worse than one that fails.
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setTolerances(rtol=rtol if rtol else 1e-12, max_it=maxiter_default())
            ksp.setFromOptions()
            xv = self._petsc_mat.createVecRight()
            bv = self._petsc_mat.createVecLeft()
            bv.setArray(b)
            if ns is not None:
                # THE RHS MUST BE MADE COMPATIBLE TOO, not just the operator. A singular system
                # has a solution only when the right-hand side is orthogonal to the nullspace;
                # leaving a constant component in b asks CG for a solution that does not exist,
                # and it converges to nothing useful or not at all. Attaching the nullspace to
                # the Mat projects the ITERATE, not the residual source.
                ns.remove(bv)
            if x0 is not None:
                xv.setArray(x0)
                ksp.setInitialGuessNonzero(True)
            ksp.solve(bv, xv)
            self.iterations = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            if reason < 0:
                # RECORD WHY. A bare `return None` here made the fallback invisible: the run
                # completed, the answer was right, and a comparison "against PETSc" was really
                # SciPy against SciPy. The reason code distinguishes a diverged solve from a
                # hit iteration cap from a breakdown, which need different fixes.
                self.petsc_fail = (int(reason), int(self.iterations), bool(singular),
                                   bool(symmetric))
                return None
            x = xv.getArray().copy()
            if really_singular:
                # Only for an operator that IS singular: its solution is fixed up to a
                # constant, and removing the mean picks the same member SciPy and AmgX pick,
                # so a backend swap cannot shift p by a constant.
                x = x - x.mean()
            return x
        except Exception:
            # A SILENT FALLBACK IS WORSE THAN A FAILURE: the run keeps going, the answer is
            # still right, and the backend under test was never exercised -- so a comparison
            # "against PETSc" is really SciPy against SciPy. PICT_PETSC_STRICT=1 re-raises so
            # the cause is visible; without it the fallback stands, which is what production
            # wants.
            import os
            if os.environ.get("PICT_PETSC_STRICT"):
                raise
            return None

    def _amgx_solve(self, A, b, x0, rtol=None):
        k = self.key(A)
        if self._amgx is not None and k != self._key:
            self._amgx.close()
            self._amgx = None
        if self._amgx is None:
            try:
                from src.amgx.binding import AmgXSolver
            except ImportError:
                return None
            try:
                self._amgx = AmgXSolver(A, config=self.config,
                                        drift_tol=self.drift_tol, rtol=rtol)
            except Exception:
                return None
            self._key = k
        x = self._amgx.solve(A.data, b, x0=x0)
        self.iterations = self._amgx.iterations
        return x

    def close(self):
        if self._amgx is not None:
            self._amgx.close()
            self._amgx = None
