"""U3 of the unstructured adjoint plan: differentiable sparse direct solves.

The 2D unstructured solver runs `SolveCache(backend="splu")`: sparse LU, factored once per
distinct matrix values. The adjoint of x = A^-1 b is

    lambda = A^-T g,     dL/db = lambda,     dL/dA = -lambda x^T  (on the pattern only),

and with the LU already in hand A^-T g is `lu.solve(g, trans="T")` -- two triangular sweeps, no
second factorisation and no iterative solve that could stop short. That last point is the
FluidGym finding (backward solves returning unconverged iterates silently); here every forward
and backward solve is residual-checked and a failure RAISES (plan test A7).

SINGULAR (all-Neumann pressure) systems are solved exactly as `SolveCache._direct_solve` does:
row 0 replaced by e_0 with b_0 = 0 (pins the constant), then the mean removed. That is a linear
map b -> x for fixed A, x = P Ã^-1 E b with E zeroing entry 0 and P = I - 11^T/n, so its adjoint
is exact too: lambda = Ã^-T P g, dL/db = E lambda, and dL/dA vanishes on row 0, whose entries the
pin overwrites.
"""
import weakref

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

RESID_TOL = 1e-10          # relative residual every solve must meet, forward and backward


class SolveFailed(RuntimeError):
    pass


def _check(A, x, b, what):
    r = np.linalg.norm(A @ x - b)
    s = max(np.linalg.norm(b), 1e-300)
    if not np.isfinite(r) or r > RESID_TOL * s and r > 1e-13:
        raise SolveFailed(f"{what}: relative residual {r / s:.2e} > {RESID_TOL:.0e}")
    return r / s


class LUFactor:
    """One factorisation of a (possibly pinned) matrix, shared by every solve with those values.

    `LUFactor.live_bytes` / `.peak_bytes` track the memory of the factors currently alive (L and U
    as CSC: 12 bytes per nonzero plus the permutations) -- the part of a tape that saved-tensor
    hooks cannot see, because the factor sits in the solve's ctx, not in a tensor (gate A18)."""
    live_bytes = 0
    peak_bytes = 0

    @classmethod
    def reset_peak(cls):
        cls.peak_bytes = cls.live_bytes

    @classmethod
    def _free(cls, nbytes):
        cls.live_bytes -= nbytes

    def __init__(self, pattern, vals, singular=False):
        v = vals.detach().numpy() if torch.is_tensor(vals) else np.asarray(vals)
        A = sp.csr_matrix((v, pattern.idx), shape=pattern.shape)
        A.sort_indices()
        self.singular = bool(singular)
        if self.singular:
            A = A.copy()
            r0, r1 = A.indptr[0], A.indptr[1]
            A.data[r0:r1] = 0.0
            d = np.flatnonzero(A.indices[r0:r1] == 0)
            if len(d) == 0:
                raise ValueError("cannot pin DOF 0: no diagonal entry in row 0")
            A.data[r0 + d[0]] = 1.0
        self.A = A
        self.lu = spla.splu(A.tocsc())
        nbytes = 12 * (self.lu.L.nnz + self.lu.U.nnz) + 16 * A.shape[0]
        LUFactor.live_bytes += nbytes
        LUFactor.peak_bytes = max(LUFactor.peak_bytes, LUFactor.live_bytes)
        weakref.finalize(self, LUFactor._free, nbytes)
        # pattern entries overwritten by the pin get no gradient
        self.last_resid = (None, None)

    def solve(self, b):
        b = np.array(b, dtype=np.float64)
        if self.singular:
            b[0] = 0.0
        xt = self.lu.solve(b)
        rf = _check(self.A, xt, b, "forward solve")
        x = xt - xt.mean() if self.singular else xt
        self.last_resid = (rf, self.last_resid[1])
        return x, xt

    def solve_T(self, g):
        g = np.array(g, dtype=np.float64)
        if self.singular:
            g = g - g.mean()                     # P^T g, P symmetric
        lam = self.lu.solve(g, trans="T")
        rb = _check(self.A.T.tocsr(), lam, g, "adjoint solve")
        self.last_resid = (self.last_resid[0], rb)
        return lam                               # dL/dA uses it whole; dL/db zeroes entry 0


class _LUSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vals, b, pattern, factor):
        x, x_pre = factor.solve(b.detach().numpy())
        ctx.save_for_backward(torch.as_tensor(x_pre))
        ctx.x_pre = torch.as_tensor(x_pre)
        ctx.pattern, ctx.factor = pattern, factor
        return torch.as_tensor(x)

    @staticmethod
    def backward(ctx, g):
        (x_pre,) = ctx.saved_tensors                 # x̃: the pinned solve's answer, before P
        f, pat = ctx.factor, ctx.pattern
        lam = f.solve_T(g.detach().numpy())          # Ã^-T P g
        grad_vals = None
        if ctx.needs_input_grad[0]:
            rows, cols = pat.idx
            gv = -lam[rows] * x_pre.numpy()[cols]    # dÃ enters as -Ã^-1 dÃ x̃
            if f.singular:
                gv[rows == 0] = 0.0                  # row 0 is the pin, not A
            grad_vals = torch.as_tensor(gv)
        lb = lam.copy()
        if f.singular:
            lb[0] = 0.0                              # E^T: b_0 never reached the solve
        return grad_vals, torch.as_tensor(lb), None, None


    @staticmethod
    def jvp(ctx, dvals, db, _p, _f):
        """Forward mode (plan P7 needs direct modes): x̃ = Ã^-1 E b, so
        dx̃ = Ã^-1 (E db - dÃ x̃) and dx = P dx̃, with dÃ = dA off row 0 when pinned."""
        f, pat = ctx.factor, ctx.pattern
        n = pat.shape[0]
        rhs = np.zeros(n) if db is None else db.detach().numpy().copy()
        if dvals is not None:
            dv = dvals.detach().numpy().copy()
            rows, cols = pat.idx
            if f.singular:
                dv[rows == 0] = 0.0
            rhs -= np.bincount(rows, weights=dv * ctx.x_pre.numpy()[cols], minlength=n)
        if f.singular:
            rhs[0] = 0.0
        dx = f.lu.solve(rhs)
        if f.singular:
            dx = dx - dx.mean()
        return torch.as_tensor(dx)


def lu_solve(vals, b, pattern, factor=None, singular=False):
    """x = A(vals)^-1 b, differentiable in vals and b. Pass `factor` to reuse a factorisation
    across solves with the same values (production reuses one LU per matrix)."""
    if factor is None:
        factor = LUFactor(pattern, vals, singular)
    return _LUSolve.apply(vals, b, pattern, factor)
