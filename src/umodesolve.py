"""Batched iterative solves for the per-mode Helmholtz families of the 2.5D solver (LES plan L4, step 1).

Every implicit system of `PISO25` has the form   (A + s_k D) x_k = b_k,  k = 0..nk-1,
with A one sparse matrix shared by all modes, D a positive diagonal (cell volume times a coefficient)
and s_k = k_z^2 (times nu*beta for momentum). The matrices differ only by that diagonal, so the whole
family is solved AT ONCE as a block: the matvec is one sparse product on an (N, nk) block plus a
diagonal scaling, and the preconditioner is one algebraic-multigrid hierarchy (pyamg smoothed
aggregation, built from the k = 0 matrix) whose level operators are shifted per mode EXACTLY:
    A_l(k) = A_l + s_k D_l,     D_l = R_l D_{l-1} P_l   (Galerkin, precomputed once per level).
Smoothing is damped Jacobi with the per-mode diagonal (vectorised over modes), the coarsest level is
a batched dense solve. Conjugate gradients then run on all columns together, each column with its
own scalars, until every column's residual is below the tolerance. Complex right-hand sides are
carried as two real columns per mode. The operators are negative (semi)definite as assembled by
`uops.laplacian`; the class works on the negated, SPD, family.

Why: the direct factorisations (one SuperLU per stage per mode) were 40% of the step and scale
superlinearly with the plane size; V3 (10^5 cells x 64 modes) needs an O(N) solve per mode, and a
block iterative solve is also the form that maps onto a GPU (cupy: same code, same sparse products).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)


def _axpy2(a, P, X, AP, R):
    """X + a P, R - a AP in one pass on the device (cupy.fuse), plain numpy on the host."""
    try:
        import cupy as cp
        if isinstance(X, cp.ndarray):
            return _axpy2_dev(a, P, X, AP, R)
    except ImportError:
        pass
    return X + a * P, R - a * AP


def _zpbp(Z, b, P):
    try:
        import cupy as cp
        if isinstance(Z, cp.ndarray):
            return _zpbp_dev(Z, b, P)
    except ImportError:
        pass
    return Z + b * P


try:
    import cupy as _cp
    @_cp.fuse()
    def _axpy2_dev(a, P, X, AP, R):
        return X + a * P, R - a * AP
    @_cp.fuse()
    def _zpbp_dev(Z, b, P):
        return Z + b * P
except ImportError:
    pass


class ModeFamily:
    """SPD family  M_k = Apos + s_k D,  Apos = -A (sparse, N x N), D >= 0 diagonal (N,), s (nk,)."""

    def __init__(self, A_neg, dvec, shifts, singular_k0=False, max_coarse=60, presmooth=2, postsmooth=2, omega=None, hierarchy="ruge_stuben", device="cpu", precond="amg"):
        """device "cpu" (numpy/scipy) or "gpu" (cupy / cupyx.scipy.sparse): the hierarchy is always built on the
        host with pyamg, then every level operator and the coarse inverses are moved to the device; `solve`
        accepts host or device blocks and returns the kind it was given."""
        import pyamg
        self.A = (-A_neg).tocsr(); self.d = np.asarray(dvec, float); self.s = np.asarray(shifts, float); self.nk = len(self.s)
        self.singular_k0 = bool(singular_k0) and float(self.s[0]) == 0.0
        N = self.A.shape[0]
        # Classical (Ruge-Stuben) coarsening: on the wall-clustered channel operator (cell aspect 6:1 at the
        # wall) smoothed aggregation needed 55 block-Jacobi PCG iterations to 1e-8 and 35 even with pyamg's
        # own Gauss-Seidel; Ruge-Stuben with Gauss-Seidel needed 11 (section 57). The hierarchy, not the
        # smoother, was the problem.
        self.precond = precond
        if precond == "jacobi":
            # diagonally dominant families (momentum: V/dt on the diagonal) converge in ~5 PCG iterations with
            # the per-mode diagonal alone; the V-cycle costs ~10 matvecs per iteration for nothing there
            # (measured on the GB10 at 256^2 x 64: 354 ms per momentum solve with AMG, section 57)
            ml = _NS(levels=[_NS(A=self.A)])
        else:
            ml = (pyamg.ruge_stuben_solver(self.A, max_coarse=max_coarse) if hierarchy == "ruge_stuben"
                  else pyamg.smoothed_aggregation_solver(self.A, max_coarse=max_coarse, symmetry="symmetric"))
        self.levels = []
        Dl = sp.diags(self.d).tocsr()
        for i, lv in enumerate(ml.levels):
            Al = lv.A.tocsr(); diagA = np.asarray(Al.diagonal()); diagD = np.asarray(Dl.diagonal())
            rho = self._rho(Al, diagA + 1e-300)
            self.levels.append(dict(A=Al, D=Dl, diagA=diagA, diagD=diagD, omega=(omega or 4.0 / (3.0 * rho)),
                                    P=(lv.P.tocsr() if hasattr(lv, "P") else None), R=(lv.R.tocsr() if hasattr(lv, "R") else None)))
            if hasattr(lv, "P"):
                Dl = (lv.R @ Dl @ lv.P).tocsr()
        # coarsest: dense per mode
        if precond == "jacobi":
            self.coarse_lu = np.zeros((self.nk, 1, 1)); self.presmooth, self.postsmooth = presmooth, postsmooth; self.iterations = 0; self.last_hist = []
            self._finish_device(device); return
        c = self.levels[-1]; Ac = c["A"].toarray(); Dc = c["D"].toarray()
        self.coarse = np.stack([Ac + sk * Dc for sk in self.s])                       # (nk, nc, nc)
        if self.singular_k0:
            for k in np.flatnonzero(self.s == 0.0): self.coarse[k] += 1e-8 * np.abs(np.diag(Ac)).mean() * np.eye(Ac.shape[0])   # regularise the Neumann k = 0 block
        self.coarse_lu = np.stack([np.linalg.inv(m) for m in self.coarse])              # (nk, nc, nc) explicit inverses, applied as one einsum
        self.presmooth, self.postsmooth = presmooth, postsmooth
        self.iterations = 0; self.last_hist = []
        self._finish_device(device)

    def _finish_device(self, device):
        self.device = device; self.xp = np
        if device == "gpu":
            import cupy as cp; from src.ucuda import DevCSR
            self.xp = cp
            for lv in self.levels:
                for key in ("A", "D", "P", "R"):
                    if lv[key] is not None: lv[key] = DevCSR(lv[key])
                lv["diagA"] = cp.asarray(lv["diagA"]); lv["diagD"] = cp.asarray(lv["diagD"])
            self.coarse_lu = cp.asarray(self.coarse_lu); self.s = cp.asarray(self.s)

    @staticmethod
    def _rho(A, diag, it=12):
        """spectral radius of D^-1 A by power iteration (for the Jacobi damping)."""
        x = np.random.default_rng(0).standard_normal(A.shape[0]); x /= np.linalg.norm(x)
        for _ in range(it):
            y = (A @ x) / diag; r = np.linalg.norm(y); x = y / r
        return r

    # ------------------------------------------------------------------ operator on blocks
    def matvec(self, X, level=0):
        lv = self.levels[level]
        return lv["A"] @ X + (lv["D"] @ X) * self.s[None, :] if lv["D"].shape[0] == X.shape[0] else lv["A"] @ X

    def _mv(self, lv, X):
        return lv["A"] @ X + (lv["D"] @ X) * self.s[None, :]

    def _jacobi(self, lv, X, B, sweeps):
        dk = lv["diagA"][:, None] + lv["diagD"][:, None] * self.s[None, :]
        for _ in range(sweeps):
            X = X + lv["omega"] * (B - self._mv(lv, X)) / dk
        return X

    def vcycle(self, B, level=0):
        lv = self.levels[level]
        if self.precond == "jacobi":
            return B / (lv["diagA"][:, None] + lv["diagD"][:, None] * self.s[None, :])
        if level == len(self.levels) - 1:
            return np.einsum("kij,jk->ik", self.coarse_lu, B)                          # per-mode dense solve
        X = self._jacobi(lv, np.zeros_like(B), B, self.presmooth)
        r = B - self._mv(lv, X)
        Xc = self.vcycle(lv["R"] @ r, level + 1)
        X = X + lv["P"] @ Xc
        return self._jacobi(lv, X, B, self.postsmooth)

    # ------------------------------------------------------------------ block PCG
    def solve(self, B, X0=None, rtol=1e-8, atol=0.0, maxiter=200, check_every=1):
        """B (N, nk) real block (call twice, or stack real/imag columns with duplicated shifts, for complex).
        Returns X with ||M_k x_k - b_k|| <= rtol ||b_k|| for every k. Host input -> host output; device
        input -> device output. `check_every` > 1 skips the device->host convergence test on some iterations."""
        xp = self.xp; host_in = isinstance(B, np.ndarray)
        B = xp.asarray(B, dtype=xp.float64); X = xp.zeros_like(B) if X0 is None else xp.array(xp.asarray(X0), dtype=xp.float64)
        sm = (xp.asarray(self.s) == 0.0) if self.singular_k0 else None            # singular columns: shift 0 on an all-Neumann operator
        proj = (lambda M: M.__setitem__((slice(None), sm), M[:, sm] - M[:, sm].mean(axis=0, keepdims=True))) if self.singular_k0 else (lambda M: None)
        B = B.copy(); proj(B)
        R = B - self._mv(self.levels[0], X); proj(R)
        bn = xp.maximum(xp.linalg.norm(B, axis=0), 1e-300); tol = xp.maximum(rtol * bn, atol)
        Z = self.vcycle(R); proj(Z); P = Z.copy(); rz = (R * Z).sum(axis=0); hist = [xp.linalg.norm(R, axis=0) / bn]
        for it in range(maxiter):
            if it % check_every == 0 and bool((xp.linalg.norm(R, axis=0) <= tol).all()): break
            AP = self._mv(self.levels[0], P); proj(AP)
            alpha = rz / xp.maximum((P * AP).sum(axis=0), 1e-300)
            X, R = _axpy2(alpha[None, :], P, X, AP, R)                                  # fused: X += a P ; R -= a AP
            Z = self.vcycle(R); proj(Z)
            rz_new = (R * Z).sum(axis=0); beta = rz_new / xp.maximum(rz, 1e-300); rz = rz_new
            P = _zpbp(Z, beta[None, :], P)                                                # fused: P = Z + b P
            hist.append(xp.linalg.norm(R, axis=0) / bn)
        self.iterations = it; self.last_hist = np.array([np.asarray(h.get() if hasattr(h, "get") else h) for h in hist])
        proj(X)
        return (X.get() if (host_in and hasattr(X, "get")) else X)

    def solve_complex(self, Bc, X0=None, **kw):
        """Bc (N, nk) complex -> (N, nk) complex, real and imaginary parts as two blocks of columns."""
        xp = self.xp; host_in = isinstance(Bc, np.ndarray); Bc = xp.asarray(Bc)
        Bs = xp.concatenate([Bc.real, Bc.imag], axis=1)
        s_save = self.s; self.s = xp.concatenate([s_save, s_save])
        coarse_save = self.coarse_lu; self.coarse_lu = xp.concatenate([coarse_save, coarse_save], axis=0)
        sing = self.singular_k0
        # the imaginary part of mode 0 is zero for a real field; keep the mean-projection on column 0 only
        X0s = None if X0 is None else xp.concatenate([xp.asarray(X0).real, xp.asarray(X0).imag], axis=1)
        X = self.solve(Bs, X0s, **kw)
        self.s = s_save; self.coarse_lu = coarse_save; self.singular_k0 = sing
        out = X[:, :self.nk] + 1j * X[:, self.nk:]
        return (out.get() if (host_in and hasattr(out, "get")) else out)
