"""Gates U1-U3 of reference/unstructured_adjoint_plan.md: operators and solves in torch.

A1  constant operators (both gradients) == production, random fields
A2  adjoint identity <y, A x> == <A^T y, x> for the constant operators
A3  dual Green-Gauss gradient is the exact adjoint of the interpolated-flux divergence
A4  measured linearity: convection in F (fixed mask), pressure Laplacian and its deferred term in
    gam, Rhie-Chow flux in p (fixed aP, fixed history)
A5  solution-dependent operators == production at a developed state: convection matrix and its
    deferred right-hand side, pressure Laplacian and its deferred right-hand side
A6  LU solve: dL/db and dL/dA == FD, non-symmetric (momentum), symmetric (pressure), singular
A7  residual certificate raises on a solve that does not solve

    python test_uadj_ops.py
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.uadj_cases import CASES
from src.uadj_step import TorchUPISO
from src.uadj_ops import Masks, convection_vals, convection_rhs, laplacian_vals, laplacian_rhs, _t
from src.uadj_solve import LUFactor, lu_solve, SolveFailed
from src.uops import convection, laplacian, divergence

torch.set_default_dtype(torch.float64)
FAILS = []


def check(name, val, tol):
    ok = val <= tol
    print(f"    {'PASS' if ok else 'FAIL'}  {name:60s} {val:.2e}  (<= {tol:.0e})")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def run(case):
    s = CASES[case]()
    T = TorchUPISO(s)
    m, tm, pat = s.m, T.tm, T.pat
    rng = np.random.default_rng(1)
    phi, phib = rng.standard_normal(m.ncell), rng.standard_normal(m.nbface)

    # A1 / A2 -- constant operators
    e1 = e2 = 0.0
    for name, G, TG in (("lsq", s.grad, T.grad), ("ggdual", s.grad_p, T.grad_p)):
        e1 = max(e1, rel(TG(_t(phi), _t(phib)).numpy(), G(phi, phib)))
        for M in (TG.Gx, TG.Gy, TG.Bx, TG.By):
            x = torch.randn(M.shape[1]); y = torch.randn(M.shape[0])
            # normalised by sum |y_i A_ij x_j|: the scale of the terms being summed, not of their
            # (possibly cancelling) total
            scale = float((y[M.rows].abs() * M.vals.abs() * x[M.cols].abs()).sum())
            if scale > 0:
                e2 = max(e2, abs(float(y @ (M @ x)) - float(M.rmatvec(y) @ x)) / scale)
    check(f"A1 {case}: gradients (LSQ, dual GG) torch vs production", e1, 1e-14)
    check(f"A2 {case}: <y, A x> == <A^T y, x>, gradient matrices", e2, 1e-13)

    # A3 -- duality: sum V u.(G p) + sum V p (D u) == 0. Every interior face contributes
    # (u_O p_O - u_N p_N).S_f; it telescopes by cell closure (sum of a cell's S_f = 0) once the
    # boundary faces carry p_b = p_owner in the gradient and zero flux in the divergence -- the
    # all-Neumann pressure / no-penetration pairing the projection actually uses.
    u, v, p = (torch.randn(m.ncell) for _ in range(3))
    zb = torch.zeros(m.nbface)
    Gp = T.grad_p(p, p[tm.o_bf])
    uf = tm.interp(u, zb); vf = tm.interp(v, zb)
    Fu = (uf * tm.normal[:, 0] + vf * tm.normal[:, 1]) * tm.span
    a = float((tm.vol * (u * Gp[:, 0] + v * Gp[:, 1])).sum()); b = float((tm.vol * p * tm.divergence(Fu)).sum())
    check(f"A3 {case}: dual GG duality, relative", abs(a + b) / max(abs(a), abs(b)), 1e-13)

    # A5 -- convection matrix and deferred rhs vs production at the developed flux
    F = s.Ff.copy()
    Cp, Crhs = convection(m, F, s.bc_u.kind, scheme="central")
    Cv, pos = convection_vals(tm, pat, _t(F), Masks())
    e = rel(pat.to_scipy(Cv).toarray(), Cp.toarray())
    gphi = s.grad(phi, phib)
    er = rel(convection_rhs(tm, _t(F), pos, _t(phi), _t(phib), _t(gphi), "central").numpy(), Crhs(phi, phib, gphi))
    check(f"A5 {case}: convection matrix torch vs production", e, 1e-14)
    check(f"A5 {case}: convection deferred rhs torch vs production", er, 1e-14)
    gam = np.abs(rng.standard_normal(m.nface)) + 0.1
    Ap, Aprhs = laplacian(m, gam, s.bc_p.kind)
    Lv = laplacian_vals(tm, pat, _t(gam), T.dir_p)
    check(f"A5 {case}: pressure Laplacian torch vs production", rel(pat.to_scipy(Lv).toarray(), Ap.toarray()), 1e-14)
    lr = laplacian_rhs(tm, _t(gam), T.dir_p, _t(gphi), _t(phib)).numpy()
    check(f"A5 {case}: Laplacian deferred rhs torch vs production", rel(lr, Aprhs(gphi, phib)), 1e-14)

    # A4 -- linearity
    Mk = Masks("record"); convection_vals(tm, pat, _t(F), Mk)
    F1, F2 = _t(F) + 0.1 * torch.randn(m.nface), _t(F) + 0.1 * torch.randn(m.nface)
    def cv(x):
        M = Mk.replay(); M.straddle = False        # fully pinned: ties included
        return convection_vals(tm, pat, x, M)[0]
    lin = cv(2.0 * F1 - 3.0 * F2) - 2.0 * cv(F1) + 3.0 * cv(F2)
    check(f"A4 {case}: convection values linear in F (fixed mask)", float(lin.abs().max() / cv(F1).abs().max()), 1e-14)
    g1, g2 = _t(gam), _t(np.abs(rng.standard_normal(m.nface)))
    def lv(x): return laplacian_vals(tm, pat, x, T.dir_p)
    lin = lv(2 * g1 - 3 * g2) - 2 * lv(g1) + 3 * lv(g2)
    check(f"A4 {case}: pressure Laplacian linear in gam", float(lin.abs().max() / lv(g1).abs().max()), 1e-14)
    st = T.state_from_solver()
    aP = torch.rand(m.ncell) + 1.0
    def rc(pp): return T._rhie_chow(st["u"], st["v"], aP, pp, T.grad(pp, torch.zeros(m.nbface)), st)[0]
    p1, p2 = torch.randn(m.ncell), torch.randn(m.ncell)
    base = rc(torch.zeros(m.ncell))
    lin = (rc(2 * p1 - 3 * p2) - base) - 2 * (rc(p1) - base) + 3 * (rc(p2) - base)
    check(f"A4 {case}: Rhie-Chow flux affine in p (fixed aP)", float(lin.abs().max() / (rc(p1) - base).abs().max()), 1e-13)

    # A6 -- LU solve gradients: momentum-like (non-symmetric), pressure (symmetric), singular
    Amom = pat.zeros().index_add(0, pat.p_diag, tm.vol / s.dt) + Cv - laplacian_vals(tm, pat, T.nu_t.expand(tm.nface), T.dir_u)
    Apre = lv(g1)
    for label, A, sing in (("momentum", Amom, False), ("pressure", Apre, T.p_singular)):
        vals = A.clone().requires_grad_(True)
        bb = torch.randn(m.ncell)
        if sing:
            bb = bb - bb.mean()
        b = bb.clone().requires_grad_(True)
        w = torch.randn(m.ncell)
        L = (w * lu_solve(vals, b, pat, singular=sing)).sum()
        gA, gb = torch.autograd.grad(L, (vals, b))
        dA = torch.randn(pat.nnz) * vals.detach().abs().mean(); db = torch.randn(m.ncell)
        if sing:
            db = db - db.mean()
        errs = []
        for which, d, g in (("A", dA, gA), ("b", db, gb)):
            best = 1.0
            for h in (1e-3, 1e-4, 1e-5):
                # fourth-order central difference: the solve is smooth, and the second-order
                # stencil's h^2 truncation sat at 3e-8 on the Dirichlet pressure operator
                vp = [float((w * lu_solve(vals.detach() + sg * h * d if which == "A" else vals.detach(),
                                          b.detach() + sg * h * d if which == "b" else b.detach(),
                                          pat, singular=sing)).sum()) for sg in (2, 1, -1, -2)]
                fd = (8 * (vp[1] - vp[2]) - (vp[0] - vp[3])) / (12 * h); ad = float((g * d).sum())
                best = min(best, abs(fd - ad) / max(abs(fd), abs(ad)))
            errs.append(best)
        check(f"A6 {case}: LU solve {label}{' (singular)' if sing else ''}, dL/dA and dL/db vs FD", max(errs), 1e-8)

    # A7 -- a factor from different values must fail its residual check, loudly
    f_bad = LUFactor(pat, Amom * 1.5)
    try:
        f_bad.A = pat.to_scipy(Amom); f_bad.solve(np.ones(m.ncell))
        check(f"A7 {case}: residual certificate raises on a wrong factor", 1.0, 0.0)
    except SolveFailed:
        check(f"A7 {case}: residual certificate raises on a wrong factor", 0.0, 0.0)


if __name__ == "__main__":
    t0 = time.time()
    for c in [a for a in sys.argv[1:]] or list(CASES):
        print(f"  {c}")
        run(c)
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t0:.0f}s" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
