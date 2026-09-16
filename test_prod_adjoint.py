"""Stage 9.1 gates -- differentiable PRODUCTION momentum assembly.

The path no chain has certified: gradient flow THROUGH matrix assembly.
On the coarse butterfly (the production topology):

  9.1a  exactness/linearity: vals0 + T x reproduces the production
        assembler on random velocity fields to ~1e-12 relative. This
        MEASURES the affine claim (central convection); a hidden sign
        branch or limiter fails here.
  9.1b  pattern stability: the sparsity pattern is field-independent
        (asserted inside values(); gated on a second random field).
  9.1c  autograd through assembly: d<w, A(x) y>/dx vs FD.
  9.1d  through the SOLVE: d ||A(x)^{-1} b||^2 / dx via LinearSolve's
        matrix-values gradient chained with T -- assembly -> matrix ->
        solve -> loss, the Stage 9 spine in miniature.

Run:  .venv/bin/python test_prod_adjoint.py     (~5 min; T is cached)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cylinder_rect_bc import classify
from cylinder_ring_grid import ring_rect_domain
from src.adjoint_piso import LinearSolve
from src.prod_adjoint import MomentumAssembly, TorchMomentumAssembly

torch.set_default_dtype(torch.float64)
NU, DT = 0.01, 0.01                     # production values (Re 100, D=1, U=1)
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}", flush=True)


def main():
    d, _ = ring_rect_domain(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                            wake_hold=8.0, wake_ratio=1.10)
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    for (b, fid), role in classify(d).items():
        d.blocks[b].faces[fid] = kindmap[role]

    t0 = time.time()
    asm = MomentumAssembly(d, NU, DT, verbose=True)
    print(f"  butterfly {d.n_cells:,} cells; T {asm.T.shape} nnz {asm.T.nnz:,} "
          f"({asm.ncolors} colors, {time.time()-t0:.0f}s incl. cache)", flush=True)

    rng = np.random.default_rng(0)
    N = asm.N

    # ------------------------------------------------------------------ 9.1a
    worst = 0.0
    for _ in range(3):
        x = rng.standard_normal(3 * N)
        ref = asm.values(x)
        pred = asm.vals0 + asm.T @ x
        worst = max(worst, np.abs(pred - ref).max() / np.abs(ref).max())
    check(worst < 1e-12, f"9.1a exact affine assembly on 3 random fields: "
                         f"worst rel {worst:.2e}")

    # ------------------------------------------------------------------ 9.1b
    try:
        asm.values(rng.standard_normal(3 * N))
        ok = True
    except AssertionError:
        ok = False
    check(ok, "9.1b sparsity pattern is field-independent")

    # ------------------------------------------------------------------ 9.1c
    tasm = TorchMomentumAssembly(asm)
    rows, cols = asm.idx
    w = torch.as_tensor(rng.standard_normal(asm.shape[0]))
    y = torch.as_tensor(rng.standard_normal(asm.shape[1]))
    wy = (w[torch.as_tensor(np.asarray(rows, dtype=np.int64))]
          * y[torch.as_tensor(np.asarray(cols, dtype=np.int64))])   # <w, A y>

    x = torch.tensor(rng.standard_normal(3 * N), requires_grad=True)
    (tasm.vals(x) * wy).sum().backward()
    g = x.grad.detach().numpy()
    ks = np.argsort(-np.abs(g))[:3]
    h, worst = 1e-3, 0.0
    x0 = x.detach().numpy()
    for k in ks:
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        Lp = float(((asm.vals0 + asm.T @ xp) * wy.numpy()).sum())
        Lm = float(((asm.vals0 + asm.T @ xm) * wy.numpy()).sum())
        worst = max(worst, abs((Lp - Lm) / (2 * h) - g[k]) / max(abs(g[k]), 1e-300))
    check(worst < 1e-9, f"9.1c autograd of <w, A(x) y> vs FD: worst rel {worst:.2e}")

    # ------------------------------------------------------------------ 9.1d
    b_rhs = torch.as_tensor(rng.standard_normal(asm.shape[0]))
    pattern = (asm.idx, asm.shape)

    def loss(xt):
        sol = LinearSolve.apply(tasm.vals(xt), b_rhs, pattern, False, False)
        return (sol ** 2).sum()

    x = torch.tensor(0.05 * rng.standard_normal(3 * N), requires_grad=True)
    L = loss(x)
    L.backward()
    g = x.grad.detach().numpy()
    ks = np.argsort(-np.abs(g))[:2]
    h, worst = 1e-4, 0.0
    x0 = x.detach().numpy()
    with torch.no_grad():
        for k in ks:
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            Lp = float(loss(torch.as_tensor(xp)))
            Lm = float(loss(torch.as_tensor(xm)))
            worst = max(worst, abs((Lp - Lm) / (2 * h) - g[k]) / max(abs(g[k]), 1e-300))
    check(worst < 1e-4,
          f"9.1d assembly -> solve -> loss gradient vs FD: worst rel {worst:.2e} "
          f"(L = {float(L):.4e})")

    # ================================================================== 9.2a
    from src.prod_adjoint import TorchFluxKernels
    fk = TorchFluxKernels(d)
    nb = len(d.blocks)
    xu = torch.as_tensor(rng.standard_normal(N))
    xv = torch.as_tensor(rng.standard_normal(N))
    xw = torch.as_tensor(rng.standard_normal(N))
    fu = {b: xu.numpy()[d.global_ids(b)] for b in range(nb)}
    fv = {b: xv.numpy()[d.global_ids(b)] for b in range(nb)}
    fw = {b: xw.numpy()[d.global_ids(b)] for b in range(nb)}
    with torch.no_grad():
        Ft = fk.face_fluxes_all(xu, xv, xw)
    worst_f = worst_d = 0.0
    for b in range(nb):
        Fr = d.face_fluxes(b, fu, fv, fw)
        for a in range(3):
            sc = max(np.abs(Fr[a]).max(), 1e-300)
            worst_f = max(worst_f, np.abs(Ft[b][a].numpy() - Fr[a]).max() / sc)
        dr = d.divergence(b, Fr, asm.Js[b])
        with torch.no_grad():
            dt_ = fk.divergence(b, Ft[b]).numpy()
        worst_d = max(worst_d, np.abs(dt_ - dr).max()
                      / max(np.abs(dr).max(), 1e-300))
    check(worst_f < 1e-13, f"9.2a torch face_fluxes == production on all "
                           f"{nb} blocks: worst rel {worst_f:.2e}")
    check(worst_d < 1e-13, f"9.2a torch divergence == production: "
                           f"worst rel {worst_d:.2e}")

    # autograd liveness through the flux path (linear -> FD exact)
    xg = torch.tensor(rng.standard_normal(N), requires_grad=True)
    L = sum((fk.divergence(b, Fb) ** 2).sum()
            for b, Fb in enumerate(fk.face_fluxes_all(xg, xv, xw)))
    L.backward()
    g = xg.grad.detach().numpy()
    k = int(np.argmax(np.abs(g)))
    h2 = 1e-4
    with torch.no_grad():
        def LL(val):
            xp = xg.detach().clone()
            xp[k] = val
            return float(sum((fk.divergence(b, Fb) ** 2).sum()
                             for b, Fb in enumerate(fk.face_fluxes_all(xp, xv, xw))))
        fd = (LL(float(xg[k]) + h2) - LL(float(xg[k]) - h2)) / (2 * h2)
    rel = abs(g[k] - fd) / max(abs(fd), 1e-300)
    check(rel < 1e-7, f"9.2a gradient through pad -> flux -> divergence: "
                      f"FD rel {rel:.2e}")

    # ================================================================== 9.2b
    from src.multiblock import face_axis_side
    from src.prod_adjoint import TorchPressureFlux
    # stamp pressure_pinned exactly as _step_impl does (dong outflow faces)
    dong_faces = [(bb, fid) for (bb, fid), role in classify(d).items()
                  if role == "outlet"]
    d.pressure_pinned = frozenset(
        (bb,) + face_axis_side(fid) for bb, fid in dong_faces)
    pf = TorchPressureFlux(d, fk, with_cross=True)
    pg = torch.as_tensor(rng.standard_normal(N))
    cg = torch.as_tensor(0.5 + 0.1 * rng.random(N))       # positive coefficient
    pd_ = {b: pg.numpy()[d.global_ids(b)] for b in range(nb)}
    cd_ = {b: cg.numpy()[d.global_ids(b)] for b in range(nb)}
    worst = {"rc": 0.0, "corr": 0.0, "cross": 0.0}
    for b in range(nb):
        for tag, kw in (("rc", dict(include_orth=True, include_cross=False,
                                    rhie_chow=True)),
                        ("corr", dict(include_orth=True, include_cross=True)),
                        ("cross", dict(include_orth=False, include_cross=True))):
            ref = d.pressure_face_fluxes(b, pd_, cd_[b], cd_, **kw)
            with torch.no_grad():
                got = pf(b, pg, cg, **kw)
            for a in range(3):
                sc = max(np.abs(ref[a]).max(), 1e-300)
                worst[tag] = max(worst[tag],
                                 np.abs(got[a].numpy() - ref[a]).max() / sc)
    for tag, label in (("rc", "rhie_chow dissipation"),
                       ("corr", "orth + cross correction"),
                       ("cross", "cross only")):
        check(worst[tag] < 1e-13,
              f"9.2b torch pressure_face_fluxes [{label}]: worst rel {worst[tag]:.2e}")

    # bilinear gradient: d/dp and d/dcoef of a flux functional, FD-checked
    wsum = [[torch.as_tensor(rng.standard_normal(
        [s + (1 if a == ax else 0) for ax, s in enumerate(d.blocks[b].shape)]))
        for a in range(3)] for b in range(nb)]

    def flux_L(pt, ct):
        return sum((pf(b, pt, ct, include_orth=True, include_cross=True,
                       rhie_chow=True)[a] * wsum[b][a]).sum()
                   for b in range(nb) for a in range(3))

    pt = pg.clone().requires_grad_(True)
    ct = cg.clone().requires_grad_(True)
    flux_L(pt, ct).backward()
    h3, worst_g = 1e-5, 0.0
    for var, grad in ((pt, pt.grad), (ct, ct.grad)):
        g = grad.detach().numpy()
        k = int(np.argmax(np.abs(g)))
        with torch.no_grad():
            vp, vm = var.detach().clone(), var.detach().clone()
            vp[k] += h3
            vm[k] -= h3
            args = ((vp, cg) if var is pt else (pg, vp),
                    (vm, cg) if var is pt else (pg, vm))
            fd = (float(flux_L(*args[0])) - float(flux_L(*args[1]))) / (2 * h3)
        worst_g = max(worst_g, abs(g[k] - fd) / max(abs(fd), 1e-300))
    check(worst_g < 1e-6,
          f"9.2b bilinear gradients d/dp and d/dcoef vs FD: worst rel {worst_g:.2e}")

    # ================================================================== 9.2c
    from src.prod_adjoint import DiffusionAssembly, TorchDiffusionAssembly
    # cross_diffusion port vs production
    worst_cd = 0.0
    xf = torch.as_tensor(rng.standard_normal(N))
    fdic = {b: xf.numpy()[d.global_ids(b)] for b in range(nb)}
    for b in range(nb):
        ref = d.cross_diffusion(b, fdic)
        with torch.no_grad():
            got = fk.cross_diffusion(b, xf).numpy()
        worst_cd = max(worst_cd, np.abs(got - ref).max()
                       / max(np.abs(ref).max(), 1e-300))
    check(worst_cd < 1e-13,
          f"9.2c torch cross_diffusion == production, all blocks: "
          f"worst rel {worst_cd:.2e}")

    # M(coef) sensitivity: linearity + gradient through the coefficient
    da = DiffusionAssembly(d)
    worst_m = 0.0
    for _ in range(2):
        cvec = 0.5 + 0.2 * rng.random(N)
        ref = da._values(cvec)
        worst_m = max(worst_m, np.abs(da.Tm @ cvec - ref).max()
                      / np.abs(ref).max())
    check(worst_m < 1e-12,
          f"9.2c M(coef) linear sensitivity ({da.ncolors} colors): "
          f"worst rel {worst_m:.2e}")

    tda = TorchDiffusionAssembly(da)
    ct2 = torch.tensor(0.5 + 0.2 * rng.random(N), requires_grad=True)
    wm = torch.as_tensor(rng.standard_normal(len(da.idx[0])))
    (tda.vals(ct2) * wm).sum().backward()
    g = ct2.grad.detach().numpy()
    k = int(np.argmax(np.abs(g)))
    h4 = 1e-4
    cp, cm = ct2.detach().numpy().copy(), ct2.detach().numpy().copy()
    cp[k] += h4
    cm[k] -= h4
    fd = (float((da.Tm @ cp) @ wm.numpy()) - float((da.Tm @ cm) @ wm.numpy())) / (2 * h4)
    rel = abs(g[k] - fd) / max(abs(fd), 1e-300)
    check(rel < 1e-9, f"9.2c d(M vals)/d(coef) autograd vs FD: rel {rel:.2e}")

    # ================================================================== 9.2d
    # THE DECISIVE GATE: one full torch production step vs m.step(),
    # field for field, from an identical warmed-up state. Both sides run
    # with the DC pressure iteration tightened so the lag truncation is
    # below the linear-solve tolerances being compared.
    os.environ["PICT_CROSS_DC_TOL"] = "1e-11"
    os.environ["PICT_CROSS_DC_ITERS"] = "60"
    # production's DC INNER solves default to 1e-4 (warm-started, so the
    # consecutive-iterate exit test can fire at loose-solve accuracy) and
    # its outer tolerances to 1e-9; tighten so the comparison floor is the
    # solves, not the iteration truncation
    os.environ["PICT_CROSS_DC_INNER"] = "1e-11"
    os.environ["PICT_MOM_TOL"] = "1e-12"
    from cylinder_rect_bc import U_INF, apply as apply_bc
    from cylinder_ring_grid import D as DIAM
    from src.piso_multiblock import MultiBlockPISO
    from src.prod_step import TorchProductionStep

    m = MultiBlockPISO(d, NU, DT, 2, 1e-11, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False,
                       implicit_cross=True, linear_backend="scipy")
    for b in range(nb):
        m.u[b][:] = U_INF
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d)
    t0 = time.time()
    for _ in range(3):
        m.step()                                  # warm-up: u_prev, p_flux live
    print(f"  9.2d warm-up: 3 production steps ({time.time()-t0:.0f}s)",
          flush=True)

    tps = TorchProductionStep(m)
    # the probed production gradient must reproduce d.gradient exactly
    from src.mb_adjoint import spmv as _spmv
    pr = torch.as_tensor(rng.standard_normal(N))
    prd = {b: pr.numpy()[d.global_ids(b)] for b in range(nb)}
    wg = 0.0
    for b in range(nb):
        ref = d.gradient(b, prd)
        for a in range(3):
            got = _spmv(tps.G3[a], pr).numpy()[d.global_ids(b)]
            wg = max(wg, np.abs(got - ref[a]).max()
                     / max(np.abs(ref[a]).max(), 1e-300))
    check(wg < 1e-13, f"9.2d probed production gradient == d.gradient: "
                      f"worst rel {wg:.2e}")
    st = tps.state_from_solver()
    t0 = time.time()
    m.step()
    t_prod = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        st2 = tps.step(st)
    t_torch = time.time() - t0
    worst = {}
    for f, ref in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p),
                   ("p_flux", m.p_flux)):
        r = m._flat(ref)
        g = st2[f].numpy()
        worst[f] = np.abs(g - r).max() / max(np.abs(r).max(), 1e-300)
    wmax = max(worst.values())
    check(wmax < 1e-5,
          f"9.2d torch production step == m.step(): "
          + "  ".join(f"{f} {v:.1e}" for f, v in worst.items())
          + f"  ({t_prod:.0f}s prod / {t_torch:.0f}s torch)")

    # ================================================================== 9.3
    # Gradient gates ON THE PRODUCTION STEP. DC sweep count pinned (a
    # data-dependent early exit would put a kink between the FD probes);
    # Picard runs its full 2 sweeps at these tolerances.
    os.environ["PICT_CROSS_DC_TOL"] = "0"
    os.environ["PICT_CROSS_DC_ITERS"] = "3"
    st0 = tps.state_from_solver()          # developed state, 4 steps in
    wake_w = torch.as_tensor(rng.standard_normal(N))

    def rollout_L(du, src=None, **kw):
        tps.clear_seeds()               # deterministic, state-matched solves
        st = tps._clone(st0)
        st["u"] = st["u"] + du
        st = tps.step(st, src=src, **kw)
        return (wake_w * st["u"]).sum() + (st["p"] ** 2).sum() * 1e-3

    du = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    L0 = rollout_L(du)
    L0.backward()
    g_state = du.grad.detach().numpy().copy()
    print(f"  9.3 forward+backward: {time.time()-t0:.0f}s", flush=True)
    # FD on interior entries only (Dirichlet nodes are re-imposed each step)
    g_int = np.zeros(N)
    g_int[tps.interior] = g_state[tps.interior]
    ks = np.argsort(-np.abs(g_int))[:2]
    h5, worst = 1e-5, 0.0
    with torch.no_grad():
        for k in ks:
            e = torch.zeros(N, dtype=torch.float64)
            e[k] = h5
            fd = (float(rollout_L(e)) - float(rollout_L(-e))) / (2 * h5)
            worst = max(worst, abs(g_state[k] - fd) / max(abs(fd), 1e-300))
    check(worst < 1e-4,
          f"9.3a dL/d(state) through the PRODUCTION step vs FD: "
          f"worst rel {worst:.2e}")

    src0 = [torch.zeros(N, dtype=torch.float64, requires_grad=True)
            for _ in range(3)]
    L = rollout_L(torch.zeros(N, dtype=torch.float64), src=src0)
    L.backward()
    g_src = src0[0].grad.detach().numpy().copy()
    k = int(np.argmax(np.abs(g_src)))
    with torch.no_grad():
        e = torch.zeros(N, dtype=torch.float64)
        e[k] = h5
        zero3 = [torch.zeros(N, dtype=torch.float64) for _ in range(3)]
        Lp = float(rollout_L(torch.zeros(N, dtype=torch.float64),
                             src=[e, zero3[1], zero3[2]]))
        Lm = float(rollout_L(torch.zeros(N, dtype=torch.float64),
                             src=[-e, zero3[1], zero3[2]]))
        fd = (Lp - Lm) / (2 * h5)
    rel = abs(g_src[k] - fd) / max(abs(fd), 1e-300)
    check(rel < 1e-4,
          f"9.3b dL/d(source) (production velocity_source hook) vs FD: "
          f"rel {rel:.2e}")

    # the assembly mangle: the cut path must CHANGE the gradient measurably
    du2 = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    rollout_L(du2, detach_assembly=True).backward()
    gm = du2.grad.detach().numpy()
    change = np.abs(gm - g_state).max() / max(np.abs(g_state).max(), 1e-300)
    check(change > 1e-6, f"9.3c path live: detach_assembly (the path the "
                         f"chains never had): grad change {change:.2e}")

    # the Dong path is IDENTICALLY INERT in forward-flow states: with
    # dong_copy = 1.0 the boundary velocity IS the interior velocity (the
    # viscous term nu(un - un_i)/dn vanishes structurally) and healthy
    # outflow saturates tanh to 1.0 exactly in float64 (theta = 0, killing
    # the |u|^2 term AND its derivative). pv == 0 == production here -- the
    # 6.9 lesson again: the probe must EXERCISE the path. Manufacture
    # backflow at the outlet, then detaching must change the gradient.
    xf_all = np.zeros(N)
    for b in range(nb):
        xf_all[d.global_ids(b).ravel()] = d.blocks[b].x.ravel()
    near_out = torch.as_tensor(np.where(xf_all > 27.0)[0])
    flip = torch.ones(N, dtype=torch.float64)
    flip[near_out] = -1.0

    def rollout_bf(du_, **kw):
        tps.clear_seeds()
        st = tps._clone(st0)
        st["u"] = st["u"] * flip + du_
        st = tps.step(st, **kw)
        return (wake_w * st["u"]).sum() + (st["p"] ** 2).sum() * 1e-3

    g_bf = {}
    for name, kw in (("normal", {}), ("detach", dict(detach_dong=True))):
        du3 = torch.zeros(N, dtype=torch.float64, requires_grad=True)
        rollout_bf(du3, **kw).backward()
        g_bf[name] = du3.grad.detach().numpy()
    change = (np.abs(g_bf["detach"] - g_bf["normal"]).max()
              / max(np.abs(g_bf["normal"]).max(), 1e-300))
    check(change > 1e-9, f"9.3c path live: detach_dong under manufactured "
                         f"outlet backflow: grad change {change:.2e}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
