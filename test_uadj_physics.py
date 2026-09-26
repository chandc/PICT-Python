"""Physical tests of reference/unstructured_adjoint_plan.md section 7.4 (P1, P2 so far).

Each case carries an A criterion (the adjoint equals the discrete derivative) and a P criterion
(the gradient means what the physics says).

P1  Plane Poiseuille driven by a body force f, from rest to steady state. The discrete steady
    problem is linear with u = (f/nu) g_h(y), so Q_h = sum(u V)/L_x scales EXACTLY as f/nu:
        dQ/df = Q/f,   dQ/dnu = -Q/nu        (exact at the discrete level, once converged in time)
    and against the continuum, Q = 2f/(3 nu): dQ/df -> 2/(3 nu) at second order in h.
    (The plan said the FV parabola is exact on uniform quads; it is not: the wall face is a
    one-sided half-cell difference, and test_upoiseuille.py measures order 2.00 with a nonzero
    error. The P criterion is therefore a convergence rate, and the exact check is the scaling.)

P2  2D Taylor-Green decay on the doubly periodic box [0, 2 pi]^2: E(t) = E0 exp(-4 nu t), so
        dE(T)/dnu = -4 T E(T).
    The first test of the adjoint THROUGH TIME with an exact answer; P criterion within 1% at 32^2
    and converging at second order.

    python test_uadj_physics.py
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
from src.uadj_step import TorchUPISO

torch.set_default_dtype(torch.float64)
FAILS = []


def check(name, val, tol, lower=False):
    ok = (val >= tol) if lower else (val <= tol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:66s} {val:.3e}  ({'>=' if lower else '<='} {tol:.2g})", flush=True)
    if not ok:
        FAILS.append(name)


def roll(T, st, n):
    for _ in range(n):
        st = T.step(st)
    return st


def grad_and_fd(T, f, x0, h, sweep=(1.0, 1e-1, 1e-2)):
    """Adjoint derivative of scalar f at scalar x0, and the best 4th-order central FD over a step
    sweep h * sweep, on the recorded branch (ties straddled; see uadj_ops.st_mask)."""
    x = torch.tensor(float(x0), requires_grad=True)
    T.record(); y = f(x); g = float(torch.autograd.grad(y, x)[0])
    best = None
    for k in sweep:
        hh = h * k
        vals = []
        for sg in (2, 1, -1, -2):
            T.replay()
            with torch.no_grad():
                vals.append(float(f(torch.tensor(float(x0) + sg * hh))))
        fd = (8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * hh)
        if best is None or abs(fd - g) < abs(best - g):
            best = fd
    T.live()
    return float(y.detach()), g, best


# ---------------------------------------------------------------------------------------------
def poiseuille(ny, nu=0.1, f=0.2, dt=0.5, nsteps=200, nx=4):
    LX = 1.0
    m = rect_mesh(nx, ny, 0.0, LX, -1.0, 1.0, cells="quad")
    m.make_periodic(1, 2, (LX, 0.0))
    kd = np.full(m.nbface, DIRICHLET)
    s = PISO(m, nu=nu, dt=dt, bc_u=BC(m, kd), bc_v=BC(m, kd.copy()), bc_p=BC(m, np.full(m.nbface, NEUMANN)),
             n_corr=2, n_nonorth=2, scheme="central", body_force=(np.full(m.ncell, f), np.zeros(m.ncell)))
    T = TorchUPISO(s)
    st0 = T.state_from_solver()                    # from rest
    vol = T.tm.vol

    def Q_of(nu_=None, f_=None):
        T.nu_t = torch.as_tensor(nu if nu_ is None else nu_)
        T.fx = (torch.full((m.ncell,), f) if f_ is None else f_ * torch.ones(m.ncell))
        out = roll(T, dict(st0), nsteps)
        return (out["u"] * vol).sum() / (LX * m.span)
    Q, gnu, fdnu = grad_and_fd(T, lambda x: Q_of(nu_=x), nu, 1e-3 * nu)
    _, gf, fdf = grad_and_fd(T, lambda x: Q_of(f_=x), f, 1e-3 * f)
    T.nu_t = torch.as_tensor(nu); T.fx = torch.full((m.ncell,), f)
    return Q, gnu, fdnu, gf, fdf


def run_p1():
    print("  P1 plane Poiseuille: flow-rate sensitivities", flush=True)
    nu, f = 0.1, 0.2
    res = {}
    for ny in (8, 16, 32):
        Q, gnu, fdnu, gf, fdf = poiseuille(ny, nu, f)
        res[ny] = (Q, gnu, gf)
        print(f"      ny={ny:3d}  Q {Q:.10f} (exact {2 * f / (3 * nu):.10f})  dQ/dnu {gnu:+.8e}  dQ/df {gf:.8e}")
        check(f"P1-A ny={ny}: dQ/dnu adjoint vs FD", abs(gnu - fdnu) / abs(fdnu), 1e-7)
        check(f"P1-A ny={ny}: dQ/df adjoint vs FD", abs(gf - fdf) / abs(fdf), 1e-7)
        check(f"P1-P ny={ny}: dQ/dnu == -Q/nu (discrete scaling, exact)", abs(gnu + Q / nu) / (Q / nu), 1e-9)
        check(f"P1-P ny={ny}: dQ/df == Q/f (discrete linearity, exact)", abs(gf - Q / f) / (Q / f), 1e-9)
    ex = 2.0 / (3.0 * nu)
    e = {ny: abs(res[ny][2] - ex) / ex for ny in res}
    print(f"      dQ/df vs continuum 2/(3 nu): errors {e[8]:.2e} {e[16]:.2e} {e[32]:.2e}")
    # The gradient's continuum error must be the forward solver's own, and add nothing: dQ/df = Q/f
    # exactly, so its relative error equals Q's. (First posed as "<= 1e-3 at ny = 32", which failed
    # at 1.95e-3: that is the forward discretisation error 2/ny^2 of Q itself, from the one-sided
    # wall face. Recorded in reference/unstructured_adjoint.md, finding 10.)
    eQ = {ny: abs(res[ny][0] - 2 * f / (3 * nu)) / (2 * f / (3 * nu)) for ny in res}
    check("P1-P dQ/df continuum error == forward Q's continuum error (ny = 32)", abs(e[32] - eQ[32]), 1e-9)
    check("P1-P dQ/df vs continuum: convergence order 16 -> 32", np.log2(e[16] / e[32]), 1.8, lower=True)


# ---------------------------------------------------------------------------------------------
def taylor_green(n, nu=0.05, T_end=1.0, dt=0.025):
    L = 2 * np.pi
    m = rect_mesh(n, n, 0.0, L, 0.0, L, cells="quad")
    m.make_periodic(1, 2, (L, 0.0)); m.make_periodic(3, 4, (0.0, L))
    nb = m.nbface
    s = PISO(m, nu=nu, dt=dt, bc_u=BC(m, np.full(nb, DIRICHLET)), bc_v=BC(m, np.full(nb, DIRICHLET)),
             bc_p=BC(m, np.full(nb, NEUMANN)), n_corr=2, n_nonorth=2, scheme="central")
    x, y = m.centroid.T
    s.u[:] = np.sin(x) * np.cos(y); s.v[:] = -np.cos(x) * np.sin(y)
    s.p[:] = 0.25 * (np.cos(2 * x) + np.cos(2 * y))
    s.init_flux()
    T = TorchUPISO(s)
    st0 = T.state_from_solver()
    nsteps = int(round(T_end / dt))
    vol = T.tm.vol
    Vtot = float(vol.sum())

    def E_of(nu_):
        T.nu_t = torch.as_tensor(nu_)
        out = roll(T, dict(st0), nsteps)
        return 0.5 * ((out["u"] ** 2 + out["v"] ** 2) * vol).sum() / Vtot
    E, g, fd = grad_and_fd(T, E_of, nu, 1e-3 * nu)
    T.nu_t = torch.as_tensor(nu)
    return E, g, fd


def run_p2():
    print("  P2 2D Taylor-Green: dE(T)/dnu = -4 T E(T)", flush=True)
    nu, T_end = 0.05, 1.0
    rel = {}
    # refined at FIXED dt/h, as the plan poses it. (First run at fixed dt = 0.025: order 1.61
    # from 32 to 64, because the O(dt) Rhie-Chow damping floors the error. Finding 11.)
    for n in (16, 32, 64):
        t0 = time.time()
        E, g, fd = taylor_green(n, nu, T_end, dt=0.04 * 16 / n)
        exact_rel = -4 * T_end * E          # the law applied to the solver's own E(T)
        E_exact = 0.25 * np.exp(-4 * nu * T_end)
        rel[n] = abs(g - exact_rel) / abs(exact_rel)
        print(f"      n={n:3d}  E(T) {E:.8f} (exact {E_exact:.8f})  dE/dnu {g:+.6e}  -4TE {exact_rel:+.6e}  ({time.time() - t0:.0f}s)")
        check(f"P2-A n={n}: dE/dnu adjoint vs FD, {int(round(T_end / (0.04 * 16 / n)))} steps", abs(g - fd) / abs(fd), 1e-7)
    check("P2-P dE/dnu vs -4 T E(T) at n = 32", rel[32], 1e-2)
    check("P2-P convergence order 32 -> 64", np.log2(rel[32] / rel[64]), 1.8, lower=True)


# ---------------------------------------------------------------------------------------------
def run_p5_p6():
    """P5: exact symmetry zeros at the symmetric steady Re 40 wake (mirror-snapped butterfly).
    Linear perturbation about a mirror-symmetric state: a symmetric input (both jets blowing) moves
    only symmetric outputs (C_D), an antisymmetric one (opposing jets, rotation) only antisymmetric
    outputs (C_L). So dC_L/da_sym = dC_D/da_anti = dC_D/domega = 0, resolution-independent.
    P6 (part): over 10 convective times, steady symmetric blowing through the +-90 degree slots must
    RAISE the drag and suction lower it -- the sign that makes HydroGym's shipped jet cylinder
    solvable by constant suction (record section 43)."""
    from src.uadj_cases import symmetric_steady_cylinder
    from src.uadj_control import WallForces, SlotJets, Rotation
    from src.uadj_replay import replay_grad
    print("  P5 symmetric steady wake, Re 40: exact symmetry zeros", flush=True)
    s = symmetric_steady_cylinder()
    T = TorchUPISO(s)
    base = T.state_from_solver()
    W = WallForces(T, s.wall_faces)
    jets = SlotJets.cylinder(T, s.wall_faces, (90.0, -90.0), 10.0)
    rot = Rotation(T, s.wall_faces)
    one, alt = torch.tensor([1.0, 1.0]), torch.tensor([1.0, -1.0])
    for N in (2, 20):
        q = torch.zeros(3, requires_grad=True)            # (a_sym, a_anti, omega)
        # rotation first: it writes every wall face, and the jets then overwrite their slots
        st = jets.apply(rot.apply(base, q[2]), q[0] * one + q[1] * alt)
        out = roll(T, st, N)
        cd, cl = W(out)
        gD = torch.autograd.grad(cd, q, retain_graph=True)[0]
        gL = torch.autograd.grad(cl, q)[0]
        print(f"      N={N:2d}  dC_D/d(a_sym, a_anti, omega) = {gD[0]:+.4e} {gD[1]:+.3e} {gD[2]:+.3e}")
        print(f"            dC_L/d(a_sym, a_anti, omega) = {gL[0]:+.3e} {gL[1]:+.4e} {gL[2]:+.4e}")
        check(f"P5-P N={N}: dC_L/da_sym   / dC_L/da_anti", abs(float(gL[0] / gL[1])), 1e-10)
        check(f"P5-P N={N}: dC_D/da_anti  / dC_D/da_sym", abs(float(gD[1] / gD[0])), 1e-10)
        check(f"P5-P N={N}: dC_D/domega   / dC_L/domega", abs(float(gD[2] / gL[2])), 1e-10)
    # A criteria on the non-zero ones, N = 2
    for label, f in (("dC_D/da_sym", lambda x: W(roll(T, jets.apply(base, x * one), 2))[0]),
                     ("dC_L/da_anti", lambda x: W(roll(T, jets.apply(base, x * alt), 2))[1]),
                     ("dC_L/domega", lambda x: W(roll(T, rot.apply(base, x), 2))[1])):
        _, g, fd = grad_and_fd(T, f, 0.0, 1e-4)
        check(f"P5-A N=2: {label} adjoint vs FD", abs(g - fd) / abs(fd), 1e-7)

    print("  P6 steady symmetric blowing/suction over 10 convective times (500 steps, replay)", flush=True)
    N = 500
    apply = lambda st, a: jets.apply(st, a * one)
    final = lambda st: W(st)[0]
    t0 = time.time()
    _, ga = replay_grad(T, base, [0.0] * N, apply, final_loss=final)
    g = float(sum(ga))                                   # the same a at every step
    tr = time.time() - t0
    def cd_after(a):
        st = dict(base)
        for _ in range(N):
            st = T.step(apply(st, a))
        return W(st)[0]
    # FD on the recorded branch (500 steps of masks), 4th order, swept: the same harness as P5-A
    _, _, fd = grad_and_fd(T, cd_after, 0.0, 1e-3, sweep=(1.0, 1e-1))
    print(f"      dC_D(T=10)/da_sym: adjoint {g:+.6e}  FD {fd:+.6e}  (replay {tr:.0f}s)")
    check("P6-A dC_D/da_sym over 500 steps, replay adjoint vs FD", abs(g - fd) / abs(fd), 1e-5)
    check("P6-P sign: blowing raises drag, suction lowers it (dC_D/da_sym > 0)", g, 0.0, lower=True)


# ---------------------------------------------------------------------------------------------
def run_p4(ny=100, T_end=100.0, t_fit=30.0, every=20):
    """P4: sensitivity of the Tollmien-Schlichting growth rate to viscosity, T8's case (Re 7500,
    alpha 1, 48 x ny quads, periodic x, BDF2, dt 0.05) and T8's measurement: growth = LSQ slope of
    ln|a(t)| over t >= 30, a(t) the projection of the first streamwise Fourier mode on its initial
    shape. The slope is a fixed linear combination of the sampled ln|a|, so it is differentiable;
    nu enters as a per-step 'action' of the replay (with f = 2 nu, so the base flow stays 1 - y^2).
    Reference: d sigma/d nu = -Re^2 d sigma/d Re from the Chebyshev OS solver by central difference."""
    import os
    os.environ.setdefault("OS_NX", "48"); os.environ.setdefault("OS_DT", "0.05")
    import test_uorr_sommerfeld as OS
    from orr_sommerfeld import least_stable
    from src.uadj_replay import replay_grad
    print(f"  P4 Orr-Sommerfeld growth-rate sensitivity: Re 7500, 48 x {ny}, T = {T_end}", flush=True)
    s, m, idx, c = OS.build(ny)
    s.init_flux()
    T = TorchUPISO(s)
    st0 = T.state_from_solver()
    dt = s.dt; n = int(round(T_end / dt)); nu0 = s.nu
    idx_t = torch.as_tensor(idx)
    NX = idx.shape[0]
    xc = (np.arange(NX) + 0.5) * (OS.LX / NX)
    cw, sw = torch.as_tensor(np.cos(OS.ALPHA * xc))[:, None], torch.as_tensor(-np.sin(OS.ALPHA * xc))[:, None]

    def amp(st):
        out = []
        for f in ("u", "v"):
            F = st[f][idx_t]
            Fp = F - F.mean(dim=0, keepdim=True)
            out += [(Fp * cw).mean(dim=0), (Fp * sw).mean(dim=0)]      # Re, Im of e^{-i alpha x} mode
        return out
    A0 = [a.detach() for a in amp(st0)]
    den = sum((a * a).sum() for a in A0)

    def lna(st):
        A = amp(st)
        re = sum((A0[2 * q] * A[2 * q] + A0[2 * q + 1] * A[2 * q + 1]).sum() for q in (0, 1)) / den
        im = sum((A0[2 * q] * A[2 * q + 1] - A0[2 * q + 1] * A[2 * q]).sum() for q in (0, 1)) / den
        return 0.5 * torch.log(re * re + im * im)
    ts = np.array([(k + 1) * dt for k in range(n) if (k + 1) % every == 0])
    sel = ts >= t_fit
    tt = ts[sel]; wts = (tt - tt.mean()) / ((tt - tt.mean()) ** 2).sum()   # LSQ slope weights
    wmap = {int(round(t / dt)) - 1: float(w) for t, w in zip(tt, wts)}

    def step_loss(st, k):
        return wmap[k] * lna(st) if k in wmap else torch.zeros((), dtype=torch.float64)

    def apply(st, nu_):
        T.nu_t = nu_; T.fx = 2.0 * nu_ * torch.ones(m.ncell)
        return st
    t0 = time.time()
    growth, ga = replay_grad(T, st0, [nu0] * n, apply, step_loss=step_loss)
    g_ad = float(sum(ga))
    t_ad = time.time() - t0

    def growth_of(nu_):
        with torch.no_grad():
            st, acc = dict(st0), 0.0
            for k in range(n):
                st = T.step(apply(st, torch.tensor(nu_)))
                acc += float(step_loss(st, k))
        return acc
    h = 1e-3 * nu0
    fd = (growth_of(nu0 + h) - growth_of(nu0 - h)) / (2 * h)
    Re = 1.0 / nu0; dRe = 1.0
    sig = lambda R: float(OS.ALPHA * least_stable(R, OS.ALPHA, 120)[0].imag)
    dsig_dnu = -Re ** 2 * (sig(Re + dRe) - sig(Re - dRe)) / (2 * dRe)
    print(f"      growth {growth:.6f} (OS {OS.G_REF})   d growth/d nu: adjoint {g_ad:+.5e}  FD {fd:+.5e}  "
          f"OS {dsig_dnu:+.5e}   (adjoint by replay {t_ad:.0f}s, {n} steps)")
    check(f"P4-A ny={ny}: d growth/d nu, replay adjoint vs FD (live, 2-point)", abs(g_ad - fd) / abs(fd), 1e-4)
    check(f"P4-P ny={ny}: d growth/d nu vs Orr-Sommerfeld", abs(g_ad - dsig_dnu) / abs(dsig_dnu), 5e-2)
    return g_ad, dsig_dnu


if __name__ == "__main__":
    t0 = time.time()
    if "--only" in sys.argv:
        globals()[sys.argv[sys.argv.index("--only") + 1]]()
        print(f"\n  {len(FAILS)} failure(s), {time.time() - t0:.0f}s"); sys.exit(1 if FAILS else 0)
    run_p1()
    run_p2()
    run_p5_p6()
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t0:.0f}s" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
