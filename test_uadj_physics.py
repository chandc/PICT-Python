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


def grad_and_fd(T, f, x0, h):
    """Adjoint derivative of scalar f at scalar x0, and a 4th-order central FD on the recorded branch."""
    x = torch.tensor(float(x0), requires_grad=True)
    T.record(); y = f(x); g = float(torch.autograd.grad(y, x)[0])
    vals = []
    for sg in (2, 1, -1, -2):
        T.replay()
        with torch.no_grad():
            vals.append(float(f(torch.tensor(float(x0) + sg * h))))
    T.live()
    fd = (8 * (vals[1] - vals[2]) - (vals[0] - vals[3])) / (12 * h)
    return float(y.detach()), g, fd


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


if __name__ == "__main__":
    t0 = time.time()
    run_p1()
    run_p2()
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t0:.0f}s" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
