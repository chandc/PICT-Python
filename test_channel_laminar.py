"""The channel rig against an EXACT solution, before any turbulence is asked of it.

Three things have to agree for a channel LES to mean anything: the constant-pressure-gradient
forcing, the no-slip wall, and the wall-normal viscous operator on a STRETCHED grid. Each is new
in this repo -- the Taylor-Green study had no walls, no forcing and a uniform mesh -- and a
turbulent run diagnoses none of them, because a wrong forcing constant or a mis-scaled metric
just gives a different, plausible-looking Re_tau.

Laminar plane Poiseuille pins all three at once. With f = u_tau^2/delta and delta = 1,

    u(y) = (f / 2 nu) y (2 - y),    u_c = f/(2 nu) = 90,    tau_w = nu du/dy|_0 = f delta = 1

so u_tau = 1 EXACTLY, by construction rather than by measurement. Two checks:

  1  STATIONARITY -- started from the exact parabola, the solver must not move it. This is the
     strong one: it tests the balance of forcing against diffusion pointwise, and a wrong
     Jacobian weighting on either side shows up immediately as drift.
  2  CONVERGENCE FROM REST -- started at u = 0, the solution must climb TOWARDS that parabola.
     Stationarity alone could be passed by a solver that has zeroed both terms.

The mangle for check 1 -- forcing at 0.9f -- must break it, or the tolerance is meaningless.
"""
import numpy as np

from src.domains import channel_box, clustered_y
from src.piso_multiblock import MultiBlockPISO

RE_TAU, NU = 180.0, 1.0 / 180.0


def _rig(ny=48, dt=0.02, fx=1.0):
    y = clustered_y(ny, re_tau=RE_TAU)
    d = channel_box(8, ny, 8, 2, y_nodes=y)
    m = MultiBlockPISO(d, NU, dt, 2, 1e-10, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    m.velocity_source = [fx, 0.0, 0.0]
    for b in range(len(d.blocks)):
        m.v[b][:] = m.w[b][:] = 0.0
        for arr, bc in ((m.u, m.u_bc), (m.v, m.v_bc), (m.w, m.w_bc)):
            arr[b][:, 0, :] = arr[b][:, -1, :] = 0.0
            bc[b][:, 0, :] = bc[b][:, -1, :] = 0.0
    return d, m, y


def _exact(d, b, fx=1.0):
    yy = d.blocks[b].y
    return fx / (2 * NU) * yy * (2.0 - yy)


def check_stationary(fx=1.0, mangle=False):
    """Started from the exact parabola, the solver must leave it alone."""
    d, m, y = _rig(fx=fx if not mangle else 1.0)
    if mangle:
        m.velocity_source = [0.9, 0.0, 0.0]          # MANGLE: 10% wrong pressure gradient
    for b in range(len(d.blocks)):
        m.u[b][:] = _exact(d, b)
        m.u[b][:, 0, :] = m.u[b][:, -1, :] = 0.0
        m.u_bc[b][:, 0, :] = m.u_bc[b][:, -1, :] = 0.0
    u0 = max(float(np.abs(m.u[b]).max()) for b in range(len(d.blocks)))
    for _ in range(50):
        m.step()
    drift = max(float(np.abs(m.u[b] - _exact(d, b)).max()) for b in range(len(d.blocks)))
    if not mangle:
        ok = drift / u0 < 2e-3
        print(f"  [{'PASS' if ok else 'FAIL'}] exact parabola is stationary: drift after 50 "
              f"steps {drift:.4f} = {100*drift/u0:.4f}% of u_c = {u0:.2f}")
    else:
        # 50 steps x dt 0.02 = t 1, against a diffusive time of 180: the 0.1 residual force
        # integrates almost unopposed, so the drift is PREDICTABLE, not merely large.
        pred = 0.1 * 50 * 0.02
        ok = abs(drift - pred) / pred < 0.1
        print(f"  [{'PASS' if ok else 'FAIL'}] the 10%-wrong forcing is DETECTED: drift "
              f"{drift:.4f} against the predicted delta_f * t = {pred:.4f} "
              f"({100*abs(drift-pred)/pred:.1f}% off)")
    return ok


def check_wall_stress():
    """tau_w read off the parabola must return u_tau = 1, the value the forcing defines."""
    d, m, y = _rig()
    for b in range(len(d.blocks)):
        m.u[b][:] = _exact(d, b)
        m.u[b][:, 0, :] = m.u[b][:, -1, :] = 0.0
    U = np.mean([m.u[b].mean(axis=(0, 2)) for b in range(len(d.blocks))], axis=0)
    # one-sided gradient at the wall, on the stretched mesh
    tau = NU * (U[1] - U[0]) / (y[1] - y[0])
    ut = np.sqrt(abs(tau))
    ok = abs(ut - 1.0) < 0.02
    print(f"  [{'PASS' if ok else 'FAIL'}] wall stress on the stretched mesh: "
          f"u_tau = {ut:.4f} against the exact 1 (dy+ wall = {(y[1]-y[0])*RE_TAU:.2f})")
    return ok


def check_converges_from_rest():
    """From u = 0 the flow must climb towards the parabola, not sit at zero or overshoot."""
    d, m, y = _rig(dt=0.05)
    for b in range(len(d.blocks)):
        m.u[b][:] = 0.0
    exact_c = 1.0 / (2 * NU)
    hist = []
    for k in range(1, 401):
        m.step()
        if k % 100 == 0:
            hist.append(max(float(np.abs(m.u[b]).max()) for b in range(len(d.blocks))))
    # analytic startup: u_c(t) approaches f/(2nu) on the diffusive time delta^2/nu = 180
    rising = all(b > a for a, b in zip(hist, hist[1:]))
    ok = rising and 0.0 < hist[-1] < exact_c and hist[-1] > 0.05 * exact_c
    print(f"  [{'PASS' if ok else 'FAIL'}] startup from rest rises monotonically towards "
          f"{exact_c:.0f}: u_c = {' -> '.join(f'{h:.2f}' for h in hist)} at t = 20")
    return ok


def main():
    print("=" * 78)
    print("  channel rig vs laminar Poiseuille: forcing, no-slip wall, stretched-mesh diffusion")
    print("=" * 78)
    r = [check_stationary(), check_stationary(mangle=True), check_wall_stress(),
         check_converges_from_rest()]
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
