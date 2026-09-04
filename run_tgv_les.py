"""Taylor-Green vortex at Re = 1600 -- the first LES validation against turbulence data.

WHY THIS CASE AND NOT COMTE-BELLOT & CORRSIN. Both are periodic boxes with no walls and no
driving force, which is what makes either a good first test: a failure has one candidate cause
rather than five. TGV wins on setup cost. Its initial condition is ANALYTIC and already in this
repo; CBC needs a synthetic isotropic field matched to a prescribed E(k), plus digitised
experimental spectra at three stations, and matching that initial condition is a documented
source of disagreement between published results. TGV's reference is a single curve.

WHAT IS MEASURED. The volume-averaged kinetic energy dissipation rate

    eps(t) = -dE/dt,        E = <|u|^2>/2

against the resolved viscous dissipation 2 nu Z. For Re = 1600 the flow transitions and eps
peaks near t ~ 9. An under-resolved run without a model dissipates too LITTLE at the peak,
because the cascade it cannot represent is where the dissipation lives; a model that works
supplies the difference.

THE BASELINE IS NOT FREE OF ERROR AND THAT IS QUANTIFIED SEPARATELY.
test_numerical_dissipation.py measures the scheme's own dissipation on this same case: 3.8% of
the total at 40^3, converging at second order. Any model comparison here must be read against
that number, because a model appearing to supply 4% of the dissipation would be supplying the
discretisation's error.

Single-core by construction, so a sweep runs cases in PARALLEL rather than one case faster.
"""
import argparse
import os
import time

import numpy as np

from src import sgs
from src.domains import periodic_box
from src.piso_multiblock import MultiBlockPISO


def tgv(blk):
    """u = sin x cos y cos z on [0, 2 pi]^3 with U0 = 1: E(0) = 1/8 and Z(0) = 3/8, exactly.

    THE BOX MUST BE [0, 2 pi]^3. Writing this field as sin(2 pi x) on a UNIT box gives the same
    velocities at wavenumber 2 pi instead of 1 -- the energy is identical, so E(0) = 0.125 still
    checks out, but the enstrophy and the dissipation are (2 pi)^2 = 39.5 times too large. That
    passed unnoticed until the initial state was compared against a reference: Z(0) = 14.06
    where the analytic value is 0.375. Build the domain with L = 2 pi.
    """
    x, y, z = blk.x, blk.y, blk.z
    return (np.sin(x) * np.cos(y) * np.cos(z),
            -np.cos(x) * np.sin(y) * np.cos(z),
            np.zeros_like(x))


def energy_enstrophy(d, m, nb):
    U = {b: m.u[b] for b in range(nb)}
    V = {b: m.v[b] for b in range(nb)}
    W = {b: m.w[b] for b in range(nb)}
    E = Z = vol = 0.0
    for b in range(nb):
        J = d.block_metrics_cached(b)[0]
        h = d.blocks[b].h
        dV = np.abs(J) * h[0] * h[1] * h[2]
        gu, gv, gw = d.gradient(b, U), d.gradient(b, V), d.gradient(b, W)
        wx, wy, wz = gw[1] - gv[2], gu[2] - gw[0], gv[0] - gu[1]
        E += float(np.sum(0.5 * (m.u[b] ** 2 + m.v[b] ** 2 + m.w[b] ** 2) * dV))
        Z += float(np.sum(0.5 * (wx ** 2 + wy ** 2 + wz ** 2) * dV))
        vol += float(np.sum(dV))
    return E / vol, Z / vol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--model", choices=("none", "wale", "smagorinsky"), default="wale")
    p.add_argument("--re", type=float, default=1600.0)
    p.add_argument("--t-end", type=float, default=12.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    nu = 1.0 / a.re
    tag = a.tag or f"tgv_n{a.n}_{a.model}"
    d = periodic_box(a.n, a.blocks, L=2 * np.pi)
    nb = len(d.blocks)
    m = MultiBlockPISO(d, nu, a.dt, 2, a.tol, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b in range(nb):
        m.u[b][:], m.v[b][:], m.w[b][:] = tgv(d.blocks[b])

    nsteps = int(round(a.t_end / a.dt))
    E0, Z0 = energy_enstrophy(d, m, nb)
    print(f"  TGV Re = {a.re:.0f}, {a.n}^3 in {nb} blocks = {d.n_cells:,} cells, "
          f"model = {a.model}", flush=True)
    print(f"  dt = {a.dt}, to t = {a.t_end} ({nsteps} steps)", flush=True)
    print(f"  E(0) = {E0:.6f} (exact 0.125),  Z(0) = {Z0:.6f} (exact 0.375)  "
          f"errors {abs(E0-0.125)/0.125*100:.2f}% and {abs(Z0-0.375)/0.375*100:.2f}%", flush=True)
    print(f"  {'step':>7}{'t':>8}{'E':>11}{'eps=-dE/dt':>13}{'2 nu Z':>11}"
          f"{'nu_t/nu':>10}{'s/step':>9}", flush=True)

    hist = [(0.0, E0, Z0, 0.0)]
    t0 = time.time()
    Eprev, ratio = E0, 0.0
    for i in range(1, nsteps + 1):
        if a.model != "none":
            U = {b: m.u[b] for b in range(nb)}
            V = {b: m.v[b] for b in range(nb)}
            W = {b: m.w[b] for b in range(nb)}
            nu_eff, nu_t = sgs.effective_viscosity(d, U, V, W, nu, model=a.model)
            worst = min(float(nu_eff[b].min()) for b in range(nb))
            if worst <= 0.0:
                raise SystemExit(f"  ABORT: nu_eff non-positive ({worst:.3e}) at step {i}")
            m.set_nu(nu_eff)
            ratio = float(np.mean([float(nu_t[b].mean()) for b in range(nb)])) / nu
        m.step()
        if i % 25 == 0:
            E, Z = energy_enstrophy(d, m, nb)
            eps = -(E - Eprev) / (25 * a.dt)
            hist.append((m.time, E, Z, eps))
            Eprev = E
            if i % 200 == 0:
                print(f"  {i:>7}{m.time:>8.2f}{E:>11.6f}{eps:>13.5f}{2*nu*Z:>11.5f}"
                      f"{ratio:>10.3f}{(time.time()-t0)/i:>9.3f}", flush=True)
                os.makedirs("results", exist_ok=True)
                np.save(f"results/{tag}.npy", np.array(hist))
    np.save(f"results/{tag}.npy", np.array(hist))
    H = np.array(hist)
    k = int(np.argmax(H[1:, 3])) + 1
    print(f"\n  peak eps = {H[k,3]:.5f} at t = {H[k,0]:.2f}   "
          f"(Re = 1600 reference peaks near t ~ 9)")
    print(f"  saved results/{tag}.npy", flush=True)


if __name__ == "__main__":
    main()
