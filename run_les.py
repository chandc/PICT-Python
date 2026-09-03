"""A runnable LES: Taylor-Green in a periodic box with a subgrid closure.

WHAT THIS DEMONSTRATES, and what it does not.

It demonstrates that the pieces compose: strain rate on a multi-block curvilinear grid, a
closure that turns it into nu_t, and a solver that accepts a viscosity FIELD and rebuilds its
operators from it every step. Each piece is separately validated -- the variable-nu operator by
MMS at order 1.96-1.99, the convective operator's skew-symmetry by test_energy_conservation.py,
the closures by test_sgs_models.py against analytic targets.

It does NOT demonstrate a converged LES of anything. Taylor-Green at this resolution is a
transitional flow, not a developed cascade, and the run is short. The honest claim is "the
machinery runs and behaves sensibly", which is what a first runnable is for.

THE DIAGNOSTIC THAT MATTERS IS THE VISCOSITY RATIO. nu_t / nu tells you whether the model is
doing anything: near zero and the closure is decoration, enormous and it has swamped the
physics. It is reported every interval alongside the kinetic energy, because a decaying energy
curve on its own cannot distinguish a working SGS model from a numerically dissipative scheme.
"""
import argparse
import time

import numpy as np

from src import sgs
from src.mb_adjoint import periodic_box
from src.piso_multiblock import MultiBlockPISO


def taylor_green(blk, u0=1.0):
    x, y, z = blk.x * 2 * np.pi, blk.y * 2 * np.pi, blk.z * 2 * np.pi
    u = u0 * np.sin(x) * np.cos(y) * np.cos(z)
    v = -u0 * np.cos(x) * np.sin(y) * np.cos(z)
    w = np.zeros_like(u)
    return u, v, w


def kinetic_energy(m, nb):
    e = tot = 0.0
    for b in range(nb):
        J = m.Js[b] if hasattr(m, "Js") else 1.0
        e += float(np.sum(m.u[b] ** 2 + m.v[b] ** 2 + m.w[b] ** 2))
        tot += m.u[b].size
    return 0.5 * e / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("wale", "smagorinsky", "none"), default="wale")
    p.add_argument("--n", type=int, default=16, help="cells per side")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--re", type=float, default=400.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.01)
    a = p.parse_args()

    nu_mol = 1.0 / a.re
    d = periodic_box(a.n, a.blocks)
    nb = len(d.blocks)
    m = MultiBlockPISO(d, nu_mol, a.dt, 2, 1e-8, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b in range(nb):
        m.u[b][:], m.v[b][:], m.w[b][:] = taylor_green(d.blocks[b])

    print(f"  Taylor-Green, {a.n}^3 in {nb} blocks = {d.n_cells:,} cells, "
          f"Re = {a.re:.0f}, nu = {nu_mol:.4g}")
    print(f"  subgrid model: {a.model}\n")
    print(f"  {'step':>6}{'t':>8}{'E':>12}{'nu_t/nu max':>14}{'nu_t/nu mean':>14}"
          f"{'s/step':>9}")
    t0 = time.time()
    for i in range(1, a.steps + 1):
        if a.model != "none":
            U = {b: m.u[b] for b in range(nb)}
            V = {b: m.v[b] for b in range(nb)}
            W = {b: m.w[b] for b in range(nb)}
            nu_eff, nu_t = sgs.effective_viscosity(d, U, V, W, nu_mol, model=a.model)
            # POSITIVITY IS NOT ASSUMED. Both closures are non-negative by construction and
            # test_sgs_models.py checks it, but a negative total viscosity would make the
            # diffusion operator indefinite and the momentum solve diverge in a way that looks
            # like a physical instability, so it is refused here rather than debugged later.
            worst = min(float(nu_eff[b].min()) for b in range(nb))
            if worst <= 0.0:
                raise SystemExit(f"  ABORT: nu_eff went non-positive ({worst:.3e}) at step {i}")
            m.set_nu(nu_eff)
        m.step()
        if i % max(1, a.steps // 10) == 0:
            if a.model != "none":
                r = [nu_t[b] / nu_mol for b in range(nb)]
                mx = max(float(x.max()) for x in r)
                mn = float(np.mean([float(x.mean()) for x in r]))
            else:
                mx = mn = 0.0
            print(f"  {i:>6}{m.time:>8.2f}{kinetic_energy(m, nb):>12.6f}{mx:>14.3f}"
                  f"{mn:>14.3f}{(time.time()-t0)/i:>9.3f}", flush=True)
    print(f"\n  final E = {kinetic_energy(m, nb):.6f}, "
          f"projected divergence {m.interior_divergence():.2e}")


if __name__ == "__main__":
    main()
