"""
Does the BFS go three-dimensional near Re = 400, and can this grid see it?

Armaly's measured reattachment departs from two-dimensional behaviour above Re ~ 400 -- it
stops growing and turns over as spanwise structure appears. Our five-domain runs give a
straight line (2.80 / 4.85 / 6.72 / 8.02 at Re = 100/200/300/389, increments of ~2.05 per 100
Re), which is what a 2D solution does. Two explanations, and they need separating:

  (a) the solver reproduces 2D physics correctly and the grid cannot represent the 3D mode
  (b) the solver would find the 3D mode given the resolution

A SPANWISE-UNIFORM FIELD STAYS 2D FOREVER. Every equation here is spanwise-symmetric, so with
a uniform initial condition the spanwise velocity stays identically zero to round-off and no
refinement will ever show a transition. The flow must be PERTURBED to have anything to grow
from -- otherwise this study measures nothing and would produce a confidently wrong "no 3D
transition" answer.

Diagnostic: energy in the non-zero spanwise Fourier modes, as a fraction of the total. It is
~1e-30 for a 2D field and grows if a three-dimensional instability takes hold.
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from armaly_bfs5_grid import bfs5_domain, RECIRC_L, RECOV_L
from run_armaly_bfs5 import setup, reattachment, interior_divergence
from src.linsolve import SolveCache


def seed_spanwise(m, d, amp=1e-3, seed=0):
    """Break spanwise symmetry with a small divergence-free-ish perturbation.

    Amplitude 1e-3 of U_bulk: large enough to grow within a few thousand steps if the flow is
    unstable, small enough not to be a forcing in its own right.
    """
    rng = np.random.default_rng(seed)
    for b in range(len(d.blocks)):
        nz = d.blocks[b].shape[2]
        z = d.blocks[b].z
        # one spanwise wave plus noise: a single mode may sit off the unstable wavenumber
        wave = np.sin(2.0 * np.pi * z / z.max() if z.max() > 0 else z)
        m.w[b] += amp * (wave + 0.3 * rng.standard_normal(d.blocks[b].shape))
        m.u[b] += 0.3 * amp * wave
    return m


def spanwise_energy_fraction(m, d):
    """Fraction of kinetic energy in non-zero spanwise Fourier modes."""
    num = den = 0.0
    for b in range(len(d.blocks)):
        for f in (m.u[b], m.v[b], m.w[b]):
            F = np.fft.rfft(f, axis=2)
            den += (np.abs(F) ** 2).sum()
            num += (np.abs(F[:, :, 1:]) ** 2).sum()      # drop the spanwise mean
    return num / max(den, 1e-300)


def run(nz, Re=389.0, nsteps=3000, dt=0.02, backend="scipy", amp=1e-3):
    g = dict(nx_in=20, nx_re=40, nx_rc=40, ny_lo=18, ny_up=19, nz=nz)
    d = bfs5_domain(**g)
    m = setup(d, Re, dt, dong=True)
    m._pcache = SolveCache(backend=backend, precond="jacobi")
    seed_spanwise(m, d, amp=amp)
    e0 = spanwise_energy_fraction(m, d)
    t0 = time.perf_counter()
    for k in range(nsteps):
        m.step()
        if not all(np.isfinite(m.u[b]).all() for b in range(5)):
            return dict(nz=nz, err=f"diverged at {k+1}")
    t = time.perf_counter() - t0
    e1 = spanwise_energy_fraction(m, d)
    return dict(nz=nz, cells=d.n_cells, xr=reattachment(m, d, nz),
                div=interior_divergence(m, d), e0=e0, e1=e1,
                growth=e1 / max(e0, 1e-300), s_step=t / nsteps, err=None)


if __name__ == "__main__":
    be = sys.argv[1] if len(sys.argv) > 1 else "scipy"
    Re = float(sys.argv[2]) if len(sys.argv) > 2 else 389.0
    print(__doc__.strip().split("\n\n")[0])
    print(f"\n  Re = {Re:.0f}, backend = {be}, spanwise perturbation 1e-3\n")
    print(f"  {'nz':>4}{'cells':>9}{'x_r/S':>8}{'div':>10}{'E3D in':>10}{'E3D out':>10}"
          f"{'growth':>9}{'s/step':>9}")
    for nz in (8, 16, 32):
        r = run(nz, Re=Re, backend=be)
        if r.get("err"):
            print(f"  {nz:4d}   {r['err']}"); continue
        print(f"  {nz:4d}{r['cells']:9,}{r['xr']:8.3f}{r['div']:10.1e}"
              f"{r['e0']:10.2e}{r['e1']:10.2e}{r['growth']:9.2e}{r['s_step']:9.3f}",
              flush=True)
    print("\n  growth >> 1 means the spanwise mode AMPLIFIED -- a real 3D transition.")
    print("  growth << 1 means it decayed and the flow is 2D at this Re and resolution.")
