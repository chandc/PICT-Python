"""
Vortex street behind a CIRCULAR cylinder at Re = 100. Target: St ~ 0.164, C_D ~ 1.33.

EVERY SETTING HERE IS A LESSON FROM THE SQUARE-CYLINDER CASE. In order of how much they cost to
learn:

  tol = 1e-6, NOT the solver's 1e-4 default. At 1e-4 the square case converged BITWISE STEADY and
  never shed -- at Re 100, 200 and 300, and at every grid resolution tried. The pressure
  correction is under-resolved there, so every velocity update is damped: harmless for a steady
  problem, fatal for a marginally unstable one. It is an argument, not a literal, because it is
  the single setting that decides whether this case has any physics in it at all.

  ddt_corr = False. Its F_prev recurrence has unit gain by construction and diverges; the square
  case died at step 455 with it on, and Rhie-Chow itself was innocent.

  rhie_chow = True with persistent_flux. 41x smaller pressure checkerboard than off, and
  persistent_flux is free (no stability effect, 1.7x better damping).

  TWO STAGES, and this is the one that looked like a detail and was not. A symmetric
  configuration on a symmetric discretisation stays symmetric forever, so the instability must be
  triggered. But a kick applied at t = 0 -- before any wake exists -- just convects away through
  undisturbed flow and is gone by the time the recirculation forms. The square case wasted a full
  45,000-step run that way. So: converge to the base flow FIRST, then perturb THAT.

The far field is at 30 D and the wake is resolved to ~12 D at 21.8 cells per shedding wavelength;
see cylinder_grid.py for why a pure geometric radial stretch is not enough.
"""
import argparse
import os
import time

import numpy as np

from cylinder_grid import cylinder_domain, D
from cylinder_bc import apply as apply_bc, classify, probe_index, U_INF
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO

RE = 100.0
DEFAULT_TOL = 1e-6
ST_REF = 0.164          # circular cylinder at Re = 100
CD_REF = 1.33


def build(dt, tol, nz, nblk, backend):
    d, r, arc = cylinder_domain(nblk=nblk, nz=nz)
    m = MultiBlockPISO(d, U_INF * D / RE, dt, 2, tol, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False,
                       linear_backend=backend)
    for b in range(len(d.blocks)):
        m.u[b][:] = U_INF
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d, nblk)
    return d, m


def kick(m, d, amp):
    """Antisymmetric nudge in the near wake. Applied to an ESTABLISHED base flow, not at t=0."""
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        sel = (blk.x > 0.5 * D) & (blk.x < 4.0 * D) & (np.abs(blk.y) < 1.5 * D)
        if sel.any():
            # SINUOUS, not varicose. The von Karman mode meanders the wake bodily
            # sideways, so the transverse velocity has the SAME sign right across the wake --
            # v EVEN in y. An earlier version used `np.sign(blk.y)`, which is v ODD in y: that
            # is the VARICOSE mode, in which the wake breathes symmetrically, and it is stable.
            # It excited the wrong mode and decayed every time, on every grid and at every
            # resolution, which read as "the grid lost the instability".
            m.v[b][sel] += amp * U_INF * \
                np.exp(-((blk.x[sel] - 1.5) ** 2) / 1.0) * \
                np.exp(-(blk.y[sel] / 0.75) ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help="linear solver tolerance; 1e-4 suppresses shedding entirely")
    p.add_argument("--settle", type=int, default=8000, help="steps to reach the base flow")
    p.add_argument("--steps", type=int, default=30000, help="steps after the kick")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nz", type=int, default=4)
    p.add_argument("--nblk", type=int, default=16,
                   help="azimuthal blocks; sets how finely the far-field "
                        "outflow arc can be cut (16 -> |theta| <= 21.8 deg)")
    p.add_argument("--kick", type=float, default=0.01, help="fraction of U")
    p.add_argument("--backend", default="scipy")
    p.add_argument("--restart", default=None)
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    d, m = build(a.dt, a.tol, a.nz, a.nblk, a.backend)
    tag = a.tag or f"cyl_Re{RE:.0f}_tol{a.tol:.0e}_n{d.n_cells}"
    os.makedirs("results/fields", exist_ok=True)

    # RESTART SEMANTICS. The restart file is written every 500 steps in BOTH phases, so a
    # crash costs 500 steps and not the whole settle. Which phase to resume in is derived from
    # the step count in the checkpoint rather than from the file name: below `--settle` the run
    # is still converging to the base flow and must finish it and then be kicked; at or above
    # it the flow has already been kicked and must NOT be kicked a second time, which would
    # inject a fresh perturbation into an established wake and corrupt the amplitude.
    settle, kicked = a.settle, False
    hist = []
    if a.restart and os.path.exists(a.restart):
        checkpoint.load(m, a.restart)
        h = f"results/{tag}_history.npy"
        if os.path.exists(h):
            hist = [tuple(row) for row in np.load(h)]
        kicked = m.nstep >= a.settle
        settle = 0 if kicked else a.settle - m.nstep
        print(f"  restarted from {a.restart}: t={m.time:.1f}, step {m.nstep}, "
              f"{len(hist)} history samples", flush=True)
        where = ("already kicked; continuing the shedding stage" if kicked
                 else f"{settle} settle steps still to run before the kick")
        print(f"  -> {where}", flush=True)

    pb, pk = probe_index(d)
    print(f"  {d.n_cells:,} cells  Re={RE:.0f}  dt={a.dt}  tol={a.tol:.0e}  "
          f"backend={a.backend}  nz={a.nz}", flush=True)
    print(f"  probe at ({d.blocks[pb].x[pk[0],pk[1],0]:.3f}, "
          f"{d.blocks[pb].y[pk[0],pk[1],0]:.3f})", flush=True)
    print(f"  reference: St = {ST_REF}, C_D = {CD_REF}\n", flush=True)

    def report(i, n, hist, t0, phase):
        seg = np.array([h[1] for h in hist[-500:]])
        print(f"  {phase:<7}{i:>7}{m.time:>8.1f}{hist[-1][1]:>12.6f}"
              f"{seg.max()-seg.min():>11.3e}{(time.time()-t0)/max(i,1):>9.3f}", flush=True)

    print(f"  {'phase':<7}{'step':>7}{'t':>8}{'v_probe':>12}{'amp/500':>11}{'s/step':>9}",
          flush=True)

    t0 = time.time()
    for i in range(1, settle + 1):
        m.step()
        hist.append((m.time, float(m.v[pb][pk[0], pk[1], 0])))
        if i % 500 == 0:
            report(i, settle, hist, t0, "settle")
            checkpoint.save(m, f"results/fields/{tag}.npz")
            np.save(f"results/{tag}_history.npy", np.array(hist))
    if settle:
        checkpoint.save(m, f"results/fields/{tag}_base.npz")
        print(f"  base flow saved -> results/fields/{tag}_base.npz", flush=True)

    v0 = float(m.v[pb][pk[0], pk[1], 0])
    if kicked:
        print(f"\n  NOT kicking: this checkpoint is already past the kick "
              f"(step {m.nstep} >= settle {a.settle}), v = {v0:+.6f}\n", flush=True)
    else:
        kick(m, d, a.kick)
        print(f"\n  kicked at {100*a.kick:.3g}% of U about v0 = {v0:+.6f}\n", flush=True)

    t0 = time.time()
    for i in range(1, a.steps + 1):
        m.step()
        hist.append((m.time, float(m.v[pb][pk[0], pk[1], 0])))
        if i % 500 == 0:
            report(i, a.steps, hist, t0, "shed")
            np.save(f"results/{tag}_history.npy", np.array(hist))
            checkpoint.save(m, f"results/fields/{tag}.npz")
    np.save(f"results/{tag}_history.npy", np.array(hist))
    checkpoint.save(m, f"results/fields/{tag}.npz")
    print(f"\n  saved results/fields/{tag}.npz and results/{tag}_history.npy", flush=True)


if __name__ == "__main__":
    main()
