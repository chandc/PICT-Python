"""
Vortex street behind a square cylinder at Re = 100. Target: St ~ 0.13.

WHY A PERTURBATION IS NEEDED. The configuration is symmetric about y = 0, and so is the
discretisation, so a symmetric initial condition stays symmetric forever -- it converges to the
unstable steady solution and never sheds. Real experiments are perturbed by noise; here the
symmetry has to be broken deliberately. A brief transverse pulse at the inlet does it, then is
switched off so the shedding that follows is the flow's own, not the forcing's.

WHAT IS MEASURED. A probe at (2D, 0.5D) records v(t); the shedding frequency comes from the
peak of its FFT after the transient. Lift would work too and is the more standard choice, but it
needs a surface-force integral over four block faces, and the probe answers the same question.

COST. dt = 0.01 gives CFL ~ 0.46 on the finest cell (dy = 0.032 across the body). One shedding
period is D / (St U) ~ 7.7 time units, so resolving ~20 periods after the transient needs
t ~ 200, i.e. 20,000 steps. Checkpoints every 2,000 steps make that restartable.
"""
import argparse
import os
import time

import numpy as np

from square_cylinder_grid import square_domain, D
from square_cylinder_bc import apply as apply_bc, U_INF
from src import checkpoint
from src.multiblock import face_slice, face_id
from src.piso_multiblock import MultiBlockPISO

RE = 100.0
# The single most consequential setting in this case, so it is an argument, not a literal.
# At 1e-4 -- the solver's own default -- this converges bitwise steady and NEVER sheds, at
# Re 100, 200 and 300 and at every resolution tried: the pressure correction is under-resolved,
# so every velocity update is damped, which is harmless for a steady problem and fatal for a
# marginally unstable one. See reference/linear_solver_tolerance.md.
DEFAULT_TOL = 1e-6
PROBE = (2.0 * D, 0.5 * D)          # in the shear layer, off the centreline where v is largest
PULSE_UNTIL = 4.0                   # time units of symmetry breaking, then off
PULSE_AMP = 0.05 * U_INF     # legacy inlet forcing; --kick is the better route


def build(dt, rhie_chow=True, nz=8, tol=DEFAULT_TOL, backend=None):
    d, idx = square_domain(nz=nz)
    # ddt_corr IS OFF, DELIBERATELY. It is what makes this case diverge, and neither Rhie-Chow
    # nor the persistent flux does. Isolated on the coarse grid with everything else identical:
    #
    #     rhie_chow  persist  ddt_corr   outcome
    #       True      True      True     DIVERGED at step 311
    #       True      True      False    stable, 2000 steps
    #       True      False     False    stable, 2000 steps
    #       True      False     True     DIVERGED at step 324
    #
    # ddt_corr alone decides it; persist is irrelevant either way. The full-resolution run
    # died the same way at step 455, with the correction outgrowing the flux it corrects:
    # |RC|/|F| 0.184 at step 200 -> 1.818 at step 311. That is the unit-gain F_prev recurrence
    # the fvcDdtPhiCoeff limiter in piso_multiblock was added to tame, and on this case the
    # limiter is not enough. See reference/rhie_chow_ddt_instability.md.
    #
    # Dropping it costs nothing here and keeps what matters: RC still suppresses the
    # checkerboard, amplitude 1.44e-01 -> 3.48e-03 against RC off, a factor of 41.
    # backend defaults from the environment so a container can select it without editing code;
    # 'amgx' silently falls back to scipy when libamgxsh.so is absent, so this is safe locally.
    backend = backend or os.environ.get("PICT_BACKEND", "scipy")
    m = MultiBlockPISO(d, U_INF * D / RE, dt, 2, tol, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=rhie_chow,
                       persistent_flux=rhie_chow, ddt_corr=False,
                       linear_backend=backend)
    for b in range(len(d.blocks)):
        m.u[b][:] = U_INF; m.v[b][:] = 0.0; m.w[b][:] = 0.0
    apply_bc(m, d, kind="dong")
    return d, idx, m


def inlet_pulse(m, d, amp):
    """Transverse velocity on the inlet faces only. amp=0 restores the symmetric condition."""
    for b, blk in enumerate(d.blocks):
        fid = face_id(0, 0)
        if blk.faces[fid] in ("periodic", "connected"):
            continue
        fs = face_slice(fid)
        if not np.all(np.abs(blk.x[fs] - blk.x.min()) < 1e-9) or blk.x.min() > -9.0:
            continue
        m.v_bc[b][fs] = amp
        m.v[b][fs] = amp


def wake_kick(m, d, amp):
    """SINUOUS nudge in an ESTABLISHED wake -- the only perturbation that excites shedding.

    Two mistakes are avoided here, both of which cost whole runs:

    SYMMETRY. The von Karman mode meanders the wake bodily sideways, so the transverse velocity
    has the SAME sign right across it -- v EVEN in y. A perturbation built from `np.sign(y)` is
    v ODD in y, which is the VARICOSE mode: the wake breathing symmetrically. That mode is
    stable, so such a kick decays on every grid at every resolution, and the result reads as
    "this grid lost the instability".

    TIMING. Applied at t = 0 the kick sits in undisturbed parallel flow with no wake to perturb,
    convects away, and is gone before the recirculation region forms. Settle first, kick that.
    """
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        sel = (blk.x > 0.5 * D) & (blk.x < 4.0 * D) & (np.abs(blk.y) < 1.5 * D)
        if sel.any():
            m.v[b][sel] += amp * U_INF * \
                np.exp(-((blk.x[sel] - 1.5) ** 2) / 1.0) * \
                np.exp(-(blk.y[sel] / 0.75) ** 2)


def probe_index(d, idx):
    """Block and index of the node nearest PROBE."""
    best = None
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        r = (blk.x[:, :, 0] - PROBE[0])**2 + (blk.y[:, :, 0] - PROBE[1])**2
        k = np.unravel_index(r.argmin(), r.shape)
        if best is None or r[k] < best[0]:
            best = (r[k], b, k)
    return best[1], best[2]


def run(nsteps, dt=0.01, rhie_chow=True, nz=8, every=2000, restart=None, tag=None,
        tol=DEFAULT_TOL, settle=0, kick=0.0, backend=None):
    d, idx, m = build(dt, rhie_chow, nz, tol, backend)
    tag = tag or (f"sqcyl_Re{RE:.0f}{'_rc' if rhie_chow else ''}"
                  f"_tol{tol:.0e}_n{d.n_cells}")
    os.makedirs("results/fields", exist_ok=True)
    hist = []
    if restart and os.path.exists(restart):
        checkpoint.load(m, restart)
        h = f"results/{tag}_history.npy"
        if os.path.exists(h):
            hist = list(np.load(h))
        print(f"  restarted from {restart} at t = {m.time:.3f}, step {m.nstep}")

    pb, pk = probe_index(d, idx)
    px, py = d.blocks[pb].x[pk[0], pk[1], 0], d.blocks[pb].y[pk[0], pk[1], 0]
    print(f"  {d.n_cells:,} cells, dt = {dt}, Re = {RE:.0f}, tol = {tol:.0e}, "
          f"rhie_chow = {rhie_chow}, nz = {nz}, backend = {m._pcache.backend}")
    print(f"  probe at ({px:.3f}, {py:.3f}), asked for ({PROBE[0]}, {PROBE[1]})")
    print(f"  symmetry-breaking pulse: v = {PULSE_AMP} at the inlet until t = {PULSE_UNTIL}\n")
    print(f"  {'step':>7}{'t':>9}{'v_probe':>11}{'max|u|':>9}{'max div':>11}{'s/step':>9}")

    if settle and not restart:
        print(f"  settling {settle} steps to the base flow before kicking\n", flush=True)
        st0 = time.time()
        for i in range(1, settle + 1):
            m.step()
            hist.append((m.time, float(m.v[pb][pk[0], pk[1], 0])))
            if i % 500 == 0:
                seg = np.array([h[1] for h in hist[-500:]])
                print(f"  settle{i:>7}{m.time:>9.1f}{hist[-1][1]:>11.6f}"
                      f"{seg.max()-seg.min():>11.3e}{(time.time()-st0)/i:>9.3f}", flush=True)
        checkpoint.save(m, f"results/fields/{tag}_base.npz")
        print(f"  base flow saved; v_probe = {hist[-1][1]:+.6f}", flush=True)
    if kick:
        wake_kick(m, d, kick)
        print(f"  SINUOUS wake kick at {100*kick:.3g}% of U\n", flush=True)

    pulse_on = None
    t0 = time.time()
    for n in range(nsteps):
        want = PULSE_AMP if m.time < PULSE_UNTIL else 0.0
        if want != pulse_on:
            inlet_pulse(m, d, want)
            pulse_on = want
        m.step()
        vp = float(m.v[pb][pk[0], pk[1], 0])
        hist.append((m.time, vp))
        if (n + 1) % 250 == 0 or n == 0:
            dv = max(np.abs(np.where(m.wall[d.global_ids(b)], 0.0,
                     d.divergence(b, d.face_fluxes(b, m.u, m.v, m.w),
                                  d.block_metrics_cached(b)[0]))).max()
                     for b in range(len(d.blocks)))
            print(f"  {m.nstep:>7}{m.time:>9.2f}{vp:>11.5f}"
                  f"{max(np.abs(m.u[b]).max() for b in range(len(d.blocks))):>9.4f}"
                  f"{dv:>11.2e}{(time.time()-t0)/(n+1):>9.3f}", flush=True)
        if (n + 1) % every == 0:
            checkpoint.save(m, f"results/fields/{tag}.npz")
            np.save(f"results/{tag}_history.npy", np.array(hist))
    checkpoint.save(m, f"results/fields/{tag}.npz")
    np.save(f"results/{tag}_history.npy", np.array(hist))
    print(f"\n  saved results/fields/{tag}.npz and results/{tag}_history.npy")
    return m, d, np.array(hist)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nz", type=int, default=8)
    p.add_argument("--no-rc", action="store_true")
    p.add_argument("--restart", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--backend", default=None,
                   help="scipy | amgx (default: $PICT_BACKEND, else scipy)")
    p.add_argument("--settle", type=int, default=0,
                   help="steps to reach the base flow before kicking")
    p.add_argument("--kick", type=float, default=0.0,
                   help="sinuous wake kick, fraction of U")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help="linear solver tolerance; 1e-4 suppresses shedding entirely")
    a = p.parse_args()
    run(a.steps, a.dt, not a.no_rc, a.nz, restart=a.restart, tag=a.tag, tol=a.tol,
        settle=a.settle, kick=a.kick, backend=a.backend)
