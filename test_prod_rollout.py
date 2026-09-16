"""Stage 9.4 -- multi-step ROLLOUT of the torch production step from the
R11 restart (fully developed shedding, PRODUCTION mesh, nz=4). Three
requirements, one per gate:

  9.4a  bounded and finite over 30 steps (max|u| < 2.5). The M2 frozen
        chains blew up ~12x/step from the sheared corners by step ~6; the
        instability was a frozen-operator artifact and must be ABSENT
        here -- this step reassembles A from the current velocity exactly
        as production does.
  9.4b  tracks the production trajectory: relative divergence from
        m.step()'s fields stays below 1e-2 at step 30 (both sides run
        with matched, tightened tolerances; residual drift is
        solver-tolerance chaos seeding, not error).
  9.4c  no exponential separation: div(30)/max(div(15),tiny) < 30 -- a
        genuine instability multiplies per step, tolerance drift
        accumulates roughly linearly.

Writes the divergence curve to results/prod_rollout_94.npz.
Run:  .venv/bin/python test_prod_rollout.py     (~1 h on the Mac)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# matched, tightened tolerances for BOTH sides (see doc note in prod_step)
os.environ["PICT_CROSS_DC_TOL"] = "1e-6"
os.environ["PICT_CROSS_DC_ITERS"] = "12"
os.environ["PICT_CROSS_DC_INNER"] = "1e-8"
os.environ["PICT_MOM_TOL"] = "1e-10"

import numpy as np
import torch

from cylinder_rect_bc import U_INF, apply as apply_bc, classify
from cylinder_ring_grid import ring_rect_domain
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO
from src.prod_step import TorchProductionStep

N_STEPS = 30
RESTART = "results/fields/cylrect_r11_final.npz"
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
    d, _ = ring_rect_domain(nz=4)               # the R11 production mesh
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    for (b, fid), role in classify(d).items():
        d.blocks[b].faces[fid] = kindmap[role]
    m = MultiBlockPISO(d, 0.01, 0.01, 2, 1e-8, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False,
                       implicit_cross=True, linear_backend="scipy")
    for b in range(len(d.blocks)):
        m.u[b][:] = U_INF
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d)
    checkpoint.load(m, RESTART)
    print(f"  restarted: t={m.time:.1f}, step {m.nstep}, "
          f"{d.n_cells:,} cells (production mesh)", flush=True)

    t0 = time.time()
    tps = TorchProductionStep(m, verbose=True)
    st = tps.state_from_solver()
    print(f"  TorchProductionStep built ({time.time()-t0:.0f}s incl. probes)",
          flush=True)

    # production trajectory first (fields recorded flat per step)
    traj = []
    t0 = time.time()
    for i in range(N_STEPS):
        m.step()
        traj.append((m._flat(m.u).copy(), m._flat(m.v).copy(),
                     m._flat(m.p).copy()))
    print(f"  production: {N_STEPS} steps ({time.time()-t0:.0f}s)", flush=True)

    div, umax = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_STEPS):
            st = tps.step(st)
            du = float((st["u"] - torch.as_tensor(traj[i][0])).abs().max())
            sc = max(float(np.abs(traj[i][0]).max()), 1e-300)
            div.append(du / sc)
            umax.append(float(st["u"].abs().max()))
            if (i + 1) % 5 == 0:
                print(f"    step {i+1:3d}  max|u| {umax[-1]:.4f}  "
                      f"div-from-production {div[-1]:.3e}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    div, umax = np.array(div), np.array(umax)
    np.savez("results/prod_rollout_94.npz", div=div, umax=umax)

    ok_a = bool(np.isfinite(umax).all() and umax.max() < 2.5)
    check(ok_a, f"9.4a bounded over {N_STEPS} steps: max|u| {umax.max():.3f} "
                f"(M2 chain instability ABSENT)")
    check(div[-1] < 1e-2,
          f"9.4b tracks production: divergence {div[-1]:.3e} at step {N_STEPS}")
    ratio = div[-1] / max(div[len(div) // 2], 1e-300)
    check(ratio < 30.0,
          f"9.4c no exponential separation: div({N_STEPS})/div({len(div)//2}) "
          f"= {ratio:.1f}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
