"""Stage 9.5 gates -- OBJECTIVES on the differentiable production step.

The payoff configuration: M2's jet actuator (opposing +-90 deg arcs,
parabolic profile, one scalar a) driving the PRODUCTION step, with Stage
8's traction C_D as the loss -- dC_D/da through real production rollouts.

  o.1  warm-start soundness: with slot-keyed seeds active, one step from
       the warmed state still matches m.step() field-for-field (the seeds
       change the Krylov path, never the answer).
  o.2  dC_D/d(jet a) through 2 production steps: FD vs adjoint.
  o.3  jet -> wake sensitivity through the step (loss on wake velocity),
       FD vs adjoint -- the M2 j.3 certificate, now on production.
  o.4  timing: warmed step vs the 9.2d cold number (report).

Run:  .venv/bin/python test_prod_objective.py     (~25 min on the Mac)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# phase A (equivalence + timing) runs with CONVERGED DC settings -- the
# warm-start claim is "the Krylov path changes, the converged answer does
# not", and a pinned truncated iteration cannot test it (production's slot
# seeds shift ITS truncation; both sides must reach the fixed point).
# phase B (FD gradient probes) then pins the sweep count for determinism.
os.environ["PICT_CROSS_DC_INNER"] = "1e-11"
os.environ["PICT_MOM_TOL"] = "1e-12"


def env_tight():
    os.environ["PICT_CROSS_DC_TOL"] = "1e-11"
    os.environ["PICT_CROSS_DC_ITERS"] = "60"


def env_pinned():
    os.environ["PICT_CROSS_DC_TOL"] = "0"
    os.environ["PICT_CROSS_DC_ITERS"] = "3"


env_tight()

import numpy as np
import torch

from cylinder_rect_bc import U_INF, apply as apply_bc, classify
from cylinder_ring_grid import ring_rect_domain
from src.forces_torch import coefficients, face_geometry
from src.piso_multiblock import MultiBlockPISO
from src.prod_step import TorchProductionStep
from test_mb_adjoint_jet import jet_geometry

NU, DT = 0.01, 0.01
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
    d, idx = ring_rect_domain(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                              wake_hold=8.0, wake_ratio=1.10)
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    roles = classify(d)
    body_faces = [(b, f) for (b, f), r in roles.items() if r == "body"]
    for (b, f), r in roles.items():
        d.blocks[b].faces[f] = kindmap[r]
    m = MultiBlockPISO(d, NU, DT, 2, 1e-11, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False,
                       implicit_cross=True, linear_backend="scipy")
    nb = len(d.blocks)
    for b in range(nb):
        m.u[b][:] = U_INF
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d)
    for _ in range(3):
        m.step()

    tps = TorchProductionStep(m)
    jid, juv = jet_geometry(d, body_faces)
    tps.register_jets(jid, juv)
    st0 = tps.state_from_solver()
    N = tps.N
    geom = face_geometry(d, body_faces)
    gids = {b: torch.as_tensor(d.global_ids(b).ravel()) for b in range(nb)}
    shapes = {b: d.blocks[b].shape for b in range(nb)}
    span = float(d.blocks[0].period[2])

    def blocks_of(flat):
        return {b: torch.take(flat, gids[b]).reshape(shapes[b]) for b in shapes}

    def cd_of(st):
        cd, _ = coefficients(geom, blocks_of(st["u"]), blocks_of(st["v"]),
                             blocks_of(st["w"]), blocks_of(st["p"]), NU,
                             span=span, tangential_only=True)
        return cd

    # ------------------------------------------------------------------ o.1
    t0 = time.time()
    with torch.no_grad():
        st1 = tps.step(tps._clone(st0))
    t_first = time.time() - t0
    m.step()
    worst = max(float((st1[f] - torch.as_tensor(m._flat(ref))).abs().max())
                / max(float(np.abs(m._flat(ref)).max()), 1e-300)
                for f, ref in (("u", m.u), ("v", m.v), ("p", m.p)))
    check(worst < 1e-5, f"o.1 warm-started step still == m.step(): "
                        f"worst rel {worst:.2e}")

    # a second no-grad step to fill every seed slot, then time a warmed step
    with torch.no_grad():
        st2 = tps.step(tps._clone(st1))
        t0 = time.time()
        tps.step(tps._clone(st2))
        t_warm = time.time() - t0

    # ------------------------------------------------------------------ o.2
    env_pinned()
    n_steps = 2

    def rollout_cd(a, want_st=False):
        tps.clear_seeds()
        st = tps.apply_jets(tps._clone(st0), a)
        for _ in range(n_steps):
            st = tps.step(st)
        return (st, cd_of(st)) if want_st else cd_of(st)

    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    st_j, L = rollout_cd(a, want_st=True)
    L.backward()
    ga = float(a.grad)
    t_grad = time.time() - t0
    h = 1e-3
    with torch.no_grad():
        Lp = float(rollout_cd(torch.tensor(0.3 + h, dtype=torch.float64)))
        Lm = float(rollout_cd(torch.tensor(0.3 - h, dtype=torch.float64)))
    fd = (Lp - Lm) / (2 * h)
    rel = abs(ga - fd) / max(abs(fd), 1e-300)
    check(abs(ga) > 0 and rel < 1e-4,
          f"o.2 dC_D/da through {n_steps} PRODUCTION steps: adjoint {ga:+.5e} "
          f"FD {fd:+.5e} rel {rel:.2e}")

    # ------------------------------------------------------------------ o.3
    wake_ids = torch.as_tensor(d.global_ids(idx["wake"]).ravel())

    def wake_L(a):
        tps.clear_seeds()
        st = tps.apply_jets(tps._clone(st0), a)
        for _ in range(n_steps):
            st = tps.step(st)
        return sum((torch.take(st[f], wake_ids) ** 2).sum()
                   for f in ("u", "v", "w"))

    a2 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    wake_L(a2).backward()
    gw = float(a2.grad)
    with torch.no_grad():
        Lp = float(wake_L(torch.tensor(0.3 + h, dtype=torch.float64)))
        Lm = float(wake_L(torch.tensor(0.3 - h, dtype=torch.float64)))
    fd = (Lp - Lm) / (2 * h)
    rel = abs(gw - fd) / max(abs(fd), 1e-300)
    check(abs(gw) > 0 and rel < 1e-4,
          f"o.3 jet -> wake through production seams: adjoint {gw:+.5e} "
          f"FD {fd:+.5e} rel {rel:.2e}")

    # ------------------------------------------------------------------ o.4
    # MEASUREMENT, not a gate: on the coarse mesh, x0 warm starts gave NO
    # speedup (52 -> 65 s in the first controlled look) -- unpreconditioned
    # BiCGStab at 1e-13 is dominated by its conditioning tail, and scipy's
    # rtol is relative to ||b|| regardless of x0. The plumbing stays (o.1
    # proves it never changes the answer); the real speed lever for
    # production-scale training is preconditioning, revisited when needed.
    check(True,
          f"o.4 timing REPORT: cold {t_first:.1f}s -> warmed {t_warm:.1f}s "
          f"(fwd+bwd 2-step gradient, pinned: {t_grad:.0f}s)")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
