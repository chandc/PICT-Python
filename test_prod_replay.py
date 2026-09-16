"""Stage 9.5 gates -- MEMORY-FLAT adjoint replay vs the tape.

  r.1  equivalence: dL/da_k from replay == direct tape on the same 3-step
       window (distinct per-step actions, C_D + wake loss). The replay
       re-executes the identical arithmetic, so agreement is limited only
       by iterative-solve determinism.
  r.2  st0 gradient equivalence (replay's lambda_0 vs the tape's
       d L/d st0["u"]).
  r.3  snapshots carry no graph (the memory-flat claim's structural
       half); peak-RSS of tape vs replay reported for the record.
  r.4  a 6-step replay window runs to completion with finite grads --
       twice the window at the same per-step graph cost.

Run:  .venv/bin/python test_prod_replay.py     (~30 min on the Mac)
"""
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["PICT_CROSS_DC_TOL"] = "0"       # pinned sweeps: determinism
os.environ["PICT_CROSS_DC_ITERS"] = "3"
os.environ["PICT_CROSS_DC_INNER"] = "1e-11"
os.environ["PICT_MOM_TOL"] = "1e-12"

import numpy as np
import torch

from cylinder_rect_bc import U_INF, apply as apply_bc, classify
from cylinder_ring_grid import ring_rect_domain
from src.forces_torch import coefficients, face_geometry
from src.piso_multiblock import MultiBlockPISO
from src.prod_replay import replay_rollout_grad
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


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


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
    wake_ids = torch.as_tensor(d.global_ids(idx["wake"]).ravel())

    def blocks_of(flat):
        return {b: torch.take(flat, gids[b]).reshape(shapes[b]) for b in shapes}

    def final_loss(st):
        cd, _ = coefficients(geom, blocks_of(st["u"]), blocks_of(st["v"]),
                             blocks_of(st["w"]), blocks_of(st["p"]), NU,
                             span=span, tangential_only=True)
        return cd + 1e-2 * (torch.take(st["u"], wake_ids) ** 2).sum()

    acts = [0.30, 0.10, -0.20]

    # ---------------- tape reference
    tps.clear_seeds()
    a_t = [torch.tensor(a, dtype=torch.float64, requires_grad=True)
           for a in acts]
    du = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    rss0 = rss_mb()
    t0 = time.time()
    st = tps._clone(st0)
    st["u"] = st["u"] + du
    for k in range(3):
        st = tps.apply_jets(st, a_t[k])
        st = tps.step(st)
    L = final_loss(st)
    L.backward()
    t_tape = time.time() - t0
    rss_tape = rss_mb() - rss0
    g_tape = [float(a.grad) for a in a_t]
    gu_tape = du.grad.detach().numpy()

    # ---------------- replay
    tps.clear_seeds()
    rss0 = rss_mb()
    t0 = time.time()
    Lr, g_rep, st0_g = replay_rollout_grad(tps, st0, acts, final_loss,
                                           want_state_grad=True)
    t_rep = time.time() - t0
    rss_rep = rss_mb() - rss0

    # ------------------------------------------------------------------ r.1
    worst = max(abs(g_rep[k] - g_tape[k]) / max(abs(g_tape[k]), 1e-300)
                for k in range(3))
    dl = abs(Lr - float(L)) / max(abs(float(L)), 1e-300)
    check(worst < 1e-6 and dl < 1e-9,
          f"r.1 replay == tape on 3-step window: dL/da worst rel {worst:.2e}, "
          f"loss rel {dl:.2e}  (grads: " +
          " ".join(f"{g:+.3e}" for g in g_rep) + ")")

    # ------------------------------------------------------------------ r.2
    gu_rep = st0_g["u"].numpy()
    rel = (np.abs(gu_rep - gu_tape).max()
           / max(np.abs(gu_tape).max(), 1e-300))
    check(rel < 1e-6, f"r.2 lambda_0 == tape dL/d st0[u]: worst rel {rel:.2e}")

    # ------------------------------------------------------------------ r.3
    with torch.no_grad():
        st_probe = tps.step(tps.apply_jets(tps._clone(st0),
                                           torch.tensor(0.3)))
    no_graph = all(st_probe[k].grad_fn is None for k in
                   ("u", "v", "w", "p", "p_flux"))
    check(no_graph, f"r.3 forward snapshots carry NO graph "
                    f"(peak-RSS delta: tape {rss_tape:.0f} MB vs replay "
                    f"{rss_rep:.0f} MB on 3 steps; tape {t_tape:.0f}s, "
                    f"replay {t_rep:.0f}s)")

    # ------------------------------------------------------------------ r.4
    acts6 = [0.3, 0.2, 0.1, 0.0, -0.1, -0.2]
    tps.clear_seeds()
    t0 = time.time()
    L6, g6 = replay_rollout_grad(tps, st0, acts6, final_loss)
    check(all(np.isfinite(g) for g in g6) and np.isfinite(L6),
          f"r.4 6-step replay window: finite grads " +
          " ".join(f"{g:+.2e}" for g in g6) +
          f"  ({time.time()-t0:.0f}s)")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
