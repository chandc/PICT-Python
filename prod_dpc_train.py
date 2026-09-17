"""DPC on OUR solver: policy training through the differentiable
production step, with verified gradients and memory-flat replay.

The M3 counterpart of tools/fluidgym/dpc_train.py: same policy family
(MLP over 453 probe readings -> one jet scalar), same reward shape
(-C_D - |C_L|, discounted), but the simulator is the R11-validated
butterfly solver and every gradient is the certified discrete adjoint
(Stage 9 gates) instead of an unverified tape. Long windows cost ONE
step's graph via replay_policy_grad.

Sensors: FluidGym's physical probe layout (wake grid + two rings +
extras; D = 1 matches our cylinder) mapped to nearest butterfly nodes.

  --selftest   tiny-window replay-vs-tape policy-gradient gate (~10 min)
  --train      smoke training loop (defaults sized for the Mac; the
               campaign configuration runs on Spark once preconditioned)

.venv/bin/python prod_dpc_train.py --selftest
.venv/bin/python prod_dpc_train.py --train --iters 3
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PICT_CROSS_DC_TOL", "0")     # pinned: deterministic
os.environ.setdefault("PICT_CROSS_DC_ITERS", "3")
os.environ.setdefault("PICT_CROSS_DC_INNER", "1e-11")
os.environ.setdefault("PICT_MOM_TOL", "1e-12")

import numpy as np
import torch

from cylinder_rect_bc import U_INF, apply as apply_bc, classify
from cylinder_ring_grid import ring_rect_domain
from plot_utility.plot_fluidgym_policy_gains import sensor_positions
from src import checkpoint
from src.forces_torch import coefficients, face_geometry
from src.piso_multiblock import MultiBlockPISO
from src.prod_replay import replay_policy_grad
from src.prod_step import TorchProductionStep
from test_mb_adjoint_jet import jet_geometry

NU, DT = 0.01, 0.01
GAMMA = 0.999


MESHES = {
    "coarse": dict(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                   wake_hold=8.0, wake_ratio=1.10),   # steady wake (subcritical)
    "mid": dict(n_east=49, side_dt=0.055, nz=2, wake_dx=0.28,
                wake_hold=8.0, wake_ratio=1.08),      # SHEDS (sigma ~ 0.046)
}


def build(restart=None, mesh="coarse", dt=0.01, picard=2, momdc=2, tol=1e-8):
    d, idx = ring_rect_domain(**MESHES[mesh])
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    roles = classify(d)
    body_faces = [(b, f) for (b, f), r in roles.items() if r == "body"]
    for (b, f), r in roles.items():
        d.blocks[b].faces[f] = kindmap[r]
    m = MultiBlockPISO(d, NU, dt, 2, tol, time_scheme="bdf2",
                       scheme="rotational", picard_iters=picard, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False,
                       implicit_cross=True, linear_backend="scipy")
    m.momentum_dc_iters = momdc
    for b in range(len(d.blocks)):
        m.u[b][:] = U_INF
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d)
    if restart and os.path.exists(restart):
        checkpoint.load(m, restart)
        print(f"  restart {restart}: t={m.time:.1f}", flush=True)
    else:
        for _ in range(3):
            m.step()
        print("  no restart: 3-step developing state", flush=True)
    tps = TorchProductionStep(m)
    jid, juv = jet_geometry(d, body_faces)
    tps.register_jets(jid, juv)
    return d, m, tps, body_faces


class Harness:
    """Sensors, reward and policy plumbing shared by selftest and train."""

    def __init__(self, d, tps, body_faces):
        nb = len(d.blocks)
        self.tps = tps
        self.geom = face_geometry(d, body_faces)
        self.gids = {b: torch.as_tensor(d.global_ids(b).ravel())
                     for b in range(nb)}
        self.shapes = {b: d.blocks[b].shape for b in range(nb)}
        self.span = float(d.blocks[0].period[2])
        self.cd_log = []
        # FluidGym's physical probes -> nearest butterfly nodes (z = 0)
        P = sensor_positions()
        xs = np.concatenate([d.blocks[b].x[:, :, 0].ravel() for b in range(nb)])
        ys = np.concatenate([d.blocks[b].y[:, :, 0].ravel() for b in range(nb)])
        gflat = np.concatenate([d.global_ids(b)[:, :, 0].ravel()
                                for b in range(nb)])
        sg = []
        for px, py in P.T:
            sg.append(gflat[np.argmin((xs - px) ** 2 + (ys - py) ** 2)])
        self.sgids = torch.as_tensor(np.asarray(sg, dtype=np.int64))

    def blocks_of(self, flat):
        return {b: torch.take(flat, self.gids[b]).reshape(self.shapes[b])
                for b in self.shapes}

    def forces(self, st):
        return coefficients(self.geom, self.blocks_of(st["u"]),
                            self.blocks_of(st["v"]), self.blocks_of(st["w"]),
                            self.blocks_of(st["p"]), NU, span=self.span,
                            tangential_only=True)

    def obs(self, st):
        return torch.cat([torch.take(st["p"], self.sgids),
                          torch.take(st["u"], self.sgids),
                          torch.take(st["v"], self.sgids)])

    def step_loss(self, st, k):
        cd, cl = self.forces(st)
        self.cd_log.append(float(cd))
        return (GAMMA ** k) * (cd + torch.abs(cl))


def make_policy(nin=453, hidden=64, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(nin, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1)).double()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--restart", default="results/fields/coarse_dev.npz")
    p.add_argument("--mesh", choices=("coarse", "mid"), default="coarse")
    p.add_argument("--ctrl-steps", type=int, default=4)
    p.add_argument("--substeps", type=int, default=3)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amax", type=float, default=1.0)
    p.add_argument("--init", default=None, help="warm-start policy state dict")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--picard", type=int, default=2)
    p.add_argument("--momdc", type=int, default=2)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--tag", default="prod_dpc")
    a = p.parse_args()

    d, m, tps, body_faces = build(a.restart if a.train else None, mesh=a.mesh,
                                  dt=a.dt, picard=a.picard, momdc=a.momdc,
                                  tol=a.tol)
    hz = Harness(d, tps, body_faces)
    policy = make_policy()
    if getattr(a, "init", None) and os.path.exists(a.init):
        policy.load_state_dict(torch.load(a.init))
        print(f"  policy warm-started from {a.init}", flush=True)

    def policy_action(st):
        return a.amax * torch.tanh(policy(hz.obs(st))).reshape(())

    if a.selftest:
        n_ctrl, sub = 2, 2
        # tape reference
        tps.clear_seeds()
        policy.zero_grad()
        st0 = tps.state_from_solver()
        st = tps._clone(st0)
        L = 0.0
        for k in range(n_ctrl):
            st = tps.apply_jets(st, policy_action(st))
            for _ in range(sub):
                st = tps.step(st)
            L = L + hz.step_loss(st, k)
        L.backward()
        g_tape = torch.cat([q.grad.reshape(-1) for q in policy.parameters()])
        # replay
        tps.clear_seeds()
        policy.zero_grad()
        Lr = replay_policy_grad(tps, st0, policy_action, n_ctrl, sub,
                                hz.step_loss)
        g_rep = torch.cat([q.grad.reshape(-1) for q in policy.parameters()])
        rel = float((g_rep - g_tape).abs().max()) / max(
            float(g_tape.abs().max()), 1e-300)
        dl = abs(Lr - float(L)) / max(abs(float(L)), 1e-300)
        ok = rel < 1e-6 and dl < 1e-9
        print(f"  [{'PASS' if ok else 'FAIL'}] policy-grad replay == tape: "
              f"worst rel {rel:.2e}, loss rel {dl:.2e} "
              f"(|g| max {float(g_tape.abs().max()):.3e})", flush=True)
        sys.exit(0 if ok else 1)

    if a.train:
        opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
        st0 = tps.state_from_solver()
        hist = []
        for it in range(a.iters):
            t0 = time.time()
            tps.clear_seeds()
            opt.zero_grad()
            hz.cd_log = []
            L = replay_policy_grad(tps, st0, policy_action, a.ctrl_steps,
                                   a.substeps, hz.step_loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            cd_fwd = float(np.mean(hz.cd_log[:a.ctrl_steps]))  # forward pass only
            hist.append((L, cd_fwd))
            print(f"  it {it:3d}  window loss {L:9.4f}  mean C_D {cd_fwd:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(policy.state_dict(), f"results/{a.tag}_policy.pt")
            np.save(f"results/{a.tag}_curve.npy", np.array(hist))
        print(f"  saved results/{a.tag}_policy.pt, {a.tag}_curve.npy",
              flush=True)


if __name__ == "__main__":
    main()
