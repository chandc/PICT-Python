"""M2 gates -- JET ACTUATION on the butterfly through the vector chain.

The FluidGym-parity actuator (reference/fluidgym_parity.md, milestone M2):
opposing jets at +-90 deg on the cylinder wall, 10 deg half-width, parabolic
profile, ONE scalar a scaling both with opposing signs (top blows, bottom
sucks -- mass-conserving, the Rabault/FluidGym configuration). The jet enters
`MultiBlockVecChain` as profile-weighted Dirichlet values on body wall nodes,
through the same A_ib elimination as every Dirichlet value.

This is also the vector chain's first outing on the BUTTERFLY (M1's gates ran
on the square duct), so j.1 re-certifies the vector FD there before the jet
gates use it. Gates:

  j.1  vector FD on the butterfly: FD vs adjoint through 3 vector steps,
       sources on u and v -- the M1 certificate transplanted to production
       topology.
  j.2  dL/da for the TRACTION loss (Stage 8 coefficients, tangential_only,
       jet values scattered back into the wall layer): FD-exact. The a-path
       runs boundary -> momentum -> projection -> fields -> traction AND
       directly boundary -> traction; FD checks their sum.
  j.3  jet -> wake across the seams: loss on the wake block's velocity only;
       dL/da nonzero and FD-exact. Nothing but ring -> trap -> wake seam
       transport can produce it.
  j.4  liveness with jets on: drop_pflux / detach_dong still change dL/da.
  j.5  forward sanity INSIDE THE GATE WINDOW (4 steps): jets on stays
       bounded there and measurably differs from jets off.

NOT A GATE -- a measured property: the frozen-coefficient surrogate is
UNSTABLE on the butterfly beyond ~5 steps at dt = 0.01 (~12x/step; scalar
and vector chains identically; blow-up seated in the near-body layer at a
ring-quarter seam). The dominant driver is the accumulating p_flux
Rhie-Chow feedback (RC = 0 cuts growth to ~1.3x/step) -- invisible on the
orthogonal gate ducts where RC ~ 0. Falsified remedies: DC cross sweeps in
the pressure stage (identical blow-up -- it is NOT the missing cross term),
a physical uniform-flow frozen operator (slightly worse), uniform inlet
(fixes the O(100) inlet data, not the growth). The chains are verification
instruments; gradient gates run inside the stable window, and physical
rollouts (M2's open-loop a(t)) belong to the production solver's adjoint.

Run:  .venv/bin/python test_mb_adjoint_jet.py     (~15 min on the Mac)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cylinder_ring_grid import ring_rect_domain
from cylinder_rect_bc import classify
from src.forces_torch import coefficients, face_geometry
from src.mb_adjoint import MultiBlockVecChain
from src.multiblock import face_slice

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
NU, DT = 0.05, 0.01
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}", flush=True)


def jet_geometry(d, body_faces, half_deg=10.0):
    """Jet nodes and profile-weighted unit velocities on the body wall.

    Top jet (+90 deg): radially OUT, +a. Bottom (-90 deg): radially out
    scaled by -a, i.e. suction when the top blows -- opposing jets.
    Parabolic profile 1 - (dtheta/half)^2, their get_jet_profile shape.
    """
    seen = {}
    for b, fid in body_faces:
        fs = face_slice(fid)
        gid = d.global_ids(b)[fs].ravel()
        xs = d.blocks[b].x[fs].ravel()
        ys = d.blocks[b].y[fs].ravel()
        th = np.degrees(np.arctan2(ys, xs))
        for g, x, y, t in zip(gid, xs, ys, th):
            for sgn, c in ((+1.0, 90.0), (-1.0, -90.0)):
                dth = t - c
                if abs(dth) <= half_deg:
                    w = 1.0 - (dth / half_deg) ** 2
                    r = float(np.hypot(x, y))
                    seen[int(g)] = [sgn * w * x / r, sgn * w * y / r, 0.0]
    ids = np.array(sorted(seen), dtype=int)
    uv = np.array([seen[g] for g in ids])
    return ids, uv


def main():
    d, idx = ring_rect_domain(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                              wake_hold=8.0, wake_ratio=1.10)
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    roles = classify(d)
    body_faces = [(b, fid) for (b, fid), role in roles.items() if role == "body"]
    for (b, fid), role in roles.items():
        d.blocks[b].faces[fid] = kindmap[role]

    jet_ids, jet_uv = jet_geometry(d, body_faces)
    # uniform inlet: the inherited parabola assumes y in [0,1] and yields
    # O(100) values on the butterfly (j.5 caught the bare chain at 316 by
    # step 6, jets OFF -- the jets were never the driver)
    ch = MultiBlockVecChain(d, nu=NU, dt=DT, uniform_inlet=True,
                            jet_ids=jet_ids, jet_uv=jet_uv)
    N = ch.N
    print(f"  butterfly: {d.n_cells:,} cells, {len(d.blocks)} blocks, "
          f"{len(jet_ids)} jet nodes", flush=True)

    geom = face_geometry(d, body_faces)
    gids = {b: torch.as_tensor(d.global_ids(b).ravel()) for b in range(len(d.blocks))}
    shapes = {b: d.blocks[b].shape for b in range(len(d.blocks))}

    def blocks_of(flat):
        return {b: torch.index_select(flat, 0, gids[b]).reshape(shapes[b])
                for b in shapes}

    n_steps = 3
    rng = np.random.default_rng(0)
    S0 = [0.05 * rng.standard_normal(3 * N) for _ in range(n_steps)]
    EPS = 1e-3

    # ------------------------------------------------------------------ j.1
    src = [torch.tensor(s, requires_grad=True) for s in S0]
    ch.rollout(src).backward()
    g = [s.grad.detach().numpy().copy() for s in src]
    scale = max(np.abs(x).max() for x in g)
    worst = 0.0
    for k_step in (0, n_steps - 1):
        idxs = [int(np.argmax(np.abs(g[k_step]))),
                int(np.argmax(np.abs(g[k_step][N:2 * N]))) + N]
        for k in idxs:
            pert = [s.copy() for s in S0]
            pert[k_step][k] += EPS
            Lp = float(ch.rollout([torch.tensor(s) for s in pert]))
            pert[k_step][k] -= 2 * EPS
            Lm = float(ch.rollout([torch.tensor(s) for s in pert]))
            worst = max(worst, abs((Lp - Lm) / (2 * EPS) - g[k_step][k]) / scale)
    check(worst < 1e-4,
          f"j.1  vector FD on butterfly (u- and v-sources): worst {worst:.2e}")

    # ------------------------------------------------------------------ j.2
    zeros = [torch.zeros(3 * N) for _ in range(n_steps)]

    def traction_loss(a):
        vel, pf = ch.rollout(zeros, jet_a=a, return_fields=True)
        ub = blocks_of(ch.with_jets(vel[0], 0, a))
        vb = blocks_of(ch.with_jets(vel[1], 1, a))
        wb = blocks_of(vel[2])
        pb = blocks_of(pf)
        cd, cl = coefficients(geom, ub, vb, wb, pb, NU, span=1.0,
                              tangential_only=True)
        return cd

    a = torch.tensor(0.3, requires_grad=True)
    L = traction_loss(a)
    L.backward()
    ga = float(a.grad)
    with torch.no_grad():
        Lp = float(traction_loss(torch.tensor(0.3 + EPS)))
        Lm = float(traction_loss(torch.tensor(0.3 - EPS)))
    fd = (Lp - Lm) / (2 * EPS)
    rel = abs(ga - fd) / max(abs(fd), 1e-300)
    check(abs(ga) > 0 and rel < 1e-4,
          f"j.2  dC_D/da through traction: adjoint {ga:+.5e} FD {fd:+.5e} rel {rel:.2e}")

    # ------------------------------------------------------------------ j.3
    wake_ids = torch.as_tensor(d.global_ids(idx["wake"]).ravel())

    def wake_loss(a):
        vel, _ = ch.rollout(zeros, jet_a=a, return_fields=True)
        return sum((torch.index_select(vel[c], 0, wake_ids) ** 2).sum()
                   for c in range(3))

    a = torch.tensor(0.3, requires_grad=True)
    wake_loss(a).backward()
    gw = float(a.grad)
    with torch.no_grad():
        Lp = float(wake_loss(torch.tensor(0.3 + EPS)))
        Lm = float(wake_loss(torch.tensor(0.3 - EPS)))
    fd = (Lp - Lm) / (2 * EPS)
    rel = abs(gw - fd) / max(abs(fd), 1e-300)
    check(abs(gw) > 0 and rel < 1e-4,
          f"j.3  jet -> wake across seams: adjoint {gw:+.5e} FD {fd:+.5e} rel {rel:.2e}")

    # ------------------------------------------------------------------ j.4
    def wake_grad(**kw):
        a = torch.tensor(0.3, requires_grad=True)
        vel, _ = ch.rollout(zeros, jet_a=a, return_fields=True, **kw)
        sum((torch.index_select(vel[c], 0, wake_ids) ** 2).sum()
            for c in range(3)).backward()
        return float(a.grad)

    for name, kw in (("drop_pflux", dict(drop_pflux=True)),
                     ("detach_dong", dict(detach_dong=True))):
        gm = wake_grad(**kw)
        change = abs(gm - gw) / max(abs(gw), 1e-300)
        check(change > 1e-9, f"j.4  path live with jets: {name} {change:.2e}")

    # ------------------------------------------------------------------ j.5
    # inside the gate window only -- the surrogate's butterfly instability
    # beyond ~5 steps is a documented non-gate (docstring above)
    with torch.no_grad():
        v_on, _ = ch.rollout([torch.zeros(3 * N)] * 4, jet_a=torch.tensor(0.5),
                             return_fields=True)
        v_off, _ = ch.rollout([torch.zeros(3 * N)] * 4, return_fields=True)
        mx = max(float(c.abs().max()) for c in v_on)
        diff = max(float((a_ - b_).abs().max()) for a_, b_ in zip(v_on, v_off))
    check(np.isfinite(mx) and mx < 10.0 and diff > 1e-6,
          f"j.5  bounded in gate window (4 steps): max|vel| {mx:.3f}, "
          f"on-vs-off diff {diff:.3e}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
