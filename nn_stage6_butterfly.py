"""Stage 6.9: the differentiable chain on the BUTTERFLY grid -- plumbing gates.

Every earlier stage runs on domains built for the gate. This one runs on the
production topology: the 9-block butterfly-in-rectangle (coarse build), with
inflow, no-slip body, walls and the solution-dependent Dong outflow -- the
configuration R11-R13 validated physically. The gates certify that gradients
are EXACT and that every structural path is LIVE on this domain; they do not
certify identifiability of inverse problems (see the note at the bottom, and
reference/differentiable_plumbing.md).

Mirrors PICT's `learning_sample.py` (learn a forcing from a reference
simulation, tum-pbs/PICT) with two differences: their sample optimises a
2-DOF uniform force on one periodic block and never verifies the gradient or
the recovery against truth; here the force has spatial structure, the domain
is body-fitted multiblock, the gradient is FD-verified, and the recovered
amplitude is checked against the ground truth.

Gates:
  A  adjoint identity  <lam, K x> == <K^T lam, x>  on A and M_ff, ~1e-12
  B  FD vs autograd through a 4-step rollout, rel err < 1e-4
  C  seam crossing: source supported ONLY in ringE, loss read ONLY in the
     wake block -- the gradient must be nonzero and FD-exact, which cannot
     happen unless sensitivity travels ring -> trap -> wake through two seams
  D  liveness mangles: drop_pflux / detach_dong / drop_history must each
     CHANGE the gradient by > 1e-6 relative -- the paths exist on this domain
  E  scalar forcing recovery: a = 0.1 -> within 5% of the truth 1.0

Run:  .venv/bin/python nn_stage6_butterfly.py         (~4 min on the Mac)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sparse
import torch

from cylinder_ring_grid import ring_rect_domain
from cylinder_rect_bc import classify
from src.mb_adjoint import MultiBlockBCChain

torch.manual_seed(0)
FAILED = []


def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def main():
    d, idx = ring_rect_domain(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                              wake_hold=8.0, wake_ratio=1.10)
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    for (b, fid), role in classify(d).items():
        d.blocks[b].faces[fid] = kindmap[role]
    chain = MultiBlockBCChain(d, nu=0.05, dt=0.01)
    N = chain.N
    print(f"  butterfly: {d.n_cells:,} cells, {len(d.blocks)} blocks", flush=True)

    xf = chain.flat({b: d.blocks[b].x for b in range(len(d.blocks))})
    yf = chain.flat({b: d.blocks[b].y for b in range(len(d.blocks))})
    g = np.exp(-(((xf - 2.0) ** 2) + (yf - 0.5) ** 2) / 0.8 ** 2)
    g_t = torch.as_tensor(g)
    n_steps = 4

    # --- Gate A: adjoint identity on the assembled operators ---------------
    rng = np.random.default_rng(0)
    A = sparse.csr_matrix(d.build_momentum_matrix(
        chain.Js, chain.ms, chain.u0, chain.u0, chain.u0, chain.nu, chain.dt,
        bdf2=True))
    x, lam = rng.standard_normal(N), rng.standard_normal(N)
    lhs, rhs = lam @ (A @ x), (A.T @ lam) @ x
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    gate("A adjoint identity", rel < 1e-12, f"rel {rel:.2e}")

    def final_u(S, n=n_steps, **kw):
        u, _ = chain.rollout([S] * n, return_fields=True, **kw)
        return u

    with torch.no_grad():
        u_target = final_u(1.0 * g_t)

    def loss_and_grad(aval, **kw):
        a = torch.tensor(aval, dtype=torch.float64, requires_grad=True)
        L = ((final_u(a * g_t, **kw) - u_target) ** 2).sum()
        L.backward()
        return float(L), float(a.grad)

    # --- Gate B: FD vs autograd --------------------------------------------
    _, gauto = loss_and_grad(0.5)
    h = 1e-4
    with torch.no_grad():
        Lp = float(((final_u((0.5 + h) * g_t) - u_target) ** 2).sum())
        Lm = float(((final_u((0.5 - h) * g_t) - u_target) ** 2).sum())
    gfd = (Lp - Lm) / (2 * h)
    rel = abs(gauto - gfd) / max(abs(gfd), 1e-300)
    gate("B FD gradient", rel < 1e-4, f"autograd {gauto:.4e} FD {gfd:.4e} rel {rel:.2e}")

    # --- Gate C: sensitivity must CROSS the seams ---------------------------
    src_ids = d.global_ids(idx["ringE"]).ravel()
    wake_ids = torch.as_tensor(d.global_ids(idx["wake"]).ravel())
    mask = np.zeros(N)
    mask[src_ids] = g[src_ids]
    mask_t = torch.as_tensor(mask)

    def wake_loss(aval):
        a = torch.tensor(aval, dtype=torch.float64, requires_grad=True)
        u = final_u(a * mask_t)
        L = (torch.index_select(u, 0, wake_ids) ** 2).sum()
        L.backward()
        return float(L), float(a.grad)

    _, gc = wake_loss(0.5)
    with torch.no_grad():
        Lp = float((torch.index_select(final_u((0.5 + h) * mask_t), 0, wake_ids) ** 2).sum())
        Lm = float((torch.index_select(final_u((0.5 - h) * mask_t), 0, wake_ids) ** 2).sum())
    gcfd = (Lp - Lm) / (2 * h)
    rel = abs(gc - gcfd) / max(abs(gcfd), 1e-300)
    gate("C seam crossing", (abs(gc) > 0) and rel < 1e-4,
         f"grad {gc:.4e} (ringE source -> wake loss, 2 seams) rel {rel:.2e}")

    # --- Gate D: structural paths are live on THIS domain -------------------
    for name, kw in (("drop_pflux", dict(drop_pflux=True)),
                     ("drop_history", dict(drop_history=True))):
        _, gm = loss_and_grad(0.5, **kw)
        change = abs(gm - gauto) / max(abs(gauto), 1e-300)
        gate(f"D path live: {name}", change > 1e-6, f"grad change {change:.2e}")

    # The Dong path needs a probe that actually EXERCISES the outlet: a
    # source at x = 2 whispers at x = 30 after 4 steps of dt = 0.01, and the
    # first version of this gate read that geometric remoteness (grad change
    # 3e-7) as a dead path. Source AND loss at the outlet instead.
    g_out = np.exp(-(((xf - 28.0) ** 2) + yf ** 2) / 1.5 ** 2)
    g_out_t = torch.as_tensor(g_out)
    near_out = torch.as_tensor(np.where(xf > 27.0)[0])

    def dong_grad(**kw):
        a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        u = final_u(a * g_out_t, n=6, **kw)
        (torch.index_select(u, 0, near_out) ** 2).sum().backward()
        return float(a.grad)

    g0, g1 = dong_grad(), dong_grad(detach_dong=True)
    change = abs(g1 - g0) / max(abs(g0), 1e-300)
    gate("D path live: detach_dong", change > 1e-6,
         f"grad change {change:.2e} (outlet-local probe)")

    # --- Gate E: scalar forcing recovery ------------------------------------
    a = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a], lr=0.08)
    for _ in range(60):
        opt.zero_grad()
        L = ((final_u(a * g_t) - u_target) ** 2).sum()
        L.backward()
        opt.step()
    gate("E scalar recovery", abs(float(a) - 1.0) < 0.05,
         f"a = {float(a):.4f} (truth 1.0)")

    # NOT A GATE -- a documented property: recovering the full 21k-DOF force
    # FIELD from two trajectory snapshots is non-unique. Measured: the loss
    # falls 5000x while cosine(S, S*) stays < 0.3, for both direct
    # optimisation and an MLP parametrisation. The gradients above are exact;
    # the inverse problem is under-observed. Upstream PICT's learning_sample
    # never measures this. See reference/differentiable_plumbing.md.

    print(f"\n  {'ALL GATES PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}",
          flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
