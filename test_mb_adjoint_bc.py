"""Adjoint gates on REAL boundary conditions: developing channel flow and a square duct.

Every earlier gate leaned on a boundary condition that made something vanish. Periodic: no
Dirichlet nodes at all. Walls: Dirichlet, but zero, so the elimination term A_ib u_bnd drops out
and never gets exercised. Those are convenient cases, and convenience is what hides defects.

  developing channel   2 walls, spanwise periodic, inlet and outlet
  square duct          4 walls, inlet and outlet

Both carry all three kinds of Dirichlet node at once, and a corner belongs to the most
restrictive:

  wall    u = 0                      constant
  inlet   u = a parabolic profile    non-zero DATA -- this is what makes A_ib u_bnd real
  outlet  u = the first interior node  SOLUTION-DEPENDENT -- so it carries a gradient

and the outflow additionally prescribes Dong's pressure, itself a function of the interior
velocity. So the outflow contributes TWO distinct gradient paths that a periodic or purely
walled case does not have, and the controls below check that each is actually in the graph
rather than assuming it.
"""
import numpy as np
import torch

from src.mb_adjoint import MultiBlockBCChain, developing_channel, square_duct

torch.set_default_dtype(torch.float64)
NU, DT, EPS = 0.05, 0.02, 1e-2
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def fd_vs_adjoint(chain, S0, **kw):
    src = [torch.tensor(a, requires_grad=True) for a in S0]
    chain.rollout(src, **kw).backward()
    g = [s.grad.detach().numpy().copy() for s in src]
    scale = max(np.abs(x).max() for x in g)
    worst = 0.0
    for k_step in (0, len(S0) - 1):
        gg = g[k_step]
        for k in np.argsort(-np.abs(gg))[:3]:
            pert = [a.copy() for a in S0]
            pert[k_step][k] += EPS
            Lp = float(chain.rollout([torch.tensor(a) for a in pert], **kw))
            pert[k_step][k] -= 2 * EPS
            Lm = float(chain.rollout([torch.tensor(a) for a in pert], **kw))
            worst = max(worst, abs((Lp - Lm) / (2 * EPS) - gg[k]) / scale)
    return worst, g, scale


print("\n" + "=" * 80 + "\n  Adjoint gates on real boundary conditions\n" + "=" * 80)

for name, dom in (("developing channel", developing_channel(10, 2)),
                  ("square duct", square_duct(10, 2))):
    print(f"\n  --- {name} ---")
    ch = MultiBlockBCChain(dom, NU, DT)
    print(f"      {dom.n_cells:,} cells | wall {len(ch.wall_ids)} inlet {len(ch.in_ids)} "
          f"outlet {len(ch.out_ids)} | unknowns {len(ch.interior)} | Dong {len(ch.pD)}")
    rng = np.random.default_rng(7)
    S0 = [rng.standard_normal(ch.N) * 0.05 for _ in range(3)]

    worst, g_ref, scale = fd_vs_adjoint(ch, S0)
    check(worst < 1e-6,
          f"FD vs adjoint, all three BC kinds live: worst {worst:.2e} of max|g| = {scale:.3e}")

    # the elimination term must MATTER: with a zero inlet the trajectory has to differ
    ch0 = MultiBlockBCChain(dom, NU, DT, u_inlet=0.0)
    L_in = float(ch.rollout([torch.tensor(a) for a in S0]))
    L_no = float(ch0.rollout([torch.tensor(a) for a in S0]))
    rel = abs(L_in - L_no) / abs(L_in)
    check(rel > 1e-6,
          f"the inlet's A_ib u_bnd is a real contribution: loss {L_in:.6e} with the profile vs "
          f"{L_no:.6e} without, {rel:.1%} apart. A zero wall hides this term entirely")

    # the outlet's solution-dependence carries a gradient
    srcd = [torch.tensor(a, requires_grad=True) for a in S0]
    ch.rollout(srcd, drop_outlet=True).backward()
    gd = [s.grad.detach().numpy().copy() for s in srcd]
    rel_o = max(np.abs(a - b).max() for a, b in zip(g_ref, gd)) / scale
    check(rel_o > 1e-12,
          f"the OUTLET velocity depends on the solution and that path is in the graph: "
          f"detaching it moves dL/dS by {rel_o:.2e} of max|g|")

    # Dong's pressure is a second, independent path through the same faces
    srcp = [torch.tensor(a, requires_grad=True) for a in S0]
    ch.rollout(srcp, detach_dong=True).backward()
    gp = [s.grad.detach().numpy().copy() for s in srcp]
    rel_d = max(np.abs(a - b).max() for a, b in zip(g_ref, gp)) / scale
    check(rel_d > 1e-12,
          f"and Dong's PRESSURE is a second, independent path through the same faces: "
          f"{rel_d:.2e} of max|g|")

    # sensitivity must still cross the seam, with walls and an inlet in the way
    idx_A, idx_B = ch.block_slice(0), ch.block_slice(1)
    theta = torch.tensor(rng.standard_normal(len(idx_A)) * 0.05, requires_grad=True)

    def loss_A(th):
        S = torch.zeros(ch.N, dtype=torch.float64).index_put((torch.as_tensor(idx_A),), th)
        u, _ = ch.rollout([S], return_fields=True)
        return (torch.index_select(u, 0, torch.as_tensor(idx_B)) ** 2).sum()

    loss_A(theta).backward()
    gA = theta.grad.detach().numpy().copy()
    base = theta.detach().numpy()
    w6 = 0.0
    for k in np.argsort(-np.abs(gA))[:3]:
        d = np.zeros(len(idx_A)); d[k] = EPS
        fd = (float(loss_A(torch.tensor(base + d))) - float(loss_A(torch.tensor(base - d)))) \
            / (2 * EPS)
        w6 = max(w6, abs(fd - gA[k]) / max(np.abs(gA).max(), 1e-300))
    check(np.abs(gA).max() > 1e-10 and w6 < 1e-6,
          f"sensitivity still crosses the seam: max|dL/dtheta| = {np.abs(gA).max():.3e}, "
          f"FD agrees to {w6:.2e}")

print("\n" + "=" * 80)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 80)
raise SystemExit(1 if FAIL else 0)
