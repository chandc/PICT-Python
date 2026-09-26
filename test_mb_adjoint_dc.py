"""Stage 7b -- the deferred-correction cross pressure solve, differentiated.

The production step gained `_solve_cross_dc` (the R11 stability cure) after
Stage 7 was scoped: the pressure solve became a TRUNCATED lagged iteration
M p = rhs + J div(Phi_cross(p)). `MultiBlockDCChain` differentiates that
algorithm AS EXECUTED -- the gradient is the exact discrete gradient of the
k-sweep iteration, not of the un-truncated fixed point. Gates:

  7b.0  the probed cross operator is LINEAR (it must be, for the matrix form
        to mean anything) and VANISHES on an orthogonal domain -- the same
        exact-math criterion production uses to skip zero-cross blocks.
  7b.1  the truncated iteration CONTRACTS and its limit solves the assembled
        full operator (M - C) directly: sign and scaling are not conventions
        to trust but facts to check.
  7b.2  FD vs adjoint through a 3-step rollout with sweeps=3 on a SHEARED
        duct (cross terms live), ~6 digits.
  7b.3  detaching the lagged cross term in the backward is DETECTED.
  7b.4  consistency both ways: on the orthogonal duct the DC chain must
        reproduce `MultiBlockBCChain` exactly (C = 0 => the sweeps are
        no-ops); on the sheared duct sweeps=3 vs sweeps=0 must DIFFER.
  7b.5  the adjoint stays contractive over 20 steps with DC on.

The shear x -> x + 0.35 y is volume-preserving (det J unchanged), so it
turns on the cross metrics without touching cell sizes.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import torch

from src.mb_adjoint import (MultiBlockBCChain, MultiBlockDCChain,
                            cross_divergence_matrix, square_duct)

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


def sheared_duct(ntot=10, n_split=2, alpha=0.35):
    d = square_duct(ntot, n_split)
    for blk in d.blocks:
        blk.x[:] = blk.x + alpha * blk.y
    return d


print("\n" + "=" * 76 + "\n  Stage 7b -- the deferred-correction pressure solve\n" + "=" * 76)

rng = np.random.default_rng(0)
duct = square_duct(10, 2)
sduct = sheared_duct(10, 2)

ch0 = MultiBlockDCChain(duct, NU, DT, sweeps=3)
ch = MultiBlockDCChain(sduct, NU, DT, sweeps=3)
bc = MultiBlockBCChain(duct, NU, DT)
N = ch.N

# ------------------------------------------------------------------- 7b.0
scaleM = abs(ch.M_ff).max()
c0max, cmax = abs(ch0.Cx).max() if ch0.Cx.nnz else 0.0, abs(ch.Cx).max()
p, q = rng.standard_normal(N), rng.standard_normal(N)
lin = np.linalg.norm(ch.Cx @ (0.3 * p + 0.7 * q) - 0.3 * (ch.Cx @ p) - 0.7 * (ch.Cx @ q))
check(c0max <= 1e-10 * scaleM and cmax > 1e-3 * scaleM and lin < 1e-10,
      f"7b.0  C vanishes on the orthogonal duct (max {c0max:.1e}), is substantial on the "
      f"sheared one (max {cmax:.3f} vs |M| {scaleM:.3f}), and is linear (defect {lin:.1e})")

# ------------------------------------------------------------------- 7b.1
free = ch.free
C_ff = ch.Cx[free][:, free].tocsc()
M_ff = ch.M_ff.tocsc()
r = rng.standard_normal(len(free))
lu = spla.splu(M_ff)
x = lu.solve(r)
steps = []
for k in range(12):
    x_new = lu.solve(r + C_ff @ x)
    steps.append(np.linalg.norm(x_new - x))
    x = x_new
# measure contraction over EARLY sweeps: once the iterate reaches the
# roundoff floor, consecutive step norms are noise and their ratio is
# meaningless (the first version read 1.14 off the floor and "failed" a
# converged iteration)
ratio = (steps[4] / steps[1]) ** (1.0 / 3.0)
x_direct = spla.spsolve((M_ff - C_ff).tocsc(), r)
rel = np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct)
check(ratio < 1.0 and rel < 1e-6,
      f"7b.1  the lagged iteration contracts (early-sweep factor {ratio:.3f}) and its limit "
      f"solves the assembled (M - C): rel diff {rel:.2e} after 12 sweeps")

# ------------------------------------------------------------------- 7b.2
nsrc = 3
S0 = [0.1 * rng.standard_normal(N) for _ in range(nsrc)]


def grad_of(chain, S0, **kw):
    src = [torch.tensor(a, requires_grad=True) for a in S0]
    chain.rollout(src, **kw).backward()
    return [s.grad.detach().numpy().copy() for s in src]


g = grad_of(ch, S0)
scale = max(np.abs(x).max() for x in g)
worst = 0.0
for k_step in (0, nsrc - 1):
    for k in np.argsort(-np.abs(g[k_step]))[:3]:
        pert = [a.copy() for a in S0]
        pert[k_step][k] += EPS
        Lp = float(ch.rollout([torch.tensor(a) for a in pert]))
        pert[k_step][k] -= 2 * EPS
        Lm = float(ch.rollout([torch.tensor(a) for a in pert]))
        worst = max(worst, abs((Lp - Lm) / (2 * EPS) - g[k_step][k]) / scale)
check(worst < 1e-6,
      f"7b.2  FD vs adjoint through 3 DC steps (sweeps=3, sheared duct): worst {worst:.2e} "
      f"of max|g| = {scale:.3e}")

# ------------------------------------------------------------------- 7b.3
gm = grad_of(ch, S0, detach_cross=True)
diff = max(np.abs(a - b).max() for a, b in zip(g, gm)) / scale
check(diff > 1e-8,
      f"7b.3  detaching the lagged cross term in the backward is DETECTED: gradients "
      f"differ by {diff:.2e} of max|g|")

# ------------------------------------------------------------------- 7b.4
g_bc = grad_of(bc, S0)
g_dc0 = grad_of(ch0, S0)
same = max(np.abs(a - b).max() for a, b in zip(g_bc, g_dc0)) / max(
    np.abs(np.concatenate(g_bc)).max(), 1e-300)
g_nosweep = grad_of(ch, S0, sweeps=0)
diff_s = max(np.abs(a - b).max() for a, b in zip(g, g_nosweep)) / scale
check(same < 1e-11 and diff_s > 1e-6,
      f"7b.4  C = 0 => the DC chain IS the BC chain (grad diff {same:.2e}); sheared => "
      f"sweeps matter (sweeps 3 vs 0 differ by {diff_s:.2e})")

# ------------------------------------------------------------------- 7b.5
nsteps = 20
norms = []
for k in (0, 5, 10, 15, 19):
    src = [torch.zeros(N, dtype=torch.float64) for _ in range(nsteps)]
    src[k] = torch.tensor(0.1 * rng.standard_normal(N), requires_grad=True)
    ch.rollout(src, final_only=True).backward()
    norms.append(float(torch.linalg.norm(src[k].grad)))
per_step = (norms[0] / norms[-1]) ** (1.0 / (nsteps - 1))
check(per_step < 1.05,
      f"7b.5  adjoint per-step factor going BACK in time with DC on: {per_step:.4f} "
      f"(norms earliest {norms[0]:.3e} latest {norms[-1]:.3e})")

print("\n" + "=" * 76 + f"\n  {PASS}/{PASS + FAIL} checks passed\n" + "=" * 76)
import sys
sys.exit(1 if FAIL else 0)
