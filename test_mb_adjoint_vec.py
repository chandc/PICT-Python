"""M1 gates -- the VECTOR chain (u, v, w coupled by the pressure projection).

The FluidGym-parity substrate (reference/fluidgym_parity.md): everything the
scalar chains certified, re-certified with three components and the one new
physics path -- CROSS-COMPONENT sensitivity through phi. Gates:

  v.1  FD vs adjoint through 3 vector steps on the square duct, sources
       perturbed on ALL components, ~6 digits.
  v.2  cross-component seam crossing: source on u ONLY in one block, loss on
       v ONLY in the other block. The only route is u -> divergence -> phi ->
       grad_y phi -> v, ACROSS the seam: a pass certifies the projection
       coupling and nothing else can fake it.
  v.3  the carried-state and Dong mangles still detected with vector state.
  v.4  forward sanity: 10 uncontrolled steps stay bounded. (|div| of CELL
       velocities is reported, not gated: this discretization keeps FACE
       fluxes solenoidal, not the cell-centred field -- the same semantics
       as production, see the flux-correction note in piso_multiblock.)
"""
import numpy as np
import torch

from src.mb_adjoint import MultiBlockVecChain, square_duct, to_torch_sparse, spmv

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


print("\n" + "=" * 76 + "\n  M1 -- the vector chain\n" + "=" * 76)

rng = np.random.default_rng(0)
d = square_duct(10, 2)
ch = MultiBlockVecChain(d, NU, DT)
N = ch.N
nsrc = 3
S0 = [0.1 * rng.standard_normal(3 * N) for _ in range(nsrc)]


def grad_of(S0, **kw):
    src = [torch.tensor(a, requires_grad=True) for a in S0]
    ch.rollout(src, **kw).backward()
    return [s.grad.detach().numpy().copy() for s in src]


# ------------------------------------------------------------------- v.1
g = grad_of(S0)
scale = max(np.abs(x).max() for x in g)
worst = 0.0
for k_step in (0, nsrc - 1):
    idxs = list(np.argsort(-np.abs(g[k_step]))[:2])
    idxs += [int(np.argmax(np.abs(g[k_step][N:2 * N]))) + N]   # a v-component entry
    for k in idxs:
        pert = [a.copy() for a in S0]
        pert[k_step][k] += EPS
        Lp = float(ch.rollout([torch.tensor(a) for a in pert]))
        pert[k_step][k] -= 2 * EPS
        Lm = float(ch.rollout([torch.tensor(a) for a in pert]))
        worst = max(worst, abs((Lp - Lm) / (2 * EPS) - g[k_step][k]) / scale)
check(worst < 1e-6,
      f"v.1  FD vs adjoint, 3 vector steps, u- and v-sources: worst {worst:.2e} of "
      f"max|g| = {scale:.3e}")

# ------------------------------------------------------------------- v.2
gid0 = set(d.global_ids(0).ravel().tolist())
b1_ids = torch.as_tensor(d.global_ids(1).ravel())
theta = torch.zeros(3 * N, dtype=torch.float64, requires_grad=True)
mask_u_b0 = torch.zeros(3 * N, dtype=torch.float64)
for gidx in gid0:
    mask_u_b0[gidx] = 1.0                                    # u-slots of block 0 only
src = [0.05 * mask_u_b0 * (1.0 + theta)] + \
      [torch.zeros(3 * N, dtype=torch.float64)] * (nsrc - 1)
vel, _ = ch.rollout(src, return_fields=True)
Lv = (torch.index_select(vel[1], 0, b1_ids) ** 2).sum()      # v in block 1 only
Lv.backward()
gth = float(torch.abs(theta.grad * mask_u_b0).max())
# FD on the largest entry
kbig = int(torch.argmax(torch.abs(theta.grad * mask_u_b0)))
def loss_at(dv):
    t2 = torch.zeros(3 * N, dtype=torch.float64)
    t2[kbig] = dv
    s2 = [0.05 * mask_u_b0 * (1.0 + t2)] + \
         [torch.zeros(3 * N, dtype=torch.float64)] * (nsrc - 1)
    vel2, _ = ch.rollout(s2, return_fields=True)
    return float((torch.index_select(vel2[1], 0, b1_ids) ** 2).sum())
fd = (loss_at(EPS) - loss_at(-EPS)) / (2 * EPS)
rel = abs(float(theta.grad[kbig]) - fd) / max(abs(fd), 1e-300)
check(gth > 0 and rel < 1e-5,
      f"v.2  u-source in block 0 -> v-loss in block 1 (pressure coupling across the "
      f"seam): max|g| {gth:.3e}, FD rel {rel:.2e}")

# ------------------------------------------------------------------- v.3
for name, kw in (("drop_pflux", dict(drop_pflux=True)),
                 ("drop_history", dict(drop_history=True)),
                 ("detach_dong", dict(detach_dong=True))):
    gm = grad_of(S0, **kw)
    diff = max(np.abs(a - b).max() for a, b in zip(g, gm)) / scale
    check(diff > 1e-9, f"v.3  {name} DETECTED with vector state: {diff:.2e}")

# ------------------------------------------------------------------- v.4
with torch.no_grad():
    zero = [torch.zeros(3 * N, dtype=torch.float64)] * 10
    vel, pf = ch.rollout(zero, return_fields=True)
    mx = max(float(c.abs().max()) for c in vel)
    Dall = to_torch_sparse(ch.D_flux)
    div_c = float(spmv(Dall, torch.cat(vel)).abs().max())
check(np.isfinite(mx) and mx < 10.0,
      f"v.4  10 uncontrolled steps bounded: max|vel| = {mx:.3f} (cell-field |div| "
      f"{div_c:.2e}, reported not gated -- face fluxes are the solenoidal object)")

print("\n" + "=" * 76 + f"\n  {PASS}/{PASS + FAIL} checks passed\n" + "=" * 76)
import sys
sys.exit(1 if FAIL else 0)
