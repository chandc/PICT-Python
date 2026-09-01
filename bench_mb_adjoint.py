"""Stage 9, part 1 — what the sparse operators cost, and that they changed no answer.

Plan: `reference/nn_multiblock_plan.md`. Stage 7's chain materialised three N x N DENSE matrices
because 1,728 cells made that invisible. It does not stay invisible: the same code at the square
cylinder's 82,096 cells asks for 54 GB per matrix, 162 GB for the three.

This measures three things and asserts the one that matters:

  * the gradient is UNCHANGED by the switch -- a performance change that moves an answer is a
    bug, and the whole point of the sparse form is that it is the same operator;
  * the memory the dense form would have needed against what the sparse form does need;
  * time per gradient, so Stage 9's "<= 4x a forward-only step" has a baseline.

The dense path is kept alive here only as the reference to compare against.
"""
import time

import numpy as np
import torch

from src.mb_adjoint import MultiBlockChainRC, periodic_box, to_torch_sparse

torch.set_default_dtype(torch.float64)
NU, DT = 0.05, 0.02
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def dense_rollout(chain, sources):
    """The pre-sparse implementation, kept as the reference the sparse form must reproduce."""
    from src.adjoint_piso import LinearSolve
    (Aidx, Ashape), Aval = chain.A_pat
    (Midx, Mshape), Mval = chain.M_pat
    Gt = torch.as_tensor(chain.G.toarray())
    Du = torch.as_tensor(chain.D_flux[:, :chain.N].toarray())
    RCt = torch.as_tensor(chain.RC.toarray())
    u = torch.as_tensor(chain.u_init)
    u_prev = torch.as_tensor(chain.u_init)
    p_flux = torch.zeros(chain.N, dtype=torch.float64)
    Jt = torch.as_tensor(chain.J_flat)
    L = 0.0
    for S in sources:
        rhs = Jt * (2.0 * u - 0.5 * u_prev) / chain.dt + S
        u_star = LinearSolve.apply(Aval, rhs, (Aidx, Ashape), False, False)
        phi = LinearSolve.apply(Mval, Du @ u_star - RCt @ p_flux, (Midx, Mshape), True, True)
        u_prev, u = u, u_star - chain.dt * (Gt @ phi)
        p_flux = p_flux + phi
        L = L + (u ** 2).sum()
    return L


print("\n" + "=" * 78 + "\n  Stage 9 part 1 — sparse operators: same answer, and the cost\n"
      + "=" * 78)

chain = MultiBlockChainRC(periodic_box(12, 2), NU, DT)
rng = np.random.default_rng(31)
S0 = [rng.standard_normal(chain.N) * 0.05 for _ in range(4)]

src_s = [torch.tensor(a, requires_grad=True) for a in S0]
Ls = chain.rollout(src_s)
Ls.backward()
gs = [s.grad.detach().numpy().copy() for s in src_s]

src_d = [torch.tensor(a, requires_grad=True) for a in S0]
Ld = dense_rollout(chain, src_d)
Ld.backward()
gd = [s.grad.detach().numpy().copy() for s in src_d]

dL = abs(float(Ls) - float(Ld)) / abs(float(Ld))
dg = max(np.abs(a - b).max() for a, b in zip(gs, gd)) / max(np.abs(g).max() for g in gd)
check(dL < 1e-14 and dg < 1e-12,
      f"9.a  sparse and dense agree: loss rel {dL:.2e}, gradient rel {dg:.2e} "
      f"(a speed-up that moves an answer is a bug)")

print(f"\n  {'operator':>10}{'shape':>16}{'nnz':>10}{'sparse MB':>12}{'dense MB':>11}"
      f"{'ratio':>9}")
tot_s = tot_d = 0.0
for name, mat in (("G", chain.G), ("D_flux", chain.D_flux[:, :chain.N]), ("RC", chain.RC)):
    nnz = mat.nnz
    sp = nnz * 12 / 1e6                       # 8-byte value + two 4-byte-ish indices, coalesced
    dn = mat.shape[0] * mat.shape[1] * 8 / 1e6
    tot_s += sp; tot_d += dn
    print(f"  {name:>10}{str(mat.shape):>16}{nnz:>10,}{sp:>12.2f}{dn:>11.2f}{dn/sp:>9.1f}x")
print(f"  {'total':>10}{'':>16}{'':>10}{tot_s:>12.2f}{tot_d:>11.2f}{tot_d/tot_s:>9.1f}x")

# extrapolate to the production case: nnz per row is a property of the stencil, not of N
rows_per_cell = sum(m.nnz for m in (chain.G, chain.D_flux[:, :chain.N], chain.RC)) / chain.N
for N_prod, label in ((82096, "square cylinder"), (89088, "circular cylinder")):
    sp = rows_per_cell * N_prod * 12 / 1e9
    dn = 3 * N_prod ** 2 * 8 / 1e9
    print(f"  extrapolated to {label} ({N_prod:,} cells): sparse {sp:.3f} GB, "
          f"dense {dn:.0f} GB")

check(tot_d / tot_s > 20,
      f"9.b  sparse is {tot_d/tot_s:.0f}x smaller at this size, and the ratio grows linearly "
      f"with N -- 54 GB per matrix, 162 GB for the three, at the square cylinder")

reps = 5
t0 = time.time()
for _ in range(reps):
    src = [torch.tensor(a, requires_grad=True) for a in S0]
    chain.rollout(src).backward()
t_grad = (time.time() - t0) / reps
t0 = time.time()
for _ in range(reps):
    with torch.no_grad():
        chain.rollout([torch.tensor(a) for a in S0])
t_fwd = (time.time() - t0) / reps
print(f"\n  forward only {t_fwd*1000:.1f} ms;  forward+backward {t_grad*1000:.1f} ms;  "
      f"ratio {t_grad/t_fwd:.2f}x over {len(S0)} steps")
check(t_grad / t_fwd < 4.0,
      f"9.c  a gradient costs {t_grad/t_fwd:.2f}x a forward, against the plan's bar of 4x "
      f"(the backward is two extra linear solves per step)")

print("=" * 78)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 78)
raise SystemExit(1 if FAIL else 0)
