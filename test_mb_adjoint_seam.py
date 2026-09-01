"""Stage 6 — two blocks that are one block: the seam gate for the multi-block adjoint.

Plan: `reference/nn_multiblock_plan.md`. The differentiable path (`piso_torch`, `adjoint_piso`)
runs on ONE block; every case the port simulates runs on `MultiBlockPISO`, which has no
gradients. Before any of that is built, the seam has to be shown to behave — and the way to show
it is the trick `test_multiblock.py` used on the forward: split a periodic box into blocks joined
by a connection, and require the answer to be IDENTICAL to the unsplit one. Same physics, so any
difference is indexing.

Cases 6.1-6.3 need no new solver code: they check the forward equivalence, the symmetry of the
assembled operator, and the adjoint identity on the GLOBAL momentum matrix. 6.4-6.6 need the
differentiable multi-block step and land with it.

Why the momentum matrix and not the pressure matrix for the adjoint identity: it is the
non-symmetric one, so a forgotten transpose is fatal there and invisible in the symmetric solve.
The control below asserts that forgetting it actually fails.
"""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from src.multiblock import Block, Connection, Domain, face_id
from src.piso_multiblock import MultiBlockPISO
from src.mb_adjoint import MultiBlockMiniPISO, alignment, periodic_box

torch.set_default_dtype(torch.float64)

NTOT, NU, DT, TOL = 12, 0.05, 0.02, 1e-13
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def taylor_green(blk):
    x, y, z = 2 * np.pi * blk.x, 2 * np.pi * blk.y, 2 * np.pi * blk.z
    return (np.sin(x) * np.cos(y) * np.cos(z),
            -np.cos(x) * np.sin(y) * np.cos(z),
            np.zeros_like(x))


def build_solver(d):
    m = MultiBlockPISO(d, NU, DT, 2, TOL, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b, blk in enumerate(d.blocks):
        m.u[b][:], m.v[b][:], m.w[b][:] = taylor_green(blk)
    return m


def assemble(d, m):
    """(global momentum matrix, global pressure/diffusion matrix) for the current state."""
    nb = len(d.blocks)
    Js = [d.block_metrics_cached(b)[0] for b in range(nb)]
    ms = [d.block_metrics_cached(b)[1] for b in range(nb)]
    A = d.build_momentum_matrix(Js, ms, m.u, m.v, m.w, NU, DT, bdf2=False)
    M = d.build_diffusion_matrix(Js, ms)
    return sparse.csr_matrix(A), sparse.csr_matrix(M)


print("\n" + "=" * 74 + "\n  Stage 6 — two blocks that are one block\n" + "=" * 74)

# ---------------------------------------------------------------- 6.1 forward equivalence
d1, d2 = periodic_box(NTOT, 1), periodic_box(NTOT, 2)
m1, m2 = build_solver(d1), build_solver(d2)
u0 = np.concatenate([m2.u[b] for b in range(2)], axis=0)
check(np.abs(u0 - m1.u[0]).max() == 0.0,
      f"initial states identical before any step: max|du| = {np.abs(u0 - m1.u[0]).max():.1e}")

for step in range(3):
    m1.step()
    m2.step()
u1 = np.concatenate([m2.u[b] for b in range(2)], axis=0)
v1 = np.concatenate([m2.v[b] for b in range(2)], axis=0)
p1 = np.concatenate([m2.p[b] for b in range(2)], axis=0)
eu = np.abs(u1 - m1.u[0]).max()
ev = np.abs(v1 - m1.v[0]).max()
ep = np.abs(p1 - m1.p[0]).max()
check(max(eu, ev) < 1e-14,
      f"6.1  three steps, 2 blocks vs 1: max|du| = {eu:.2e}, max|dv| = {ev:.2e} "
      f"(pressure {ep:.2e}, defined up to a constant)")

# ---------------------------------------------------------------- 6.2 operator symmetry
A2, M2 = assemble(d2, m2)
asym = np.abs((M2 - M2.T).data).max() if (M2 - M2.T).nnz else 0.0
check(asym == 0.0, f"6.2  global diffusion operator is symmetric across the seam: "
                   f"max|M - M^T| = {asym:.1e} (exactly zero required)")

A1, M1 = assemble(d1, m1)
dm = np.abs((M2 - M1).data).max() if (M2 - M1).nnz else 0.0
check(dm < 1e-12, f"     and identical to the unsplit operator: max|M_2 - M_1| = {dm:.2e}")

# ---------------------------------------------------------------- 6.3 adjoint identity
rng = np.random.default_rng(0)
N = A2.shape[0]
vv, ww = rng.standard_normal(N), rng.standard_normal(N)

# THE CONTROL MUST BE RUN WHERE THE NON-SYMMETRIC PART MATTERS, and showing that is better
# than tuning a threshold. A carries J/dt on the diagonal and its only non-symmetric part is
# convection, so in a diffusion-dominated, small-dt corner the transpose barely moves an inner
# product -- 1.1% at cell Peclet 1.7 and dt = 0.02, which a 1e-2 bar would pass on a
# technicality. The probe therefore sweeps both knobs and reports the cell Peclet, and the bar
# is set at the convection-dominated corner, where the single-block Stage 0 gate measured 24.5%.
Js = [d2.block_metrics_cached(b)[0] for b in range(2)]
ms = [d2.block_metrics_cached(b)[1] for b in range(2)]
u0d = {b: taylor_green(blk)[0] for b, blk in enumerate(d2.blocks)}
v0d = {b: taylor_green(blk)[1] for b, blk in enumerate(d2.blocks)}
w0d = {b: taylor_green(blk)[2] for b, blk in enumerate(d2.blocks)}
print(f"       {'|u| x':>7}{'dt':>7}{'cell Pe':>10}{'identity rel':>15}{'no-transpose':>15}")
ident, ctrl, hard = [], [], 0.0
for scale in (1.0, 10.0):
    us = {b: scale * u0d[b] for b in u0d}
    vs = {b: scale * v0d[b] for b in v0d}
    ws = {b: scale * w0d[b] for b in w0d}
    for dt_probe in (0.02, 2.0):
        Ap = sparse.csr_matrix(d2.build_momentum_matrix(Js, ms, us, vs, ws, NU, dt_probe,
                                                        bdf2=False))
        lu, luT = splu(Ap.tocsc()), splu(Ap.T.tocsc())
        lhs = float(lu.solve(vv) @ ww)
        rhs = float(vv @ luT.solve(ww))
        bad = float(vv @ lu.solve(ww))
        ident.append(abs(lhs - rhs) / max(abs(lhs), 1e-300))
        ctrl.append(abs(lhs - bad) / max(abs(lhs), 1e-300))
        pe = scale * (1.0 / NTOT) / NU
        if scale == 10.0 and dt_probe == 2.0:
            hard = ctrl[-1]
        print(f"       {scale:>7.0f}{dt_probe:>7.2f}{pe:>10.1f}{ident[-1]:>15.2e}"
              f"{100*ctrl[-1]:>14.1f}%")

check(max(ident) < 1e-10,
      f"6.3  adjoint identity on the global MOMENTUM matrix holds in every regime: "
      f"worst {max(ident):.2e}")
check(hard > 0.15,
      f"     control: forgetting the transpose is DETECTED at cell Pe 16.7, dt 2.0 -- "
      f"{100*hard:.1f}% error (and only {100*min(ctrl):.1f}% where J/dt dominates, which is "
      f"why the bar is set at the convective corner)")

sym = np.abs((A2 - A2.T).data).max() if (A2 - A2.T).nnz else 0.0
check(sym > 1e-6, f"     and the momentum matrix really is non-symmetric: "
                  f"max|A - A^T| = {sym:.2e}")


# ---------------------------------------------------------------- 6.4 gradient equivalence
rng2 = np.random.default_rng(7)
p1 = MultiBlockMiniPISO(d1, NU, DT)
p2 = MultiBlockMiniPISO(d2, NU, DT)
perm = alignment(d2, d1.blocks[0].shape)

S_un = rng2.standard_normal(p1.N) * 0.1                  # in the UNSPLIT layout
S1 = torch.tensor(S_un, requires_grad=True)
L1 = p1.step(S1); L1.backward()
g1 = S1.grad.detach().numpy().copy()

S_sp = np.zeros(p2.N)
S_sp[perm] = S_un                                        # same field, split layout
S2 = torch.tensor(S_sp, requires_grad=True)
L2 = p2.step(S2); L2.backward()
g2 = S2.grad.detach().numpy()[perm]                      # back into the unsplit layout

dL = abs(float(L1) - float(L2)) / max(abs(float(L1)), 1e-300)
dg = np.abs(g1 - g2).max() / max(np.abs(g1).max(), 1e-300)
check(dL < 1e-12, f"6.4  loss agrees across the split: {float(L1):.12e} vs {float(L2):.12e}, "
                  f"rel {dL:.2e}")
check(dg < 1e-12, f"     dL/dS agrees across the split: max rel diff {dg:.2e} "
                  f"(|g| up to {np.abs(g1).max():.3e})")

# ---------------------------------------------------------------- 6.5 FD vs adjoint
# eps = 1e-3, NOT the reflexive 1e-6, and the reason is worth stating. u* and phi are both
# LINEAR in S and the loss is their sum of squares, so L is exactly quadratic in S and a central
# difference has NO truncation error whatever the step. The only error is the iterative solver's
# residual, which enters divided by eps -- so a larger step is strictly better here, the
# opposite of the usual trade-off. Measured worst relative error against the adjoint:
#
#     eps        1e-5     1e-4     1e-3     1e-2     5e-2
#     6.5     2.7e-05  5.3e-06  1.3e-06  8.3e-08  1.4e-08
#     6.6                       1.2e-06  4.3e-07  9.1e-08
#
# monotone in eps over four decades, which is the signature of a noise floor rather than
# truncation. At 1e-6 the FD signal was ~1e-9 of a loss of order 170 and the comparison was
# measuring CG's stopping criterion, not the gradient.
eps = 1e-2
# SAMPLE WHERE THE SIGNAL IS, AND NORMALISE TO THE GRADIENT'S OWN SCALE. A random cell can
# carry a dL/dS near zero, and dividing by it turns a correct gradient into a huge "relative
# error" -- 3.2e-05 on one such cell here while every large entry agreed to 1e-08. The bar is
# against max|g| over the vector, which is the scale a training step actually sees.
g2_all = S2.grad.detach().numpy()
idxs = np.argsort(-np.abs(g2_all))[:6]
scale = np.abs(g2_all).max()
worst = 0.0
for k in idxs:
    dvec = np.zeros(p2.N); dvec[k] = eps
    Lp = float(p2.step(torch.tensor(S_sp + dvec)))
    Lm = float(p2.step(torch.tensor(S_sp - dvec)))
    fd = (Lp - Lm) / (2 * eps)
    worst = max(worst, abs(fd - float(g2_all[k])) / scale)
check(worst < 1e-6, f"6.5  FD vs adjoint through one 2-block step, the 6 largest-|g| cells: "
                    f"worst {worst:.2e} of max|g| = {scale:.3e}")

# ---------------------------------------------------------------- 6.6 sensitivity crosses
# The only check here that cannot pass unless the seam transmits sensitivity BACKWARDS: the
# parameter lives entirely in block 0 and the loss entirely in block 1.
idx_A = p2.block_slice(0)
idx_B = p2.block_slice(1)
theta = torch.tensor(rng2.standard_normal(len(idx_A)) * 0.1, requires_grad=True)


def loss_from_A(th):
    S = torch.zeros(p2.N, dtype=torch.float64)
    S = S.index_put((torch.as_tensor(idx_A),), th)
    return p2.step(S, loss_idx=idx_B)


Lx = loss_from_A(theta); Lx.backward()
gA = theta.grad.detach().numpy().copy()
check(np.abs(gA).max() > 1e-8,
      f"6.6  sensitivity CROSSES the seam: parameter in block 0, loss in block 1, "
      f"max|dL/dtheta| = {np.abs(gA).max():.3e} (zero would mean the backward treats blocks "
      f"independently)")

worst6, scale6 = 0.0, np.abs(gA).max()
base = theta.detach().numpy()
for k in np.argsort(-np.abs(gA))[:4]:
    dth = np.zeros(len(idx_A)); dth[k] = eps
    Lp = float(loss_from_A(torch.tensor(base + dth)))
    Lm = float(loss_from_A(torch.tensor(base - dth)))
    fd = (Lp - Lm) / (2 * eps)
    worst6 = max(worst6, abs(fd - gA[k]) / scale6)
check(worst6 < 1e-6, f"     and it is the RIGHT sensitivity: FD agrees to {worst6:.2e} of "
                     f"max|dL/dtheta|")

# ------------------------------------------------- 6.7 the autograd primitive itself
# 6.4-6.6 compare the adjoint against finite differences of the same forward, so they DO verify
# the chain end to end -- but only the RHS path. `A_val` never carries requires_grad in
# MiniPISO or in MultiBlockMiniPISO, so `LinearSolve.backward`'s dL/dA branch, -lambda x^T, had
# never been exercised by any gate. It is exercised here, and the singular case has a limit
# worth pinning down.
from scipy.sparse import random as sprandom, eye as speye, diags as spdiags
from src.adjoint_piso import LinearSolve, csr_pattern

rng3 = np.random.default_rng(3)
nn_ = 40
B = sprandom(nn_, nn_, density=0.08, random_state=1, data_rvs=rng3.standard_normal)
An = sparse.csr_matrix(B + speye(nn_) * 6.0)
idxn, shpn, valn = csr_pattern(An)
bn = torch.tensor(rng3.standard_normal(nn_))
fn = lambda av, bb: LinearSolve.apply(av, bb, (idxn, shpn), False, False)
okn = torch.autograd.gradcheck(fn, (valn.clone().requires_grad_(True),
                                    bn.clone().requires_grad_(True)),
                               eps=1e-4, atol=1e-6, rtol=1e-3, nondet_tol=1e-8)
check(bool(okn), "6.7  torch.autograd.gradcheck passes on LinearSolve over BOTH inputs "
                 "(non-symmetric): dL/dA and dL/db")

rng4 = np.random.default_rng(5)
nm = 30
offm = sprandom(nm, nm, density=0.15, random_state=2,
                data_rvs=lambda k: rng4.uniform(0.2, 1.0, k))
offm = (offm + offm.T) * 0.5
Lap = sparse.csr_matrix(spdiags(np.asarray(offm.sum(axis=1)).ravel()) - offm)
idxm, shpm, valm = csr_pattern(Lap)
bm = torch.tensor(rng4.standard_normal(nm)); bm = bm - bm.mean()
fb = lambda bb: LinearSolve.apply(valm, bb, (idxm, shpm), True, True)
okb = torch.autograd.gradcheck(fb, (bm.clone().requires_grad_(True),),
                               eps=1e-4, atol=1e-6, rtol=1e-3)
check(bool(okb), "     gradcheck passes on dL/db for the SYMMETRIC SINGULAR solve "
                 "(the pressure system), where both passes project")

avm = valm.clone().requires_grad_(True)
xm = LinearSolve.apply(avm, bm, (idxm, shpm), True, True)
(xm ** 2).sum().backward()
gm = avm.grad.numpy().copy()
rows_m, cols_m = idxm
i_, j_ = next((int(rows_m[k]), int(cols_m[k])) for k in range(len(valm))
              if rows_m[k] != cols_m[k])
D = np.zeros(len(valm))
for k in range(len(valm)):
    rc = (int(rows_m[k]), int(cols_m[k]))
    if rc in ((i_, j_), (j_, i_)):
        D[k] = -1.0                      # a face coefficient: two off-diagonals ...
    if rc in ((i_, i_), (j_, j_)):
        D[k] = +1.0                      # ... and the two diagonals that keep row sums at zero


def _loss_A(vec):
    xx = LinearSolve.apply(torch.tensor(vec), bm, (idxm, shpm), True, True)
    return float((xx ** 2).sum())


gd = float(gm @ D)
fd_dir = (_loss_A(valm.numpy() + 1e-4 * D) - _loss_A(valm.numpy() - 1e-4 * D)) / 2e-4
rel_dir = abs(fd_dir - gd) / max(abs(gd), 1e-30)
check(rel_dir < 1e-7,
      f"     dL/dA in the singular case is correct along a STRUCTURE-PRESERVING direction "
      f"(one face coefficient, row sums stay zero): FD {fd_dir:.9f} vs adjoint {gd:.9f}, "
      f"rel {rel_dir:.2e}")

Dbad = np.zeros(len(valm)); Dbad[0] = 1.0
gdb = float(gm @ Dbad)
fd_bad = (_loss_A(valm.numpy() + 1e-3 * Dbad) - _loss_A(valm.numpy() - 1e-3 * Dbad)) / 2e-3
check(abs(fd_bad - gdb) > 1e-3 * max(abs(gdb), 1e-30),
      f"     LIMITATION PINNED: along a null-space-BREAKING direction the formula does NOT "
      f"apply -- FD {fd_bad:.3e} vs adjoint {gdb:.3e}. The forward projects both b and x, so a "
      f"perturbation that destroys N(M) = span(1) barely moves the answer while -lambda x^T "
      f"still reports a large derivative. Every parameter dependence in PISO moves FACE "
      f"coefficients, which is structure-preserving; a raw gradcheck over A_val is not")

print("=" * 74)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 74)
raise SystemExit(1 if FAIL else 0)
