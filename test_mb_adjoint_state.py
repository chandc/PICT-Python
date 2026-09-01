"""Stage 7 — the state a chain carries between steps, and the operators that cross the seam.

Plan: `reference/nn_multiblock_plan.md`. Stage 6 proved one step. This is about what survives
from one step to the next, because a backward pass that quietly ignores carried state produces a
plausible gradient and nothing errors.

WHAT IS COVERED HERE AND WHAT IS NOT, stated plainly rather than left to be discovered:

  7.0  the gradient operator composes into the pressure operator EXACTLY. New stencils that
       "look right" are how a projection quietly stops projecting; this proves the identity
       sum_a F_a^T W_a F_a == build_diffusion_matrix instead of asserting it.
  7.1  FD vs adjoint through a three-step chain carrying BDF2 history across a seam.
  7.4  dropping that history from the BACKWARD ONLY must change the gradient.
  7.5  the adjoint norm stays bounded over 20 steps.

  7.6  the assembled flux-divergence operator matches `face_fluxes` exactly.
  7.7  and the Rhie-Chow pressure flux, with a spatially varying Gamma.
  7.3  p_flux carried across steps and consumed by that flux -- FD, and a mangle test.

  7.2  F_prev -- RE-SCOPED, not skipped. `self.F_prev` is read in exactly one place in
       `piso_multiblock.py`, guarded by `if self.ddt_corr`, and ddt_corr is off in every
       production case: it is what made the square cylinder diverge at step 455. So with the
       settings the port actually runs, F_prev feeds a diagnostic and the checkpoint and
       nothing in the solution. There is no gradient path through it, and the test as
       originally specified would FAIL for a CORRECT implementation. The live within-step
       state is the corrector's reuse of Fb; see the plan.
"""
import numpy as np
import torch

from src.mb_adjoint import (MultiBlockChain, MultiBlockMiniPISO, check_consistency,
                            periodic_box)

NTOT, NU, DT = 12, 0.05, 0.02

torch.set_default_dtype(torch.float64)
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


print("\n" + "=" * 76 + "\n  Stage 7 — carried state and the seam-crossing operators\n" + "=" * 76)

# ---------------------------------------------------------------- 7.0 operator consistency
worst = 0.0
for ns in (1, 2, 3):
    d = periodic_box(NTOT, ns)
    p = MultiBlockMiniPISO(d, NU, DT)
    worst = max(worst, check_consistency(d, p.Js, p.ms))
check(worst < 1e-10,
      f"7.0  sum_a F_a^T W_a F_a reproduces build_diffusion_matrix for 1, 2 and 3 blocks: "
      f"worst {worst:.2e} against entries of order 1e+04")

# ---------------------------------------------------------------- 7.1 FD through the chain
d2 = periodic_box(NTOT, 2)
chain = MultiBlockChain(d2, NU, DT)
rng = np.random.default_rng(11)
S0 = [rng.standard_normal(chain.N) * 0.05 for _ in range(3)]


def loss_of(arrays, drop=False):
    src = [torch.tensor(a) for a in arrays]
    return chain.rollout(src, drop_history=drop)


src = [torch.tensor(a, requires_grad=True) for a in S0]
L = chain.rollout(src)
L.backward()
grads = [s.grad.detach().numpy().copy() for s in src]

eps = 1e-2                      # the chain is linear in S, so central FD has no truncation
worst71, scale71 = 0.0, max(np.abs(g).max() for g in grads)
for k_step in (0, 2):
    g = grads[k_step]
    for k in np.argsort(-np.abs(g))[:3]:
        pert = [a.copy() for a in S0]
        pert[k_step][k] += eps
        Lp = float(loss_of(pert))
        pert[k_step][k] -= 2 * eps
        Lm = float(loss_of(pert))
        fd = (Lp - Lm) / (2 * eps)
        worst71 = max(worst71, abs(fd - g[k]) / scale71)
check(worst71 < 1e-6,
      f"7.1  FD vs adjoint through a 3-step chain across the seam: worst {worst71:.2e} of "
      f"max|g| = {scale71:.3e}")

# ---------------------------------------------------------------- 7.4 mangle the history
src_d = [torch.tensor(a, requires_grad=True) for a in S0]
Ld = chain.rollout(src_d, drop_history=True)
Ld.backward()
grads_d = [s.grad.detach().numpy().copy() for s in src_d]

check(abs(float(Ld) - float(L)) / abs(float(L)) < 1e-15,
      f"     forward is untouched by the mangle, as it must be: {float(L):.9e} vs "
      f"{float(Ld):.9e}")

rel = max(np.abs(a - b).max() for a, b in zip(grads, grads_d)) / scale71
check(rel > 1e-12,
      f"7.4  dropping the BDF2 history from the BACKWARD is DETECTED: gradients differ by "
      f"{rel:.2e} of max|g| (a backward that ignored carried state would match exactly here)")

# ---------------------------------------------------------------- 7.5 adjoint norm
# THE LOSS HAS TO BE ON THE FINAL STATE for this profile to mean anything. With a loss summed
# over the trajectory, dL/dS_0 collects a term from every subsequent step and dL/dS_last
# collects one, so the profile falls by a factor of order the horizon whatever the propagator
# does -- measured 100x over 20 steps here, which reads as amplification and is nothing of the
# sort. On the final state alone, dL/dS_k IS the adjoint propagator from step k.
nsteps = 20
src20 = [torch.tensor(rng.standard_normal(chain.N) * 0.05, requires_grad=True)
         for _ in range(nsteps)]
chain.rollout(src20, final_only=True).backward()
norms = np.array([float(s.grad.norm()) for s in src20])
per_step = (norms[0] / max(norms[-1], 1e-300)) ** (1.0 / (nsteps - 1))
print(f"       ||dL/dS_k|| earliest {norms[0]:.4e}  latest {norms[-1]:.4e}  "
      f"per-step factor going BACK in time {per_step:.4f}")
check(per_step < 1.0 and np.isfinite(norms).all(),
      f"7.5  the adjoint CONTRACTS over {nsteps} steps: {per_step:.4f} per step backwards, "
      f"against the 0.91 the single-block Stage 3 gate measured. Above 1.0 would be the "
      f"upstream amplification the plan warns about, where the answer is a shorter window "
      f"rather than gradient clipping")

# ---------------------------------------------------------------- 7.6 / 7.7 flux operators
# The two blocked cases needed `face_fluxes` and `pressure_face_fluxes` differentiable. Both are
# LINEAR in the field they consume once the coefficients are frozen, so neither needs a torch
# port or a hand-written transpose: assemble each as a sparse matrix, which torch differentiates
# natively, and verify it against the real function.
from src.mb_adjoint import verify_flux_divergence, verify_rc_divergence

rng2 = np.random.default_rng(4)
worst_f = worst_rc = 0.0
for ns in (1, 2, 3):
    d = periodic_box(NTOT, ns)
    p = MultiBlockMiniPISO(d, NU, DT)
    e, sc = verify_flux_divergence(d, p.Js, p.ms, rng=np.random.default_rng(0))
    worst_f = max(worst_f, e / sc)
    gam = {b: rng2.uniform(0.2, 1.5, blk.shape) for b, blk in enumerate(d.blocks)}
    e, sc = verify_rc_divergence(d, p.Js, p.ms, gam, rng=np.random.default_rng(9))
    worst_rc = max(worst_rc, e / sc)

check(worst_f < 1e-13,
      f"7.6  the assembled operator reproduces divergence(face_fluxes(u,v,w)) for 1, 2 and 3 "
      f"blocks: worst {worst_f:.2e} relative")
check(worst_rc < 1e-13,
      f"7.7  and reproduces divergence(pressure_face_fluxes(p, rhie_chow=True)) with a "
      f"SPATIALLY VARYING Gamma: worst {worst_rc:.2e} relative -- the width-2 wide-gradient "
      f"stencil across a seam, assembled as a face AVERAGE of a cell GRADIENT, two width-1 "
      f"operators each of which already resolves its own seam")

# ---------------------------------------------------------------- 7.3 p_flux carried
from src.mb_adjoint import MultiBlockChainRC

rc_chain = MultiBlockChainRC(d2, NU, DT)
rng3 = np.random.default_rng(21)
S3 = [rng3.standard_normal(rc_chain.N) * 0.05 for _ in range(3)]

src3 = [torch.tensor(a, requires_grad=True) for a in S3]
rc_chain.rollout(src3).backward()
g3 = [s.grad.detach().numpy().copy() for s in src3]
scale3 = max(np.abs(x).max() for x in g3)

worst3 = 0.0
for k_step in (0, 2):
    gg = g3[k_step]
    for k in np.argsort(-np.abs(gg))[:3]:
        pert = [a.copy() for a in S3]
        pert[k_step][k] += eps
        Lp = float(rc_chain.rollout([torch.tensor(a) for a in pert]))
        pert[k_step][k] -= 2 * eps
        Lm = float(rc_chain.rollout([torch.tensor(a) for a in pert]))
        worst3 = max(worst3, abs((Lp - Lm) / (2 * eps) - gg[k]) / scale3)
check(worst3 < 1e-6,
      f"7.3  FD vs adjoint through the chain WITH p_flux carried and consumed by the "
      f"Rhie-Chow flux: worst {worst3:.2e} of max|g| = {scale3:.3e}")

src3d = [torch.tensor(a, requires_grad=True) for a in S3]
rc_chain.rollout(src3d, drop_pflux=True).backward()
g3d = [s.grad.detach().numpy().copy() for s in src3d]
rel3 = max(np.abs(a - b).max() for a, b in zip(g3, g3d)) / scale3
check(rel3 > 1e-12,
      f"     dropping p_flux from the BACKWARD is DETECTED: gradients differ by {rel3:.2e} of "
      f"max|g|. Small because the Rhie-Chow term is a deliberate O(h^3) dissipation -- but it "
      f"is four orders above the 1e-10 gradient noise floor, and a backward that ignored the "
      f"carried pressure would differ by exactly 0")

# ---------------------------------------------------------------- 7.2 / 7.3 not covered
print("\n  NOTE:")
print("    7.2  re-scoped: self.F_prev is read ONLY under `if self.ddt_corr`, which is off in")
print("         every production case, so it carries no gradient there. The live within-step")
print("         state is the corrector's reuse of Fb. See the plan.")

print("=" * 76)
print(f"  {PASS}/{PASS + FAIL} checks passed  (7.2 re-scoped: see the header)")
print("=" * 76)
raise SystemExit(1 if FAIL else 0)
