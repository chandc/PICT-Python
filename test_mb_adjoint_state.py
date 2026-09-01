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

  7.2  F_prev (the Rhie-Chow persistent flux)  -- NOT COVERED
  7.3  p_flux (the projection pressure)        -- NOT COVERED

7.2 and 7.3 need `face_fluxes` and `pressure_face_fluxes` in torch, which is the next
increment. Recording them as absent is the point: a Stage 7 that claimed to be finished while
two of its five cases were unimplemented would be exactly the silent failure this file is
written to catch.
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

# ---------------------------------------------------------------- 7.2 / 7.3 not covered
print("\n  NOT COVERED, and deliberately not faked:")
print("    7.2  F_prev, the Rhie-Chow persistent flux")
print("    7.3  p_flux, the projection pressure")
print("    Both need face_fluxes / pressure_face_fluxes in torch. A surrogate state array "
      "would\n         pass a mangle test while proving nothing about the real one.")

print("=" * 76)
print(f"  {PASS}/{PASS + FAIL} implemented checks passed  (7.2, 7.3 not implemented)")
print("=" * 76)
raise SystemExit(1 if FAIL else 0)
