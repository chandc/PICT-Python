# FluidGym parity: scoping the adjoint-DPC paper

Target: reproduce FluidGym's DPC result on `CylinderJet2D-easy-v0`, then beat
its horizon limitation -- the hypothesis being that their DPC's 7.2% (vs
SAC's 8%) is horizon-truncation, which a memory-flat adjoint removes.

## The environment, extracted from source (repo @ 2026-09-13, code == spec)

| item | value (easy) | medium / hard |
|---|---|---|
| Reynolds number | 100 | 250 / 500 |
| angular resolution | 24 | 32 / 32 |
| solver dt | 1e-2 (adaptive CFL 0.8) | same |
| control interval | step_length 0.25 t.u. = 25 solver steps per action | same |
| episode | 80 env steps = 20 t.u. ~ 3.3 shedding periods | same |
| jets | opposing, top/bottom (+-90 deg), half-width 10 deg, fixed smooth profile x scalar action | same |
| action | ONE scalar (SARL) scaling both jet profiles (opposing signs) | same |
| sensors | 151 wake/near probes x (u, v) = 302 (irrelevant for adjoint control) | same |
| reward | r = C_D,ref - <C_D>_0.25 - 1.0 * |<C_L>_0.25| | same |
| baselines | SAC ~8% drag reduction; DPC ~7.2%, 1-2 orders faster | -- |

Lineage: Rabault et al. JFM 2019 / Ren, Rabault & Tang PoF 2021 -- the same
configuration family as our validated butterfly campaign (Re 100 shedding,
C_D ~ 1.32, St ~ 0.167 at our confinement).

## What we already have

- The physics regime, validated four ways (R11-R13 closing table) with
  quantified blockage laws -- environment ground truth with a pedigree.
- Differentiable force objective: Stage 8 (10/10) -- torch `surface_force`,
  dC_D/d(actuation) through a wall-bounded chain (8.6), the tangential-only
  loss guard (8.3), Dong non-singular solve (8.7/8.8).
- Verified adjoint machinery through every production ingredient: seams,
  BCs, Dong, persistent state (Stage 7, 9/9), DC cross solve (7b, 6/6),
  butterfly topology (6.9, 7/7).
- Jet actuation is a small addition to `cylinder_rect_bc`: Dirichlet arcs at
  +-90 deg +-10 deg on the ring wall faces, profile x a(t) -- same machinery
  as the slip/pin treatments.

## The two honest gaps

1. **The dynamics chains are SCALAR.** `MultiBlockMiniPISO` and descendants
   evolve one field; the force objective is vector but the differentiable
   rollout is not. Full-vector NS differentiable rollout is the outstanding
   build (it IS the plan's Stage 9/10 substrate). Options: (a) extend the
   chain family to (u, v, w, p) -- mechanical but sizeable, all gates
   re-run; (b) adjoint of the PRODUCTION solver step (harder, but then the
   physics is exactly R11's). (a) first: the chain architecture and its
   gate style transfer directly.
2. **The horizon hypothesis has a cheaper competitor.** Their tape's
   across-step memory growth can be attacked with plain gradient
   checkpointing inside THEIR stack (O(sqrt T) memory, 2x compute). If
   FluidGym's DPC merely never enabled checkpointing, the "memory-flat
   adjoint enables long horizons" claim weakens to an efficiency argument.
   MUST be settled empirically before building: run their DPC with torch
   checkpointing at 2-4x horizon on easy; if the 7.2 -> 8% gap closes there,
   the paper pivots to the verified-gradients + validated-environments
   contribution (still publishable, smaller).

## Milestones

  M0  settle gap 2: their DPC + checkpointing at longer horizons (their
      stack, GPU box). Decision gate for the whole framing.
  M1  vector NS chain (u,v,p in 2D first) + gates (FD, seam, BC, forces
      loss) -- the Stage 9/10 substrate regardless of M0's outcome.
  M2  jet actuation in bc module + dC_D/da through the vector chain on the
      butterfly; open-loop optimal a(t) at Re 100 (our own physics, our
      figures).
  M3  FluidGym env parity run: match DPC 7.2% at their horizon, then extend
      horizon at flat memory; report against their published curves.
  M4  the 3D/TCF frontier (their declared future work) if M3 lands.

## M0 setup: DONE (2026-09-13) -- FluidGym runs and differentiates on the GB10

Recipe (tools/fluidgym/Dockerfile, image `fluidgym:m0` on Spark): no aarch64
wheel exists on PyPI, so source-build in `cuda:12.8.1-cudnn-devel` with torch
2.9 cu128 aarch64 wheels and **TORCH_CUDA_ARCH_LIST="12.0"** -- nvcc 12.8
rejects compute_121; same-major SASS compatibility runs sm_120 kernels on the
sm_121 GB10 (the AmgX lesson, re-earned in miniature). Runtime deps installed
explicitly (the source install skipped them). Smoke (tools/fluidgym/
m0_smoke.py): env builds + resets (initial domain from HF hub), uncontrolled
steps step, and differentiable mode returns d(reward)/d(action) = -1.845
end-to-end. Actions must be torch CUDA tensors, not numpy.

Next: the M0 experiment proper -- DPC at their horizon (reproduce ~7.2%),
then 2-4x horizon with torch gradient checkpointing. Decision gate per the
milestones above.
