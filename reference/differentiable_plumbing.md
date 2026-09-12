# Differentiable plumbing on the butterfly grid: verification and its limits

`nn_stage6_butterfly.py` runs the multiblock differentiable chain
(`MultiBlockBCChain`: BDF2 history, persistent Rhie-Chow flux, walls, inflow,
solution-dependent Dong outflow) on a coarse build of the PRODUCTION topology --
the 9-block butterfly-in-rectangle that R11-R13 validated physically -- and
gates the gradient machinery on it. Upstream context: tum-pbs/PICT's
`learning_sample.py` learns a forcing from a reference simulation, but on one
periodic block, with a 2-DOF uniform force, and without ever verifying the
gradient or the recovery against ground truth. This stage does all three on a
body-fitted multiblock domain.

## Gates (all PASS, 2026-09-12; ~4 min on the Mac)

| gate | what it certifies | measured |
|---|---|---|
| A | adjoint identity <lam, Ax> = <A^T lam, x> on the assembled operators | 7.4e-16 |
| B | autograd == finite differences through a 4-step rollout | rel 1.3e-07 |
| C | sensitivity CROSSES seams: source only in ringE, loss only in the wake -- two seams between them | grad 2.1e+03, FD rel 9.0e-08 |
| D | structural gradient paths live on THIS domain: p_flux carry, BDF2 history, Dong pressure | changes 6e-2 / 2.3e-1 / 1.9e-4 |
| E | scalar forcing recovery through the solver, a = 0.1 -> truth 1.0 | a = 1.030 |

Gate-calibration lesson (kept in the script): the first Dong-liveness probe put
the source at x = 2 and read the gradient change at 3e-7 -- geometric
remoteness (4 steps of dt = 0.01 move information 0.04 D; the outlet is 28 D
away), not a dead path. A liveness gate must EXERCISE the path it tests:
source and loss at the outlet give 1.9e-4.

## The documented non-gate: field-level identifiability

Recovering the full 21k-DOF force FIELD from trajectory snapshots is
NON-UNIQUE, and no gradient quality fixes that. Measured (both direct
optimisation of the field and an MLP(x, y) parametrisation, two-horizon
trajectory loss): the loss falls ~5000x while cosine(S, S*) stays below 0.3 --
the optimiser finds A force reproducing the observations, not THE force.
This is a property of the observation operator, not the plumbing (the same
gradients pass gates B/C to 1e-7), and it is the ambiguity PICT's paper
sec. 3.1 discusses for learned corrections -- their own sample never measures
it. Consequence for any real learning task on this stack: identifiability
must be engineered -- dense-in-time observations, multiple initial
conditions, statistics losses (what PICT's channel-SGS training uses), or
physically structured parametrisations.

## What this does NOT certify, and the remaining verification ladder

The chain is the mini physics: no corrector loop, no implicit-cross DC
pressure step, frozen coefficients. Certification of gradients through the
PRODUCTION solver is Stage 7. The ladder, in order of value:

1. **Adjoint-norm boundedness on the butterfly** (7.5-style): profile
   ||dL/dS_k|| backwards over 20+ steps with a final-state loss at a STABLE
   dt; bound the backward growth on this stiff, fine-walled grid.
2. **Full 9x9 block sensitivity map**: source in block i, loss in block j,
   FD-checked for every pair -- certifies every seam and the wake handoff,
   not just the ringE->wake path of gate C.
3. **Checkpointed == non-checkpointed on the butterfly** (stage-3 gate (b)
   re-run here) before any long-rollout training.
4. **Gradient-path decomposition** (PICT paper sec. 2.4): measure J_Adv /
   J_P / J_none contributions on this domain to know which paths a training
   loop may truncate, and what that buys at 21k cells.
5. **Mini-vs-production trajectory gap**: same coarse butterfly, same
   parameters, chain vs MultiBlockPISO forward -- quantifies what the mini
   physics misses (correctors, implicit_cross) so learning results transfer
   knowingly.
6. **Identifiability restoration**: repeat the field recovery with loss at
   EVERY step and 3+ initial conditions -- cosine(S, S*) should climb toward
   1, confirming the non-uniqueness diagnosis experimentally.
7. **Stage 7 proper**: adjoint (or tape) through the production step --
   Rhie-Chow correctors, BDF2, and now the deferred-correction cross solve,
   which post-dates the original Stage-7 plan and needs its own treatment
   (frozen-operator adjoint of the DC fixed point is the natural first cut).
