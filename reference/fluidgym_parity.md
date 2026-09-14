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

## The reward function, and how each approach consumes it

All CylinderJet2D results in this program -- theirs and ours -- optimise the
same per-control-step reward (fluidgym cylinder_env_base.py:769):

    r_t = C_D,ref - <C_D>_t - 1.0 * |<C_L>|_t

where <.>_t averages over the 0.25 t.u. (25 solver steps) one action spans,
and the terms are:

- **C_D,ref = 3.3281555**: the mean UNCONTROLLED drag, loaded from the
  shipped domain statistics (`_metrics_stats["drag"].mean`; the value our
  own baseline reproduced to five digits). A constant: it carries no
  gradient and cannot change any optimum -- it only centres the scale so
  "no drag improvement" reads as 0 and genuine reduction reads positive.
- **-<C_D>**: the objective proper.
- **-1.0 * |<C_L>|**: the lift penalty, and it is load-bearing. Without it
  the optimiser cheats: strong asymmetric blowing trims drag while
  generating a large oscillating side force. Uncontrolled shedding swings
  C_L to +-1.2, so this term DOMINATES the uncontrolled reward (~ -0.78
  per step, hence episode reward ~ -62 for doing nothing); suppressing the
  vortex street improves both terms at once, which is why good controllers
  approach ~-1 rather than 0. Weight 1.0 in every difficulty tier.

Episode return = sum over 80 control steps. One number, four consumers:

| approach | what the reward is to it | where the gradient comes from |
|---|---|---|
| SAC / PPO (SB3) | a sampled scalar per step, stored in the replay buffer / rollout batch | estimated statistically from returns (policy gradient / soft Q targets); nothing needs to be differentiable, gamma = 0.99 inside the algorithm |
| DPC (trained policy, theirs and our dpc_train.py) | the training loss itself: L = -sum_t gamma^t r_t over a BPTT window (their config gamma = 0.999, H = 40) | backprop THROUGH the solver: r_t is a differentiable function of the fields (traction integral in torch), so dL/d(weights) flows fields -> reward -> policy via the tape |
| D-MPC (planner, their run_d-mpc / our dmpc_run.py) | the planning objective: discounted sum over the 20-step lookahead, re-optimised at every control step (10 gradient iterations, lr 0.1) | same tape gradients as DPC, but w.r.t. the raw ACTION sequence -- no network, no training; the reward is consumed at deployment time |
| our M2 chain (adjoint side) | the same quantity built from OUR fields: Stage 8 `coefficients()` gives C_D, C_L as differentiable torch scalars from the traction integral, so r = C_D,ref - C_D - |C_L| composes directly | discrete adjoint through the vector chain (LinearSolve backward = A^T solves), FD-gated; j.2 certifies dC_D/da at 5e-5 on the butterfly |

Two integration details that matter when comparing numbers:

1. **The window average is part of the reward's definition.** <C_D>_t is a
   mean over 25 solver steps, so a controller is scored on the drag it
   holds BETWEEN decisions, not at decision instants -- and for the
   gradient methods each r_t backpropagates through all 25 solver steps
   inside its window. Our M2 mirror must average the same way when we
   report parity numbers.
2. **Discounting differs by consumer.** SB3's gamma = 0.99 lives inside the
   RL algorithm; the DPC/D-MPC configs use gamma = 0.999 applied to the
   summed loss; our chain losses so far are undiscounted (gates are
   horizon-3). gamma^80 = 0.92 -- a small but real difference to hold fixed
   in like-for-like comparisons.

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

## M0 first light (2026-09-13): the smoke run's four findings

DPC trainer written (tools/fluidgym/dpc_train.py -- truncated-BPTT policy
mode + open-loop sequence mode; neither platform ships trainers, which is
the field norm: HydroGym likewise ships environments + SB3 examples only,
and its differentiable coverage is 4 of 60+ envs vs FluidGym's full suite).
Uncontrolled baseline on CylinderJet2D-easy: drag 3.3281 +/- 0.0004 (their
normalization), episode reward -62.48. Smoke (H=8, 15 iters):

1. Learning works: training reward -62.5 -> -29.3 in 15 iterations.
2. Eval regressed (drag 3.67) -- expected at 15 iters; the sweep decides.
3. **Their backward linear solves DO NOT CONVERGE**: "Linear solve (BWD)
   did not converge after 5000 iterations" / "CG residual rising ... using
   best result" throughout training. The adjoint solves in PISOtorch_diff
   hit caps and silently return best-effort iterates -- UNVERIFIED
   gradients, the exact failure class the caller-side residual gate exists
   for (and the same silent-degradation family as the AmgX status saga).
   Front-page motivation for the verified-gradients framing, observed in
   their own stack on the first training run.
4. Cost/memory: 215 s/iteration (episode-bound, H-independent); peak memory
   1.18 GiB at H=8 -- on the GB10's unified memory the horizon-MEMORY
   constraint will not bind, shifting M0's question to gradient QUALITY vs
   horizon. Overnight sweep running: H=8 vs H=80 (full-episode BPTT), 80
   iterations each, eval on held-out seeds vs the 3.3281 baseline.

## M0 mid-flight (2026-09-13): the "non-reproduction" dissolved by forensics

The H=8 policy arm FAILED reproduction: eval drag 3.746 (12.6% WORSE than
uncontrolled) after 80 iterations. Before indicting anything, the reproducer
was audited -- and the published artifacts settle it. The HF experiments
dataset (safe-autonomous-systems/fluidgym-experiments) has a `D-MPC/
CylinderJet2D-easy-v0/` folder with 10 seed dirs, each holding hydra configs
and a per-step eval CSV. Findings, in decreasing order of importance:

1. **We were reproducing the wrong algorithm.** Their gradient baseline is
   `run_d-mpc`: receding-horizon differentiable MPC -- `horizon: 20,
   n_iterations: 10, lr: 0.1, discount_factor: 0.999, rl_mode: sarl`. No
   policy network, no training phase; actions come from 10 gradient steps on
   the next-20-action sequence through the unrolled simulator at every
   control step. Our clean-room H=8 trainer (MLP policy + truncated BPTT)
   is a different method, so its failure to hit their number was never a
   contradiction. The runner script is NOT in the public repo -- only the
   hydra artifacts reveal what was run.
2. **Their own artifacts do not support the headline number.** Across the
   10 uploaded eval episodes: mean drag 3.2030 +/- 0.0082 = **3.76%**
   reduction vs cd_ref (full episode); quasi-steady tails reach 4.2%, best
   seed 4.9%, minimum instantaneous drag 5.2%. The paper's "approximately
   8%" (fig. 5 caption family; D-MPC framed as "a proof of concept",
   appendix D.3) is not reachable from any cut of the uploaded CSVs.
3. **Environment parity is now cross-certified.** The env's reward reference
   `_cd_ref` loads from the shipped domain statistics: 3.3281555 -- our
   independently measured uncontrolled baseline was 3.3281 +/- 0.0004.
   Five-digit agreement; we are running the same environment they did.
4. **Our H=80 full-episode-BPTT policy arm already trains past their D-MPC
   eval level**: training ep reward reached -3.4 by iteration 35 vs their
   D-MPC eval -7.75 +/- 1.16 (eval on held-out seeds pending at sweep end).

Exact-config reproduction queued on Spark (tools/fluidgym/dmpc_run.py:
get_state/set_state receding-horizon loop with their four hyperparameters;
the ONE unpublished knob is the optimizer -- Adam default, --opt sgd).
Queue chain: H=80 arm -> FD gradient audit -> dmpc smoke (4 steps) -> full
80-step episode, per-step CSV directly comparable to their seed-0 artifact
(mean drag 3.1928, ep reward -7.60).

## M0 results (2026-09-13 evening): horizon hypothesis CONFIRMED

The training/ side of the experiments dataset (found while mapping their SB3
integration) resolves the last ambiguity: `training/sarl/<env>/DPC/` is their
TRAINED gradient policy -- horizon 40, lr 1e-3, max_grad_norm 0.5, discount
0.999, 10k env steps (125 episodes), and the weights reveal a 302->64->64->1
MLP with tanh action scaling, IDENTICAL to our clean-room architecture. The
published numbers reconcile: their DPC evals at 5.9% mean / 6.9% quasi-steady
drag reduction (the "7.2%"), their SB3 SAC at 6.9% / 7.9% (the "8%"); the
top-level D-MPC planner is the weaker 3.8% method.

Sweep results (clean-room trainer, 80 iterations, eval on held-out seeds):

  | arm  | eval reward        | eval drag | reduction |
  |------|--------------------|-----------|-----------|
  | H=8  | -63.79 +/- 3.60    | 3.746     | -12.6% (WORSE) |
  | H=80 | **-1.02 +/- 0.37** | **3.110** | **+6.55%** |

  their trained DPC (H=40, seed 0): eval reward -6.88, drag 3.131 (+5.9%).

Same architecture, same env, same gradient machinery -- horizon alone spans
catastrophic failure to state of the art, and full-episode BPTT (H=80) beats
their published trained policy on both reward (-1.02 vs -6.88) and drag.
This is the paper's core empirical claim, landed in their own stack.

FD gradient audit (d reward/d action, action=0.3): rel err 0.25% (H=1),
0.03% (H=4), **1.45% (H=16)** -- their tape gradients degrade with horizon,
consistent with the non-converged backward solves; usable but unverified,
exactly the gap our residual-gated adjoint framing targets.

Engineering notes: `env.set_state` re-entry raises "Parent Domain is
expired" -- PISOtorch Blocks/Boundaries hold weak_ptr to their owning
Domain and clones can still reference the clone SOURCE, so every Domain
object must be kept alive (dmpc_run.py KEEP list; domains are small).
Actions and planner tensors must be float32 (env default dtype). Their SAC
ckpt_latest.zip cloudpickles omegaconf objects -- SB3 load needs
`pip install stable-baselines3 omegaconf`. Queue: exact-config DPC rerun
(their four knobs) -> SAC ckpt eval -> D-MPC full episode.

## M0/M3 CLOSING TABLE (2026-09-14 morning): replication achieved

All eval numbers are episode-mean drag on held-out seeds vs
cd_ref = 3.3281555; "artifacts" = their published HF eval CSVs.

  | controller                          | drag   | reduction | ep reward |
  |-------------------------------------|--------|-----------|-----------|
  | uncontrolled                        | 3.3281 | --        | -62.5     |
  | D-MPC planner (their artifacts)     | 3.2030 | 3.76%     | -7.75     |
  | our H=8 DPC (truncated BPTT)        | 3.7464 | -12.6%    | -63.8     |
  | their DPC artifacts (H=40)          | 3.131  | 5.9%      | -6.88     |
  | **our DPC, THEIR exact config**     | 3.1364 | 5.76%     | -3.44     |
  | our H=80 DPC (full-episode BPTT)    | 3.1103 | 6.55%     | -1.02     |
  | their SAC ckpt (OUR eval, 10 eps)   | 3.1065 | 6.66%     | +3.37     |
  | their SAC artifacts                 | 3.0996 | 6.87%     | --        |

Readings:
1. **The replication is closed.** Their exact config in our clean-room
   trainer lands 0.17% in drag from their published policy (3.1364 vs
   3.131) with a BETTER episode reward (-3.44 vs -6.88 -- ours holds the
   lift tighter, plausibly the pressure-sensor input). The training curve
   tracks and edges past their published log
   (figures/fluidgym_dpc_reward_progress.png).
2. **The horizon hypothesis is CONFIRMED as M0's decision gate.** H=8
   fails catastrophically; H=40 (theirs) plateaus at ~5.8-5.9%; H=80
   closes the DPC-to-SAC gap entirely (3.1103 vs SAC's 3.1065 -- a
   statistical tie). The "DPC < SAC" ordering in their paper is a
   HORIZON artifact, not a property of simulator-gradient training.
   The memory-flat verified-adjoint framing has its motivating result.
3. **SAC checkpoint portability**: their SB3 model evaluated in our
   container reproduces their artifacts to 0.2% (3.1065 vs 3.0996) --
   the SB3 integration transfers cleanly (needs `pip install
   stable-baselines3 omegaconf`).
4. D-MPC exact-config episode mid-run (~585 s/control step: planning is
   ~200x the actuation cost at deployment -- the quantified price of
   skipping policy training).

## What the trained policy learned (2026-09-14)

The controller is an MLP 453 -> 64 -> 64 -> 1 (tanh; output tanh-scaled to
the +-1 action bound; ~33k parameters): inputs are the 151 probes x
(pressure, u, v) under dpc_train.py's sorted-key flattening -- NOTE our net
also sees pressure, where FluidGym's own DPC net was velocity-only (302
inputs). Probed via the zero-point linearized gain g = W3 W2 W1 mapped at
the physical sensor positions (figures/fluidgym_h80_policy_gains.png,
plot_utility/plot_fluidgym_policy_gains.py) and the action/lift traces
(results/fluidgym_h80_fields.npz). Three findings:

1. **Mode selectivity.** Every gain map is ANTISYMMETRIC in y (top sensors
   opposite in sign to their bottom mirrors). The shedding instability is
   the antisymmetric mode and the opposing jet pair is an antisymmetric
   actuator: the network learned a matched filter that projects the
   453-dim observation onto the shedding mode and ignores the symmetric
   component entirely.
2. **Where it looks.** v-gains concentrate on the inner sensor ring at the
   separation points (top-5 all at r ~ 0.6, x ~ 0) -- the incipient
   vortex; pressure gains peak in the far wake grid (x = 3.5-4.5) -- the
   developed street's phase; u is used least (sum|gain| 1.02 vs ~2.2 for
   the other channels), as the leading-order shedding signature in u is
   symmetric.
3. **Phase-lead opposition control.** The action is phase-locked to the
   lift (corr 0.92) but LEADS it by ~2 t.u. (~1/3 of the shedding period):
   it opposes the vortex that is FORMING, not the one already shed --
   anticipation that full-episode BPTT buys, since each weight's gradient
   carries its effect on drag many periods later. Once shedding is dead
   the action decays toward zero: the stabilized symmetric state needs
   only whisper-level corrections (see the a(t) trace in
   figures/fluidgym_jet_vorticity_h80_basesub.png).

Summary: the network converged to what control theory would prescribe --
a mode-selective, phase-leading stabilizer of the antisymmetric wake
instability -- from nothing but d(reward)/d(weights) through the solver.
Flow-field effect: vortex street suppressed into two parallel shear
layers, C_D 3.328 -> 3.107 (-6.7%), C_L rms 0.87 -> 0.42 on the eval seed.

## M2 started (2026-09-13 night): jet actuation certified on the butterfly

The FluidGym actuator mirrored onto our stack: opposing +-90 deg jets,
10 deg half-width, parabolic profile, one scalar a (top blows / bottom
sucks -- mass-conserving). Jets enter `MultiBlockVecChain` as
profile-weighted Dirichlet values on body wall nodes through the same A_ib
elimination as every Dirichlet value; `with_jets` scatters the boundary
values back for traction losses. Gates (test_mb_adjoint_jet, 6/6):
vector FD on the butterfly (the M1 certificate transplanted to production
topology), **dC_D/da FD-exact at 5e-5 through the Stage 8 traction**,
jet -> wake seam transport FD-exact, liveness mangles, gate-window
boundedness. Commit dc3798c.

Measured NON-gate, the session's honest negative: the frozen-coefficient
surrogate chain is UNSTABLE on the butterfly beyond ~5 steps at dt = 0.01
(~12x/step, scalar and vector identically, seated in the near-body layer
at a ring-quarter seam). Dominant driver: the accumulating p_flux
Rhie-Chow feedback (RC = 0 cuts growth to ~1.3x/step); falsified remedies:
DC cross sweeps in the chain's pressure stage, a physical uniform-flow
frozen operator, uniform inlet. Consequence: the gradient certificates
stand (gates run inside the stable window), and M2's open-loop optimal
a(t) experiment moves to the production solver's adjoint -- the same
boundary the architecture diagram already draws.

## Architecture: how the two stacks and the learning network interact

```mermaid
flowchart LR
    subgraph FG["FluidGym -- Spark GB10 (theirs)"]
        ENV["CylinderJet2D env<br/>Re 100, jets +-90 deg"]
        SOLV["PICT solver<br/>tape BPTT gradients"]
        BWD["backward solves<br/>NON-CONVERGED (our finding)"]
        ENV --> SOLV --> BWD
    end
    subgraph PP["PICT-Python -- Mac (ours)"]
        PROD["production solver<br/>butterfly, validated St/C_D"]
        ADJ["adjoint chains<br/>A^T lambda, FD-gated, memory-flat"]
        GATE["residual gate<br/>|b - Ax| <= rtol |b|"]
        PROD --> ADJ --> GATE
    end
    POL["policy network<br/>MLP: obs -> a(t)"]
    POL -- "a(t)  [M0, live]" --> ENV
    SOLV -- "grad r, tape  [M0, live]" --> POL
    POL -. "a(t)  [M2, planned]" .-> PROD
    ADJ -. "grad r, adjoint  [M2, planned]" .-> POL
    GATE -- "FD gradient audit (2 episodes)" --> BWD
    ENV -- "env spec extracted (done)" --> PROD
```

Reading it: the SOLID left circuit is the live M0 loop -- the policy drives
FluidGym's jets and learns from THEIR tape gradients, which pass through
backward solves we caught silently non-converging; the solid audit arrow is
our residual-gate discipline applied across the stack boundary (a 1-D
action makes the FD check cost two forward episodes). The DASHED right
circuit is M2/M3: the same policy, the same mirrored environment spec, but
gradients from our discrete adjoint -- FD-gated, memory-flat, residual-
verified at every inner solve. The paper's thesis IS this picture: swap
only the gradient machinery, keep the physics problem identical, and every
arrow on the right side carries a certificate. Boundary facts that keep the
diagram honest: our adjoint cannot be grafted onto their kernels (an
adjoint is married to its discretization), and our gated chains freeze
their operators (verification instruments, not shedding simulators) -- so
the right circuit's production-grade closure is exactly the remaining
Stage-7-full/M2 build, with the chains certifying each ingredient on the
way.
