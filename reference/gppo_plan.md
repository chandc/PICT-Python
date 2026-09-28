# A gradient-enhanced PPO for UniFlow: implementation plan

## 0. What this is, and what it is not

HydroGym's paper (`arXiv:2512.17534`, Nature) names "Gradient-Enhanced PPO" (GPPO) and reports it
improves sample efficiency 45–65% over standard PPO by backpropagating through their differentiable
solver and "incorporating" that gradient into "PPO's clipped surrogate objective." No equations, no
released code (checked both the paper text and `dynamicslab/hydrogym` directly — nothing under that
name in either). So there is nothing to port; what follows is a from-scratch design that does the same
*kind* of thing — mixing a policy's own analytic gradient through a differentiable solver with PPO's
score-function estimator — grounded in the closest published lineage, Stochastic Value Gradients (Heess
et al., NeurIPS 2015): a reparameterized stochastic policy, differentiated through the (true or learned)
dynamics, combined with a value-function baseline. The design choices below (how the two gradients are
combined, the annealing schedule) are ours, stated as such, not attributed to HydroGym.

**Why we are unusually well placed to build this.** GPPO's core requirement — a solver that
backpropagates a reward trajectory to policy parameters — is exactly `src/uadj_step.py` /
`src/uadj_replay.py`, already certified to round-off against production UniFlow and already carrying a
memory-flat replay mechanism (masks and Poisson-iteration counts recorded forward, replayed backward
exactly). The stochastic-policy extension below reuses that same record/replay log, just with one more
thing cached per step: the action noise.

## 1. Equations

**Policy.** A reparameterized Gaussian, the same shape SB3's `MlpPolicy` uses (`net_arch=[64,64]`,
learned `log_std_init`), so a like-for-like comparison against the existing PPO baseline stays valid:

  a_t = mu_theta(s_t) + sigma_theta(s_t) ⊙ eps_t,   eps_t ~ N(0, I)

  log pi_theta(a_t | s_t) = -sum_i [ log sigma_theta,i(s_t) + 0.5 log(2 pi) + eps_t,i^2 / 2 ]

**Two gradient estimators over the same rollout, same sampled eps_t's:**

*Pathwise (analytic) term* — backprop through the reparameterized action AND through the solver
(`replay_policy_grad_sub` with eps_t cached at record time, replayed at the same value on the backward
pass, exactly like the branch-mask log already is):

  R_path(theta) = sum_{t=0}^{H-1} r(s_t, a_t(theta)),   s_{t+1} = f_solver(s_t, a_t)  [exact, via the adjoint]

*Score-function (PPO) term* — solver treated as a black box, standard GAE(lambda) advantage from a
learned value function V_phi:

  delta_t = r_t + gamma V_phi(s_{t+1}) - V_phi(s_t)
  A_t = sum_{l=0}^{inf} (gamma lambda)^l delta_{t+l}
  r_t(theta) = exp( log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t) )
  L_CLIP(theta) = E_t[ min( r_t(theta) A_t,  clip(r_t(theta), 1-eps_clip, 1+eps_clip) A_t ) ]
  L_VF(phi) = E_t[ (V_phi(s_t) - V_t_targ)^2 ],   V_t_targ = A_t + V_phi(s_t)
  H[pi_theta](s_t) = sum_i 0.5 log(2 pi e sigma_theta,i(s_t)^2)

**Combined objective (minimized):**

  L_GPPO(theta, phi) = -L_CLIP(theta) + c1 L_VF(phi) - c2 H[pi_theta] - lambda(k) R_path(theta)

  lambda(k) = lambda_0 * max(0, 1 - k / K_anneal)        [linear decay to 0 over K_anneal updates]

lambda=0 recovers plain PPO exactly (a required regression check, G1 below); lambda held large and
K_anneal -> infinity recovers plain DPC (G2). The decay is the answer to a failure mode we have already
seen in pure DPC — deterministic gradient descent converging to the same degenerate basin (the free-jets
suction loophole) regardless of horizon or seed spread. Annealing lambda -> 0 hands control back to
PPO's stochastic exploration once the analytic term has done its job of bootstrapping useful behaviour
quickly, rather than letting it lock in a local optimum for the rest of training.

## 2. Stages and tollgates

Each gate is falsifiable and cheap to check before the next stage is built on top of it — the same
discipline the adjoint plan (`unstructured_adjoint_plan.md`) used, for the same reason: a bug two
stages down is expensive to trace back through a combined stochastic/analytic gradient.

### G0 — Reparameterized policy + noise-cached replay (2–3 days)
Extend `replay_policy_grad_sub`'s log (currently `(masks, poisson_counts)` per control step) with the
sampled `eps_t`; the policy becomes `policy_action(st) -> (mu, sigma)`, action drawn with `eps_t` cached
forward and reused backward.
**Gate:** replay == tape for `dR_path/dtheta` through the stochastic policy, same tolerance already met
for the deterministic case (1e-15 relative, `test_uadj_train.py`'s A20 pattern). This isolates "did the
noise-caching break replay-exactness" from everything downstream.

### G1 — PPO half, standalone, lambda = 0 (1 week)
Value network `V_phi`, GAE(lambda), the clipped surrogate, entropy bonus — a from-scratch PPO using the
same reparameterized `log pi_theta` as G0, no analytic term at all.
**Gate:** trained at lambda=0 on the NACA gust task, this reproduces the existing SB3-PPO baseline's
learning curve (`figures/naca_ppo_vs_dpc_training.png`) to within seed variance — same order of return
at matched env-step budgets. A buggy from-scratch PPO would confound every result built on top of it, so
this must pass on its own before G2.

### G2 — Pathwise half, standalone, lambda -> infinity (2–3 days)
Same combined loop, PPO term weighted to zero, sigma held small (near-deterministic).
**Gate:** reproduces the existing DPC results (`results/uadj_dpc/dpc_naca_h10_s*`) — same final return,
same solver-step count — confirming the reparameterization didn't change the analytic pathway's
behaviour or cost.

### G3 — Combined objective, fixed lambda, no annealing (1 week)
Both terms live, `lambda` a fixed constant, swept over 2–3 values.
**Gate:** at some lambda > 0, the hybrid reaches a target return (matching DPC's final return, or
beating pure PPO's return at matched budget) in fewer environment steps than pure PPO (lambda=0) at the
same iteration count — the direct, falsifiable "does mixing help at all" test, before spending effort on
the schedule.

### G4 — Annealing schedule (3–4 days)
Add `lambda(k)`'s decay.
**Gate, stress test on a known failure mode:** run on the cylinder free-jets ZNMF task, where pure DPC's
outcome is already known (record §65: every DPC run converges to the same degenerate basin regardless of
horizon). The annealed hybrid must do at least as well as pure PPO's asymptotic behaviour on that task
(i.e., retain the ability to escape a basin DPC alone cannot), while still converging faster than pure
PPO from a cold start. Failing this gate means the anneal is too slow (locked into DPC's basin) or too
fast (no benefit over PPO); adjust `lambda_0` / `K_anneal` and re-run, don't proceed to G5 until it holds.

### G5 — Full three-way comparison, NACA gust task, >= 3 seeds (1 week)
**Gate:** plot return vs. solver steps for PPO, DPC, and GPPO-hybrid together (extending
`plot_naca_ppo_vs_dpc_training.py`), >= 3 seeds each. Success is GPPO reaching PPO's recorded return in
materially fewer solver steps than pure PPO needs — not necessarily matching DPC's ~194k, since some of
DPC's efficiency is inseparable from its determinism (record §"training histories": part of the 57x gap
is the eval-vs-training-metric asymmetry, which GPPO's stochastic policy will not have) — while showing
lower seed-to-seed variance in the final return than pure DPC, evidence the exploration term is doing
its job of avoiding basin lock-in.

## 3. Estimated effort

| stage | time | cumulative |
|---|---|---|
| G0 reparameterized policy + noise-cached replay | 2–3 days | ~3 days |
| G1 PPO half standalone | 1 week | ~1.5 weeks |
| G2 pathwise half standalone | 2–3 days | ~2 weeks |
| G3 combined, fixed lambda | 1 week | ~3 weeks |
| G4 annealing schedule | 3–4 days | ~3.5 weeks |
| G5 full comparison, >= 3 seeds | 1 week | ~4.5 weeks |

## 4. Open risks

* **The combination rule (a weighted sum of losses) is the simplest defensible choice, not the only
  one.** The SVG lineage and later work (e.g. control-variate formulations that use the pathwise
  gradient to reduce the score-function estimator's variance rather than just adding to it) suggest a
  control-variate combination could beat a plain weighted sum — flagged as a G3-or-later refinement, not
  attempted before the simple version is shown to work at all.
* **Horizon mismatch.** The pathwise term is memory-flat but still horizon-limited by how far the
  replay window extends each update (the same limitation DPC has); the score-function term has no such
  limit (GAE handles arbitrarily long credit assignment). Early in training, when lambda is large, the
  hybrid inherits DPC's horizon sensitivity (record: sub-period windows gave near-zero effect on the
  cylinder ZNMF task; two-period windows were needed). G3's lambda sweep should include a horizon sweep,
  not just a lambda sweep, before concluding the combination doesn't help.
* **No literature equations to check against.** Every gate above checks internal consistency (replay ==
  tape, recovers PPO at lambda=0, recovers DPC at lambda->infinity) rather than agreement with a
  published number, since none exists for this exact method. The G4/G5 gates are the closest thing to
  an external check we have: does it behave the way the *idea* should, not does it match a number
  HydroGym published.
