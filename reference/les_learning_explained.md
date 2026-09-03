# How learning works in an LES — the short version

Plain-language companion to `les_learning_plan.md`. Written 2026-09-03.

---

## The idea in one line

**Run the cheap simulation, compare its statistics to the truth, and let the differentiable
solver tell you which way to nudge the network so the statistics move closer.**

## The four steps

**1. The problem.** A coarse grid cannot represent small eddies, and those eddies drain energy
from the large ones. Leave them out and the flow keeps too much energy. In PICT's channel that
shows up as `Re_τ = 390` where the answer is 550.

**2. The classical patch.** Guess the missing drain from what you can see. Smagorinsky adds
viscosity wherever the resolved flow is strained: `ν_t = (C_s Δ)² |S|`. One formula, one
constant. It gets 452 against a target of 550.

**3. Replace the formula with a network.** Same idea, more capacity. The network reads the local
flow and outputs the correction — either as extra viscosity, or as a small extra force in the
momentum equation, which is what PICT actually does for the channel.

**4. The part that is genuinely different: training.**

You cannot train by comparing snapshots. Turbulence is chaotic, so a coarse run and a fine run
drift apart within a few time units however good the model is. "Make this field match that
field" is asking for something impossible, and the network would chase noise. PICT states the
problem directly: *"matching individual realizations of the flow from simulations at different
resolutions no longer provides a physically meaningful learning target."*

So you compare **statistics** — mean profile, Reynolds stresses, energy budget — which are
reproducible even though the instantaneous fields are not.

Then the differentiable solver does the work: run forward with the network in the loop, measure
how wrong the statistics are, and ask *if the network's output had been slightly different, how
would that error have changed?* That derivative runs backwards through every timestep, through
the pressure and momentum solves, and lands on the weights.

**The differentiable solver is the only reason step 4 is possible.** Without it there is no way
to know which direction to nudge, and you are back to hand-tuning a handful of constants.

---

## Where eddy viscosity fits, and whether it belongs in the loss

### It enters the matrix, not the loss

A learned **force** is added to the right-hand side, `A u = b + S`. A learned **viscosity**
changes the operator, `A(ν_t) u = b`. That single difference is why the viscosity path is harder
to differentiate:

```
  force:      ∂u/∂b = A⁻¹                     any linear solve already gives you this
  viscosity:  ∂u/∂ν = −A⁻¹ (∂A/∂ν) u          the adjoint must know how the MATRIX ENTRIES
                                              depend on ν
```

This repo has the first and not the second: `build_momentum_matrix` is NumPy and only its
assembled *values* reach torch. PICT has both — *"any derived intermediate quantities like
matrices and RHS of the linear systems"* are differentiable there.

### ν_t is not observable, so it is not a target

There is no measurement of eddy viscosity. No DNS field of it. Nothing to compare against. It is
a latent quantity whose only job is to make the velocity statistics come out right, and putting
it in the loss would mean already knowing the answer. `nn_eddy_viscosity.py` says exactly this:
*"Nothing about nu_t appears in the loss. It is recovered only through the momentum balance,
which is exactly the position a closure network is in."*

### Where it legitimately does appear — as a regulariser, never as a target

| term | why |
|---|---|
| magnitude, `λ‖ν_t‖²` | resolves NON-UNIQUENESS: many ν_t fields give nearly the same statistics, so ask for the smallest correction that works. PICT uses exactly this on their force, `L = L_stats + λ_S (1/N) Σ‖S_θⁿ‖²` |
| positivity | better enforced BY CONSTRUCTION (`softplus`) than by penalty — a construction is exact, a penalty is a suggestion |
| smoothness, `‖∇ν_t‖²` | an oscillatory viscosity field is unphysical and destabilises the solve |

### The one regime where ν_t really is in the loss, and why it disappoints

Filter a DNS, compute the exact subgrid stress `τ_ij`, fit ν_t to it directly. That is
**a-priori** training — `nn_stage5a_apriori.py` here, against `nn_stage5b_aposteriori.py`.

The catch is structural: an eddy viscosity assumes the subgrid stress is aligned with the
resolved strain rate, and in real turbulence it largely is not. A model can fit the stress well
and still run badly. That mismatch is the main reason the field moved to training through the
solver on statistics.

### The limit we measured ourselves

ν_t is **unrecoverable where the strain vanishes**. `nn_eddy_viscosity.py` recovers it to 1.5%
of peak in `5 < y⁺ < 108` and fails completely near the centreline, because `dU/dy → 0` and the
total stress → 0 together: ν_t multiplies something that vanishes, so the velocity is
insensitive to it. More data does not fix that. Put ν_t in a loss there and you fit noise; leave
it out and you simply learn nothing there, which is the honest outcome.

---

## What this means for the build

**It simplifies it, and by more than it first appears.** Three consequences:

**The expensive item is not on the critical path.** The differentiable `ν` → matrix work (two to
four days, §2.4 of the plan) is only needed for an eddy-viscosity closure. PICT's channel model
is a **force**, which enters the right-hand side — a path this repo already has and already
gates (`test_sgs_net_seam.py`, FD vs adjoint on the network's own weights at 1.76e-09 of
`max|g|`). A trainable closure does not require it.

**No training dataset is required.** Because the target is statistics rather than fields, there
is nothing to precompute — PICT: *"no pre-computed training data sets are required in this
case."* That takes the a-priori data pipeline (`make_sgs_data.py`) off the critical path too.

**What is left is two small pieces**, both exactly testable:

1. **Divergence-free projection of the learned source.** Our Stage 2 measured that only the
   solenoidal part of a source is identifiable from velocity data — the velocity matched to
   8.9e-04 while the recovered source was 18% wrong, and both were correct. PICT projects the
   ambiguity out by construction. We already own the projection: it is the pressure Poisson
   solve, applied to `S` instead of `u*`.
2. **A statistics loss** — moments of a trajectory, differentiable. Cheaper than it sounds,
   since the moments are weighted sums over the rollout.

So the order inverts. Instead of *build the hard differentiable-ν path, then train*, it is
*project the source, add a statistics loss, train* — with the ν_t head and its matrix gradient
as a later refinement, worth doing to compare closure FORMS rather than to make learning
possible at all.

---

Full citations: `reference/bibliography.md`. Detailed work items and their tests:
`reference/les_learning_plan.md`.
