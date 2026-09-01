# Connecting a CNN to PISO — staged implementation and verification plan

Mirrors what PICT does (`velocitySource` / `m_viscosity` hooks, discrete adjoint through every
solver stage, training against reference data). Built in six stages, smallest first, each with
a **gate** that must pass before the next stage starts.

The governing principle: **a wrong gradient still trains, just to the wrong place.** Every
stage's gate is therefore a gradient check, not a falling loss curve.

Theory in [`nn_piso_coupling.md`](nn_piso_coupling.md). Stage 0 is already done.

---

## Stage 0 — the linear-solve adjoint ✅ done

**Build.** `LinearSolve(torch.autograd.Function)`: forward solves $A\mathbf{x}=\mathbf{b}$ and
saves $(A_{\text{val}}, \mathbf{x})$; backward solves $A^{\mathsf T}\lambda = \bar{\mathbf g}$
and returns $\partial L/\partial\mathbf b = \lambda$, $\partial L/\partial A = -\lambda\mathbf{x}^{\mathsf T}$
on the sparsity pattern only.

**Gate — passed** ([`adjoint_piso.py`](../adjoint_piso.py)):

| check | result |
|---|---|
| adjoint identity on the **non-symmetric** momentum matrix | exact to 1e-10 (and forgetting the transpose is 24.5% off) |
| finite differences through a full PISO step | agree to ~7 digits |
| pressure gradient invariant to a constant shift of $\bar{\mathbf g}$ | 8.9e-16 |

---

## Stage 1 — one step, one scalar parameter

The smallest thing that can possibly be trained: a single learnable scalar $c$ scaling a fixed
forcing shape, $S = c\,\Phi(\mathbf{x})$, injected into the momentum RHS. No network yet.

**Build.** `nn_hooks.py`: a `MomentumSource` wrapper that adds $J\,S$ to the predictor RHS and
routes the gradient back through `LinearSolve`. Mirrors PICT's `velocitySource` +
`SetupAdvectionVelocityEulerImplicitRHS_GRAD`.

**Gate.**
1. $\partial L/\partial c$ matches central finite differences to 6 digits (float64).
2. **Recovery test** — set $c_\text{true}=0.7$, generate a target field, start from $c=0$, and
   confirm gradient descent recovers $0.7$ to 4 digits. This is the first end-to-end proof that
   the sign and scale of the gradient are right, not just its magnitude.

*Why this first:* if the sign is flipped, a 1-parameter recovery makes it obvious immediately,
whereas a CNN would just train to something plausible-looking.

---

## Stage 2 — one step, a small CNN

Replace the scalar with a genuinely small network: 2 conv layers, ~200 weights, taking
$(\mathbf{u}^n)$ and predicting a 3-component source field.

**Build.** `sgs_net.py` — a 3D CNN with periodic padding (matching the solver's periodic BCs;
use replicate padding on wall axes). Keep it tiny so finite differences over *all* weights is
affordable.

**Gate — passed** ([`nn_stage2_cnn.py`](../nn_stage2_cnn.py)): FD on all 173 weights, max
rel. err **4.6e-08**; shift-equivariance **1.1e-16**; target velocity reproduced to **8.9e-04**.

**Gate (b) is stated on VELOCITY, not on the source field, and that is a physics point, not a
weakened bar.** Only the *solenoidal* part of $S$ is identifiable from velocity data — the
projection removes any gradient component, so $S$ itself is not uniquely recoverable. Measured:
the velocity matches to 8.9e-04 while the source field still differs by 1.8e-01, exactly as that
non-identifiability predicts. Asking for source recovery would be asking for something the
physics does not determine.

**A test-design trap worth recording.** The first version scaled the target network's weights by
2 "to make the target non-trivial". That pushed `tanh` into saturation — a known-hard
optimisation regime — and capped the achievable fit at 2.9e-03 against a 1e-3 bar. The gradient
was correct the whole time (gate (a) passed at 5e-08). The fix was to the *target*, not the bar:
an unsaturated target of the same architecture reaches 8.9e-04. When a gradient check passes and
a training check fails, suspect the optimisation setup before the coupling.

---

## Stage 3 — multi-step rollout with checkpointing

Training signal comes from a trajectory, not one step. Roll out $N$ steps, loss on the final
state (or accumulated).

**Build.** `rollout.py`: forward over $N$ steps; backward accumulating adjoints in reverse.
Add gradient checkpointing — store the state every $k$ steps and recompute the forward within
a segment, since each step must otherwise retain $A$, $M$, $\mathbf{u}^*$, $\phi$.

**Gate — passed** ([`nn_stage3_rollout.py`](../nn_stage3_rollout.py)): FD **6.1e-09**;
checkpointed gradient identical to **exactly 0** difference; 16-step peak memory **0.8 MB**
checkpointed vs **13.8 MB** not (a 17× reduction); adjoint norm ratio **0.91**.

**Gate (a) had to be redesigned, and the reason matters.** As first written it compared the
rollout gradient against finite differences — but the rollout uses the frozen-coefficient
approximation, while FD measures the *exact* derivative. The gate was therefore testing
something the implementation deliberately does not compute, and it failed at 16% relative
error. The fix was **not** a looser tolerance: the gate now runs with `rebuild=False`, holding
$A$ genuinely constant so the adjoint gradient *is* exact and FD must match — isolating the
time chain (Stage 3's actual subject) from the frozen-coefficient bias (Stage 4's).

**Watch for.** Adjoint instability over long rollouts: the adjoint of an advection-dominated
flow transports sensitivity *upstream* and can amplify. Plot $\lVert\lambda\rVert$ per step;
if it grows exponentially, shorten the window rather than clipping gradients silently.

---

## Stage 4 — the frozen-coefficient decision

Currently $A$ depends on $\mathbf{u}^n$ through the SOU coefficients, so the exact gradient
needs $\partial L/\partial A \to \partial L/\partial\mathbf{u}^n$ (PICT's
`SetupAdvectionMatrixEulerImplicit_GRAD`). Dropping it is cheaper and often still a usable
descent direction — but it is **not** the true gradient.

**Build.** Implement the term; make it a flag.

**Gate.** Measure, don't assume: report the angle between the exact and frozen-coefficient
gradients, and the difference in converged loss after a short training run. **Publish the number
either way** — if the shortcut is used in later stages, its bias must be a stated quantity, not
an unexamined convenience.

**Gate — complete** ([`nn_stage4_bias.py`](../nn_stage4_bias.py)), and it **overturned the
criterion this plan originally specified.**

| | frozen-coefficient | `exact_A` |
|---|---|---|
| max rel. error vs finite differences | 1.16e-01 | **1.99e-02** |
| angle between the two gradients | — | **0.395°** |
| **converged loss after identical training** | 2.70e-02 | **2.15e-02** |

The angle is 0.4°, which by the "< 5°" rule written above would pass comfortably. Yet the
converged loss is **25.4% worse**. **An angle threshold is therefore not sufficient**, and the
original criterion was wrong. A small but *systematic* bias barely tilts the gradient at any one
point in parameter space, but it accumulates across the optimisation and shifts the fixed point
that training converges to. The converged-loss comparison is the binding test; the angle is a
cheap screen, nothing more.

**Recommendation: use `exact_A=True`.** The cost is one differentiable matrix assembly per step,
and it recovers ~6× more of the true gradient.

**Residual, stated rather than hidden:** even `exact_A` is not exact. $\Gamma = J/A_\text{diag}$,
and hence $M$ and $G$, are still detached — that is what the remaining ~2% against finite
differences represents. Closing it would need torch versions of the pressure operators.

Worth noting *why the angle is the right metric*: per-component relative error between the two
gradients reaches **16%**, which looks alarming, yet the gradient *direction* — the only thing
descent uses — differs by under 2°. Small components can be badly wrong in relative terms while
contributing nothing to the direction. Had the criterion been per-component error, this shortcut
would have been rejected on a misleading number.



---

## Stage 5 — a real closure task

Only now attempt something with physical content. Two sub-steps, in order:

**5a — a-priori.** Filter a fine-grid solution onto a coarse grid, compute the exact
sub-grid term, and train the CNN to predict it directly (no solver in the loop). This is a
plain regression problem and isolates *network capacity* from *solver coupling*.

**5b — a-posteriori.** Put the solver back in the loop: train so that the coarse-grid
trajectory matches the filtered fine-grid trajectory over $N$ steps. This is the PICT
configuration and the whole point of differentiability — the network sees the solver's actual
response rather than a one-step surrogate.

**5a — passed** ([`nn_stage5a_apriori.py`](../nn_stage5a_apriori.py)): held-out correlation
**0.850**, against a trivial "SGS ∝ resolved velocity" baseline of **−0.21**. So the mapping is
learnable and the network has the capacity.

**5b — one gate failed, and the failure is the result** ([`nn_stage5b_aposteriori.py`](../nn_stage5b_aposteriori.py)):

| | 6-step | 30-step |
|---|---|---|
| no model | 0.0589 | 0.0774 |
| **exact SGS force (oracle)** | **0.0591** | **0.0779** |
| a-priori (5a) model, used a-posteriori | 0.0591 | 0.0781 |
| a-posteriori trained | **0.0573** | **0.0737** |

**The oracle is the key row, and it is why gate (a) was never reachable.** Injecting the
*exact* sub-grid force changes the error by −0.3% — no closure, however perfect, can beat that.
The sub-grid term is only ~6% of the tendency $\partial u/\partial t$, while the 16³ coarse
solver carries several percent of its own discretisation error. **The numerics dominate the
physics the closure is meant to supply.**

Two conclusions follow, both worth stating plainly:

1. **The 30% bar was miscalibrated**, not merely missed. An absolute improvement target is
   meaningless without knowing how much improvement is available; the oracle measures that and
   should have been part of the criterion from the start.
2. **The trained model beats the oracle** (2.7% vs −0.3%). A closure cannot out-model the exact
   sub-grid term, so it is compensating **coarse-grid numerical error**, not learning physics —
   precisely the confound Stage 3.5 flagged, now demonstrated rather than hypothesised.

Gate (b) *does* pass: a-posteriori training beats the a-priori model (0.0573 vs 0.0591), which
is the core claim for differentiable-solver training. Note also that a **0.85 a-priori
correlation produced no a-posteriori benefit at all** — the a-priori model is marginally worse
than no model. That disconnect is well known in the LES closure literature and is reproduced
here.

**What a meaningful closure test needs:** higher Reynolds number so the sub-grid term carries
more of the dynamics, a larger filter ratio, and a coarse discretisation whose own error is well
below the sub-grid contribution. None is reachable at the resolutions a NumPy solver can afford
— a limitation of this port, not of the method.

**Test case.** Decaying Taylor-Green on a fully periodic grid — we already have exact
solutions, periodic BCs, and verified 2nd-order spatial accuracy there, so the coarse-grid
error is attributable to the closure rather than to boundary treatment.

---

---

## Stage 5c — the eddy-viscosity hook  ⛔ not started, and Stage 5 needs it

**§1 of [`nn_piso_coupling.md`](nn_piso_coupling.md) lists three hooks. Every stage above uses
the first one.** `SGSNet` emits three channels — a force vector — and `make_sgs_data.sgs_force`
returns $-\nabla\!\cdot\tau$, so Stages 2, 3, 4, 5a and 5b all train an explicit SGS FORCE. The
eddy-viscosity hook, $\nu_t = \mathrm{NN}(\mathbf u^n)$ entering the momentum matrix, has never
been built.

That matters because the standard SGS closure — Smagorinsky and everything descended from it —
IS an eddy viscosity. A force closure is a legitimate formulation and it is what PICT's
SGS-stress hook does, but it is not the one most of the literature reports, and it cannot
reproduce a model whose whole content is $\nu_{\rm eff}(\mathbf x) = \nu + \nu_t(\mathbf x)$.

**What the solver supports today.**

| | variable coefficient? | where |
|---|---|---|
| pressure / diffusion operator | **yes** — `build_diffusion_matrix(..., coefs)` takes a per-block array and face-averages it, $\tfrac12(J g_{lo} + J g_{hi})$ | used for the $1/A$ weighting, `piso_multiblock.py:299` |
| momentum operator, multi-block | **no** — `nu` is a scalar: `return nu * Js[b] * g` | `multiblock.py:build_momentum_matrix` |
| momentum operator, differentiable | **no** — `self.nu` scalar into `build_conservative_diffusion_matrix` | `piso_torch.py` |

So the pattern for a spatially varying, symmetry-preserving diffusion coefficient already exists
and is exercised every step; it has simply never been applied to $\nu$ in the momentum matrix.

**THE ONE THING THAT CHANGES THE ADJOINT REQUIREMENT.** With a force closure, freezing the
matrix coefficients drops a *correction* — Stage 4 measured 0.40° of angle error and a 25.4%
worse converged loss. With a viscosity closure, $\nu_t$ enters **only** through $A$. Freeze $A$
and $\partial L/\partial\nu_t$ is **identically zero**: the network receives no gradient at all,
the loss does not move, and nothing errors. `exact_A=True` stops being a recommendation and
becomes a precondition.

**Build.** Costed in [`implementation_plan.md`](implementation_plan.md) §5.2, which also carries
the model formulas and the filter width. (a) accept an array $\nu$ in the momentum assembly,
face-interpolated the way `build_diffusion_matrix` already interpolates `coefs`, so the operator
stays symmetric; (b) route $\partial L/\partial A$ through $\partial A/\partial\nu_t$ in
`MomentumAssembler`; (c) a positive output map — softplus, not a clip, so positivity holds inside
the graph.

**AND ONE PIECE THAT IS NEW PHYSICS, NOT PLUMBING.** Our operator is
$\nabla\!\cdot(\nu\nabla\mathbf u)$. The full stress is
$\nabla\!\cdot\big(\nu_{\rm eff}(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T})\big)$. For
**constant** $\nu$ the transpose part vanishes by continuity, which is why nothing in the port
has ever needed it. With a varying $\nu_t$ it leaves $\nabla\nu_t\cdot(\nabla\mathbf u)^{\mathsf T}$,
which is not small — it is largest exactly where the model is most active. §5.2 flagged this;
it is the only part of this stage that is not threading an array through existing machinery, and
it needs its own MMS (5c.7b below).

**Test cases and success criteria.**

| # | Test case | Config | Success criterion | Failure means |
|---|---|---|---|---|
| 5c.1 | constant $\nu_t$ reproduces the scalar path | 16³ periodic, $\nu_t \equiv c$ | state identical to `nu = nu + c` to < 1e-14 | face interpolation or assembly is wrong before any gradient is involved |
| 5c.2 | operator stays symmetric | random positive $\nu_{\rm eff}$ field | `max|A_diff - A_diff^T| == 0` for the diffusion part | the face average was replaced by a one-sided value; CG on the pressure system would silently degrade |
| 5c.3 | $\partial L/\partial\nu_t$ vs central FD | 8³, one step, 8 sampled cells, tol 1e-12 | ≥ 6 digits | $\partial A/\partial\nu_t$ is wrong or missing |
| 5c.4 | **frozen $A$ gives exactly zero** | same, `exact_A=False` | $\lVert\partial L/\partial\nu_t\rVert$ **== 0**, and the test asserts it | if it is non-zero, something else is feeding the gradient and the result is not the viscosity sensitivity |
| 5c.5 | positivity is enforced | adversarial input driving $\nu_t$ negative | $\min(\nu+\nu_t) > 0$ always; solver never raises | the output map is not guarding the operator |
| 5c.6 | Smagorinsky recovery | $\nu_t = (C_s\Delta)^2\lvert\bar S\rvert$, $C_s$ the only parameter, target from $C_s=0.16$ | descent from $C_s=0$ recovers 0.16 to < 1e-3 | the sign or scale of the viscosity path is wrong — the Stage 1 test, moved to the other hook |

### How $\nu_{\rm eff}$ is actually computed

$$\nu_{\rm eff}(\mathbf x) = \nu + \nu_t(\mathbf x)$$

$\nu$ is molecular and constant. Everything below is about $\nu_t$, and every candidate needs
the same two ingredients.

**1. The velocity gradient tensor, through the metrics.** On this grid $\partial u_i/\partial x_j$
is not a finite difference in $x$:

$$\frac{\partial u_i}{\partial x_j} = \sum_{a\in\{\xi,\eta,\zeta\}} \frac{\partial u_i}{\partial a}\,\frac{\partial a}{\partial x_j}$$

with $\partial a/\partial x_j$ the `xi_x, xi_y, ... zeta_z` entries `block_metrics_cached` already
returns. `src/forces.py` does exactly this at a wall, where the two tangential terms drop; here
all nine survive. Then $\bar S_{ij} = \tfrac12(\partial_j u_i + \partial_i u_j)$ and
$\lvert\bar S\rvert = \sqrt{2\bar S_{ij}\bar S_{ij}}$.

**2. The filter width.** On a stretched curvilinear grid $\Delta$ is the local cell size, which
is the cube root of the cell volume:

$$\Delta = \left(J\,h_\xi h_\eta h_\zeta\right)^{1/3}$$

since $J$ is physical volume per unit computational volume and $h$ are the computational
spacings. `implementation_plan.md` §5.2 previously gave $\Delta = J^{1/3}$, which omits the $h$
factors and is **44× too large** on a measured square-cylinder cell — and $\nu_t \propto \Delta^2$,
so ~2000× in the eddy viscosity. Corrected there. This matters more here than in a cube: our wall cell is 0.006 D and the outer cell is
2.18 D, a factor of 360, so a constant $\Delta$ would be wrong by that factor somewhere.

**The candidates, and why the obvious one is the wrong default.**

| model | $\nu_t$ | note |
|---|---|---|
| **Smagorinsky** | $(C_s\Delta)^2\lvert\bar S\rvert$, $C_s \approx 0.17$ isotropic, $\approx 0.1$ in shear | does **not** vanish at a wall — $\lvert\bar S\rvert$ is largest there — so it needs van Driest damping, which needs a wall distance, which on a multi-block O-grid is another thing to build |
| **Dynamic Smagorinsky** | $C_s^2$ from the Germano identity with a test filter | no tuned constant, but $C_s^2$ can go negative and needs averaging over a homogeneous direction. **We have one**: the span is periodic in every case, `span = 4 D`, so spanwise averaging is available |
| **WALE** | $(C_w\Delta)^2\dfrac{(S^d\!:\!S^d)^{3/2}}{(\bar S\!:\!\bar S)^{5/2}+(S^d\!:\!S^d)^{5/4}}$, $C_w\approx0.5$ | $\nu_t\to0$ like $y^3$ at a wall **by construction**, no damping function and no wall distance. The natural default for wall-bounded cases |
| **Vreman** | positive by construction, from $\alpha_{ij}=\partial_i u_j$ | cheap, no averaging, no wall distance |
| **Network** | $\nu_t = \mathrm{softplus}\big(\mathrm{NN}(\text{invariants of }\nabla\bar{\mathbf u},\ \Delta)\big)$ | softplus, not a clip: positivity must hold in the graph or the gradient is discontinuous where it is needed most |

Constants and exact tensor forms above are from memory and must be checked against the source
papers before they are coded.

**Where it is evaluated and where it is used.** $\nu_t$ is a NODE field, like $u$ and $p$. The
operator needs it on FACES, and the interpolation is the thing 5c.1 cannot test — see 5c.7 and
5c.8 below.

**The honest caveat.** Every production case in this repo is laminar: Re = 100, steady or a clean
limit cycle. A correct SGS model returns $\nu_t \approx 0$ there, so the cylinders cannot test one
— they can only test that it does no harm. The case that can test it is filtered DNS on the
periodic box, which is where Stage 5 already lives, and TGV, where Stage 3.5's energy budget
supplies the bar.

### How to test a variable $\nu$ — three layers, and the first one is not about gradients

**5c.1 is necessary and blind.** Setting $\nu_t \equiv c$ and checking the answer matches
`nu + c` cannot detect a wrong face interpolation, because interpolating a constant is exact
however you do it. It catches assembly and indexing, nothing else. The interpolation is the most
likely thing to be wrong and needs a $\nu$ that actually varies.

**Layer 1 — the operator, before any network exists.**

| # | Test | Why it catches what 5c.1 cannot |
|---|---|---|
| 5c.7b | **MMS on the FULL stress** $\nabla\!\cdot(\nu(\nabla u + \nabla u^{\mathsf T}))$, solenoidal $u$, varying $\nu$ | second order; and a control that OMITS the transpose term must show O(1) error, not merely a degraded rate. For constant $\nu$ the term vanishes by continuity, so no existing test can detect its absence |
| 5c.7 | **MMS with a varying $\nu$**: choose $u(\mathbf x)$ and $\nu(\mathbf x)$ analytically, form $f = -\nabla\!\cdot(\nu\nabla u)$ by hand, refine the grid | second-order convergence of the discrete operator against $f$. A face interpolation that is first-order, or that uses the cell value instead of the face value, shows up as a rate near 1. `accuracy_verification.md` already reports 2.0-2.5 for constant $\nu$ over $\nu \in [0.01, 10]$, so the harness and the bar both exist |
| 5c.8 | **steep $\nu$ jump**, 10x over three cells, against the 1D two-layer analytic solution | arithmetic vs harmonic face averaging. Flux continuity across a jump wants the harmonic mean; arithmetic is second-order for smooth $\nu$ and wrong at a jump. An SGS field near a wall is closer to a jump than to smooth |
| 5c.9 | **row sums are zero** for any $\nu(\mathbf x) > 0$ | a constant field must lie in the operator's null space, which is momentum conservation. Exact, cheap, and independent of any solution |
| 5c.10 | **negative semi-definite**: $u^{\mathsf T} A_{\rm diff} u \le 0$ on 100 random fields | diffusion must remove energy at every wavenumber. If a face average can go negative — which it can if $\nu_t$ is unclipped — this is where it surfaces, not in a diverging run three hours later |

**Layer 2 — does it do the right physics?**

| # | Test | Bar |
|---|---|---|
| 5c.11 | **two-layer Couette**: $\nu_1$ for $y<0$, $\nu_2$ above, steady | velocity slope ratio equals $\nu_2/\nu_1$ to < 1e-10; the exact piecewise-linear solution |
| 5c.12 | **variable-$\nu$ Poiseuille**: $\nu(y)$ smooth, integrate the analytic profile twice | matches to discretisation order |
| 5c.13 | **Smagorinsky on TGV**, reusing the Stage 3.5 energy budget | $\nu_t \ge 0$ everywhere; $\nu_t \to 0$ where $\lvert\bar S\rvert \to 0$; measured $-\mathrm{d}E/\mathrm{d}t$ equals $2\langle(\nu+\nu_t) Z\rangle$ within the 5% the budget already achieves |

5c.13 is the one that says the closure is wired in correctly rather than merely differentiable:
the energy budget already built for Stage 3.5 becomes the instrument, and an eddy viscosity that
does not increase dissipation is not an eddy viscosity.

**Layer 3 — the gradient.** 5c.3 to 5c.6 above.

### FOUR PLACES ASSUME A SCALAR $\nu$, and only one is the matrix

Found by reading `piso_multiblock.py` rather than by assuming:

| line | what | with $\nu(\mathbf x)$ |
|---|---|---|
| 238 | `build_momentum_matrix(..., self.nu, ...)` | the obvious one |
| 278 | `rhs = base + Jg * (self.nu * cd)` — deferred-correction cross term | must use the same face-interpolated field, or the correction and the matrix disagree and the deferred correction stops contracting |
| 439 | `self.p = ... - self.nu * div_star[b]` — the rotational scheme's pressure | becomes $\nu(\mathbf x)\,\nabla\!\cdot\mathbf u^*$, a field multiply |
| 520 | `pv = self.nu * (un - un_i)/dn - ...` — Dong outflow pressure | needs $\nu_{\rm eff}$ AT the boundary, where an SGS model is least reliable |

Lines 278, 439 and 520 are the ones that would be silently wrong: each still runs, still
converges, and quietly mixes a molecular $\nu$ into a field that is no longer molecular. A
grep for `self.nu` is part of this stage's build, not an afterthought.

**One coupling to watch.** $\Gamma = J/A_{\rm diag}$ and $A_{\rm diag}$ now varies with
$\nu_t$, so the Rhie-Chow damping and the pressure operator's weighting become spatially varying
too. Nothing is wrong with that, but `pressure_checkerboard.md`'s measurements were taken with a
constant $\nu$, and the checkerboard amplitude should be re-measured once $\nu_t$ is live.

5c.4 is the one worth writing first. It is a test that a gradient is **zero**, which is unusual,
and it exists because that zero is the silent failure this hook invites: train with the default
`exact_A=False` and the model simply never learns, with no error anywhere.

5c.6 is Stage 1 repeated through the viscosity hook — recover one scalar, prove sign and scale —
before any field-valued network is attached.

---

## Test problems and acceptance criteria at a glance

Every stage names one concrete problem and a numeric bar. Stages 1–3 are deliberately tiny —
they are debugging instruments, not experiments, and should run in seconds to minutes.

| Stage | Test problem | Config | Acceptance criteria |
|---|---|---|---|
| **0** ✅ | adjoint of one linear solve | 8³ warped, one PISO step | (a) adjoint identity on the **non-symmetric** momentum matrix, rel. err < 1e-8; (b) FD through a full step, ≥ 6 digits; (c) pressure gradient invariant to a constant shift of $\bar g$, < 1e-12 |
| **1** ✅ | recover a scalar forcing amplitude | 8³ warped, 1 step, $S = c\,\Phi(\mathbf x)$, $c_{\text{true}}=0.7$ | (a) $\partial L/\partial c$ vs central FD, rel. err < 1e-6; (b) **sign** correct (negative when $c < c_{\text{true}}$); (c) descent from $c=0$ recovers $0.7$ to < 1e-4 |
| **2** ✅ | tiny CNN predicts a source field | 16³ **periodic** Cartesian, 1 step, 173 weights, target from the same architecture with different weights | (a) FD on **every** weight, max rel. err < 1e-5 → **4.6e-08**; (b) reproduce the target **velocity** to < 1e-3 → **8.9e-04**; (c) shift-equivariance < 1e-10 → **1.1e-16** |
| **3** ✅ | same CNN, 5-step rollout | 12³ periodic, loss on final state | (a) FD on 5 sampled weights, rel. err < 1e-5 → **6.1e-09**; (b) checkpointed == non-checkpointed → **exactly 0**; (c) peak memory at 16 steps **0.8 MB vs 13.8 MB**; (d) $\lVert\lambda\rVert$ ratio early→late **0.91** (bounded) |
| **3.5** | **3D Taylor-Green energy budget** | 48³ periodic, $\nu=0.01$, $t\in[0,2]$ | (a) $-\mathrm{d}E/\mathrm{d}t$ vs $2\nu Z$ agree within **5%** for central; (b) numerical dissipation *quantified* for SOU; (c) $E$ monotone decreasing; (d) flux divergence < 1e-9 throughout |
| **4** ✅ | frozen-coefficient bias | 10³ periodic, 5-step rollout | Measured: angle **0.40°**, but converged loss **25.4% worse**, and FD error 1.16e-01 → 1.99e-02 with `exact_A`. **The angle criterion proved insufficient** — see below. Recommendation: `exact_A=True` |
| **5a** ✅ | a-priori SGS regression | filter 48³ → 16³, random solenoidal field, no solver in the loop | held-out correlation > 0.8 → **0.850** (trivial baseline −0.21) |
| **5b** ⚠️ | a-posteriori closure training | 16³ coarse vs filtered 48³, 6-step train / 30-step check | (a) **FAIL** 2.7% vs 30% bar — *but the oracle headroom is −0.3%, so the bar was unreachable*; (b) **PASS** 0.0573 vs 0.0591; (c) **PASS** stable at 5× horizon. See the diagnosis below |

**Why Stage 3.5 sits where it does.** It is not a gradient check, it is a *prerequisite* for
Stage 5 to mean anything. MMS verifies that each operator approximates its differential
counterpart; it says nothing about whether the assembled nonlinear scheme conserves the
quadratic invariants that govern a cascade. The periodic identity

$$\frac{\mathrm{d}E}{\mathrm{d}t} = -2\nu Z, \qquad E=\tfrac12\langle|\mathbf u|^2\rangle,\quad Z=\tfrac12\langle|\boldsymbol\omega|^2\rangle$$

holds exactly in the continuum, so the gap between measured $-\mathrm{d}E/\mathrm{d}t$ and
$2\nu Z$ **is** the scheme's numerical dissipation. That matters directly for Stage 5: SOU is
dissipative by construction, so training a closure on an SOU baseline asks the network to
correct numerics as well as physics, and the two become inseparable. Measure the numerical
dissipation first, then decide which convection scheme the closure work should use.

**Measured** (`run_tgv3d.py`, 48³, ν = 0.01, rotational + BDF2, t ∈ [0,2]):

| scheme | mean numerical / physical dissipation | max | energy at t=2 |
|---|---|---|---|
| 2nd-order upwind | **1.10 %** | 1.48 % | 6.66e-04 |
| central | **0.56 %** | 0.66 % | 6.77e-04 |

Energy is monotone decreasing for both, and flux divergence stays at 7.8e-15 (SOU) /
4.3e-14 (central). SOU carries **roughly twice** central's numerical dissipation and removes
1.56 % more total energy by t = 2 — the dissipative error is real, quantified, and exactly the
sort of thing a closure would otherwise be asked to absorb.

**Resolution honesty — this is not a turbulence benchmark.** The canonical TGV case is
$Re=1600$, needing ~$256^3$ for DNS, far beyond a NumPy solver. At ν = 0.01 the flow is fully
resolved and laminar: **enstrophy peaks at t = 0 and decays monotonically**, so there is no
vortex stretching and no cascade (the real TGV peaks near t ≈ 9). What this validates is the
*energy-budget machinery* and the relative dissipation of the two convection schemes — not
turbulence. Any write-up must say so.


## Verification tooling to build alongside

| Tool | Purpose |
|---|---|
| `check_gradient(fn, params, idxs)` | central FD vs adjoint, float64, with tolerance tied to solver tolerance |
| `check_adjoint_identity(A)` | $\langle A^{-1}v, w\rangle = \langle v, A^{-\mathsf T}w\rangle$ — run it on the **momentum** matrix, where a missing transpose is fatal and detectable |
| adjoint-norm logger | $\lVert\lambda\rVert$ per step, to catch adjoint blow-up early |

---

## Standing risks

- **Solver tolerance caps gradient accuracy.** The adjoint inherits the forward residual, so a
  forward tolerance of $10^{-6}$ limits gradient accuracy to roughly the same level. Tighten to
  $10^{-12}$ during FD checks or the residual dominates the comparison.
- **Non-convergence must raise, not warn.** A silently unconverged adjoint solve produces a
  plausible but wrong gradient. The deferred-correction contraction check already in the port
  must be applied to the adjoint loop too.
- **Grid warp ≲ 0.15.** The deferred correction stops contracting beyond that, forward and
  adjoint alike. Keep training cases inside it, or make the cross terms implicit first.
- **Wall-bounded cases are 1st-order in space.** Prefer periodic training cases until the
  half-cell boundary-flux stencil is upgraded, so closure error is not confounded with
  boundary error.

---

## Order of work

```
Stage 0  done
Stage 1  done  scalar recovery        <- proves sign and scale
Stage 2  done  tiny CNN, all weights  <- proves the field mapping
Stage 3  done  rollout + checkpoint   <- proves the time chain
Stage 3.5 TGV energy budget     <- proves the baseline is not numerically polluted
Stage 4  done   frozen-coeff bias     <- angle 0.4 deg BUT loss 25% worse; use exact_A
Stage 5a done  a-priori regression   <- corr 0.85; capacity confirmed
Stage 5b part  a-posteriori training <- (b),(c) pass; (a) unreachable (oracle -0.3%)
Stage 5c  plan  EDDY-VISCOSITY hook   <- not started; every stage above uses the FORCE hook
Stage 6+  plan  MULTI-BLOCK          <- see nn_multiblock_plan.md; not started
```

Everything above runs on ONE block, 16^3, periodic, Cartesian. Every case the port actually
simulates -- backward-facing step, square cylinder, circular cylinder -- runs on
`MultiBlockPISO`, which has no gradients at all. Connecting the two is planned in
[`nn_multiblock_plan.md`](nn_multiblock_plan.md) and has not been started.

Each gate is cheap relative to the stage it protects; stages 1–3 should run in seconds to
minutes on a 16³ grid, which is deliberate — they are debugging instruments, not experiments.
