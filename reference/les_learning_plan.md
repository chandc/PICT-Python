# Learning in LES — what is missing here, and how the original PICT does it

Written 2026-09-03. The PICT statements below are quoted from the paper itself
(arXiv:2505.16992v2, Franz, Wei, Guastoni & Thuerey, TUM) after a first automated summary of
the same PDF returned the **opposite** answer on the central question. It claimed the learned
model outputs an eddy viscosity; the paper says a correcting force. Everything here was checked
against the pages rather than a summary.

---

## 1. How the original PICT actually does it

### The model is a forcing, not an eddy viscosity

> "we train an SGS model in form of a corrector `G_θ` that is tasked to estimate a correcting
> force `S_θ` at a low spatial resolution of 64 × 32 × 32."

So for the turbulent channel, PICT learns a **momentum source**, which is the same form as our
`TinySGSNet`. Our architecture is not behind theirs on this point; it matches.

That is not a limitation of the solver. Appendix A.5 is explicit that viscosity is
differentiable too:

> "The differentiable quantities are `u`, `ν`, `ρ`, and `S`, where boundary values for `u` and
> `ρ` are also differentiable, as well as any derived intermediate quantities like matrices and
> RHS of the linear systems."

**`ν` → matrix is a differentiable path in PICT.** They simply chose the forcing form for the
channel. This is the one capability we genuinely lack: our `build_momentum_matrix` is NumPy and
only its *values* enter torch, so `dL/dν_t` does not exist here.

### The inputs are raw velocity plus wall distance, not invariants

> "For the TCF, `G_θ` receives the instantaneous velocity and the normalized wall distance
> `1 − |y/δ|` as inputs, i.e. `S_θ = G_θ(u^n, 1 − |y/δ|)`."
>
> "The term `1 − |y/δ|` is added to inform the network of the grid refinement in regions near
> the wall."

**This contradicts the advice I gave earlier in this project**, which was to feed strain-rate
invariants for Galilean invariance. PICT does not, and it works. Two honest readings: for a
channel at fixed Re in a fixed frame the invariance buys nothing and costs capacity; or the
model is frame-dependent and would not transfer, which for their stated goal it does not need
to. **Invariants remain defensible for a closure meant to generalise across geometries, but
they are a departure from PICT, not a correction to it.**

### The architecture

> "a simple CNN with layers using 8, 64, 64, 32, 16, 8, 4, and 3 filters, each having a kernel
> size of 3³. Only the last layer uses a kernel size of 1³. This gives 198931 trainable
> parameters in total. ReLU activations are used for all but the last layer."

199k parameters against our `TinySGSNet`'s 173. Ours is deliberately tiny so finite differences
on every weight are cheap; theirs is a production model.

### The loss is on STATISTICS, not on matching a field

> "the dynamics in the 3D TCF are highly turbulent and matching individual realizations of the
> flow from simulations at different resolutions no longer provides a physically meaningful
> learning target. Hence, we instead aim for matching the turbulence statistics via the
> statistic loss from eq. (13), complemented with a regularization term on the generated
> forcing."

    L = L_stats + λ_S (1/N) Σ_n ‖S_θ^n‖²₂                                    (their eq. 16)

> "We additionally constrain the forcing to the [−2, 2] range to stabilize early training."

And from the introduction:

> "A distinctive feature of our work is that the PICT solver allows for training the SGS model
> while only supervising in terms of velocity moments. I.e., no pre-computed training data sets
> are required in this case."

### The forcing is projected divergence-free

> "it proved essential to prevent un-physical network outputs, which in our case means violating
> the incompressibility assumption. To ensure divergence free flow motions, we include the
> gradient modification from eq. (11) for `S_θ`."

**This is the same fact this repo measured independently and recorded in
`measurement_traps.md` §7**: only the SOLENOIDAL part of a momentum source is identifiable from
velocity data, which is why Stage 2 matched the velocity to 8.9e-04 while the recovered source
was 18% wrong and both were correct. PICT removes the ambiguity by construction instead of
living with it. **We should do the same** — it converts an unidentifiable parameter into an
identifiable one rather than merely documenting the problem.

### They test excluding vs including the linear-solve gradients

> "we consider two different approaches for the training of the CNN model: in the first one, we
> exclude the gradients of the linear solves from the optimization (i.e. only using `J^none`
> from eq. (8)). In the second variant, we start in the same way, but include the terms at a
> later [stage]"

That is precisely our Stage 5c question — the frozen-coefficient approximation — treated as a
tunable training strategy rather than as a correctness issue. Worth knowing before we treat
"exact gradients or nothing" as the only defensible position.

### The result

Average `Re_τ` from the three simulations at a target of 550:

| model | achieved `Re_τ` |
|---|---|
| no SGS | 390 |
| Smagorinsky | 452 |
| learned CNN SGS | **548** |

---

## 2. What we would have to build

Ordered so the cheap, exactly-testable parts come first and the expensive one is judged against
something.

### 2.1 Divergence-free projection of the learned source — small, and highest value per hour

Our Stage 2 result says the non-solenoidal part of a source is invisible to velocity data. PICT
projects it out. This is a projection we already have — the pressure Poisson solve — applied to
`S` instead of to `u*`.

*Test:* a source with a deliberate irrotational component must project to the same velocity
trajectory as one without it, and the recovered source must then be unique. That converts the
18%-wrong-but-correct outcome of Stage 2 into an exact recovery test.

### 2.2 A statistics loss — small

Mean profile, Reynolds stresses, and the energy budget terms as a differentiable function of a
trajectory. Cheaper than it sounds because the moments are just weighted sums over the rollout.

*Test:* on a trajectory generated by a known model, the statistics loss must be minimised at
that model's parameters — an inverse-crime control of the kind `nn_fit_constants.py` already
uses, and for the same reason: it separates "the machinery recovers parameters" from "the model
is right".

### 2.3 The `ν_t` head — half a day, given the machinery now exists

    nu_t = Δ² |S| · softplus(net(...))

Dimensionally correct by construction, non-negative, with Smagorinsky as the special case
`net ≡ C_s²`. Pointwise (1×1×1) needs no halo at all; a stencil version uses the
`pad_mode="none"` path added today.

*Tests:* scaling (double Δ → 4×), Galilean invariance, Smagorinsky recovery to round-off,
positivity, seam invariance. All exact, all cheap — the same shape as `test_sgs_models.py`.

### 2.4 The differentiable `ν` → matrix path — two to four days, and the real work

Currently impossible here: `build_momentum_matrix` is NumPy, and `mb_adjoint` takes only the
assembled *values* into torch. Either assemble the diffusion operator in torch (as
`nn_eddy_viscosity.py` does in 1-D) or write a custom `autograd.Function` carrying the analytic
`dA/dν`.

*Test:* FD vs adjoint on the network weights with `ν_t` entering `A`, **plus a frozen-coefficient
mangle that must make the gradient collapse to zero.** Without that mangle the gate cannot
distinguish "working" from "silently disconnected", which is exactly the failure Stage 5c
predicted: freeze the coefficients and the gradient is identically zero, the loss does not move,
and nothing errors.

PICT's finding above matters here: they train *without* the linear-solve gradients first and add
them later. So this path may be a refinement rather than a prerequisite — which argues for
building 2.1–2.3 first and measuring how far they get.

### 2.5 What we would still lack

Resolution. PICT's channel is 64 × 32 × 32 over 36 eddy-turnover times of warm-up plus 22 more.
Our cases run `nz = 4`. That is a compute question, not a code question, but it is the gap that
decides whether any of this can be validated against turbulence data rather than against
manufactured targets.

---

## 3. Sources

* Paper: [PICT – A Differentiable, GPU-Accelerated Multi-Block PISO Solver for Simulation-Coupled Learning Tasks in Fluid Dynamics](https://arxiv.org/pdf/2505.16992), Franz, Wei, Guastoni & Thuerey (TUM), arXiv:2505.16992v2, and the [journal version](https://www.sciencedirect.com/science/article/pii/S0021999125007156).
* Code: [github.com/tum-pbs/PICT](https://github.com/tum-pbs/PICT), with the channel-flow learning recipe in `TCF.md`.

---

Full citations for every source named here: `reference/bibliography.md`.
