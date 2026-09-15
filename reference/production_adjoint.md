# Stage 9: the PRODUCTION-solver adjoint

The one build behind three deliverables: M2's open-loop optimal a(t) on our
own physics (fluidgym_parity.md), the FluidGym parity closure with verified
gradients, and the first non-JAX differentiable HydroGym backend
(hydrogym_backend.md). The frozen-coefficient chains (Stages 6-8, M1, M2)
certified every ingredient piecewise -- seams, BCs, Dong, DC cross solve,
vector coupling, jets, forces, LinearSolve's dL/dA = -lambda x^T -- but the
chains freeze the momentum operator, and the M2 instability study showed
that boundary is hard: physical rollouts need the PRODUCTION step. What no
chain has certified is gradient flow THROUGH MATRIX ASSEMBLY, because the
production step reassembles A from the current velocity (Picard).

## The structural fact 9.1 exploits

`build_momentum_matrix` is CENTRAL-convection only, so A is AFFINE in the
convecting velocity: A(u,v,w) = A0 + reshape(T x), x = (u|v|w) flat, with
T a CONSTANT sparse (nnz x 3N) sensitivity operator. Differentiable
assembly is therefore EXACT: build T once per mesh by colored probing of
the production assembler (the probe-the-production-code method that built
cross_divergence_matrix), and torch assembly is vals0 + T @ x -- fully in
the autograd graph, composing directly with LinearSolve's existing
matrix-values gradient. Linearity is not assumed: gate 9.1a MEASURES it,
and any hidden nonlinearity (a sign branch, a limiter) fails the gate.

Coloring: probing cell c perturbs entries whose faces read c's padded
velocity (dist <= 2 in the stencil graph S = pattern(A)); two probe cells
collide when their influence regions overlap an entry span (dist <= 5).
Conflict graph = S^5, greedy-colored; per color, one production assembly
per velocity component, changed entries attributed to the unique in-batch
cell within S^2 of the entry's row or column. T is cached per
(grid, nu, dt) fingerprint.

## Roadmap

  9.1  Differentiable momentum assembly (THIS increment).
       Gates: (a) exactness/linearity of vals0 + T x vs production
       assembly on random fields; (b) pattern stability across fields;
       (c) autograd of <w, A(x) y> vs FD; (d) THROUGH THE SOLVE:
       d/dx of ||A(x)^{-1} b||^2 via LinearSolve, FD-checked -- the
       assembly -> matrix -> solve -> loss path that no chain has.
  9.2  One full production step in torch: Picard(2) x corrector(2) with
       fresh A per Picard sweep, Rhie-Chow flux, DC cross pressure, Dong.
       Gate: field-for-field equivalence with m.step() on the butterfly
       (the decisive gate -- the torch step IS production or it fails).
  9.3  Gradient gates on the torch step: FD vs adjoint through 1-3 steps
       with assembly in the graph; NEW mangle: detach the assembly path
       (frozen-A) and require the gradient to CHANGE -- measuring exactly
       what the chains were missing.
  9.4  Multi-step rollout from the R11 restart: bounded, tracks production
       trajectory; the M2 surrogate instability must be ABSENT (it is a
       frozen-operator artifact; this step is not frozen).
  9.5  Objectives on top: dC_D/d(jet a) and d(reward)/d(policy weights) on
       production physics; wire as the differentiable mode of
       hydrogym_pict and the M3 parity runs.

## Status

- 9.1 GATED 4/4: affine assembly measured exact at 2e-18 (46 colors, 3 s
  T build, cached); assembly -> solve -> loss gradient FD-exact at 2e-9.
- 9.2a GATED 3/3: pad_field probed into exact affine gather maps (PadMap
  asserts the selection structure); torch face_fluxes + divergence
  BIT-IDENTICAL to production on all 9 butterfly blocks (0.00e+00, seams
  included, axis-aligned branch); gradient through pad -> flux ->
  divergence FD-exact at 3e-11. src/prod_adjoint.py TorchFluxKernels.
- 9.2b GATED 4/4: pressure_face_fluxes in torch, machine-exact (9e-18 ..
  1.7e-16) in all three step call patterns; bilinear d/dp, d/dcoef
  FD-exact.
- 9.2c GATED 3/3: cross_diffusion machine-exact; M(coef) linear
  sensitivity (10 colors) with FD-exact gradient.
- 9.2d GATED 2/2 -- THE MILESTONE: TorchProductionStep (src/prod_step.py)
  is field-for-field equivalent to m.step() at u 3e-9 / p 5e-9 (both
  sides tightened to ~1e-11 solves; production env-var DC semantics,
  Picard inf-norm exit, Dong copy, persistent flux). The gate CAUGHT a
  real landmine on first run: the chains' cell_gradient_matrix differs
  O(1) from d.gradient on the butterfly; production_gradient_ops now
  probes the genuine operator (probe_cell_operator, reach-2 coloring for
  edge_order=2 stencils). The step contains ZERO re-derived operators --
  every linear map is probed from the production code it mirrors.
  Timing: 52 s/torch step vs 17 s production at tight tolerances (coarse
  butterfly, Mac).
- 9.3 GATED 4/4 (suite 20/20): dL/d(state) FD-exact at 7e-7 and
  dL/d(source) at 8e-5 through the full production step; detach_assembly
  mangle LIVE at 5e-4 (the through-assembly path measured at last). Two
  earned findings: torch's backward walks the FULL padded arrays, so
  numpy-inert ghost corners (J = 0 / nonfinite metrics, E-trapezoid)
  manufacture NaN grads via 0*inf -- geometry constants sanitized, the
  9.2 bit-level gates re-verify the forward untouched; and production's
  Dong pressure is IDENTICALLY inert in forward-flow states (copy BC
  zeroes the viscous term structurally, saturated tanh zeroes theta and
  its derivative), so the liveness probe manufactures outlet backflow
  (grad change 3e-3 once exercised). DC sweep count pinned during FD
  probes (a data-dependent exit puts a kink between probes).
- NEXT (9.4): multi-step rollout from the R11 restart -- bounded, tracks
  the production trajectory; the M2 chain instability must be ABSENT.
  Then 9.5 (objectives + HydroGym/FluidGym differentiable wiring).
  Butterfly uses the axis-aligned seam branch of face_fluxes; only that
  branch is ported (the other raises).
