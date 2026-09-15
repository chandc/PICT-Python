# hydrogym_pict: the production solver as a HydroGym backend

PICT-Python behind HydroGym's solver-agnostic contract (`PDEBase` /
`TransientSolver` / `FlowEnv`, dynamicslab/hydrogym): the R11-validated
butterfly cylinder at Re 100 with rotary control, mirroring their firedrake
`RotaryCylinder` conventions -- observations (C_L, C_D), objective C_D,
reward -dt*C_D, MAX_CONTROL pi/2, damped actuator TAU 0.0556. Strategic
rationale and the FluidGym-vs-HydroGym assessment: fluidgym_parity.md.

## Layout

- `hydrogym_pict/core_shim.py` -- imports the real `hydrogym.core` when the
  package is importable; otherwise the vendored copy of the same file
  (`_vendored_core.py`, main branch fetched 2026-09-14). PyPI 0.1.2.1 is
  the stale 2023 release whose `__init__` hard-imports firedrake; git main
  needs python >= 3.10, so the Mac's 3.9 venv develops against the vendored
  contract. `HYDROGYM_SOURCE` records which one loaded.
- `hydrogym_pict/flow.py` -- `RotaryCylinder(PDEBase)`: production
  `MultiBlockPISO` (rotational scheme, Rhie-Chow, persistent flux,
  **implicit_cross ON by default** -- see gotchas), meshes "coarse" (the
  21k-cell gate build) and "production" (the R11 campaign build).
  Rotation enters as tangential Dirichlet u = omega x r on the body wall.
  `copy_state`/`set_state` snapshot the FULL restart state (fields, BDF2
  history, Rhie-Chow p_flux and F_prev, clock) so env resets are exact;
  file checkpoints delegate to src/checkpoint.
- `hydrogym_pict/solver.py` -- `PICTTransientSolver`: per step, integrate
  the actuator lag, write the rotation BC, advance one production step.
- `make_env(mesh, dt, tol, restart, num_substeps, ...)` -- one-liner
  returning a configured `FlowEnv`.

## Gates (test_hydrogym_pict.py, 6/6 on the Mac, ~10 min)

  hg.1 FlowEnv conformance (spaces, 5-tuple, reward = -dt*C_D)
  hg.2 uncontrolled 40 steps: C_D 1.485 decaying toward the campaign band,
       C_L = -0.0000 (symmetric state -- machine-zero lift)
  hg.3 exact reset: |obs - fresh env| = 0.0 after 40 steps
  hg.4 Magnus response: dC_L(omega=+1) = -0.3532, dC_L(-1) = +0.3532,
       antisymmetry defect 0.00%, C_D split 1.9e-7
  hg.5 actuator lag follows 1 - exp(-dt/tau) exactly
  hg.6 num_substeps=5 advances 5 solver steps, aggregates the reward

## Gotchas earned during the build

1. **implicit_cross is required physics on the butterfly.** The first gate
   run defaulted it off (mirroring the runner's env-var pattern) and
   reproduced the R11 growth signature verbatim: ~2x/step from the sheared
   trapezoid corners, NaN by step ~17. On this grid the orthogonal-only
   pressure solve is not an approximation, it is unstable. Default ON.
2. **A short repro can sit inside the knee.** The blow-up starts ~step 9;
   2-3-step bisections all passed and pointed away from the real cause.
   The traceback's loop line reports WHICHEVER iteration fails, not the
   first.
3. The rotating wall violates the no-slip assumption in the traction
   integral: forces use `check_wall=False` and the spurious viscous-normal
   share is tracked separately (`_spurious_normal`).

## Shedding validated through the env (validate_hydrogym_shedding.py)

No settle run was needed: the R11 final checkpoint
(results/fields/cylrect_r11_final.npz, t = 380, full restart state) loads
straight through `make_env(mesh="production", restart=...)` -- the grid
fingerprint and config both match the backend defaults. 1800 uncontrolled
env steps (~3 shedding periods, ~4.7 s/step on the Mac):

  St     0.1673  vs R11 0.1673   (0.00%)
  C_D    1.3216  vs R11 1.321    (0.05%, whole-period mean)
  C_L rms drift 0.08% first-to-last period (restart is lossless)

The campaign physics round-trips through HydroGym's API exactly. Trace in
results/hydrogym_shedding_validation.npz.

## Status / next

- Non-differentiable backend DONE and PHYSICS-VALIDATED (above). SB3
  baselines can train against `make_env(mesh="production",
  restart="results/fields/cylrect_r11_final.npz")` as-is; budget
  ~4.7 s/solver step on the Mac. Coarse-mesh restarts for cheap RL
  experiments TBD.
- Differentiable mode = the production-adjoint build (Stage 9/10); the
  interface has no gradient slot (their differentiable envs are JAX
  end-to-end), so gradients will be exposed as a PICT extension -- the
  first non-JAX differentiable HydroGym backend.
- Genuine-package validation (python >= 3.10): run the same gates in a
  container with `pip install git+https://github.com/dynamicslab/hydrogym`;
  the shim flips to the real classes with no code change.
