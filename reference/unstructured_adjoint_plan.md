# Discrete adjoint for the unstructured PISO solver, in PyTorch — implementation plan

*Draft 2026-09-26. Written against `origin/unstructured` @ `70b817e`. All paths are on that branch;
the work itself should be done on a branch cut from it, not from `main`, which has neither the
unstructured solver nor the Stage 9 adjoint.*

---

## 1. What exists, and what carries over

**The structured adjoint (Stage 9, `reference/production_adjoint.md`)** differentiates one
`MultiBlockPISO` step:

| piece | file | carries over? |
|---|---|---|
| `LinearSolve` autograd Function: x = A⁻¹b, backward λ = A⁻ᵀg, dL/dA = −λxᵀ on the pattern, singular (all-Neumann) handling | `src/adjoint_piso.py` | **yes, as is**, plus a factor-reuse extension (§3, U3) |
| Probe-built affine momentum assembly `vals = vals0 + T x` (colouring at distance 5) | `src/prod_adjoint.py` `MomentumAssembly` | **no** — needed because production assembly was opaque; the unstructured operators are explicit (below) |
| `PadMap`, `TorchFluxKernels`, `TorchPressureFlux`, `probe_cell_operator` | `src/prod_adjoint.py` | **no** for the kernels (ghost padding, seams, block metrics do not exist on the unstructured mesh); **yes** for `probe_cell_operator` as an *independent cross-check* |
| `TorchProductionStep`: the whole step in torch, field-for-field equal to production | `src/prod_step.py` | **pattern yes, code no** |
| Memory-flat replay: forward under `no_grad` with detached snapshots, reverse sweep rebuilding one step's graph | `src/prod_replay.py` | **yes**, with a different state key set |
| Gate discipline: equivalence to production first, FD second, "mangle" (detach a path, require the gradient to change) third, residual-gated solves throughout | `test_prod_*.py` | **yes** — the plan below is written as gates of the same kind |
| DPC trainer (`prod_dpc_train.py`), HydroGym backend (`hydrogym_pict/`) | | **yes** once a force objective and actuator exist on the unstructured step |

**The unstructured solver was built adjoint-first.** `src/uops.py` states it: *"EVERYTHING IS AN
EXPLICIT SPARSE MATRIX. The discrete adjoint needs to transpose these operators, so nothing here is
matrix-free and nothing mutates its inputs."* Concretely:

* **Constant, geometry-only** (assembled once per mesh): LSQ gradient `Gx, Gy, Bx, By`
  (`uops.lsq_gradient`), dual Green–Gauss pressure gradient (`ugrad.GGDualGradient`, same four
  matrices), orthogonal Laplacian part and its deferred cross-term operators `Cx, Cy, Bd`
  (`uops.laplacian`), divergence (face→cell scatter over `owner`/`neigh`), face interpolation.
* **Solution-dependent**, and each a short, explicit function of face arrays:
  * convection matrix — linear in the face flux F, with an upwind *sign mask*
    (`uops.convection`);
  * SIMPLEC coefficients `aP = diag(A)`, `aC = max(rowsum(A), a_t V)` (`PISO._momentum`);
  * pressure Laplacian — linear in `gam`, the face-interpolated `V/aC`;
  * Rhie–Chow face flux — bilinear in `D = V/aP` and p, plus the time-step-independent history
    term (`PISO._rhie_chow`);
  * pressure-correction flux (`PISO._pressure_flux`);
  * boundary values `BC.effective` — linear, with a Dirichlet/Neumann mask.

So the torch port is a **direct transcription** onto `index_add_` over face arrays and constant
torch sparse matrices. The probing machinery that Stage 9.1 needed is not needed; it is kept only
as a second, independent check on the transcription.

**What is harder here than on the structured grid**:

1. **Sign-dependent sparsity.** The implicit convection matrix is upwind, so which column a face
   writes to depends on sign(F_f). The pattern can change from one step to the next.
2. **Non-smooth points.** The upwind switch at F_f = 0, the `max` in `aC`, and the `1e-300` floors
   are kinks. The gradient exists almost everywhere and is piecewise-defined.
3. **Tolerance-terminated loops.** `solve_poisson` exits its non-orthogonal loop on a tolerance;
   `n_inner`, `n_corr` and `n_outer` are fixed counts. The graph must be of the iteration actually
   run (the Stage 7b lesson).
4. **Two integrators and a 2.5D solver.** `PISO.step` (BDF2-PISO, the laminar and control cases)
   and `PISO.step_rk3` (the LES integrator), then `PISO25` (complex Fourier modes, CuPy, block
   AMG-PCG).

**What is easier:** no ghosts, no padding, no seams or corner pins, no Dong outflow, no multiblock
bookkeeping, and the 2D production solves are sparse direct (`backend="splu"`), so the adjoint
solve can reuse the forward LU factorisation.

---

## 2. Scope and order

Target the BDF2 PISO step first. It is the integrator behind every control case validated on this
solver: the Re 100 cylinder on HydroGym's triangles and the butterfly quads (T9), NACA0012 at
α 20°/40° (§45–46 of `skew_unstructured_literature.md`), the pinball, and `naca_env.py`. RK3 comes
second (LES and transitional control), and the 2.5D solver last.

```
U0 decisions ─ U1 constant ops ─ U2 solution-dependent ops ─ U3 LinearSolve for splu
  ─ U4 torch BDF2 step (equivalence) ─ U5 gradient gates ─ U6 forces + actuation
  ─ U7 replay ─ U8 first control result
  ─ U9 RK3 step ─ U10 2.5D ─ U11 cross-code check against the structured adjoint
```

---

## 3. Stages and gates

Every stage ends on a gate. Nothing starts until the previous gate passes. Tolerances are stated
up front so a gate cannot be retuned after the fact. The gates below are summarised; §7 is the
full test catalogue, with the physical tests that sit beside them and the acceptance criteria that
decide the whole plan is done.

### U0 — Decisions (1–2 days)

* **Upwind pattern.** Assemble convection on a fixed **superset pattern**: for every interior face
  write both the owner and the neighbour columns, with the inactive one carrying an exact zero.
  The pattern is then geometric, `LinearSolve`'s pattern and SuperLU's symbolic analysis are
  reused, and the sign mask becomes a detached, per-step value mask. *Alternative rejected:*
  rebuilding the pattern per step, which invalidates every cache and makes gradient bookkeeping
  per-step.
* **Kinks.** Every mask (upwind sign, `aC` floor, BC kind) is computed from **detached** values and
  applied as a multiplier. The gradient is the one of the branch actually taken. FD gates hold
  masks fixed across ± probes and report the count of faces within δ of a switch.
* **Loop counts.** Pin `n_nonorth` to a fixed count in differentiable mode (production default 2–3)
  and add a production flag that runs the same fixed count, so equivalence is exact. The
  convergence test stays a logged diagnostic.
* **Precision and device.** float64 on the CPU for U1–U8. The GPU arrives with U10, through the
  existing CuPy path via DLPack.
* **Module layout.** `src/uadj_ops.py` (U1–U2), `src/uadj_solve.py` (U3), `src/uadj_step.py` (U4,
  U9), `src/uadj_replay.py` (U7), `src/uadj_forces.py` (U6); tests `test_uadj_*.py`.

### U1 — Constant operators in torch (2–3 days)

Convert each geometry-only scipy matrix to a torch sparse CSR constant once per mesh, keyed on the
existing grid fingerprint. Gradients (LSQ and dual GG), Laplacian orthogonal part, `Cx/Cy/Bd`,
face interpolation, face→cell scatter and divergence, plus the mesh arrays `owner, neigh, wf,
normal, span, ef_over_d, Tf, vol, bface_index`.

**Gate U1** (cavity quads, clustered rect mesh, HydroGym triangles, fine butterfly):

* torch vs numpy on random fields: ≤ 1e-15 relative, every operator;
* adjoint identity ⟨y, A x⟩ = ⟨Aᵀy, x⟩ to 1e-14 for each (checks that sparse transpose is wired
  in the right orientation);
* the duality that the dual GG gradient was built for: Σ V u·(G p) + Σ V p (D u) = 0 to 1e-14 on
  the perturbed mesh, in torch. This is a structural check, not only a numerical one.

### U2 — Solution-dependent operators in torch (1 week)

Transcribe, as pure functions of torch tensors:

* `convection_vals(F, mask)` on the superset pattern, and the deferred central-minus-upwind right-hand
  side including the skewness correction and the boundary-outflow extrapolation;
* `momentum(comp, …)` → A values, aP, aC (SIMPLEC floor as a masked `max`);
* `pressure_laplacian_vals(gam)` — the orthogonal coefficient is linear in `gam`, so this is one
  constant sparse map applied to `gam`; the cross-term operators scale the same way;
* `rhie_chow(u, v, aP, p, gp, history)` including the fixed-flux boundary mask and the
  dt-independent term with its **previous-step** history (not previous-corrector — the record's
  bug (18));
* `pressure_flux(gam, pp, gpp)` including the non-orthogonal part;
* `BC.effective` as a masked select.

**Gate U2:**

* each function equals its numpy counterpart on a mid-run state of each test mesh: ≤ 1e-14;
* **measured linearity** where linearity is claimed (convection in F at fixed mask, pressure
  Laplacian in `gam`, Rhie–Chow in p at fixed D): the residual of f(αa + βb) − αf(a) − βf(b)
  ≤ 1e-14. The same "measure, don't assume" gate as 9.1a;
* FD vs autograd on each function at a state with no face within 1e-6 of a sign switch: ≤ 1e-7;
* cross-check: `probe_cell_operator` on the numpy convection at fixed F gives the same matrix as the
  torch superset assembly to 1e-15.

### U3 — `LinearSolve` for the unstructured solves (3–4 days)

Extend `LinearSolve` with a `factor` path. The forward solve factors with SuperLU (as
`SolveCache(backend="splu")` does) and stores the factor in `ctx`. The backward pass calls
`factor.solve(g, trans="T")`, a transpose solve at the cost of two triangular sweeps with no
refactorisation. Symmetric systems (the pressure operator, RK3 momentum) take the plain solve.
Singular all-Neumann systems keep the existing mean projection in both passes.

Memory note: a stored LU per solve per step is what makes a long tape expensive. That is the
reason U7 exists; within one step it is fine.

**Gate U3:**

* dL/db and dL/dA vs FD on the momentum, pressure and singular-pressure systems of each mesh:
  ≤ 1e-8;
* residual certificate on every forward **and** backward solve, ‖b − Ax‖ ≤ 1e-12‖b‖, raised as an
  error rather than logged — the FluidGym finding was unconverged backward solves returning
  silently;
* timing: backward ≤ 0.3× forward per solve on the fine butterfly (no refactorisation).

### U4 — The BDF2 PISO step in torch (1–2 weeks)

`TorchUPISOStep(piso)` built from a **configured** `PISO`, so boundary kinds, masks, `n_corr`,
`n_inner`, `n_nonorth`, `scheme`, `conv_flux_extrap` and the gradient choices are the production
ones by construction. It mirrors `PISO.step` line for line: extrapolated convecting flux
2Fⁿ − Fⁿ⁻¹, `n_inner` deferred momentum sweeps, `n_corr` correctors each with Rhie–Chow →
pressure Poisson with pinned non-orthogonal sweeps → velocity and flux correction → `p += pp`.

State carried across steps (all tensors):
`u, v, p, Ff, Ff_prev, Ff_old, Fbar_old, u_old, v_old`, plus the boundary-value vectors of
`bc_u, bc_v, bc_p`, which become the actuation inputs in U6. `n_outer > 1` (PIMPLE) is supported
but not required for the gates.

**Gate U4 — the decisive one**, as 9.2d was:

* field-for-field equivalence with `PISO.step()` on each mesh from a developed state:
  u, v, p, Ff ≤ 1e-10 after 1 step and ≤ 1e-9 after 10, with both sides at the same solve
  tolerance;
* the torch step contains no re-derived geometry: every constant operator comes from U1's
  conversion of the production matrix;
* 30-step rollout on the fine butterfly: drift from production grows at most linearly (the 9.4
  criterion: tolerance-seeded, not error);
* `max|div F|` on the torch side equals production's to 1e-14 each step.

### U5 — Gradient gates through the step (1 week)

With U4 passed, the gradient is taken of a step known to be the production step.

* FD vs adjoint through 1, 2, 3 steps for dL/d(state), dL/d(body force), dL/d(boundary values):
  ≤ 1e-6 relative on every mesh; mask-switch count reported per probe;
* **liveness mangles** — detach each path in turn and require the gradient to change by more than
  1e-6 relative: (a) convecting flux → momentum matrix; (b) aP → Rhie–Chow D; (c) aC → `gam` →
  pressure Laplacian; (d) the dt-independent history (Ff_old, Fbar_old); (e) the BDF2 history;
  (f) the deferred non-orthogonal corrections. A liveness probe has to *exercise* its path: put the
  source and the loss where that path matters (the Dong-outlet lesson of `differentiable_plumbing.md`);
* adjoint identity through one full step at the linearised level (a JVP/VJP pair):
  ⟨w, J v⟩ = ⟨Jᵀ w, v⟩ to 1e-10.

### U6 — Force objective and actuation (4–5 days)

* **Forces:** port the one-sided wall traction used by `run_ucylinder.py` and `naca_env.forces()` to
  torch (pressure owner value, wall-normal derivative u_P/d_n). Gate: equals the numpy value to
  1e-14; FD of C_D w.r.t. the state to 1e-8.
* **Actuation as boundary values:** `JetSet` (`src/ujets.py`) writes Dirichlet values on jet faces,
  and the gust modulates the inlet profile. In torch both become differentiable functions of the
  action writing into `bc_u.value`, `bc_v.value`, including `JetSet.ramp`'s tanh ramp. Rotary
  cylinder: tangential wall velocity ω × r.
* **Gate U6:** dC_D/da and dC_L/da FD-exact (≤ 1e-6) through 2 steps on the butterfly cylinder with
  the HydroGym jet layout and on the NACA α 40° mesh with the three leading-edge jets. Jet → wake
  transport: source at the jet, loss on a wake probe, FD-exact (the M2 j-gates).

### U7 — Memory-flat replay (3–4 days)

Port `prod_replay.py` with the U4 state keys. Forward under `no_grad` storing detached snapshots,
reverse sweep rebuilding one step's graph (and its LU factors) at a time.

**Gate U7:** replay == tape to 1e-9 on a 5-step window; measured peak memory flat in the window
length (report MB at H = 3, 10, 30); `replay_policy_grad` equals tape policy gradients to 1e-10 on
a 33k-parameter MLP.

### U8 — First control result on the unstructured adjoint (1–2 weeks)

Pick the case the unstructured solver owns and the structured one cannot run: **HydroGym's
NACA0012 α 40° gust task** (`naca_env.py`), where PPO scored −30.3 against −64.2 doing nothing
(record §56).

* DPC through the replay adjoint, same observation and reward as the PPO run;
* report against PPO with the same evaluation protocol, ≥ 3 seeds each;
* a horizon sweep, since horizon was the controlling variable in the FluidGym study.

This is a paper-grade result independent of the structured stack, and it tests the adjoint on a
mesh with triangles (the airfoil far field) as well as quads.

### U9 — RK3 step (1 week)

Mirror `PISO.step_rk3`: three stages, explicit convection on the stage flux, Crank–Nicolson
diffusion, per-stage Rhie–Chow **without** the dt-independent term, per-stage projection. The
stage momentum matrices `V/dt − βL` are symmetric and constant for fixed dt, so their factors are
built once and reused forward and backward. Gates U4–U5 repeated against `step_rk3`, plus the
inviscid energy-conservation property (T11) checked in torch.

### U10 — The 2.5D solver (3–4 weeks)

`PISO25` in torch:

* `torch.fft.rfft/irfft` along the span (differentiable), 3/2-rule padding and truncation,
  complex per-mode fields carried as real pairs;
* the per-mode families (A + s_k D) x_k = b_k: a custom autograd Function around
  `umodesolve.ModeFamily`'s block AMG-PCG. Forward and backward are both SPD block solves with the
  same hierarchy (RK3 momentum and pressure are symmetric), so the backward reuses the
  preconditioner. Residual-gated in both directions;
* WALE / Smagorinsky ν_t (`src/usgs.py`) differentiable, with an ε floor in the WALE denominator
  (a kink at zero strain otherwise);
* GPU: CuPy ↔ torch through DLPack, keeping the CuPy raw kernels (`src/ucuda.py`) in the forward
  and wrapping each with its transpose kernel for the backward.

Gates: mode 0 equals the 2D torch RK3 step to 1e-14; equivalence with `PISO25.step` ≤ 1e-9;
FD gates per mode family; periodic-span cylinder gradient dC_D/da equal to the 2D one when the
actuation is span-uniform. Replay memory and a one-step cost estimate at the Re_τ 180 channel size
(24×80×32).

### U11 — Cross-code check (3–4 days)

The fine butterfly exists on both solvers. Run the same control input on both and compare
dC_D/da from the structured Stage 9 adjoint and from the unstructured one. They should agree to
the discretisation difference between the two solvers, which the forward C_D comparison already
bounds (the T9 record). This is the check that the unstructured adjoint is the gradient of the
same physics, not just of its own code.

---

## 4. Estimated effort

| stage | effort | cumulative |
|---|---|---|
| U0–U3 | 2.5–3 weeks | 3 weeks |
| U4–U5 | 2–3 weeks | 6 weeks |
| U6–U7 | 1.5 weeks | 7.5 weeks |
| U8 (first control result) | 1–2 weeks | **≈ 9 weeks** |
| U9 | 1 week | 10 weeks |
| U10 | 3–4 weeks | 14 weeks |
| U11 | ½ week | ≈ 14.5 weeks |
| §7 physical tests beyond the stage gates (P3, P5, P7, P9 mostly) | 2 weeks, spread over U5–U10 | **≈ 16.5 weeks** |

Stage 9 on the structured solver took the path through probing, ghost padding and Dong; none of
that recurs, which is why U0–U5 is estimated shorter than 9.1–9.3 were.

## 5. Risks and how each is contained

| risk | containment |
|---|---|
| Upwind switches during a rollout make gradients jump | superset pattern + detached mask (U0); report switch counts; RK3's explicit central convection (U9) has no switch at all in the implicit matrix |
| Truncated deferred loops give a gradient of the wrong iteration | pin counts in both production and torch (U0); U4 equivalence then guarantees the torch graph is of the iteration actually run |
| LU factors on the tape blow memory | replay (U7) holds one step's factors; if one step is too large, refactor in the backward instead of storing |
| Kinks near a switch corrupt FD gates | masks held across ± probes; probes placed where no face is within δ of a switch; count reported |
| Triangle meshes carry the non-bipartite pressure mode (record §34, §55) | not an adjoint problem, but a *gradient of a mesh artefact* would look like a real sensitivity. U5 includes a gradient-of-checkerboard-content diagnostic on triangle meshes |
| Silent solver failure in the backward | U3's residual certificate raises, never logs |
| torch sparse CSR on CPU is slow for index-heavy assembly | use `index_add_` on face arrays for assembly and torch sparse only for constant operators; profile at U4 and move to CuPy kernels at U10 if needed |

## 6. Deliverables

* `src/uadj_*.py`, `test_uadj_*.py` (gates U1–U10, each runnable alone and in under 10 minutes on
  the coarse meshes);
* `reference/unstructured_adjoint.md`, the running record in the style of `production_adjoint.md`;
* an unstructured `hydrogym_pict` backend with a differentiable mode, and a DPC trainer on
  `naca_env.py`;
* for paper 1 (`reference/paper1_outline.md`): U8 supplies the "training on our own certified
  solver" result that framing B needs, on a task the published benchmarks' own backends do not
  differentiate.

---

## 7. Test cases and acceptance criteria

### 7.1 How the tests are organised

Every test belongs to one of two families, and most physical cases carry **one criterion of
each**:

* **Algorithmic (A)**: the adjoint is the exact derivative *of the discrete solver*. Criteria are
  tight (round-off, linear-solve tolerance or finite-difference floor) and are measured against the
  torch step's own finite differences or against production numpy.
* **Physical (P)**: the gradient means what the physics says it should. Criteria are measured
  against an analytic result, an exact symmetry, a conservation law, an independent reference
  solver or published data. They are **discretisation-level**: a P criterion passes at a stated
  resolution and must converge under refinement. It is never tightened to round-off.

A physical case that passes A but fails P shows the adjoint is right and the discretisation is
wrong for that physics, which the forward validation should already have found. A case that passes
P but fails A is a bug hidden by a coincidence. Both must pass.

**Finite-difference convention.** FD is central, with a relative step h = 1e-6 of the perturbed
quantity's scale. Its floor is about h² truncation plus ε/h round-off, roughly 1e-9 to 1e-10 in
float64. That is why FD gates sit at 1e-6 to 1e-8 and never at round-off. Every FD probe holds the
detached masks (upwind sign, `aC` floor) fixed across the ± pair and reports how many faces are
within 1e-6·max|F| of a sign switch. A probe with a non-zero count is re-run at a different state
rather than loosened.

**Linear-solve tolerance.** 1e-12 relative for every forward and backward solve in gate runs.
Production runs keep their own tolerance, and equivalence gates tighten production to match.

### 7.2 Meshes

| id | mesh | why it is in the matrix |
|---|---|---|
| M1 | uniform quads, `rect_mesh(nx, ny, cells="quad")` | orthogonal: every cross term is zero, the simplest correct answer |
| M2 | wall-clustered quads (`cluster_y`) | aspect ratio and grading; the channel mesh family |
| M3 | perturbed triangles (`perturb > 0`) | skewness and non-orthogonality: every deferred correction live; non-bipartite |
| M4 | HydroGym's cylinder triangles (T9) | a production triangle mesh with a body |
| M5 | fine butterfly quads (T9) | the production quad cylinder; mirror-symmetric; shared with the structured solver |
| M6 | NACA0012 α 40° mesh (§46) | hybrid mesh, jets, gust inflow: the U8 target |
| M7 | periodic channel, `make_periodic` | periodic seams (`dcc` carries the period shift) |
| M8 | pinball mesh | three bodies, several wall tags |

M1, M3 and M7 at coarse resolution form the **fast suite**: every A test on them runs in under 10
minutes on a laptop and runs on every commit.

### 7.3 Algorithmic test catalogue

| id | test | meshes | acceptance | stage |
|---|---|---|---|---|
| A1 | Constant operators, torch vs numpy, random fields | all | ≤ 1e-15 relative | U1 |
| A2 | Adjoint identity ⟨y, Ax⟩ = ⟨Aᵀy, x⟩ for every operator and every solution-dependent map at a fixed state | all | ≤ 1e-14 | U1–U2 |
| A3 | Discrete gradient–divergence duality, Σ V u·(G p) + Σ V p (D u) = 0, with the dual GG gradient | M1, M3, M5 | ≤ 1e-14 (fails at O(1) with the ordinary weights on M3: the test is sensitive) | U1 |
| A4 | Measured linearity where claimed (convection in F at fixed mask, pressure Laplacian in `gam`, Rhie–Chow in p at fixed D, `BC.effective`) | M1, M3, M4 | residual of f(αa+βb) − αf(a) − βf(b) ≤ 1e-14 | U2 |
| A5 | Each solution-dependent map, torch vs numpy, at a developed state | all | ≤ 1e-14 | U2 |
| A6 | `LinearSolve` dL/db and dL/dA vs FD: momentum (non-symmetric), pressure (symmetric), all-Neumann pressure (singular) | M1, M3, M5 | ≤ 1e-8 | U3 |
| A7 | Residual certificate on every forward and backward solve | all | ‖b − Ax‖ ≤ 1e-12‖b‖ or an exception; a logged warning is a failure | U3 on |
| A8 | **Step equivalence**, torch vs `PISO.step()` from a developed state | all | u, v, p, F ≤ 1e-10 after 1 step, ≤ 1e-9 after 10; max\|div F\| equal to 1e-14 each step | U4 |
| A9 | Rollout drift, 30 steps, torch vs production | M5, M6 | drift grows at most linearly (ratio at 30 vs 15 steps ≤ 2.2) | U4 |
| A10 | FD vs adjoint through 1, 2, 3 steps: dL/d(state), dL/d(body force), dL/d(boundary values), dL/dν | all | ≤ 1e-6 relative | U5 |
| A11 | **Liveness mangles**: detach each path in turn and require the gradient to change: convecting flux, aP → D, aC → gam, dt-independent history, BDF2 history, deferred non-orthogonal terms, deferred central convection | M3, M5 | change ≥ 1e-6 relative for each; source and loss placed where the path acts | U5 |
| A12 | JVP/VJP consistency of one step, ⟨w, J v⟩ = ⟨Jᵀw, v⟩, random v, w | M3, M5 | ≤ 1e-10 | U5 |
| A13 | Mask robustness in a recirculation bubble: gradient at a steady Re 40 state, where faces with F ≈ 0 exist | M5 | FD agreement ≤ 1e-6 on probes with zero switch count; switch count reported | U5 |
| A14 | Periodic seam: source on one side of the seam, loss on the other | M7 | FD ≤ 1e-6, gradient non-zero across the seam | U5 |
| A15 | Boundary-type coverage: gradient w.r.t. Dirichlet values on walls, inflow, symmetry (mixed u Neumann / v Dirichlet), jets; Neumann outlet | M4, M6 | FD ≤ 1e-6 per boundary type | U5–U6 |
| A16 | Force functional, torch vs numpy, and its FD | M4, M5, M6, M8 | ≤ 1e-14; FD ≤ 1e-8 | U6 |
| A17 | Replay == tape on the same window | M5, M6 | ≤ 1e-9 over 5 steps; policy gradients ≤ 1e-10 | U7 |
| A18 | Memory flat in horizon with replay | M5 | peak memory at H = 30 within 10% of H = 3 | U7 |
| A19 | Cost | M5 | tape backward ≤ 2.5× forward per step; replay ≤ 3.5× | U7 |
| A20 | Determinism: two identical gradient runs | M3 | bitwise equal on the CPU | U5 |
| A21 | RK3: A8–A12 repeated against `step_rk3`; stage factors reused, not rebuilt | M1, M3, M5 | as A8–A12 | U9 |
| A22 | 2.5D: mode 0 equals the 2D torch RK3 step; A8, A10, A12 against `PISO25.step` | M1×nz, M7×nz | mode 0 ≤ 1e-14; others as A8/A10/A12 | U10 |
| A23 | 2.5D block solve, forward and backward, on every mode | M2×nz | per-mode residual ≤ 1e-10 both directions; iterations reported | U10 |

### 7.4 Physical test catalogue

Each case lists the physical reason it is in the plan, its A criterion (against the discrete
solver) and its P criterion (against the physics).

**P1. Plane Poiseuille, flow-rate sensitivities (analytic).** Channel y ∈ [−1, 1], body force f,
laminar steady state. Q = ∫u dy = 2f / (3ν) per unit span, so ∂Q/∂f = 2/(3ν) and ∂Q/∂ν = −Q/ν.
*Why:* the simplest statement that the adjoint of the viscous operator and of the pressure–velocity
coupling carries the right physics. Walls, forcing and viscosity all enter.
* A: adjoint ∂Q/∂f and ∂Q/∂ν vs FD of the torch steady state ≤ 1e-8.
* P: vs the analytic values ≤ 1e-8 on M1 (second-order FV is exact for the parabola) and ≤ 1e-3
  on M3 and M2, converging at order ≥ 1.8.

**P2. Two-dimensional Taylor–Green decay (analytic time dependence).** On [0, 2π]² with
u = sin x cos y, v = −cos x sin y, the energy decays as E(t) = E₀ e^(−4νt), so
∂E(T)/∂ν = −4T E(T).
*Why:* the first test of the adjoint *in time*. It is an unsteady, periodic, pressure-coupled
flow whose exact gradient is known for all T. It exercises the BDF2 or RK3 history in the
reverse sweep.
* A: vs FD ≤ 1e-7, at T = 1, 5, 20.
* P: vs −4T E(T) ≤ 1% at 64² (M1 and M3), converging at order ≈ 2 under refinement at fixed dt/h.

**P3. Energy conservation in the reverse direction (inviscid Taylor–Green, RK3).** For an exactly
energy-conserving flow E(T) = E(0) for every initial state, so ∇_{u₀}E(T) = ∇_{u₀}E(0) = V u₀.
*Why:* this is the test of the **convection adjoint**, the hardest piece (upwind masks, deferred
central correction, skewness correction). A convection adjoint that is wrong in any of those shows
up as a mismatch here, and the mismatch is bounded by a measured physical quantity: the scheme's
own energy loss.
* A: vs FD ≤ 1e-7.
* P: ‖∇_{u₀}E(T) − V u₀‖ / ‖V u₀‖ ≤ 5 × the forward relative energy loss over [0, T]. The record
  gives that loss as 0.04 %/turnover for RK3 at dt 0.005 on 64² quads, and about 4 % for BDF2-PISO,
  so the test is informative for RK3 and loose for BDF2.

**P4. Orr–Sommerfeld growth-rate sensitivity (T8 case, spectral reference).** Plane Poiseuille at
Re 7500, α = 1. The least-stable mode grows at σ = α Im(c) = 0.00223497 (`orr_sommerfeld.py`). With
the body force f = 2ν the base flow is independent of ν, so for a window [T₁, T₂] inside the linear
phase, ∂/∂ν ln(E(T₂)/E(T₁)) = 2(T₂ − T₁) ∂σ/∂ν. The reference ∂σ/∂ν comes from the Chebyshev
eigen-solver by central difference in Re, and it is effectively exact.
*Why:* a growth rate of 2e-3 is a severe test of dissipation. Its sensitivity to ν is a severe test
of the viscous adjoint along a long horizon (T₂ ≈ 100), on a periodic seam (M7).
* A: vs FD ≤ 1e-6.
* P: vs the OS derivative ≤ 5 % at ny = 200 and converging with ny. The forward growth rate is
  itself within 0.2 % (RK3) on the recorded T8 runs, and a derivative is expected to be less
  accurate than the value.

**P5. Cylinder at Re 40, steady symmetric wake: exact symmetry of sensitivities.** On the
mirror-symmetric butterfly (M5) the steady base flow is symmetric about y = 0. Linear perturbation
theory then forces:
* ∂C_L/∂a_sym = 0 and ∂C_D/∂a_anti = 0, for symmetric and antisymmetric jet actuation at ±90°;
* ∂C_D/∂ω = 0 for cylinder rotation (the Magnus force is odd in ω, the drag even);
* ∂C_L/∂ω ≠ 0, with the sign of the Magnus force.

*Why:* these are exact consequences of the physics, independent of resolution, and they catch
symmetry-breaking bugs (a sign error on one side of the mesh, a wrong neighbour index at a seam)
that FD agreement alone does not catch.
* A: every sensitivity vs FD ≤ 1e-6.
* P: the "zero" sensitivities ≤ 1e-10 of the non-zero ones on M5. On M4, which is not exactly
  symmetric, the ratio is reported as a measure of the mesh asymmetry.

**P6. Cylinder at Re 40: sensitivity to Reynolds number and to steady suction (independent
reference).**
* ∂C_D/∂Re from the adjoint (through ν).
  * A: vs FD of the torch steady state ≤ 1e-6.
  * P: vs the slope of HydroGym's Firedrake C_D(Re) on the same domain at Re 39 and 41 (the §38
    set-up, runnable on the Spark) ≤ 2 %. Also vs the unstructured solver's own C_D(Re) slope
    ≤ 1e-4 (an A-type check on the ν path through a converged steady state).
* ∂C_D/∂a_sym for symmetric suction/blowing at the ±90° slots. P: the sign is positive (suction
  lowers drag, blowing raises it), consistent with the constant-actuation references of §43, and
  the magnitude equals the linear fit of the solver's own C_D(a) at |a| ≤ 0.01 to 1 %.

**P7. Global stability of the cylinder wake: direct and adjoint eigenmodes.** At a Reynolds number
above onset (Re 60, on the base flow computed by the mirror-averaged steady run that
`run_ucylinder.py --steady` already provides), the linearised step's leading eigenvalue μ gives
the growth rate σ = ln|μ|/dt and the frequency. Direct modes come from Arnoldi on the linearised
forward step: a JVP rule for `LinearSolve` (solve A dx = db − dA x), or FD matvecs as in the
time-stepper method. Adjoint modes come from Arnoldi on the VJP.
*Why:* this is where the adjoint has a physical meaning that the literature has mapped. The
adjoint eigenmode is the receptivity of the wake, and the product |û||û†| is the structural
sensitivity, the "wavemaker" (Giannetti & Luchini 2007 [U]). It is also a direct test that the
VJP is the transpose of the JVP over the long-time dynamics, not just over one step.
* A: the adjoint eigenvalue equals the complex conjugate of the direct one to 1e-8 relative
  (the same operator's spectrum). ⟨û†, û⟩ ≠ 0.
* P:
  * σ and the frequency match the linear phase of a forward kick run from the same base flow to
    2 % and 1 %;
  * the direct mode's energy centroid lies downstream of the adjoint mode's (the non-normality of
    the convective wake: direct modes peak in the far wake, adjoint modes near the body);
  * the structural sensitivity peaks inside the recirculation bubble, as two lobes symmetric about
    the axis to 1e-6 on M5.
* Optional: the growth rate crosses zero at the solver's own measured onset Re to 2 %. Onset
  depends on blockage, so compare with the solver's forward onset estimate, not with the unconfined
  literature value of about 47.

**P8. Periodic shedding at Re 100: time-shift invariance on the limit cycle.** On a periodic orbit,
shifting the initial state along the flow advances the solution in time. Take δx = x₁ − x₀, the
full state difference over one step, including every history array. Then ⟨∇_{x₀} C_L(t_N), δx⟩
must equal C_L(t_{N+1}) − C_L(t_N) up to O(|δx|²).
*Why:* it tests the adjoint against the dynamics' own symmetry (autonomy) along a long unsteady
horizon, which is the regime every control run lives in, without any external reference. A second
limit-cycle property is checked at the same time: the neutral phase mode means ∇_{x₀}J(T) must
neither decay nor grow exponentially.
* A: ∇_{x₀}C_L vs FD in random directions ≤ 1e-6 over 1 period.
* P:
  * the time-shift identity holds to ≤ 1 % at production dt, and the error halves when dt halves
    (first order in |δx|);
  * over 10 shedding periods, ‖∇_{x₀}C_L(T)‖ grows at most linearly (fitted exponent in T within
    [0, 1.2]);
  * St from the forward run within 0.5 % of the T9 value, so the orbit is the validated one.

**P9. Jet and rotary control sensitivities on the validated shedding case, and the cross-code
check (U11).** Same geometry, Re 100, the same actuator on both solvers (M5, butterfly quads).
*Why:* the gradient a control study actually uses, on the flow both solvers were validated on.
* A: dC_D/da and dC_L/dω vs FD ≤ 1e-6 through 2 steps and 1 control interval.
* P:
  * the unstructured adjoint's dC_D/da, dC_L/da and dC_L/dω agree with the structured Stage 9
    adjoint to within the two solvers' forward difference in C_D and C_L at that state (the T9
    record: about 1 %);
  * the rotary Magnus sensitivity has the sign measured on the structured backend (hg.4:
    ΔC_L = −0.3532 for ω = +1) and is antisymmetric in ω to 1e-6.

**P10. NACA0012 α 40° gust (U8 target): physically sensible control gradients.**
* A: dC_L/da_j for each of the three jets vs FD ≤ 1e-6.
* P: at the gust peak, the sign of dC_L/da for the upper-surface jet agrees with the direction the
  trained PPO policy used (suction on the upper jet to hold C_L, record §56). The DPC policy trained
  through the adjoint reaches a return at least as good as PPO's −30.3 within the same evaluation
  protocol, over ≥ 3 seeds.

**P11. 2.5D: symmetry of spanwise modes.** At a span-uniform (2D) state, any objective built from
mode-0 quantities (C_D, C_L, the mean profile) has **zero first-order sensitivity** to every
spanwise mode k ≠ 0. The modes couple to mode 0 only quadratically.
*Why:* it is exact and resolution-independent, and it catches mode-mixing errors in the FFT,
dealiasing and per-mode solve adjoints.
* A: per-mode FD ≤ 1e-6.
* P: ‖∂J/∂(mode k ≠ 0)‖ ≤ 1e-12 × ‖∂J/∂(mode 0)‖. With span-uniform actuation the periodic-span
  cylinder's dC_D/da equals the 2D value to 1e-10 (the G2 cylinder test, in reverse).

**P12. Turbulent channel Re_τ 180: where adjoint gradients stop being meaningful
(characterisation, not a pass/fail on the physics).** For a chaotic flow, the sensitivity of a
long-time average cannot be obtained by a plain adjoint: the adjoint grows at the leading Lyapunov
exponent. The plan does not attempt shadowing methods. It measures the limit so that nobody uses
the tool past it.
* A: FD ≤ 1e-5 for horizons shorter than one Lyapunov time.
* P, reported rather than gated: the growth rate λ of ‖∇_{x₀}J(T)‖ (positive, exponential); the
  horizon at which the gradient of a window-averaged wall stress loses agreement with FD; the ratio
  λ / (u_τ²/ν). The laminar channel at the same box (P1) must show no growth, which confirms that
  the growth is the physics and not the adjoint.

### 7.5 Acceptance criteria for the whole plan

The adjoint is **accepted for control studies on 2D unstructured meshes** when all of these hold:

1. A1–A20 pass on every mesh listed for them, and the fast suite runs clean on every commit.
2. P1, P2, P4, P5, P6, P8 and P9 pass both their A and P criteria. P3 passes for RK3 once U9 lands.
3. P7 passes its A criterion and the σ/frequency part of its P criterion. The wavemaker shape is
   reported with figures, compared qualitatively with the literature, and marked [U] until that
   comparison has been made against the source.
4. U8 delivers its control result (P10) with ≥ 3 seeds and the evaluation protocol written down
   before the runs.
5. The record `reference/unstructured_adjoint.md` states every measured number of 1–4, with the
   command that reproduces it.

It is **accepted for 2.5D/LES use** when, in addition, A21–A23 and P3, P11 pass, and P12's
characterisation is published in the record with the horizon limit stated beside every 2.5D
gradient result.

**Standing rule.** A test that fails is reported as failed, with its number, and is not re-posed.
If a criterion turns out to be mis-posed (as G1's T9 amplitude criterion was, record §50), the
correction and the reason go into the record *before* the test is re-run.
