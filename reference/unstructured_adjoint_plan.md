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
up front so a gate cannot be retuned after the fact.

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
