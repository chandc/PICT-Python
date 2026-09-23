# Replicating HydroGym's JET cylinder with our solver and discrete adjoint

**Goal.** HydroGym's `Cylinder` -- the blowing/suction case, not `RotaryCylinder` -- run on
PICT-Python with gradient-based control through our discrete adjoint, on Spark.

**Why this plan exists.** `hydrogym_pict` implements `RotaryCylinder`. `prod_dpc_train.py` uses
jets but NOT HydroGym's. The two are disjoint, and neither is the benchmark.

## The target, pinned from source (dynamicslab/hydrogym, `firedrake/envs/cylinder/flow.py`)

| quantity | HydroGym `Cylinder` |
|---|---|
| Reynolds | 100 |
| RADIUS | 0.5 (D = 1) |
| DEFAULT_DT | 1e-2 |
| MAX_CONTROL | **0.1** |
| TAU | 0.0556 (0.01 x shedding period) |
| num_inputs | **1** -- ONE scalar drives both jets |
| observations | `lift_drag`, num_outputs 2 -> (C_L, C_D) |
| jet centres | theta = +90 and -90 deg |
| jet width | omega = pi/18 -> **10 deg TOTAL** (+-5 deg) |
| profile | `pi/(2*omega*R^2) * cos((pi/omega)(theta - theta_jet))` |
| direction | `(x, y) * (A_up + A_lo)` -- radial, **both OUTWARD** |

## Two properties of the target that must be replicated, not corrected

1. **The actuation is SYMMETRIC.** Both jets blow, or both suck, together. It therefore has
   almost no direct lift authority -- unlike a pair of opposing jets. Our current
   `jet_geometry` is explicitly opposing ("suction when the top blows"), which is a DIFFERENT
   control problem.
2. **It is NOT zero-net-mass-flux**, despite citing Rabault et al. (2018), whose jets are
   opposed and ZNMF. `A_up + A_lo` with a radial `(x, y)` is outward at both stations, so a
   positive control injects net mass. **Replicate it as written and record the discrepancy** --
   silently "fixing" it would mean benchmarking against something HydroGym does not run. Raise
   it upstream rather than diverging locally.

## What our current jets get wrong

| | HydroGym | ours (`test_mb_adjoint_jet.jet_geometry`) |
|---|---|---|
| width | 10 deg total | **20 deg** (`half_deg=10`) |
| profile | cosine | parabola `1 - (dtheta/half)^2` |
| symmetry | both outward | **opposing (+a / -a)** |
| bound | MAX_CONTROL 0.1 | unbounded in `prod_dpc_train` |

**This explains the current DPC result.** The policy found -1.9% drag by driving |C_L| to
0.48-0.51 against a 0.26 baseline -- steering the wake with the asymmetry. That lever does not
exist under HydroGym's symmetric jets, so the present training curve does not transfer.
`fluidgym_parity.md` already names the failure mode: "the optimiser cheats: strong asymmetric
blowing trims drag".

## Observations: the default is TWO SCALARS, not a sensor array

`configure_observations` defaults to `obs_type="lift_drag"`, `num_outputs=2` -> **(C_L, C_D)**.
Probes exist only as OPTIONAL `probe_obs_types` you opt into.

| | HydroGym | ours (`prod_dpc_train`) |
|---|---|---|
| default observation | **(C_L, C_D)** -- 2 numbers | **151 positions x 3 fields = 453 readings** |
| velocity probes (optional) | 64: x = linspace(1,10,16) x y = linspace(-2,2,4) | 56 wake pts, x in [1,5) step .5, y in [-1.5,1.75) step .5 |
| pressure probes (optional) | **20, ON the cylinder surface**, theta = 0..342 deg uniform | 72 on rings at r = 1.0 and 0.625 -- **1.25 and 2 D off the body** -- plus 23 scattered |

Three consequences:

1. The canonical benchmark policy reads **two scalars**. `hydrogym_pict` already implements this
   correctly for the rotary case. A 453-input policy is reading near-full-field state, which is a
   materially EASIER control problem and is not the benchmark.
2. HydroGym's pressure probes sit ON the body -- the physically realisable sensor. Ours are out
   in the flow, and its velocity probes reach x = 10 D where ours stop at 5 D.
3. **This compounds the jet error.** The current -1.9% was won with an antisymmetric actuator AND
   near-full-field observations: one gives the policy a lift lever it should not have, the other
   gives it information it should not have. Both must go for the number to mean anything.

## The mesh divergence, stated once

HydroGym ships a Firedrake "medium" mesh; we use the R11-validated butterfly
(10 D upstream, 30 D downstream, +-10 D lateral, beta = 0.05; C_D 1.3167, St 0.1656 on "mid").
`hydrogym_backend.md` already took this decision for the rotary case: mirror the CONVENTIONS,
keep our validated grid. Consequence: **absolute C_D is not comparable to published HydroGym
numbers; percentage change under control is.** Every result must be quoted as a delta against
our own uncontrolled baseline on the same mesh.

---

# Gates

### G0 -- Pin the contract
Vendor `cylinder/flow.py` alongside `_vendored_core.py` with the fetch date, and write the
constants table above into a test as literals so an upstream change breaks a gate rather than
drifting silently.
**Success:** every constant asserted against the vendored source.

### G1 -- The jet-resolved grid  [BUILT, needs re-validation]

HydroGym's jet is 10 deg wide. Our grids put **3** points across it on "production" and **2** on
the DPC arena -- and worse, the butterfly's tangential clustering is at the EAST corner, so
theta = +-90 (where the jets live) is the COARSEST part of the ring. Two or three points cannot
represent a cosine; control authority would be a discretisation artefact.

Built: `side_dt = 0.008, nz = 2` -> **130,592 cells, 473 azimuthal points, 11 across the jet**
(spacing 0.86-0.91 deg at +-90). nz = 2 because HydroGym's cylinder is 2D and that halves the
cost against nz = 4 for nothing lost. Figure: `figures/hydrogym_jet_grid.png`, whose right panel
shows the 11 points tracing the analytic cosine including both zero crossings.

**Success:** >= 8 points inside the 10-deg jet AND the uncontrolled baseline back in the campaign
band (C_D 1.32-1.34, St 0.167-0.170).
**This gate has a real pass/fail, not just a number.** Refining tangentially changed cell aspect
ratios around the WHOLE ring, not only in the jets, so the shedding may have moved. If C_D comes
back at 1.28 or St at 0.175, the grid altered the physics rather than the actuator and must be
redone -- and any timing measured on it was measuring the wrong mesh.

**Known bluntness, deliberate:** this refines all four ring sides to resolve two 10-deg patches.
A midpoint-clustered `t_N`/`t_S` would reach 11 points at ~90-100k cells instead of 130k, at the
cost of new code in `ring_rect_domain`. Taken as a parameter change first.

### G1b -- Observation configuration

**Primary = HydroGym's default: observation is (C_L, C_D).** That is the benchmark and what SAC
is compared against. **Secondary, optional: HydroGym's 20 surface pressure probes**, since a
2-scalar observation may simply be too impoverished for DPC to learn anything and it is worth
knowing WHICH of the two failed. The 473-point surface discretisation from G1 can host the 20
probes at theta = 0, 18, ... 342 deg by interpolation -- a small addition, not a new mesh.
**Retire the 453-probe array for this benchmark**; keep it, clearly labelled, for the separate
differentiable-control study.

### G2 -- The jet actuation
`JetCylinder(CylinderBase)` in `hydrogym_pict/flow.py`, reusing the rotary plumbing already
gated 6/6 (actuator lag, BC write, exact reset, reward).

**Success criteria, and the middle one is the point:**
- discrete jet profile integrates to the analytic `int A dtheta` within 1%;
- **dC_L/du ~ 0 while dC_D/du != 0** -- the symmetry test. This is the mirror of hg.4's Magnus
  antisymmetry check and is precisely what would catch an accidentally-opposing implementation,
  i.e. the bug our existing jets have. A replication that reproduces drag authority but also
  produces lift authority has NOT replicated this case;
- actuator lag follows `1 - exp(-dt/tau)`, reset exact, as hg.3/hg.5.
**Abort:** if the symmetry test cannot be met on the butterfly body discretisation, the jet
cannot be represented on this grid and the mesh question reopens.

### G3 -- Forward validation
Uncontrolled shedding must reproduce our own validated butterfly numbers (C_D 1.3167,
St 0.1656 on "mid"); open-loop steady blowing and suction at +-MAX_CONTROL give a dC_D curve.
**Success:** uncontrolled within the campaign band; open-loop response monotone and
sign-consistent between blowing and suction.

### G4 -- The discrete adjoint through the jet control
The jet is a scaled Dirichlet BC on the body wall, so dJ/du flows through the BC. `implicit_cross`
is required physics on the butterfly, so the gradient must go through `MultiBlockDCChain`
(Stage 7b, 6/6) and not only the orthogonal chain.
**Success:** FD vs adjoint on the scalar control, normalised to max|g|, no worse than the
existing 1e-8; the detached-jet mangle DETECTED; gradient non-zero at a symmetric state (where
the lift path is degenerate, the drag path must still carry signal).
**Abort:** FD-vs-adjoint worse than 10x the serial norm.

### G5 -- The backend decision on Spark, MEASURED
**Do not assume the GPU pays.** The cylinder solver shootout already found: warm-started in-run
solves leave the cylinder at **2.71 s/step either way, GPU 14%, and the floor is 16-block CPU
assembly**.

**CORRECTION to an earlier draft of this plan.** It said a GPU forward "necessarily" pairs with a
CPU adjoint. That overstated it. The adjoint backward is scipy-only TODAY -- `adjoint_piso._solve`
takes scipy.sparse and numpy and does `op = A.T if transpose else A` -- but nothing makes it so:

* the 4 PRESSURE adjoint solves per step are symmetric, so **A^T = A**: they could reuse the
  hierarchy AmgX has already built for the forward, with no transpose and no new capability;
* the 12 MOMENTUM adjoint solves need A^T, which means a SECOND `AmgXSolver` instance holding
  `A.T` (the binding creates one matrix handle per instance, with `drift_tol` governing
  rebuilds), doubling GPU matrix and preconditioner memory -- **and those are exactly the solves
  where AmgX has no upside**: the shootout found AMG DIVERGES on the momentum operator (20k
  iterations) while Jacobi converges in 1 iteration at 0.014 s.

So 75% of the backward needs the expensive machinery and would gain nothing, while the 25% that
would gain needs no new machinery at all. Tolerance is a further wrinkle: it is baked into the
AmgX JSON at solver creation (`_config_with_tolerance`), and the adjoint runs at
`PICT_ADJ_TOL = 1e-11` against the forward's 1e-6.

**The measurement that settles it costs nothing to take:** profile one training iteration and
report the split between pressure-adjoint solves, momentum-adjoint solves, and assembly. If the
pressure-adjoint share is small, the GPU question is closed for this arena without writing a line
of binding code.

Measure one full training iteration (forward rollout + adjoint) three ways:
1. AmgX on the GB10 (pressure: aggregation AMG; momentum: Jacobi -- the shootout's per-SYSTEM
   configs, not one config per process);
2. PETSc MPI at 8 ranks on the 20 CPU cores, with the Mat/KSP cache and jacobi defaults;
3. scipy serial, as the control.
**Success:** a backend chosen on measured wall-clock per training iteration, with the adjoint
handoff included. **This gate may legitimately conclude the GPU is not the right device for a
30k-cell arena** -- that is a result, not a failure, and it is cheaper to find here than after
a training campaign.

### G5b -- The training environment is READY, and the interpreter is not the default

Verified, not assumed (`reference/amgx_momentum_fix.md`): AmgX and torch-CUDA coexist in one
process on the GB10 -- 40 AmgX solves between two successful torch GPU matmuls, with
`REJECT`/`unhealthy`/`RECONSTRUCT`/`FULL RESET` all zero.

**Run G6 under `/usr/bin/python3`, not `python3`.** The PATH interpreter in `pict-amgx:1.0`
(`/opt/cpn/bin/python3`) has numpy and scipy but NO torch; `/usr/bin/python3` has
torch 2.10 + numpy 2.1 + scipy 1.16 + CUDA. The PATH default fails as an ImportError seconds
into a job rather than at submission.

`mpi4py` is absent from that interpreter and this does not matter: the discrete adjoint has NO
MPI path (Gate 7 of the DD plan, unstarted), so G6 is serial whatever we do. That is also why
AmgX matters here far more than it did for the baseline -- serially it is **4.7x** over scipy
(2.400 against 11.285 s/step), not the 3% it was worth when the baseline could use 8 ranks.

### G6 -- Training, and the comparison that is the actual benchmark
DPC through the discrete adjoint against SAC on the IDENTICAL env (same `FlowEnv`, same reward
`-dt*C_D`, same MAX_CONTROL, same mesh). That is the comparison HydroGym exists to support.
**Success:** both learners run to a stated iteration budget with per-iteration checkpoints and
a recorded config; drag change quoted as a delta against our own uncontrolled baseline with the
baseline's own scatter band shown.
**Expectation to state in advance:** with the lift lever removed, drag authority will be
SMALLER than the current -1.9%. A smaller honest number on the right problem is the deliverable.

### G7 -- Report
Table of C_D, C_L, St uncontrolled and controlled; the open-loop response curve; the learning
curves; and an explicit statement of the mesh divergence and the ZNMF discrepancy.

---

## Risks, ranked

| risk | cost | mitigation |
|---|---|---|
| GPU does not pay at 30k cells | a campaign on the wrong device | G5 measures it before G6 |
| symmetric jets have little drag authority | a null result | it is the correct problem; report the null |
| 2-scalar observation too impoverished to learn | DPC stalls | G1b's optional 20 surface probes separate actuator failure from sensor failure |
| tangential refinement moved the shedding | a day on the wrong mesh | G1's re-validation is pass/fail against the campaign band |
| absolute C_D not comparable to HydroGym | mis-stated claims | quote deltas only; G0 records it |
| HydroGym's non-ZNMF jets | replicating an upstream quirk | replicate as written, record, raise upstream |
| adjoint through a scaled Dirichlet BC untested at this config | G3 slips | `test_mb_adjoint_jet.py` already exercises the path |

## Sequencing

G0 and G1b are hours. G1's grid is BUILT and needs only its re-validation run. G2 reuses
plumbing already gated 6/6. G4 (the adjoint through a scaled Dirichlet BC) is the technical
risk. G5 is one afternoon and gates the whole GPU premise. G6 is the long pole and must not
start until G5 has named the device and G3 has a baseline WITH ITS SCATTER BAND -- a control
result quoted against a baseline whose own spread is unknown says nothing.

## Status 2026-09-23: HydroGym's own jet environment, trained with its own stack

Both variants trained on the Spark with HydroGym's Firedrake solver and SB3 PPO (their script and
defaults, 100k steps, (C_L, C_D) observations, 5000-step episodes from their published checkpoint).
Full account in `skew_unstructured_literature.md` sections 43-44; archives in `results/hydrogym_rl/`.

| | shipped jets (symmetric, net flux) | ZNMF opposed jets (`hg_znmf.py`) |
|---|---|---|
| C_D uncontrolled -> controlled | 1.486 -> 1.032 (-30.6%) | 1.486 -> 1.487 (0.0%) |
| C_L rms | 0.25 -> 0.002 | 0.25 -> 11.9 (slot pressure from reversing jets) |
| learned action | constant -0.1 (suction bound) | +-0.1 alternating every step, mean 0 |
| episodes to converge | 2 | never improved |
| wake | no vortex; base flow held by suction | natural Karman street |

Consequences for this plan: (1) the shipped `Cylinder` reward is solved by steady maximal suction,
so a drag-reduction number on it is not a wake-control result and the ">20%" in their docs is that
solution; (2) the physically constrained (ZNMF) jets need Rabault's ingredients to be learnable --
probes (`--obs-type velocity_probes`), an action hold (`--num-substeps 50`) and ~10x the budget --
and the constant references (steady asymmetric deflection, 5-11%) are the floor a policy must beat;
(3) for our own adjoint/DPC comparison against HydroGym, the ZNMF variant with an actuation cost is
the target, and the reproduction of section 2 above stands (jets as written, discrepancy recorded).
