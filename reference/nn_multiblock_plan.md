# Connecting the network to the MULTI-BLOCK solver — plan and gates

Continuation of [`nn_piso_plan.md`](nn_piso_plan.md), which took the differentiable path from a
linear-solve adjoint to a-posteriori closure training. Everything there runs on **one block,
16³, periodic, Cartesian**. Everything the port actually simulates — backward-facing step, square
cylinder, circular cylinder — runs on `MultiBlockPISO`, which has **no gradients at all**.

This plan connects them. Same governing principle as before: *a wrong gradient still trains,
just to the wrong place*, so every gate is a gradient check, not a falling loss curve.

---

## Where we actually stand

Audited by reading imports, not by reading intentions.

| | differentiable path | production path |
|---|---|---|
| solver | `src/piso_torch.DifferentiablePISO`, `src/adjoint_piso.py` | `src/piso_multiblock.MultiBlockPISO` |
| blocks | 1 | 8-16, with connections |
| grid | 16³ Cartesian, `warp=1e-9`, periodic | curvilinear, stretched, wall-bounded |
| cells | 4,096 | 82,096 (square) / 89,088 (cylinder) |
| backend | torch + SciPy | NumPy + SciPy or AmgX |
| autograd | yes | **none** |
| used by | `nn_stage1..5b`, `make_sgs_*` | `run_*`, every physics result in the repo |

No file under `nn_*` imports anything multi-block, and "multiblock" appears zero times in the
NN reference documents. This is not blocked work; it is unsequenced work.

## The one structural fact that makes it tractable

`MultiBlockPISO` assembles **one global sparse matrix over all blocks** — a seam contributes
off-diagonal entries exactly as an interior face does (`build_diffusion_matrix`,
`build_momentum_matrix`). `LinearSolve` does not care where a row came from.

**So the adjoint needs no per-block gradient exchange.** The block coupling is already inside
$A$, and $A^{\mathsf T}\lambda = \bar{\mathbf g}$ carries sensitivity across a seam by the same
entries that carry flux across it forwards. That is the single largest risk in this kind of work
and it is already retired by a design decision taken for other reasons.

What is *not* retired: the state PISO carries between steps, the scale, and the walls.

---

## Stage 6 — two blocks that are one block

The seam gate. Take a periodic 16³ case, split it into two blocks joined by a connection, and
require the gradient to be **identical** to the unsplit one.

**Build.** The minimum differentiable multi-block path for a SINGLE step: assemble the global
momentum and pressure matrices from `MultiBlockPISO`, hand them to the existing `LinearSolve`,
and do the cheap algebra (interpolation, gradients, corrector updates) in torch. No new adjoint
mathematics — this stage is entirely about indexing and seams.

**Gate.**

| check | target |
|---|---|
| forward: 2-block state vs 1-block state | machine precision, as `test_multiblock` already requires |
| $\partial L/\partial S$, 2-block vs 1-block | ≤ 1e-12 relative |
| adjoint identity on the global **momentum** matrix | ≤ 1e-10, and the missing-transpose control must fail loudly |
| FD vs adjoint through one step, tol 1e-12 | ~7 digits, matching Stage 1 |

**Why this ordering.** `test_multiblock.py` validated the forward this way — connect a block to
itself and demand the single-block answer exactly. The same trick isolates seam handling in the
backward from every other difference, because the two configurations are the same physics.

**Stop condition.** If the split gradient differs from the unsplit one by more than solver
tolerance, stop and fix indexing. Do not proceed with a tolerance widened to accommodate it.

---

## Stage 7 — the state PISO carries between steps

`piso_torch` differentiates one step of a stateless configuration. `MultiBlockPISO` in production
carries **three** things across a step, all of which the adjoint must chain through:

* `u_prev` — BDF2 history,
* `p_flux` — the projection pressure the face flux actually carries,
* `F_prev` — the persistent Rhie–Chow face flux.

`ddt_corr` is off in every production case and stays off here; its unit-gain recurrence is a
forward instability (`rhie_chow_ddt_instability.md`) and would be an adjoint one too.

**Gate.**

| check | target |
|---|---|
| FD vs adjoint over 3 steps, `rhie_chow=True, persistent_flux=True` | ~6 digits |
| dropping `F_prev` from the backward must be **detected** | gradient error > 1e-12 |
| adjoint norm $\lVert\lambda\rVert$ per step | bounded over 20 steps |

The second row mirrors `test_checkpoint.py`, which mangles the restart state and requires the
error to show. A backward pass that quietly ignores persistent state produces a plausible
gradient, which is the failure mode this whole document exists to prevent.

---

## Stage 8 — an objective that lives on a wall

Every loss in Stages 1–5 is a field difference on a periodic box. The reason to want gradients
on the multi-block solver is that the interesting objectives live on solid surfaces: drag, lift,
separation.

`src/forces.py` already computes $C_D$ and $C_L$ by integrating traction over wall faces, and it
is pure array arithmetic over a face — a torch version is mechanical, not novel.

**Build.** `surface_force` in torch, sharing the metric and area-vector code.

**Gate.**

| check | target |
|---|---|
| the four analytic checks in `test_forces.py`, re-run through the torch path | same tolerances |
| $\partial C_D/\partial S$ adjoint vs central FD | ~6 digits |
| $\partial C_D/\partial S$ with the viscous normal stress dropped | must differ — it is 1.7% of $C_D$ |

**A caveat this stage inherits.** The measured viscous normal stress is discretisation error of
the same size as the genuine friction drag (`src/forces.py`). A network trained to reduce $C_D$
can reduce that error instead of the drag. The loss should use the tangential traction, or the
normal part must be shown not to move under training.

---

## Stage 9 — scale

16³ is 4,096 cells. The square cylinder is 82,096, in eight blocks, with 38,000 steps to
saturation.

**Build.** Checkpointed rollout on the multi-block state — Stage 3's `rollout.py` generalised to
a state that is a dict of per-block arrays plus fluxes.

**Gate.**

| check | target |
|---|---|
| peak memory, checkpointed vs not, 50 steps | measured, with the ratio reported |
| checkpointed vs non-checkpointed gradient | exactly 0 difference, as Stage 3 achieved |
| wall-clock per gradient step | measured against forward-only |

**Budget it before building it.** One stored state is 4 fields × 89,088 nodes × 8 bytes ≈ 2.8 MB,
plus fluxes. A 500-step window checkpointed every 25 is ~56 MB of states and 25 steps of
recompute per segment — fine. The same window unchecked is not.

---

## Stage 10 — an actual learning task

Deliberately last, and not specified here beyond the two candidates:

* **SGS closure on a multi-block LES case**, continuing Stage 5's line of work into a geometry
  that is not a periodic box.
* **Drag reduction / flow control** — a boundary or body-force actuation trained against $C_D$
  from Stage 8.

Choosing between them is a question for when Stages 6–9 have passed, not now.

---

## Decisions to take before Stage 6 starts

**D1. Torch port of the step, or one hand-written adjoint for the whole step?**
Recommendation: **port the step**, keep `LinearSolve` for the two solves. A hand-written whole-step
adjoint duplicates the PISO algebra in a second place, and the two copies will drift the first
time the corrector changes. The linear solves are the only part where the adjoint shortcut is
needed, and that part already exists.

**D2. Which backend in the backward?**
SciPy, always, to begin with. AmgX has no adjoint path, so a GPU forward would pair with a CPU
backward; measure that mismatch before deciding whether it matters.

**D3. Which case first?**
The split periodic box (Stage 6), then a wall-bounded case. `nn_piso_plan.md` warns to prefer
periodic training cases because wall-bounded ones are first-order in space — and every
multi-block case in the repo is wall-bounded. That tension is real and is the reason Stage 8
comes with its own gate rather than being folded into Stage 6.

---

## Standing risks

* **Solver tolerance caps gradient accuracy**, unchanged from the single-block plan: tighten to
  1e-12 for FD checks. Production runs at 1e-6 and that is the gradient's ceiling too.
* **The adjoint of an advection-dominated flow transports sensitivity upstream.** At Re = 100
  with a shedding wake this is a real amplification risk over long windows. Log
  $\lVert\lambda\rVert$; shorten the window rather than clipping.
* **Dong outflow makes the pressure system non-singular**, while the periodic training cases are
  singular. `LinearSolve`'s `singular` flag must follow the case, and the compatibility
  projection must be applied in the backward exactly where it is applied in the forward.
* **Every production multi-block case is wall-bounded and first-order at the wall.** Closure
  error and boundary error are confounded until the half-cell boundary-flux stencil is upgraded.
* **Nothing in Stages 6-9 produces a physics result.** They are instruments. The temptation to
  skip to Stage 10 because the gates are "just tests" is exactly what Stage 0's gate was written
  to resist.

---

## Order of work

```
Stage 6   two blocks that are one block   <- proves seams in the backward
Stage 7   persistent state                <- proves the PISO state chain
Stage 8   force objective in torch        <- proves a wall-bounded loss
Stage 9   scale + checkpointing           <- proves it is affordable
Stage 10  a learning task                 <- the first thing that is not an instrument
```

Stages 6 and 7 should run in seconds on 16³ split blocks; they are debugging instruments and
must stay cheap enough to run on every change. Stage 9 is the first that needs the GB10.
