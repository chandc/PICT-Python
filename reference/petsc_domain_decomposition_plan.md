# Domain decomposition with PETSc — implementation plan

Status: **PROPOSED, NOT STARTED.** No code written. Awaiting approval.

Written 2026-09-03. Every number below is measured on this repo at this commit, not estimated;
where a figure is an order-of-magnitude guess it says so.

---

## 1. Why, and the honest size of the prize

Measured on the cylinder case actually running (160,768 cells, `dt = 0.005`, scipy backend):

```
  one step                    4.767 s
    inside bicgstab           3.662 s   76.8%
    everything else           1.105 s   23.2%   assembly, BCs, Rhie-Chow, seams
  4 GLOBAL linear systems per step, 160,640 rows each
  one core of sixteen in use; the solver is single-threaded by construction
```

Two routes were measured before proposing this one:

| route | speedup | why not |
|---|---|---|
| threaded sparse matvec (torch CSR, 8 threads) | 2.76× on the matvec, **~1.96× overall** | Amdahl ceiling of 4.31× from the serial 23% |
| AmgX on the GB10 GPU | **4.38×**, already integrated | needs the GPU box; also latency-bound at this size |
| **full decomposition** | **removes the 23% floor**, because assembly is already per block | this plan |

The case for decomposition is *not* that it is the fastest route to a faster cylinder run — the
GPU already exists and is faster today. It is that decomposition is the only route that scales
past one device, and the NN closure work will need 3-D turbulent channel and duct resolutions
where a single node is not enough. **If that requirement disappears, so does the case for this
plan.**

---

## 2. What makes this repo unusually ready, and what does not

### Favourable, and decisive

**Block ownership is already row ownership.** `MultiBlock.offsets = cumsum(block sizes)`, so
every block occupies one contiguous range of global matrix rows:

```
  block  0: rows       0 to  10,048   contiguous
  block  1: rows  10,048 to  20,096   contiguous
  ... 16 blocks, 160,768 rows
```

That is exactly how PETSc distributes rows across ranks. **No renumbering, no permutation, no
index translation** — the step that usually consumes the first month of an MPI retrofit is
already done, as a side effect of the multiblock design.

**The seam exchange is funnelled through two methods.** `pad_field` and `pad_coords` in
`src/multiblock.py`, 20 call sites between them. Everything that reads across a block boundary
goes through there. That is the entire halo exchange surface.

**The repo already enforces exact reproducibility.** `checkpoint.py` verifies exact restart,
`grid_fingerprint` refuses a mismatched grid, and `measurement_traps.md` documents nine ways a
plausible number has been wrong. The discipline needed to keep a parallel path honest is in
place and habitual.

### Unfavourable, and measured

**The cylinder's blocks are thin in the seam direction.** Blocks are `(157, 16, 4)` and the
seams are azimuthal — axis 1, extent 16. A halo of width 2 on both sides is:

```
  cylinder, split on axis 1 (the seam axis):  2,512 of 10,048 cells = 25.0% of the block
  square,   split on axis 0 or 1:              ~10.8-13.3%
```

**25% halo overhead is high.** It does not forbid decomposition, but it caps efficiency and
means communication will be a first-order cost, not a rounding error.

**The square's blocks are badly imbalanced.** 8 blocks ranging 4,440 to 20,572 cells. One block
per rank puts 20,572 on the slowest rank against an ideal 10,262, so **the best possible speedup
on 8 ranks is 4×, not 8×** — 50% efficiency before a single message is sent. The cylinder's 16
blocks are all 10,048 and perfectly balanced; the square would need splitting or grouping.

**The adjoint is the real blocker, and it is the project's purpose.** `src/mb_adjoint.py` runs
torch autograd through the solve, gated by 38 gradient checks. A distributed forward pass needs
a distributed backward pass — a custom `autograd.Function` whose backward is itself a PETSc
solve on the transposed operator. This is the deepest part of the codebase and is Gate 7 for
that reason.

---

## 3. Scope

**In:** `MultiBlockPISO`, `MultiBlock`, the linear solves, halo exchange, global reductions
(forces, divergence), checkpoint I/O, and the adjoint path.

**Out, explicitly:** the single-block solvers; plotting utilities (they read checkpoints, which
stay serial-readable); the NN training loop beyond making the adjoint work; multi-GPU; any
change to the discretisation. **A decomposition that changes an answer is a bug, not a feature.**

---

## 4. The cross-cutting invariant

Every gate is judged against one rule:

> **An N-rank run must reproduce a 1-rank run to round-off, on the same grid, from the same
> checkpoint, for the same number of steps.**

Not "to plotting accuracy", not "to engineering tolerance". Round-off, because anything looser
hides exactly the class of error this project keeps catching. Concretely: `max |u_N − u_1|` over
all cells and all fields, required below `1e-12` relative to `max|u_1|`, after 10 steps from a
converged checkpoint.

A test harness `test_mpi_equivalence.py` implements this once and every gate reuses it. It is
built in Gate 0, before anything can break.

**Abort rule for the whole project:** if any gate fails its criteria twice after remediation,
stop and re-evaluate rather than weaken the criterion. The criterion being inconvenient is not
evidence that it is wrong.

---

## 5. Gates

### Gate 0 — Baseline, harness, and build

**Deliverable.** Reference solutions and a profile captured before any change; `petsc4py` and an
MPI implementation building on both the Mac (ARM) and inside the GB10 container;
`test_mpi_equivalence.py` written and passing trivially at N = 1.

**Test plan.**
1. Save 10-step reference trajectories from `sqcyl_v3_forces.npz` and `cyl_shed_mac.npz`.
2. Record per-step profile (solve vs assembly) for both cases.
3. `mpirun -n 4 python -c "from petsc4py import PETSc; print(PETSc.COMM_WORLD.rank)"` on both
   machines.
4. Run the full existing suite and record which tests pass, as the regression baseline.

**Success criteria.**
- petsc4py imports under `mpirun -n {1,2,4,8}` on both machines.
- Reference trajectories stored with their grid fingerprints.
- Existing suite result recorded; any test already failing is documented, not silently inherited.

**Abort criteria.** PETSc will not build on ARM macOS with a working MPI after one day of
effort — in which case the plan continues GB10-only, or stops.

---

### Gate 1 — `Comm` abstraction, serial only

**Deliverable.** A `Comm` object owning rank/size and block ownership, with a serial
implementation. `pad_field` and `pad_coords` route through it. Blocks still all owned by rank 0.
**No MPI yet.**

**Test plan.**
1. Full existing suite.
2. `test_mpi_equivalence.py` at N = 1 against the Gate 0 references.
3. Explicit check that `Comm.owner(b) == 0` for all b and no message is sent.

**Success criteria.**
- **Bitwise identical** to the Gate 0 references — not 1e-12, bitwise. Nothing has changed yet,
  so anything else means the refactor altered an operation order.
- Every existing test passes.
- The 20 seam call sites reduced to routing through 2 methods, verified by grep.

**Abort criteria.** Bitwise identity cannot be achieved — indicates a hidden dependence on
global state that must be understood before proceeding.

---

### Gate 2 — Distributed fields and halo exchange

**Deliverable.** Blocks distributed across ranks. `pad_field` performs a real MPI exchange for
non-local neighbours. Explicit operators only — gradients, divergence, the deferred cross terms.
No distributed linear solve yet; the implicit solves still gather to rank 0.

**Test plan.**
1. `test_mpi_equivalence.py` at N = 1, 2, 4, 8, 16 for 10 steps on both cases.
2. A halo-content test: for a field with a known analytic value, every halo cell after exchange
   equals the analytic value at that coordinate — checks the exchange moved the *right* data,
   not merely *some* data.
3. A deliberately-wrong-neighbour mangle test: corrupt one seam's exchange and require the
   equivalence test to fail. **A test that cannot fail is worth nothing** — this is section 3 of
   `measurement_traps.md` applied deliberately.
4. Halo volume measured and compared against the 25% prediction.

**Success criteria.**
- Equivalence to < 1e-12 at every rank count.
- Halo-content test exact.
- Mangle test fails as intended.
- No deadlock at any rank count, verified by a timeout in CI.

**Abort criteria.** Equivalence fails at N > 1 and the cause is not found within two days.

---

### Gate 3 — Distributed pressure solve

**Deliverable.** The four pressure systems per step assembled directly into a distributed PETSc
`Mat` using the existing offsets, solved with KSP. Serial path retained behind the backend flag,
as `amgx` already is.

**Test plan.**
1. Assemble the same matrix serially and distributed; compare `A.norm()` and 20 random rows
   exactly.
2. Solve the same right-hand side both ways; compare solutions to the solver tolerance.
3. `test_mpi_equivalence.py` at N = 1, 2, 4, 8, 16 for 10 and 100 steps.
4. Iteration counts recorded per rank count — **a preconditioner whose strength depends on the
   partition changes the answer's cost, and block-Jacobi does exactly that.**
5. The Dong outflow's prescribed-pressure rows must survive distribution: verify the reduced
   system is still non-singular and the prescribed values land on the right global rows.

**Success criteria.**
- Matrix identical serial vs distributed (exact, it is the same assembly).
- Trajectory equivalence < 1e-12 at all rank counts over 100 steps.
- Iteration count at N ranks within 20% of serial, or the difference explained and recorded.

**Abort criteria.** Iteration count grows more than 2× with rank count — meaning the
preconditioner has become partition-dependent, which needs an algorithmic answer before
proceeding.

---

### Gate 4 — Distributed momentum solves

**Deliverable.** The momentum systems distributed the same way.

**Test plan.** As Gate 3, plus a check that the velocity BC elimination (inflow, wall, Dong)
applies to the correct global rows under every partition.

**Success criteria.** Equivalence < 1e-12 over 100 steps at all rank counts; BC rows verified by
direct inspection, not inferred from the trajectory matching.

---

### Gate 5 — Reductions, I/O and restart

**Deliverable.** Surface force integration, divergence diagnostics, the far-field watchdog and
checkpoint write/read all correct under MPI.

**Test plan.**
1. Forces from an N-rank run vs 1-rank, on a field where the body spans a rank boundary — the
   case that breaks a naive reduction.
2. Checkpoint written at N ranks, read at M ranks, for every (N, M) in {1,2,4,8,16}²  — the
   file must not encode the partition.
3. Exact-restart test under MPI: 20 steps continuous vs 10 + restart + 10.
4. Watchdog metrics compared serial vs distributed.

**Success criteria.**
- Forces agree to 1e-12.
- Checkpoints portable across all 25 (N, M) combinations; `grid_fingerprint` unchanged by rank
  count.
- Exact restart holds under MPI as it does serially.

---

### Gate 6 — Performance

**Deliverable.** A scaling study, and the decision on whether to continue to Gate 7.

**Test plan.** Both cases at N = 1, 2, 4, 8, 12, 16 ranks on the Mac and in the GB10 container;
time per step, time in solve, time in exchange, all recorded separately.

**Success criteria.**
- **≥ 4× on 8 ranks for the cylinder** (16 balanced blocks, 25% halo). Below that the halo cost
  is dominating and the block topology needs revisiting before more effort is spent.
- Communication time measured and reported as a fraction — no target, but it must be *known*.
- The square's imbalance quantified against the predicted 4× ceiling on 8 ranks.

**Abort criteria.** Under 3× on 8 ranks. That is the point to stop, because Gate 7 is the
expensive one and is only worth paying for if Gates 1–6 delivered.

---

### Gate 7 — The adjoint under MPI

**Deliverable.** `mb_adjoint.py` working distributed: a custom `autograd.Function` whose forward
is the distributed solve and whose backward is a distributed solve on the transposed operator,
with gradients reduced correctly across ranks.

**Test plan.**
1. All 38 existing gradient gates, under `mpirun` at N = 1, 2, 4.
2. Finite-difference vs adjoint on a distributed field, sampled where the gradient is large —
   normalised to `max|g|`, not per-cell, for the reason recorded in `measurement_traps.md` §4.
3. The existing mangle tests (zeroing `F_prev`, `p_flux`, `u_prev`, the Dong pressure in the
   backward only) must still change the gradient.
4. A new mangle specific to MPI: **drop one rank's contribution to the gradient reduction and
   require the FD comparison to fail.** Without this, a missing reduction would look like a
   small gradient error rather than a bug.

**Success criteria.**
- All 38 gates pass at every rank count.
- FD-vs-adjoint agreement no worse than serial (currently 1e-8 on `max|g|`).
- Every mangle test fails as intended, including the new rank-drop one.

**Abort criteria.** Gradient agreement degrades by more than 10× versus serial.

---

### Gate 8 — Production validation

**Deliverable.** The square cylinder rerun end to end under MPI and compared against the values
this repo has already validated against Sohankar et al.

**Test plan.** Full run at 8 ranks; measure `St` by zero crossings, `C_D`, `C_L` rms, surface
`C_p`, the half-period symmetry residuals.

**Success criteria.**
- `St` within 1e-4 of the serial 0.148811 — this is a limit cycle converged to six figures, so
  agreement should be near-exact, not "within a few percent".
- `C_D` within 1e-4 of 1.4529.
- Half-period symmetry residuals no worse than the serial 6e-06 and 6e-05.
- The far-field and near-body watchdog metrics behave identically.

---

## 6. Risks, ranked by what they would cost

| risk | consequence | mitigation |
|---|---|---|
| **Adjoint under MPI proves impractical** | the project's actual purpose does not survive the port | Gate 6 is the decision point; do not start Gate 7 without the scaling result |
| 25% halo on the cylinder caps speedup | effort spent for 2-3× | measured at Gate 2, decided at Gate 6 |
| Square block imbalance (4.6×) | poor scaling on the square specifically | known now; may need block splitting, which is a grid change and therefore a fingerprint change |
| Partition-dependent preconditioner | answers cost differently at different rank counts, silently | iteration count tracked from Gate 3 |
| PETSc build on ARM macOS | Mac path unavailable | Gate 0 establishes this before anything depends on it |
| Silent divergence of serial and parallel paths | the worst outcome — wrong results that look right | the round-off equivalence invariant, enforced at every gate, plus mangle tests that must fail |

---

## 7. Effort, and what I am not confident about

Gates 0–2 are a few days. Gates 3–5 are the bulk of the mechanical work, perhaps two weeks.
Gate 6 is a day. Gate 7 is genuinely uncertain and I would not estimate it — distributed
autograd through a PETSc solve is the part of this plan I have least basis to size, and the
honest thing is to say so rather than produce a number.

The performance predictions in §1 assume assembly parallelises cleanly because it is per-block.
That is true of the loops as written, but it has not been profiled per block, and Rhie-Chow and
the deferred cross terms both touch seams. **The 6-10× figure is a projection, not a
measurement**, and Gate 6 exists to replace it with one.
