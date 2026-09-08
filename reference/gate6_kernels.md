# Which kernels dominate the 8-rank step, and which of them is worth optimising

PETSc `-log_view` at 8 ranks (96 KSPSolve over 6 steps), plus the Python-level cProfile in
`gate6_profile_8rank.md`. Log: `results/logs/petsc_logview.log` on Spark.

## The kernels

| kernel | s/step | % of step | % flops | Mflop/s | max/min |
|---|---|---|---|---|---|
| **MatMult** -- sparse matvec | 0.221 | **13%** | 34% | 12,685 | 1.2 |
| **MatSolve** -- ILU triangular solves | 0.204 | **12%** | 34% | 13,512 | 1.3 |
| **VecTDot** -- global dot products | 0.128 | **8%** | 10% | 6,554 | **2.5** |
| MatAssemblyEnd | 0.067 | 4% | 0% | -- | 1.0 |
| PCSetUpOnBlocks + MatLUFactorNum | 0.027 | 2% | 2% | ~4,000 | 1.0 |
| VecNorm | 0.018 | 1% | 5% | 23,506 | 1.4 |
| VecAXPY | 0.010 | 0.6% | 10% | 82,379 | 1.1 |

Read the Flop column as MAX PER RANK and the Mflop/s column as the global total -- they differ by
the rank count, which is easy to misread by 8x.

**A CORRECTION TO AN EARLIER CLAIM IN THIS PROJECT.** "MatMult runs at 13 GFlop/s against
VecAXPY's 82, therefore it is memory-bandwidth-bound" is WRONG. VecAXPY's implied traffic is
988 GB/s, well above this machine's DRAM bandwidth, which proves its vectors are CACHE-RESIDENT
(20,096 doubles = 160 KB per rank). It is not a bandwidth yardstick. MatMult is limited by
indirect addressing, and MatSolve by the sequential dependency in a triangular solve.

## VecTDot is where the "imbalance" lives

15,636 global `MPI_Allreduce` calls carrying 63% of all reductions. Same flop count as VecAXPY
(6.28e8 against 6.32e8) and **12.5x the time** (0.767 s against 0.061 s). Its max/min ratio is
**2.5**, the worst of any kernel, against 1.1 for the non-synchronising ops.

That resolves what `gate6_profile_8rank.md` left open. The reported ~30% "imbalance" is NOT
unequal work -- every block is 10,048 cells, every rank owns 20,096, and pinning all ranks to
identical Cortex-X925 cores did not change it. It is ranks waiting at Krylov synchronisation
points, and PETSc measures it directly.

## The preconditioner: MEASURED, and it inverts the textbook answer

8 ranks, pinned to X925 cores, 2 reps, distributed momentum:

| preconditioner | s/step | pressure solve | vs default |
|---|---|---|---|
| jacobi | 1.737 | 0.619 | -8% **(WRONG -- see below)** |
| bjacobi | 1.884 | 0.752 | -- |
| gamg (AMG) | 3.166 | 1.190 | +68% |
| asm | 78.7 | 77.4 | +4080% |

**AMG loses badly, and jacobi beats the bjacobi default.** One explanation covers both: there are
**16 solves per step**, so every preconditioner SETUP is paid 16 times per step on only 20k cells
per rank. bjacobi's ILU setup costs 0.20 s across the run and AMG's hierarchy build far more,
while jacobi has essentially none. Jacobi wins despite needing more iterations.

`petsc_domain_decomposition_plan.md` already suspected this for AmgX ("latency-bound at this
size"). It holds on the CPU path too, and is now measured rather than suspected.

## What is worth optimising, in order

1. **Switch the parallel default from `bjacobi` to `jacobi`.** 8% off the step for a default
   change. `linsolve.py` also records that bjacobi is "the last partition-dependent thing left" --
   it factorises a different block per rank, which is why Gate 3's pressure iteration counts
   differ (serial 303 against 8-rank 211).
2. **Attack the SOLVE COUNT.** MatMult + MatSolve + VecTDot is a third of the step and all three
   are strictly proportional to it. 16 solves/step comes from `corrector_steps=2` x
   `picard_iters=2` x components, and whether both are needed at this dt has never been examined.
   This is the largest untouched lever in the solver.
3. **NOT the kernels themselves** -- indirection and sequential dependencies, nothing tunable
   here. **NOT AMG** -- measured, it is the worst viable option at this size. **NOT assembly
   caching** -- measured at 0.4%, see `gate6_profile_8rank.md`.


---

# RETRACTION: the preconditioner timings above were taken under contention

The whole table ran while another user's process held a core at 100%. `--cpu-set` CONFINES our
ranks to the named cores; it does not RESERVE them. So we time-shared with that process, and
pinning additionally removed the OS's ability to migrate away from it -- pinning made the
measurement more vulnerable to contention, not less, which is the opposite of what was claimed
for it. Everything measured in that window ran about 2.6x slow.

Re-measured on an idle machine, 3 reps, non-overlapping:

| ranks | jacobi | bjacobi | jacobi vs bjacobi |
|---|---|---|---|
| 1 | 3.733 | 2.190 | **+70.5% slower** |
| 8 | 1.095 | 0.985 | **+11.2% slower** |

reps: jacobi n8 [1.092, 1.095, 1.100], bjacobi n8 [0.984, 0.985, 0.990].

**The sign flipped.** jacobi is slower at every rank count, not 8% faster at 8. The conclusion
that "jacobi is the one clear performance win" is withdrawn.

What SURVIVES: jacobi is still the right default, because it is what makes Gates 3 and 4 agree to
~1e-14 across rank counts instead of ~1e-7. That is a correctness property and no timing affects
it. The 11.2% at 8 ranks is the price of partition independence, not a bonus.

What ALSO survives, because it is a ratio taken within one contended window and the contention
was steady: AMG and asm lose badly, and the reason -- 16 preconditioner setups per step on 20k
cells per rank -- is structural. Worth re-measuring idle before being quoted.

---

# The distributed solve path had NO caching at all

Looking for a preconditioner-reuse win turned up something larger: `_petsc_solve_mpi` created a
brand-new `PETSc.Mat` AND a brand-new `PETSc.KSP` on **every call**, while the serial path has
cached both since Gate 3 (`_petsc_key`). With 16 solves per step that is 16 matrix allocations,
16 symbolic assemblies and 16 preconditioner setups per step -- visible directly in `-log_view`,
where `MatAssemblyEnd`, `MatILUFactorSym`, `MatLUFactorNum` and `PCSetUpOnBlocks` all report
exactly 96 calls over 6 steps.

It also explains the preconditioner ranking measured earlier: AMG lost by 68% because its
hierarchy was rebuilt 16 times per step, and jacobi won relatively because it has nothing to
rebuild. Setup cost was being paid per SOLVE rather than per PATTERN.

The cache is safe because the sparsity pattern is topology-only -- verified, not assumed: the
pressure matrix is identical under a changed coefficient field and the momentum matrix under a
REVERSED flow. `NEW_NONZERO_ALLOCATION_ERR` makes a pattern change fail loudly rather than
silently mallocing to a correct-but-slow answer.

## Measured, 8 ranks, pinned, 3 reps

| | s/step | pressure | momentum |
|---|---|---|---|
| cache off | 1.462 / 1.493 / 1.459 | 0.585 | **0.207** |
| cache on | 1.268 / 1.263 / 1.259 | 0.524 | **0.074** |

**13.6% off the step; the momentum bucket down 64%.** Non-overlapping reps.

## Bitwise, and verified before defaulting

Gates 3, 4 and 5 at 2, 4 and 8 ranks return values identical to the uncached run to every printed
digit: 2.360e-14 / 2.266e-14 / 3.287e-14 (Gate 3), 1.219e-14 / 9.783e-15 / 1.176e-14 (Gate 4),
3/3 on Gate 5 at every rank count. Now on by default; `PICT_MPI_CACHE=0` disables it.

## It also makes the CORRECT default free

Uncached, jacobi cost 11.2% against bjacobi at 8 ranks -- the price of partition independence.
Cached, with the preconditioner ranking re-measured:

| preconditioner | s/step (cached, 8 ranks) |
|---|---|
| **jacobi** | **1.290** |
| bjacobi | 1.303 |
| gamg | 1.77 |

jacobi and bjacobi are within 1%, and jacobi is the faster of the two. **The 11% penalty is
gone**, so partition independence no longer trades against speed.

gamg still loses, and the reason is now precise: `setOperators` on a cached KSP still triggers
`PCSetUp`, so the hierarchy is still rebuilt per solve. Amortising it needs
`setReusePreconditioner`, which reuses a preconditioner built from STALE values -- convergent but
along a different Krylov path, so **not bitwise**. Not pursued: it would trade back the
partition-independence property that was just secured.
