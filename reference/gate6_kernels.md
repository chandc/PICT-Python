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
| **jacobi** | **1.737** | **0.619** | **-8%** |
| bjacobi (current default) | 1.884 | 0.752 | -- |
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
