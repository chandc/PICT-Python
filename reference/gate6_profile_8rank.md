# Profiling the 8-rank step

`cProfile` around exactly the region Gate 6 times -- build, one warm step, barrier, then the same
5-step loop -- on all 8 ranks, in the container, with momentum distributed. Profiles in
`results/prof/g6_r*.prof` on Spark.

**The instrumentation is honest here.** Profiled 1.733 s/step against the sweep's uninstrumented
1.668, i.e. ~1.0x overhead, because most of the time is in PETSc C code and NumPy where cProfile
taxes per-CALL and the call counts are low. Absolute numbers below can be read as real.

## Where the step goes (rank 0, 1.668 s/step)

| bucket | s/step | % |
|---|---|---|
| linear solve (PETSc), 16 solves/step | 0.803 | 46.3% |
| **sparse matrix construction (scipy)** | **0.226** | **13.0%** |
| **geometry / flux operators** | **0.236** | **13.6%** |
| communication | 0.128 | 7.4% |
| other Python + NumPy | 0.340 | 19.6% |

Rank 3 is within 0.1% of rank 0 on every bucket, so this shape is not a rank-0 artefact.

## Ranked bottlenecks

### 1. The linear solve -- 46.3%, and NOT a defect

16 solves per step at ~0.050 s each. This is the bucket the port exists to distribute and it
already scales 5.85x (3.591 -> 0.614 s from 1 to 8 ranks). It is the floor unless the SOLVE COUNT
or the preconditioner changes, neither of which is a Gate 6 question.

### 2. Sparse matrix construction -- 13.0%, and provably wasted

`build_momentum_matrix` runs 2x/step at 0.189 s/step cumulative; `build_diffusion_matrix` 2x/step
at 0.141 s/step. Underneath, the scipy pattern kernels cost 0.128 s/step:

| kernel | calls/step | s/step |
|---|---|---|
| `coo_tocsr` | 4 | 0.0475 |
| `csr_row_index` | 6 | 0.0311 |
| `csr_sort_indices` | 4 | 0.0291 |
| `csr_column_index2` | 6 | 0.0206 |

**The sparsity pattern is invariant, and this was TESTED rather than assumed:**

* pressure, two different coefficient fields: nnz 1,123,328 both, `indices` and `indptr`
  identical, values differ. Pattern is topology-only.
* momentum, two OPPOSITE flow fields: identical. The doubt was real -- an upwind-biased
  convective stencil reaches to i-1 or i+1 with the sign of the flux -- but the multi-block
  momentum path raises `NotImplementedError` for anything but `convection='central'`, so a
  symmetric stencil is the only one it can build.

So the pattern can be built once and only `.data` refilled. **Est. 0.13-0.20 s/step.**

### 3. Static geometry recomputed -- part of the 13.6%

| site | calls/step | s/step | static? |
|---|---|---|---|
| `Jg_of` (momentum) | 160 | 0.035 | `g` yes, `nu` no |
| `Jg_of` (pressure) | 160 | 0.039 | `g` yes |
| `jg_field` | 36 | 0.057 | already cached, still 3.3% |
| `contravariant_components` | 96 | 0.064 | **no** -- depends on u,v,w |

Both `Jg_of` closures recompute `g = xi_x^2 + xi_y^2 + xi_z^2` from the mesh every call.

**CACHE `g` ALONE, NEVER `nu*Js*g` OR `Js*g`.** The expression evaluates left to right, and
floating-point multiplication is not associative: caching the product and re-associating broke
bitwise identity against the Gate 0 reference on all 640 arrays while being mathematically
identical. This is the same trap `jg_field` already documents. **Est. 0.06 s/step.**

`contravariant_components` is genuine per-step work and is not on this list.

### 4. `gather_blocks` -- 0.107 s/step, 6.4%

28 calls/step. This is the Gate 2 allgather shortcut that replicates fields to make "distributed
equals serial" provable. Removing it is item 2 of `gate6_verdict.md` and the profile confirms the
size. **Est. 0.107 s/step, but it is a CORRECTNESS scaffold -- removing it changes what the
bitwise gates prove, so it is not free in the way the other two are.**

### 5. Load imbalance -- 15.4% spread, two groups of four

| ranks | local work | comm |
|---|---|---|
| 0, 1, 3, 6 | 0.81 s | 0.12 s |
| 2, 4, 5, 7 | 0.70 s | 0.21 s |

Totals are identical to 0.001 s because every step ends in collectives: the light ranks simply
wait longer, and the wait shows up as `comm`. Independent confirmation of the sweep's 17.2%.

16 unequal blocks paired contiguously onto 8 ranks gives 4 heavy and 4 light. A balanced
assignment brings the slowest rank from 0.817 to the 0.758 mean. **Est. 0.06 s/step.**

## What this adds up to

| fix | est. saving |
|---|---|
| cache the sparsity pattern, refill `.data` | 0.13-0.20 |
| cache `g` in both `Jg_of` closures | 0.06 |
| balanced block assignment | 0.06 |
| remove the allgather (correctness cost) | 0.107 |

1.668 s/step less 0.30-0.37 gives **1.30-1.37 s/step, i.e. 3.75-3.95x** -- against Gate 6's 4%
success bar of 4x and its 3x abort bar.

**These are estimates derived from a profile, not measurements.** What they say is that 4x is
plausibly reachable WITHOUT revisiting the block topology or the decomposition, and only by doing
all of items 2, 3 and 5 -- no single one gets there. The cheapest and safest is the sparsity
pattern cache: largest single prize, provably safe, and it touches no correctness scaffold.

---

# The three items were implemented and MEASURED. They delivered ~nothing.

## Result

Pinned to 8 identical Cortex-X925 cores, 5 reps, median. Both trees carry the Rhie-Chow fix, so
the caches are the only difference.

| | n1 | n8 | speedup | solve | momentum | assembly |
|---|---|---|---|---|---|---|
| without items 1+3 | 5.312 | 1.731 | 3.07x | 0.744 | 0.245 | 0.609 |
| with items 1+3 | 5.325 | 1.721 | 3.09x | 0.745 | 0.245 | 0.607 |

n8 reps: 1.719/1.722/1.731/1.731/1.747 against 1.712/1.721/1.721/1.727/1.733. Means 1.7300 vs
1.7228, **+0.4%** -- inside the spread. The assembly bucket the items target moved 0.609 ->
0.607. **Estimated 0.19-0.26 s/step; delivered 0.007.**

## Why the estimate was wrong

Profile time is not removable time when the work is MEMORY-BANDWIDTH-BOUND.

* The CSR cache does remove scipy's sort, and replaces it with `v[order]` -- a 2.2M-element
  gather, 18 MB -- plus `np.add.reduceat` over the same 2.2M. Same memory traffic, different
  library. Nothing was saved because the sort was never the cost.
* The `g` cache removes three squares and two adds per call, but still reads the cached array and
  writes the product. On a machine where sparse matvec runs at ~0.2 flops/byte, arithmetic that
  touches the same bytes is free either way.

`measurement_traps.md` already records the bandwidth argument for the SOLVE. It applies to
ASSEMBLY as well, and this is the case that shows it: a bucket can be 13% of the step, be
provably redundant, be removed correctly, and cost nothing.

## Item 2's premise was FALSE

The cylinder's blocks are all **10,048 cells**, every rank owns exactly **20,096**, and every
rank has exactly **4 physical faces**. The decomposition is perfectly balanced by every static
measure, so the 15-30% "imbalance" is not block imbalance and no reassignment can address it.

The next hypothesis -- CPU heterogeneity -- was real but is also NOT the cause. The Spark is
20 cores: 10x Cortex-X925 at 3900 MHz and 10x Cortex-A725 at 2808 MHz, and `--bind-to none` lets
the OS scatter ranks across both. **Pinning all 8 ranks to identical X925 cores left the reported
imbalance unchanged at ~30%.**

So the imbalance is real, reproducible, and **unexplained**. It is not block size, not physical-
face count, and not core type. Worth its own investigation; the likeliest remaining candidate is
that `local = wall - tp - tm - te - tg` mis-attributes time when ranks wait inside PETSc rather
than at the barrier.

## What pinning IS worth

A large cut in run-to-run variance, which is a real methodological gain for every future Gate 6
measurement:

* unpinned, 3 reps: **1.535, 1.736, 1.728** (and an earlier set at 2.137)
* pinned, 5 reps: **1.712, 1.721, 1.721, 1.727, 1.733**

The unpinned spread is wide enough to hide a 10% effect, which is exactly what happened on the
first attempt to measure items 1+3. **Pin both arms, and pin the 1-rank baseline too** -- pinning
only the parallel arm inflates the speedup the way `mpirun -n 1` on a single core once faked a 2x.

## Recommendation

Both caches are bitwise-verified (8/8 digests over a wall/inflow/outflow domain and a periodic
one) and marginally positive, but they add ~60 lines of subtle code -- including a hand-derived
duplicate-summation order that must match scipy's -- for 0.4%. **On this hardware they do not
earn their complexity.** Keep them only if the port is expected to run where memory bandwidth is
not the limit; otherwise revert and leave this section as the record of why.

The real ceiling is unchanged and is not addressable by removing redundant work: 46% of the step
is the PETSc solve and the rest is bandwidth-bound.
