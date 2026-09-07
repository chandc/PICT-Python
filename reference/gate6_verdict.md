# Gate 6: measured on Spark, and it trips its own abort

Measured on the DGX Spark, 20 idle cores, inside `pict-petsc:latest` (pinned OpenMPI 4.1.9a1,
PETSc/petsc4py 3.25.5). Cylinder, five timed steps after a warm-up, one BLAS thread per rank,
`--bind-to none`. Momentum replicated -- the configuration that passes Gate 4.

## The numbers

**nz = 4 (160,768 cells)**

| ranks | s/step | speed-up | pressure | momentum | comm % | imbalance |
|------:|-------:|---------:|---------:|---------:|-------:|----------:|
| 1  | 2.443 | 1.00x |  1.389 | 0.164 |  0.9% |  0.0% |
| 2  | 1.683 | 1.45x |  0.960 | 0.167 |  2.4% |  0.6% |
| 4  | 1.119 | 2.18x |  0.487 | 0.205 |  4.0% |  9.6% |
| 8  | 0.937 | **2.61x** | 0.270 | 0.258 | 11.1% | 10.1% |
| 12 | 1.642 | 1.49x |  0.500 | 0.492 |  5.6% | 75.4% |
| 16 | 1.909 | 1.28x |  0.458 | 0.618 | 14.4% | 80.1% |

**nz = 8 (321,536 cells)** — 8 ranks: 2.308 s, **2.15x**, comm 19.6%, imbalance 25.7%.

## Verdict: ABORT by the plan's own criterion

**2.23x on 8 ranks**, against an abort threshold of 3x and a success bar of 4x.

That figure is the mean of five repeats, and the repeats matter: the single-shot sweep above
recorded 2.61x, which turned out to be near the top of the range.

    1 rank    2.361 s  +/- 0.064   (5.5% spread)   stable
    8 ranks   1.058 s  +/- 0.107  (24.6% spread)   noisy
    speed-up  2.11x  2.19x  2.09x  2.71x  2.16x    mean 2.23x, median 2.16x

Four of five repeats sit at 2.09-2.19x; one outlier reaches 2.71x. **No repeat reaches 3x.**
Re-measuring was worth doing and moved the number DOWN by 15%, which is a reminder that a
single sample under a threshold decision is not evidence -- the first value happened to flatter
the port.

Note the asymmetry: the 1-rank measurement is stable at 5.5% spread while the 8-rank one varies
24.6%. Contention among eight replicated momentum solves for memory bandwidth is the likely
source, and it means multi-rank timings on this machine need repeats as a matter of course.

## Three findings that matter more than the verdict

**1. A LARGER PROBLEM DOES NOT HELP -- it hurts.** The obvious objection to a strong-scaling
result is that 20,096 cells per rank is too small to scale. Doubling the problem made 8-rank
scaling WORSE, 2.61x -> 2.15x. Granularity is not the constraint, so "run a bigger case" is not
the answer.

**2. THE GATE 4 / GATE 6 TENSION DOES NOT EXIST.** Replicating the momentum solve was believed to
cost ~13% throughput, measured on the Mac. On idle hardware it costs nothing and is slightly
FASTER than distributing it:

    ranks   DM=0 (replicated)   DM=1 (distributed)
      4          1.119               1.063
      8          0.937               1.012
     12          1.642               1.718

The momentum solve itself is 0.258 s replicated against 0.288 s distributed at 8 ranks:
distributing a small, easy system costs more in communication than it saves in work. The earlier
13% figure was an artefact of eight ranks contending for cores on a 16-core Mac already carrying
~2.5 cores of unrelated load. **The configuration that passes Gate 4 is also the faster one.**

**3. THE CEILING IS 3.67x AND IT IS NOT THE SOLVE.** At 8 ranks:

    pressure   1.389 -> 0.270   5.14x   28.8% of the step
    momentum   0.164 -> 0.258   0.64x   27.5%
    assembly   0.867 -> 0.305   2.84x   32.6%
    comm       0.022 -> 0.103   0.21x   11.0%

The pressure solve -- the thing this port was built to distribute -- scales at **5.14x**. It is
everything else that does not: 71% of the step is now momentum, assembly and communication, which
caps the achievable speed-up at 3.67x even if the pressure solve became free. **Gate 6's 4x
success bar is unreachable by construction with this decomposition**, and its 3x abort is only
marginally reachable.

**4. BEYOND 8 RANKS IT COLLAPSES, AND IMBALANCE IS WHY.** 10% at 8 ranks becomes 75-84% at 12 and
16. With 16 blocks over 12 ranks, contiguous assignment gives some ranks two blocks and others
one, so half the ranks idle at every collective. This is a BLOCK-COUNT problem, not a code
problem: 16 blocks only balances on rank counts that divide 16.

## What would actually raise the number

In order of measured payoff:

1. **The assembly bucket** (32.6% of the step, scaling only 2.84x). Larger than the pressure solve
   at 8 ranks and nobody has looked at it.
2. **Remove the allgather.** It is the Gate 2 shortcut that replicates fields to make
   "distributed equals serial" provable, and it is most of the communication growth
   (0.020 -> 0.070 s from 1 to 8 ranks).
3. **Block count divisible by the rank count**, or a non-contiguous assignment that balances.
   Free, and it recovers the 12- and 16-rank points.

Distributing the momentum solve is NOT on this list: measured, it makes things worse.

## Recommendation

The plan says abort means stop, "because Gate 7 is the expensive one and is only worth paying for
if Gates 1-6 delivered". Gates 0-5 delivered: the port is correct, bitwise, at every rank count
tested. Gate 6 did not deliver the performance the plan required.

Whether to stop is a judgement about goals rather than a technical one. What is technically clear
is that continuing to Gate 7 on the current decomposition would buy an adjoint that runs at 2.6x,
and that the three items above are cheaper than Gate 7 and address the actual limits.

---

# SUPERSEDED: the re-run never tested distributed momentum, and with it Gate 6 clears the abort

## What was wrong with the earlier measurement

`gate6_scaling.py` line 36 passes `distribute_momentum=False` explicitly, overriding the default
that had just been changed to `True`. The Spark copy is byte-identical to local (same md5), so
**`g6_redo.log` measured the REPLICATED configuration throughout** -- the verification that was
asked for never happened.

Worse, the driver prints a `DM` label taken from an environment variable while the BEHAVIOUR
stayed hardcoded, so a run labelled `DM1` would not have been one. `gate6_scaling_dm.py` wires
both to `PICT_DIST_MOM`; the original driver is untouched.

Two further method notes, both of which would have corrupted the comparison:

* **The container is ~5% slower than bare metal** (1-rank: 5.13 s/step in-container against
  `g6_redo`'s 5.13... measured 5.391 on an earlier partial run). `g6_redo` and this sweep are
  different environments and must not be mixed, which is why BOTH arms were re-run rather than
  reusing the old DM0 column.
* **16 ranks SEGFAULTED inside the container** (signal 11) on the container's default 64MB SHMEM
  limit, and the sweep script had `2>/dev/null`, so the failures appeared as silently missing
  rows rather than errors. `--ipc=host` fixes it; the follow-up script does not suppress stderr.

## The measurement

Cylinder nz=4, in-container, median of 3 reps, base = 1 rank replicated. Same `PICT_MOM_TOL`
(default) in both arms -- this is not the rigged comparison that tied tolerance to decomposition.

| ranks | DM0 s/step | x | DM1 s/step | x | mom DM0 | mom DM1 | imbal |
|---|---|---|---|---|---|---|---|
| 1 | 5.133 | 1.00 | 5.141 | 1.00 | 0.297 | 0.298 | 0.0% |
| 2 | 3.553 | 1.44 | 4.234 | 1.21 | 0.318 | **0.974** | 0.7% |
| 4 | 2.226 | 2.31 | 2.347 | 2.19 | 0.358 | 0.415 | 0.8% |
| **8** | 1.910 | 2.69 | **1.668** | **3.08** | 0.441 | **0.233** | 17.2% |
| 12 | 2.657 | 1.93 | 2.096 | 2.45 | 0.742 | 0.254 | 47.5% |
| 16 | 2.919 | 1.76 | 2.061 | 2.49 | 0.524 | 0.207 | 55.2% |

Per-rep spreads do not overlap at any rank count (n8 totals: DM0 1.783/1.910/2.019 against DM1
1.668/1.668/1.759), so the pattern is not noise.

## Verdict

The plan's criteria are **success at >= 4x on 8 ranks, abort under 3x**. The peak is **3.08x at 8
ranks with the momentum solve distributed**. That **lifts the abort** and does **not** meet
success: Gate 6 moves from "stop" to "continue, below target". Whether to continue is a judgement
about goals, not a technical conclusion.

## Corrections to the text above this section

* "Distributing the momentum solve is NOT on this list: measured, it makes things worse" is
  **WRONG at 8 ranks and above** and is retracted. It came from the rigged comparison. Measured
  fairly, distribution halves the momentum bucket at 8 ranks (0.441 -> 0.233 s) and cuts it by
  60-72% at 12 and 16.
* It is **right below 8 ranks**, and by more than was thought: at 2 ranks distribution makes the
  momentum bucket THREE TIMES worse (0.318 -> 0.974 s, no overlap across reps) and costs 19%
  overall. The crossover is between 4 and 8 ranks.

**So `distribute_momentum=True` as an UNCONDITIONAL default is wrong.** It should be conditional
on rank count -- on this problem, distribute at >= 8 ranks and replicate below. The 2-rank spike
is not understood and is worth its own look before the threshold is hardcoded.

## What still caps the result

Unchanged, and the three items above still address it. The pressure solve scales 5.85x (3.591 ->
0.614 s at 8 ranks); assembly scales ~1.9x, so 68% of the 8-rank step is non-solve work. Beyond
8 ranks the binding constraint is the BLOCK TOPOLOGY: 16 blocks of unequal size give 47.5% and
55.2% imbalance at 12 and 16 ranks, which is why both are slower than 8.
