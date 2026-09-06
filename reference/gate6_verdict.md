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
