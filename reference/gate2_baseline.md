# Gate 2 baseline — what the port costs before the solve is distributed

Measured on the Re=100 cylinder, 160,768 cells in 16 blocks of 157x16x4, five timed steps
after a warm-up step. One BLAS thread per rank (`OMP_NUM_THREADS=1`) and `--bind-to none`.

| ranks | s/step | solve | % | other (explicit + comm) | blocks/rank |
|------:|-------:|------:|--:|------------------------:|------------:|
| 1     |  6.865 | 5.088 | 74.1 | 1.776 | 16 |
| 2     |  5.524 | 4.504 | 81.5 | 1.021 |  8 |
| 4     |  6.786 | 5.858 | 86.3 | 0.928 |  4 |
| 8     | 12.702 | 11.275 | 88.8 | 1.427 |  2 |

## What this says

**The explicit work distributes.** `other` falls 1.776 -> 1.021 -> 0.928, which is Gate 2
doing exactly what it was for. It turns back up at 8 ranks: with two blocks per rank the
communication no longer amortises over enough local work.

**Everything else gets worse, and must.** The pressure solve is REPLICATED at this gate --
every rank solves the whole system -- so ranks add contention and remove no work. The solve
share climbs 74% -> 89% as the parallel part shrinks around a fixed serial core. This is the
hole Gate 3 exists to climb out of, and it is now measured rather than assumed.

**Halo traffic is not the problem.** On this decomposition each rank exchanges 2 messages
regardless of rank count -- the cylinder is a 1-D ring, so every rank has two neighbours --
and total halo volume reaches 25.00% of interior only at 16 ranks, one block each. That
matches the plan's 25% prediction exactly.

## A measurement trap this exercise walked into

The first single-threaded run showed 1 rank at 10.6 s/step against 2 ranks at 5.4 -- an
apparent halving of SOLVE time, which is impossible when the solve is replicated. It
reproduced across runs, so it was not noise. It was `mpirun -n 1` **pinning to one core** on a
machine with 12 performance and 4 efficiency cores; `--bind-to none` brought 1 rank to 6.9
s/step and the speed-up vanished.

Recorded because the number was self-consistent, reproducible, and flattering, and only
contradicted by knowing what the code actually distributes. Any future scaling figure here
must state its binding and thread settings, or it means nothing.

## Serial profile, unchanged by Gate 2

72.7% solve / 27.3% assembly, 4.539 s/step against 4.512 before the port began -- the
distributed machinery costs nothing serially, because `exchange_halos`, `local_blocks()` and
`gather_blocks` are all no-ops there.
