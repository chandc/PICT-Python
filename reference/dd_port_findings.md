# The PETSc domain-decomposition port: what Gates 0-4 established

Companion to `petsc_domain_decomposition_plan.md`, which says what was intended. This says what
was found. Every number here is measured on the Re=100 cylinder, 160,768 cells in 16 blocks of
157x16x4, unless stated otherwise.

## Status

| Gate | Deliverable | Result |
|------|-------------|--------|
| 0 | Baselines, harness, build | PASS — all five criteria |
| 1 | `Comm` abstraction, serial | PASS — bitwise, 640/640; suite 34/34 |
| 2 | Distributed fields and halo exchange | PASS — bitwise at 2/4/8 ranks; halo 25.00% at 16 |
| 3 | Distributed pressure solve | PASS — all five test items |
| 4 | Distributed momentum solves | PASS — 8.2e-14, with `distribute_momentum=False` |
| 5-8 | Reductions/IO, performance, adjoint, validation | not started |

## What the port actually buys

Measured with both solves distributed, one BLAS thread per rank, `--bind-to none`:

| ranks | s/step | speed-up | pressure | momentum | other |
|------:|-------:|---------:|---------:|---------:|------:|
| 1 | 3.729 | 1.00x | 2.351 | 0.255 | 1.122 |
| 2 | 2.882 | 1.29x | 1.530 | 0.714 | 0.637 |
| 8 | 1.195 | 3.12x | 0.534 | 0.204 | 0.457 |

Against the pre-port serial baseline of 6.865 s/step the end-to-end gain is 5.7x: PETSc is
1.84x faster than SciPy at one rank, and scales 3.12x on top of that.

**The bottleneck has moved.** `other` -- assembly, halo exchange and the Gate 2 allgather --
scaled 2.5x while the pressure solve scaled 4.4x, so it has grown from 30% to 38% of runtime.
The allgather inside it is pure overhead that exists only because Gate 2 replicated the fields
to make "distributed equals serial" provable. Removing it is worth more than anything left in
the solves.

## The seam is two lines, not twenty

The plan budgeted "20 seam call sites", correctly counting CALLERS of `pad_field`/`pad_coords`.
But those all ask for a block's OWN padded data. The place where one block reads ANOTHER's is
inside the padding recursion, and there are exactly two such lines in the codebase. The right
boundary is `(block, k)` -- a block padded along its first k axes -- not `pad_field`, because
padding axis 1 needs the neighbour ALREADY padded along axis 0.

## Three structural decisions and why

**The exchange is a COLLECTIVE EPOCH, not lazy.** Triggered from inside `pad_field`, a rank
owning three blocks enters it three times and one owning two enters twice, and MPI hangs -- in a
way that looks like a solver stall rather than a protocol error. `Domain.exchange_halos` is
called once per pass by every rank, and `pad_field` communicates not at all.

**It is LEVEL-SYNCHRONOUS.** Padding is recursive: computing a neighbour's level-1 data can
require a third rank's level-0 data. Chasing that depth-first across ranks deadlocks. Three
rounds, one per axis, each supplying what the next consumes; every rank computes `upto` only for
blocks it owns, and remote data always arrives as an already-extracted slab.

**The MESH IS REPLICATED and only the FIELDS are distributed.** Every rank constructs the same
`Domain`, so coordinates never need to move. Exchanging them was moving data every rank already
had -- and worse, the coordinate path served only local blocks while the solver's constructor
builds metrics for ALL of them, so it faulted before a step ran.

## Halo traffic

On this decomposition each rank exchanges 2 messages regardless of rank count -- the cylinder is
a 1-D ring, so every rank has two neighbours however many exist. That is ideal weak scaling of
communication. Total halo volume reaches 25.00% of interior at 16 ranks, one block each, which
matches the plan's 25% prediction exactly; the prediction turns out to describe the
one-block-per-rank limit. An earlier 44-79% was an artefact of 12^3 test blocks.

## `distribute_momentum` is a flag, not a decision

The momentum system is ~7% of serial runtime and converges in ~26 iterations a step against the
pressure system's ~1200. Distributing it makes its iteration path partition-dependent, so serial
and distributed reach different but equally valid solutions within tolerance, and the trajectory
comparison degrades from 8.2e-14 to 1.0e-12.

    distribute_momentum=1   8 ranks 1.708 s/step   Gate 4 equivalence FAILS
    distribute_momentum=0   8 ranks 1.963 s/step   Gate 4 PASSES

13% throughput against provable partition-independence. Replicating is NOT free: every rank does
the whole solve and they contend, so the momentum solve costs 0.253 s at one rank but 0.527 s at
eight. An earlier estimate of 4.3% assumed it stayed flat and was wrong.

## The Dong outflow amplifies pressure-solve error ~500x

Measured against the solve tolerance, the trajectory sensitivity is:

    singular (no outflow)   rtol 1e-9  ->  7.8e-14      ratio 7.8e-5
    Dong outflow            rtol 1e-9  ->  4.4e-11      ratio 4.4e-2

The prescribed-pressure boundary feeds error back through the advective outflow update rather
than having it projected away. Production runs with Dong outflow need a tighter pressure
tolerance than the periodic case for the same trajectory accuracy -- which applies to the
cylinder and square cases, not only to the gates.

## The bitwise reference is MACHINE-SPECIFIC

Verified by running the same code, same configuration, on two machines:

    Spark run vs SPARK-captured reference   640/640   PASS
    Spark run vs MAC-captured reference       0/640   FAIL

Both are ARM64, but the toolchains differ -- numpy 2.0.2 / scipy 1.13.1 on the Mac against
2.4.6 / 1.16.3 in the Spark container -- and with them the BLAS paths and reduction orders. The
last bits move, and a digest comparison is all-or-nothing.

**This does not weaken the gates.** What Gates 1-5 verify is that DISTRIBUTION does not change
the answer on a given machine, which is the property that matters for a domain-decomposition
port and is exactly what the per-machine test shows. Cross-machine bit-reproducibility is a
different and much harder property, and was never claimed.

But it does mean **each machine needs its own Gate 0 baseline**, captured with
`gate0_baseline.py` before the gate tests are run there. Anyone reproducing this port on a third
machine will otherwise see 0/640 and reasonably conclude the port is broken.

## Reference and test must build the SAME solver

Three separate "0/640" alarms in this port were configuration drift rather than regressions:

  1. `momentum_tol` was tightened from `tol` to 1e-14 in Gate 4 to stop partition-dependent
     iteration paths diverging, and `test_mpi_equivalence.py` was not re-run afterwards. It
     reported a deliberate change as a regression on every run for hours.
  2. The same mismatch again on Spark, once the test had been pinned back to `tol` but
     `gate0_baseline.py` -- the reference GENERATOR -- had not.
  3. A missing reference file, which the test reported as a failed check rather than as a
     missing input.

The pattern is a check reporting something other than what it claims to measure, and it is the
same shape as the mangles that silently stopped intercepting (`measurement_traps.md` §14). The
structural fix is for the reference generator and its consumer to construct the solver through
one shared path instead of each building their own; both are currently pinned by hand, which
works and will drift again.

## What the criteria turned out to mean

**"Bitwise identical" is achievable and was verified first.** Gate 1's criterion only means
something if unmodified serial code reproduces itself, which nobody had checked. Two independent
captures: 640/640 arrays and all reductions identical. Had that failed, the whole gate structure
would have rested on an unachievable criterion.

**"< 1e-12" is a statement about a converged solve.** Agreement tracks the pressure tolerance
and then saturates: 1.36e-9 at rtol 1e-6, 7.8e-14 at 1e-9, 8.3e-14 at 1e-12. The plateau is the
round-off floor of ten steps' accumulation. At the production tolerance of 1e-6 the distributed
and serial runs already agree as exactly as two 1e-6 solves can; comparing against 1e-12 there
is a category error.

**Iteration growth was a red herring.** The plan's 2x abort exists because block-Jacobi is
partition-dependent by construction. The 2.8x growth that tripped it was not that at all -- it
was a dropped initial guess (below). The genuine partition effect is 1.08x for pressure and
1.14-1.23x for momentum: mild.
