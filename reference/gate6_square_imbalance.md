# Gate 6, third criterion: the square's imbalance does not bind

Gate 6 asks for "the square's imbalance quantified against the predicted 4x ceiling on 8 ranks".
Quantified, the prediction is right about the geometry and wrong about the consequence.

## The geometry does predict a 4x ceiling

    8 blocks, 164,192 cells
    sizes  [10952, 8880, 41144, 8880, 33360, 10952, 8880, 41144]     max/min = 4.63x

One block holds **25.1%** of the mesh, and that is irreducible: no assignment can put fewer than
41,144 cells on the busiest rank. So a speed-up bounded by cell count saturates at
164192/41144 = **3.99x** however many ranks are used.

| ranks | per-rank cells (contiguous) | max/mean | cell-bound ceiling |
|------:|-----------------------------|---------:|-------------------:|
| 2 | 69856, 94336 | 1.149x | 1.74x (ideal 2x) |
| 4 | 19832, 50024, 44312, 50024 | 1.219x | 3.28x (ideal 4x) |
| 8 | one block each | 2.005x | 3.99x (ideal 8x) |

## But the expensive part never sees that partition

The distributed pressure matrix is built with

    M = PETSc.Mat().createAIJ(size=((PETSc.DECIDE, n), (PETSc.DECIDE, n)), comm=comm)

`PETSC_DECIDE` splits matrix ROWS EVENLY across ranks. It does not follow the block partition.
The plan specified "using the existing offsets"; an even split was chosen instead at Gate 3, on
the grounds that aligning to block offsets is a locality optimisation on top of a working solve
and doing both at once would mean debugging two things.

The consequence was not thought through at the time: **the pressure solve, which is the dominant
cost, is perfectly balanced however uneven the blocks are.** Block imbalance touches only the
LOCAL work -- assembly and halo packing -- which is a minority of runtime.

## Measured

| ranks | s/step | speed-up | cell imbalance | local-work spread | idle cost | % of step |
|------:|-------:|---------:|---------------:|------------------:|----------:|----------:|
| 1 | 3.019 | 1.00x | 0.0% | 0.0% | — | — |
| 2 | 2.392 | 1.26x | 14.9% | 3.5% | 0.020 s | **0.8%** |
| 4 | 1.588 | 1.90x | 21.9% | 10.8% | 0.053 s | **3.3%** |

Imbalance costs about **3% of runtime at 4 ranks**, not a factor of 4. Note also that the
local-work spread is roughly HALF the cell imbalance: local work is not purely proportional to
cell count, since some per-block overheads are fixed.

**So re-blocking the square mesh would buy ~3%.** That matters, because "one block is 25% of the
mesh" is the kind of fact that invites an expensive re-meshing effort. What actually limits this
case is the same thing that limits the cylinder: the replicated momentum solve, and a pressure
solve that scales at 2.2x on 4 ranks rather than 4x.

## A metric that could not fail, and its replacement

The first version reported per-rank WALL time spread and read **0.0% at every rank count** on a
mesh whose blocks differ by 4.63x. That is not a healthy result, it is a meaningless one: every
step ends in collectives, so all ranks block until the slowest arrives and then report identical
wall time however uneven the work was. Imbalance appears as idle time AT the barrier, not as
spread in total wall time.

Subtracting the collectives -- solve, exchange, gather -- leaves per-rank LOCAL work, and the
spread of that is the quantity wanted. It now reads 0.0%, 3.5%, 10.8% at 1, 2 and 4 ranks, which
tracks the cell imbalance as it should.

This is the third metric in this port to have been vacuous on first writing (see
`measurement_traps.md` §14 for the mangles, and the Gate 5 reduction check that reported
"0.000e+00 on n/a"). The common shape: a number that is small for a good reason and small for a
bad reason, with nothing in the output distinguishing them.
