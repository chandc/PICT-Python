"""Gate 6, third criterion: quantify the SQUARE case's load imbalance.

WHY THE SQUARE AND NOT THE CYLINDER. The cylinder's 16 blocks are equal, so it measures load
imbalance at exactly 0.0% and says nothing about it. The square's 8 blocks are not:

    sizes  [10952, 8880, 41144, 8880, 33360, 10952, 8880, 41144]     max/min = 4.63x

One block holds 25.1% of the mesh, and that is IRREDUCIBLE -- no assignment can put fewer than
41,144 cells on the busiest rank. So the speed-up ceiling saturates at 164192/41144 = 3.99x
however many ranks are used, which is the plan's predicted "4x ceiling on 8 ranks", derived
rather than assumed:

    ranks   per-rank cells (contiguous)          max/mean   ceiling
      2     69856, 94336                          1.149x     1.74x   (ideal 2x)
      4     19832, 50024, 44312, 50024            1.219x     3.28x   (ideal 4x)
      8     one block each                        2.005x     3.99x   (ideal 8x)

WHAT IS MEASURED HERE is whether the achieved speed-up tracks those ceilings. A run that falls
well short of them has a problem BEYOND imbalance; one that meets them is doing as well as this
block topology permits, and the answer is to re-block the mesh rather than to optimise the code.

The field is the free stream rather than a converged solution: a scaling measurement needs a
representative workload, not a physically meaningful one, and loading a converged field would
tie this to a particular saved state.
"""
import argparse
import time

import numpy as np


def main():
    from mpi4py import MPI
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    a = p.parse_args()
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

    from square_cylinder_grid import square_domain
    from square_cylinder_bc import apply as apply_bc
    from src.comm_mpi import MPIComm
    from src.piso_multiblock import MultiBlockPISO

    out = square_domain()
    d = out[0] if isinstance(out, tuple) else out
    d.comm = MPIComm(d)
    d.prepare_geometry()
    m = MultiBlockPISO(d, 1.0 / 48.0, 0.01, 2, 1e-6, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False,
                       linear_backend="petsc", distribute_momentum=False)
    for b in range(len(d.blocks)):
        m.u[b][:] = 1.0
        m.v[b][:] = 0.0
        m.w[b][:] = 0.0
    apply_bc(m, d)

    sz = np.array([b.size for b in d.blocks])
    ow = np.array(d.comm.owners)
    load = np.array([sz[ow == r].sum() for r in range(size)])
    ceiling = sz.sum() / load.max()

    m.step()                                        # warm caches
    c = d.comm
    p0, mm0 = m._pcache.t_solve, m._mcache.t_solve
    e0, g0 = c.t_exchange, c.t_gather
    MPI.COMM_WORLD.Barrier()
    ta = time.perf_counter()
    for _ in range(a.steps):
        m.step()
    MPI.COMM_WORLD.Barrier()
    N = a.steps
    wall = (time.perf_counter() - ta) / N
    # IMBALANCE MUST BE MEASURED ON LOCAL WORK, NOT WALL TIME. Every step ends in collectives,
    # so all ranks block until the slowest arrives and then report IDENTICAL wall time however
    # uneven the blocks are -- the first version of this metric read 0.0% at every rank count on
    # a mesh whose blocks differ 4.63x, which is not a healthy result but a meaningless one.
    # Imbalance appears as idle time at the barrier. Subtracting the collectives (solve,
    # exchange, gather) leaves the per-rank LOCAL work, and the spread of that is the quantity.
    local = wall - (m._pcache.t_solve - p0) / N - (c.t_exchange - e0) / N \
            - (c.t_gather - g0) / N
    locals_ = MPI.COMM_WORLD.allgather(local)
    spread = (max(locals_) - min(locals_)) / max(np.mean(locals_), 1e-30)
    walls = MPI.COMM_WORLD.allgather(wall)
    if rank == 0:
        print(f"  {size:>2} | {wall:7.3f} | cells/rank {load.min():>6,}-{load.max():>6,} "
              f"| load max/mean {load.max()/load.mean():.3f}x | ceiling {ceiling:.2f}x "
              f"| LOCAL-work spread {100*spread:5.1f}% | press {(m._pcache.t_solve-p0)/N:.3f} "
              f"| comm {((c.t_exchange-e0)+(c.t_gather-g0))/N:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
