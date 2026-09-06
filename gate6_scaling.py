"""Gate 6: the scaling study, with communication split out.

Gate 6 asks for time per step, time in solve and time in exchange RECORDED SEPARATELY, and it
sets no target for communication -- only that it be known. The buckets are therefore:

    pressure solve | momentum solve | halo exchange | allgather | assembly (remainder)

EXCHANGE AND GATHER ARE SEPARATE, and that is the point of the split. They scale oppositely:
the halo exchange moves a fixed 2 messages per rank on this ring topology however many ranks
exist, while the allgather moves every block to every rank and so grows with rank count. Lumped
together as "communication" they would hide which one stops the speed-up.

The allgather is not inherent to the port. It exists because Gate 2 replicated the fields to
make "distributed equals serial" provable, and the implicit path still assembles globally.
Measuring it separately is what turns "should we remove it?" into a costed question.

Run:  mpirun -n N python gate6_scaling.py [--case cylinder|square]
"""
import argparse
import time

import numpy as np


def build_cylinder():
    from cylinder_grid import cylinder_domain
    from src import checkpoint
    from src.comm_mpi import MPIComm
    from src.piso_multiblock import MultiBlockPISO
    d, _, _ = cylinder_domain(nz=4)
    d.comm = MPIComm(d)
    d.prepare_geometry()
    m = MultiBlockPISO(d, 1.0 / 100.0, 0.005, 2, 1e-6, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False, linear_backend="petsc",
                       distribute_momentum=False)
    checkpoint.load(m, "results/fields/cyl_shed_mac.npz")
    return d, m


def main():
    from mpi4py import MPI
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="cylinder")
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

    d, m = build_cylinder()
    c = d.comm
    m.step()                                   # warm caches and the PETSc setup
    p0, mm0 = m._pcache.t_solve, m._mcache.t_solve
    e0, g0 = c.t_exchange, c.t_gather
    MPI.COMM_WORLD.Barrier()
    ta = time.perf_counter()
    for _ in range(a.steps):
        m.step()
    MPI.COMM_WORLD.Barrier()
    N = a.steps
    wall = (time.perf_counter() - ta) / N
    tp = (m._pcache.t_solve - p0) / N
    tm = (m._mcache.t_solve - mm0) / N
    te = (c.t_exchange - e0) / N
    tg = (c.t_gather - g0) / N
    asm = wall - tp - tm - te - tg

    # LOAD IMBALANCE: the spread of per-rank wall time is what a block topology mismatch shows
    # up as, and it is invisible in a rank-0-only measurement.
    walls = MPI.COMM_WORLD.allgather(wall)

    if rank == 0:
        print(f"  {size:>2} | {wall:7.3f} | {tp:7.3f} {tm:6.3f} | {te:6.3f} {tg:6.3f} | "
              f"{asm:7.3f} | comm {100*(te+tg)/wall:5.1f}% | imbal {100*spread:5.1f}% | "
              f"blk/rank {len(c.local_blocks())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
