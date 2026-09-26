"""Gate 5 item 2: a checkpoint written at N ranks must read at M, for every (N, M).

WHY BOTH SIDES MATTER. That the FILES come out byte-identical regardless of writing rank count
is necessary but not sufficient: it proves the writer does not encode the partition, not that
the reader is partition-independent. A reader that distributed blocks differently at M ranks
would load the same bytes into the wrong places, and the result still runs -- it just has a
permuted field. So each file is loaded at each rank count and the resulting SOLVER STATE is
compared against the serial load, not the file against the file.

Run:  mpirun -n M python test_gate5_matrix.py
"""
import glob
import os
import sys

import numpy as np

from cylinder_grid import cylinder_domain
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO


def main():
    from mpi4py import MPI
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

    files = sorted(glob.glob("results/gate5/ckpt_n*.npz"),
                   key=lambda p: int(p.split("_n")[1].split(".")[0]))
    if not files:
        if rank == 0:
            print("  no checkpoints — run test_gate5.py at 1,2,4,8,16 ranks first")
        return 1
    ref = np.load("results/gate5/ckpt_n1.npz", allow_pickle=True)

    worst_all, rows = 0.0, []
    for f in files:
        n_written = int(f.split("_n")[1].split(".")[0])
        d, _, _ = cylinder_domain(nz=4)
        if size > 1:
            from src.comm_mpi import MPIComm
            d.comm = MPIComm(d)
        d.prepare_geometry()
        m = MultiBlockPISO(d, 1.0 / 100.0, 0.005, 2, 1e-9, time_scheme="bdf2",
                           scheme="rotational", picard_iters=2, rhie_chow=True,
                           persistent_flux=True, ddt_corr=False, linear_backend="petsc",
                           distribute_momentum=False)
        checkpoint.load(m, f)
        worst = 0.0
        for fld, arr in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p)):
            for b in range(len(d.blocks)):
                worst = max(worst, float(np.abs(arr[b] - ref[f"{fld}_{b}"]).max()))
        worst = MPI.COMM_WORLD.allreduce(worst, op=MPI.MAX)
        worst_all = max(worst_all, worst)
        rows.append((n_written, worst))

    ok = worst_all == 0.0
    if rank == 0:
        cells = "  ".join(f"N={n}:{w:.0e}" for n, w in rows)
        print(f"  [{'PASS' if ok else 'FAIL'}] read at M={size}: {cells}   "
              f"worst {worst_all:.3e} (must be exactly 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
