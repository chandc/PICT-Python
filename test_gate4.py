"""Gate 4: the momentum systems distributed, and the BC rows verified by INSPECTION.

WHY INSPECTION AND NOT THE TRAJECTORY. Gate 4's success criteria say the boundary-condition
rows must be "verified by direct inspection, not inferred from the trajectory matching", and
that distinction is the whole point. Velocity BC elimination moves known wall, inflow and Dong
values out of the unknown set and onto the right-hand side. If the elimination were applied to
the WRONG global rows under some partition, the resulting system is still well posed and still
converges -- it just solves a different problem, with the wall condition imposed somewhere other
than the wall. A trajectory comparison catches that only if the two runs disagree; when both the
serial and the distributed run use the same (wrong) index set, they agree perfectly with each
other and with nothing physical.

So this checks the index set against GEOMETRY: every row marked boundary must belong to a node
that actually lies on a wall face, and every node on a wall face must be marked. That is
independent of any run.

Run:  mpirun -n {1,2,4,8} python test_gate4.py
"""
import os
import sys

import numpy as np

from cylinder_grid import cylinder_domain
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO

RTOL = float(os.environ.get("GATE4_RTOL", "1e-9"))
NSTEPS = int(os.environ.get("GATE4_STEPS", "10"))
REF = f"reference/gate4_serial_{RTOL:.0e}_{NSTEPS}.npz"


def build(distributed):
    d, _, _ = cylinder_domain(nz=4)
    if distributed:
        from src.comm_mpi import MPIComm
        d.comm = MPIComm(d)
    d.prepare_geometry()
    m = MultiBlockPISO(d, 1.0 / 100.0, 0.005, 2, RTOL, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False, linear_backend="petsc")
    checkpoint.load(m, "results/fields/cyl_shed_mac.npz")
    return d, m


def check_bc_rows(d, m):
    """The eliminated rows are exactly the geometric wall nodes. Independent of any run."""
    from src.multiblock import face_axis_side, face_slice

    # geometry: mark every node lying on a face the domain calls a wall
    marks = []
    for b, blk in enumerate(d.blocks):
        w = np.zeros(blk.shape, dtype=bool)
        for fid, kind in enumerate(blk.faces):
            if kind == "wall":
                w[face_slice(fid)] = True
        marks.append(w)
    geo = np.concatenate([w.ravel() for w in marks])
    geo_rows = set(np.where(geo)[0].tolist())
    bnd_rows = set(m.bnd.tolist())

    missing = geo_rows - bnd_rows          # a wall node the solver does NOT eliminate
    extra = bnd_rows - geo_rows            # an eliminated row that is not on a wall
    n_int = int(m.interior.size)
    n_bnd = int(m.bnd.size)
    total_ok = (n_int + n_bnd) == d.n_cells
    return len(missing), len(extra), n_bnd, len(geo_rows), total_ok


def trajectory(m, d, n):
    its_p, its_m = [], []
    p0 = m._pcache.total_iterations
    m0 = m._mcache.total_iterations
    for _ in range(n):
        m.step()
        # PER-STEP TOTALS, not the last solve: a step runs 4 pressure solves and 6 momentum
        # solves, and the final one often converges in 0 iterations from a good Picard guess.
        its_p.append(m._pcache.total_iterations - p0)
        its_m.append(m._mcache.total_iterations - m0)
        p0, m0 = m._pcache.total_iterations, m._mcache.total_iterations
    nb = len(d.blocks)
    return ({f"{f}_{b}": arr[b].copy()
             for f, arr in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p))
             for b in range(nb)}, its_p, its_m)


def main():
    from mpi4py import MPI
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

    d, m = build(size > 1)
    miss, extra, n_bnd, n_geo, total_ok = check_bc_rows(d, m)
    ok_bc = (miss == 0 and extra == 0 and total_ok)
    if rank == 0:
        print(f"  [{'PASS' if ok_bc else 'FAIL'}] BC rows by inspection ({size} rank(s)): "
              f"{n_bnd:,} eliminated rows, {n_geo:,} geometric wall nodes, "
              f"{miss} missing, {extra} spurious, interior+bnd == n_cells: {total_ok}")

    if size == 1:
        st, ip, im = trajectory(m, d, NSTEPS)
        os.makedirs("reference", exist_ok=True)
        np.savez_compressed(REF, its_p=np.array(ip), its_m=np.array(im), **st)
        print(f"  serial reference: {NSTEPS} steps, pressure its {ip[:4]}..., "
              f"momentum its {im[:4]}...")
        print(f"  wrote {REF}")
        return 0 if ok_bc else 1

    if not os.path.exists(REF):
        if rank == 0:
            print(f"  reference {REF} missing — run at 1 rank first")
        return 1
    ref = np.load(REF)
    st, ip, im = trajectory(m, d, NSTEPS)
    worst = max(float(np.abs(st[k] - ref[k]).max()) for k in st)
    scale = max(float(np.abs(ref[k]).max()) for k in st)
    rel = MPI.COMM_WORLD.allreduce(worst / max(scale, 1e-30), op=MPI.MAX)
    rm = sum(im) / max(sum(list(ref["its_m"])), 1)
    ok_traj = rel < 1e-12
    ok_its = rm < 2.0
    if rank == 0:
        print(f"  [{'PASS' if ok_traj else 'FAIL'}] {size} ranks, {NSTEPS} steps: max relative "
              f"difference {rel:.3e}  (criterion < 1e-12)")
        print(f"  [{'PASS' if ok_its else 'FAIL'}] momentum iterations {im[:6]}... "
              f"total ratio {rm:.2f}x of serial")
    return 0 if (ok_bc and ok_traj and ok_its) else 1


if __name__ == "__main__":
    sys.exit(main())
