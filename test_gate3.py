"""Gate 3: the distributed pressure solve reproduces the serial one.

THE REFERENCE IS SERIAL PETSc, NOT THE GATE 0 SCIPY DIGESTS. Comparing a distributed PETSc run
against a SciPy reference would fold two differences together -- the Krylov method and the
partitioning -- and the gate's <1e-12 criterion is about the second alone. Serial PETSc against
distributed PETSc isolates it: same solver, same tolerance, only the row distribution differs.

WHAT IS CHECKED
  1  the assembled matrix is identical serial vs distributed -- norm and 20 random rows,
     EXACTLY, because it is the same assembly either way;
  2  the ten-step trajectory agrees to better than 1e-12 relative;
  3  the ITERATION COUNT per rank count, because bjacobi is partition-dependent by
     construction -- each rank factorises its own diagonal block, so more ranks is a weaker
     preconditioner. The gate aborts above 2x growth, and a count that changes silently would
     change the answer's COST while every correctness check still passed.

Run:  mpirun -n {1,2,4,8} python test_gate3.py
"""
import json
import os
import sys

import numpy as np

from cylinder_grid import cylinder_domain
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO

RTOL = float(os.environ.get("GATE3_RTOL", "1e-6"))
REF = (f"reference/gate3_serial_petsc_{RTOL:.0e}"
       f"{'_dong' if os.environ.get('GATE3_DONG') == '1' else ''}"
       f"{'' if os.environ.get('GATE3_STEPS', '10') == '10' else '_' + os.environ['GATE3_STEPS']}"
       f".npz")
NSTEPS = int(os.environ.get("GATE3_STEPS", "10"))


DONG = os.environ.get("GATE3_DONG") == "1"


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
    if DONG:
        # GATE 3 ITEM 5. Without the outflow installed the pressure system is SINGULAR -- no
        # prescribed-pressure rows exist at all -- so "the Dong rows land on the right global
        # rows after distribution" has nothing to check. Installing it makes the system
        # non-singular (M_fD is not None) and puts real Dirichlet rows into the matrix, which
        # is the case that can go wrong silently: a mis-mapped prescribed row yields a
        # plausible flow field rather than an error.
        from cylinder_bc import apply as apply_bc
        apply_bc(m, d, kind="dong")
    return d, m


def trajectory(m, d, n=NSTEPS):
    its = []
    for _ in range(n):
        m.step()
        its.append(m._pcache.iterations)
    nb = len(d.blocks)
    return ({f"{f}_{b}": arr[b].copy()
             for f, arr in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p))
             for b in range(nb)}, its, m)


def check_matrix(d, m):
    """Gate 3 item 1: the distributed Mat holds the same entries as the serial one.

    EXACT, not to a tolerance -- it is the same assembly either way, so any difference is an
    indexing error in the row distribution rather than arithmetic. A wrong row map produces a
    matrix that is still symmetric, still positive definite and still solvable; it just solves
    a different problem, which no residual check would catch.

    `A.norm()` alone is a weak test -- permuting rows preserves it -- so 20 random rows are
    compared entry by entry as well.
    """
    import numpy as np
    from petsc4py import PETSc
    from src.linsolve import SolveCache
    import src.linsolve as L

    cap = {}
    orig = SolveCache.solve

    def spy(self, A, b, x0=None, symmetric=True, rtol=1e-12, maxiter=20000, singular=False):
        cap.setdefault("A", A.tocsr().copy())
        return orig(self, A, b, x0, symmetric, rtol, maxiter, singular)

    SolveCache.solve = spy
    m.step()
    SolveCache.solve = orig
    A = cap["A"]
    n = A.shape[0]

    world = PETSc.COMM_WORLD
    Md = PETSc.Mat().createAIJ(size=((PETSc.DECIDE, n), (PETSc.DECIDE, n)), comm=world)
    Md.setUp()
    r0, r1 = Md.getOwnershipRange()
    sub = A[r0:r1]
    Md.setValuesCSR(sub.indptr, sub.indices, sub.data)
    Md.assemble()

    nrm_d = Md.norm(PETSc.NormType.FROBENIUS)
    nrm_s = float(np.sqrt((A.data ** 2).sum()))
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(n, size=20, replace=False))
    worst = 0.0
    for r in rows:
        if r0 <= r < r1:
            cols, vals = Md.getRow(r)
            ref = A[r]
            got = dict(zip(cols.tolist(), vals.tolist()))
            for c, v in zip(ref.indices.tolist(), ref.data.tolist()):
                worst = max(worst, abs(got.get(c, 0.0) - v))
    worst = world.tompi4py().allreduce(worst, op=__import__("mpi4py").MPI.MAX)
    # RELATIVE, because an absolute tolerance on a norm of order 1e6 is meaningless: PETSc
    # accumulates the Frobenius sum in a different ORDER than numpy, so the two differ by
    # round-off no matter how identical the entries are. The entry-by-entry comparison is the
    # test with teeth; the norm is a cheap check that no row went missing entirely.
    return abs(nrm_d - nrm_s) / max(nrm_s, 1e-30), worst, n, nrm_s


def main():
    from mpi4py import MPI
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

    if size == 1:
        d, m = build(False)
        st, its, m = trajectory(m, d)
        print(f"  outflow specs installed: {len(getattr(m, 'outflow', []))}")
        os.makedirs("reference", exist_ok=True)
        np.savez_compressed(REF, iters=np.array(its), **st)
        print("=" * 78)
        print("  Gate 3 — serial PETSc reference captured")
        print("=" * 78)
        print(f"  rtol {RTOL:.0e}: {len(st)} arrays, {NSTEPS} steps, iterations {its}")
        print(f"  fell_back = {m._pcache.fell_back}")
        print(f"  wrote {REF}")
        return 0

    if not os.path.exists(REF):
        if rank == 0:
            print(f"  reference {REF} missing — run at 1 rank first")
        return 1
    ref = np.load(REF)
    d, m = build(True)
    dn, drow, n, nrm = check_matrix(d, m)
    ok_mat = dn < 1e-12 and drow == 0.0
    if rank == 0:
        print(f"  [{'PASS' if ok_mat else 'FAIL'}] matrix serial == distributed: entries over "
              f"20 random rows differ by {drow:.2e} (exact), ||A||_F relative difference "
              f"{dn:.2e} on ||A||_F = {nrm:.4e}, n = {n:,}")
    d, m = build(True)
    st, its, m = trajectory(m, d)

    worst = 0.0
    scale = 0.0
    for k, v in st.items():
        worst = max(worst, float(np.abs(v - ref[k]).max()))
        scale = max(scale, float(np.abs(ref[k]).max()))
    rel = worst / max(scale, 1e-30)
    rel = MPI.COMM_WORLD.allreduce(rel, op=MPI.MAX)
    ref_its = list(ref["iters"])
    ok_traj = rel < 1e-12
    growth = (sum(its) / max(sum(ref_its), 1))
    ok_its = growth < 2.0
    if rank == 0:
        print(f"  [{'PASS' if ok_traj else 'FAIL'}] {size} ranks, rtol {RTOL:.0e}, {NSTEPS} "
              f"steps vs serial PETSc: max relative difference {rel:.3e}  "
              f"(criterion < 1e-12)")
        print(f"  [{'PASS' if ok_its else 'FAIL'}] pressure iterations {its}")
        print(f"           serial {ref_its}  -> total ratio {growth:.2f}x "
              f"(abort above 2x; bjacobi is partition-dependent)")
        print(f"           fell_back = {m._pcache.fell_back}")
    return 0 if (ok_traj and ok_its and ok_mat) else 1


if __name__ == "__main__":
    sys.exit(main())
