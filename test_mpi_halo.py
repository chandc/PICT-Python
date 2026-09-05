"""Gate 2: the distributed halo exchange moves the RIGHT data to the RIGHT place.

Run under mpirun:  mpirun -n {1,2,4,8} python test_mpi_halo.py

THIS IS THE SERIAL HALO-CONTENT TEST, DISTRIBUTED. Same analytic field, same requirement --
every halo cell equals f at that cell's own coordinate -- but with blocks owned by different
ranks, so a halo can only be right if the exchange moved the correct neighbour's correct face,
in the correct orientation, with the correct tangential extent. A test that merely compared
against a serial run would pass on a byte-identical exchange of the WRONG face, provided the
serial reference had the same bug.

WHAT EACH RANK CHECKS. Only its own blocks: that is the whole point of distributing them. The
verdict is then reduced across ranks, so one bad halo on one rank fails the run rather than
being averaged away.

N = 1 IS NOT A TRIVIAL CASE HERE. At one rank every block is local, no message is sent, and the
result must be bitwise what the serial path produces -- which is what makes N = 1 the control
that separates "the exchange is wrong" from "the distribution is wrong".

THE MANGLE REVERSES LAYER ORDER within the slab -- "nearest-first" becoming "furthest-first".
That is the archetypal halo bug: right size, right rank, right block, right face, right bytes,
wrong arrangement, and the resulting field is smooth. Only a content check catches it.

THE COORDINATES ARE CHECKED INDEPENDENTLY, and that is not belt-and-braces. Comparing a padded
FIELD against f evaluated at padded COORDINATES is blind to any bug that corrupts both the same
way -- and since both travel through the same exchange, symmetric corruption is the LIKELY case,
not the exotic one. This was found the hard way: a mangle that reversed layer order in every
exchanged slab produced a perfectly passing run, because the field halo and the coordinate halo
were wrong identically and the errors cancelled. `check_coords` closes it by comparing the padded
coordinates against the closed form a uniform periodic box must have -- a ramp of constant pitch
that continues straight through every seam -- which depends on no other part of the exchange.

THREE EARLIER MANGLES WERE WRONG AND ARE RECORDED BECAUSE THEY FAILED INSTRUCTIVELY. Flipping the
requested face SIDE changed the schedule KEY, so the receiver asked for a slab nobody had sent
and got a clean RuntimeError instead of bad data -- a crash is not the failure mode being
tested. And at N = 1 no mangle of the exchange can do anything at all, because no message is
sent: every block is local. So the mangle is SKIPPED, not passed, at one rank -- reporting it as
a pass would be claiming evidence that the run cannot produce. And corrupting BOTH fields and
coordinates cancelled, as above; the mangle now corrupts fields only.
"""
import sys

import numpy as np

from src.comm_mpi import MPIComm
from src.domains import periodic_box
from test_halo_content import f

WIDTH = 2
L = 2 * np.pi


def build(ntot=12, n_split=4, mangle=False):
    d = periodic_box(ntot, n_split, L=L)
    comm = MPIComm(d)
    if mangle:
        class Bad(type(comm)):
            def _run(self, upto, width, ncomp, kind):
                super()._run(upto, width, ncomp, kind)
                if kind != "field":
                    return          # FIELDS ONLY: corrupting both cancels, see the docstring
                # reverse the layer order of every received slab: nearest-first becomes
                # furthest-first. Same bytes, same size, same sender -- wrong arrangement.
                c = self._slabs[kind]
                for key, (lay, olo, ohi) in list(c.items()):
                    c[key] = (lay[::-1], olo, ohi)
        comm = Bad(d)
    d.comm = comm
    return d, comm


def run(n_split, mangle=False):
    d, comm = build(n_split=n_split, mangle=mangle)
    nb = len(d.blocks)
    # each rank holds ONLY its own blocks' data
    fields = {b: f(d.blocks[b].x, d.blocks[b].y, d.blocks[b].z, (L, L, L))
              for b in comm.local_blocks()}
    # every block must be addressable by the recursion, but remote ones are served from the
    # exchange cache and their entries are never read; None makes an accidental read fail loudly
    for b in range(nb):
        fields.setdefault(b, None)

    d.prepare_geometry(width=WIDTH)
    d.exchange_halos(fields, width=WIDTH)

    worst = 0.0
    cells = 0
    for b in comm.local_blocks():
        got, lo, hi = d.pad_field(b, fields, width=WIDTH)
        X, Y, Z, _, _ = d.pad_coords(b, width=WIDTH)
        want = f(X, Y, Z, (L, L, L))
        n = got.shape
        mask = np.ones(n, bool)
        mask[tuple(slice(lo[a], n[a] - hi[a]) for a in range(3))] = False
        worst = max(worst, float(np.abs(got - want)[mask].max()))
        cells += int(mask.sum())
    tot = comm.mpi.allreduce(worst, op=__import__("mpi4py").MPI.MAX)
    ncell = comm.mpi.allreduce(cells)
    nmsg = comm.mpi.allreduce(comm.messages)
    nbytes = comm.mpi.allreduce(comm.bytes_moved)
    return tot, ncell, nmsg, nbytes, comm


def check_coords(n_split):
    """Padded coordinates against closed form, independent of the field path.

    A uniform periodic box has a padded coordinate array that is a ramp of constant pitch h,
    continuing straight through every seam -- the connection shift exists precisely so it does.
    So the expected value at padded index i along axis a is x0 + (i - lo[a]) * h, with x0 the
    block's own first node. Nothing about the field exchange enters this, which is the point.
    """
    d, comm = build(n_split=n_split)
    dummy = {b: np.zeros(d.blocks[b].shape) for b in comm.local_blocks()}
    for b in range(len(d.blocks)):
        dummy.setdefault(b, None)
    d.prepare_geometry(width=WIDTH)
    d.exchange_halos(dummy, width=WIDTH)
    worst = 0.0
    for b in comm.local_blocks():
        blk = d.blocks[b]
        X, Y, Z, lo, hi = d.pad_coords(b, width=WIDTH)
        for a, (got, first) in enumerate(((X, blk.x), (Y, blk.y), (Z, blk.z))):
            n = got.shape[a]
            idx = np.arange(n) - lo[a]
            shape = [1, 1, 1]
            shape[a] = n
            origin = first.reshape(-1)[0] if a == 0 else None
            # take the origin from the unpadded block along this axis
            sl = [0, 0, 0]
            origin = {0: blk.x, 1: blk.y, 2: blk.z}[a][tuple(sl)]
            want = origin + idx.reshape(shape) * blk.h[a]
            worst = max(worst, float(np.abs(got - want).max()))
    tot = comm.mpi.allreduce(worst, op=__import__("mpi4py").MPI.MAX)
    return tot


def check_metrics(n_split=4):
    """Distributed metrics must equal serial metrics, block for block.

    THE SHARPEST TEST OF THE GEOMETRY EXCHANGE, because metrics are DERIVED from padded
    coordinates: the Jacobian and the covariant basis are differences of neighbouring ghost
    coordinates, so a halo that is off by one cell, mis-oriented, or missing its period shift
    produces a wrong Jacobian rather than a wrong-looking coordinate. And a wrong Jacobian does
    not announce itself -- it rescales the volume of cells near a seam, which looks like a
    slightly different mesh rather than like a bug.

    The serial reference is built on each rank from the same domain with an ordinary serial
    Comm, so the comparison needs no communication and no stored baseline.
    """
    d, comm = build(n_split=n_split)
    d.prepare_geometry(width=WIDTH)

    ref = periodic_box(12, n_split, L=L)          # serial Comm by default
    worst_J, worst_m = 0.0, 0.0
    for b in comm.local_blocks():
        Jd, md = d.block_metrics_cached(b)
        Jr, mr = ref.block_metrics_cached(b)
        worst_J = max(worst_J, float(np.abs(Jd - Jr).max()))
        # block_metrics returns (J, dict); zipping the dicts iterates their KEYS, which
        # silently compares strings. Compare by key.
        assert set(md) == set(mr), "metric dictionaries disagree on their keys"
        worst_m = max(worst_m,
                      max(float(np.abs(np.asarray(md[k]) - np.asarray(mr[k])).max())
                          for k in md))
    wJ = comm.mpi.allreduce(worst_J, op=__import__("mpi4py").MPI.MAX)
    wm = comm.mpi.allreduce(worst_m, op=__import__("mpi4py").MPI.MAX)
    return wJ, wm


def main():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    ok = []
    if rank == 0:
        print("=" * 78)
        print(f"  Gate 2 — distributed halo content, {size} rank(s)")
        print("=" * 78)
    for n_split in (4, 12):
        err, cells, msgs, nbytes, comm = run(n_split)
        good = err < 1e-13
        ok.append(good)
        if rank == 0:
            block_cells = 12 ** 3
            print(f"  [{'PASS' if good else 'FAIL'}] {n_split} blocks / {size} ranks: "
                  f"{cells:,} halo cells, max error {err:.2e}, {msgs} messages, "
                  f"{nbytes/1024:.1f} KiB")
            print(f"           halo volume {100*cells/(n_split*block_cells):.1f}% of "
                  f"interior ({n_split*block_cells:,} cells)")
    wJ, wm = check_metrics(4)
    mgood = wJ < 1e-13 and wm < 1e-13
    ok.append(mgood)
    if rank == 0:
        print(f"  [{'PASS' if mgood else 'FAIL'}] distributed metrics == serial metrics: "
              f"max |dJ| {wJ:.2e}, max |d(basis)| {wm:.2e}")

    cerr = check_coords(4)
    cgood = cerr < 1e-12
    ok.append(cgood)
    if rank == 0:
        print(f"  [{'PASS' if cgood else 'FAIL'}] padded coordinates vs closed form "
              f"(independent of the field path): max error {cerr:.2e}")
    if size > 1:
        err, _, _, _, _ = run(4, mangle=True)
        good = err > 1e-13
        ok.append(good)
        if rank == 0:
            print(f"  [{'PASS' if good else 'FAIL'}] reversed-layer mangle DETECTED: "
                  f"max error {err:.2e}")
    elif rank == 0:
        print("  [SKIP] mangle not applicable at 1 rank: no message is sent, so nothing "
              "the exchange does can be corrupted")
        print("=" * 78)
        print(f"  {sum(ok)}/{len(ok)} checks passed")
        print("=" * 78)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
