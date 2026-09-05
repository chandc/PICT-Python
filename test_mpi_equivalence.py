"""Gate 1: the Comm abstraction changed nothing. Bitwise, not approximately.

WHY BITWISE AND NOT 1e-12. At this gate nothing has been distributed and no arithmetic has been
reordered -- the two cross-block reads simply go through a method that calls what they used to
call directly. So the only correct answer is the SAME BITS. A tolerance here would hide exactly
the failure this gate exists to catch: a refactor that quietly changed an operation order, which
at Gate 3 would be indistinguishable from an MPI reduction-order difference and far harder to
attribute. Gate 0 established that the serial code is bitwise reproducible against itself
(640/640 arrays across independent captures), so this criterion is achievable rather than
aspirational.

The comparison is against DIGESTS committed at Gate 0, not against a freshly captured reference.
Comparing a run against a reference produced by the same run's code proves only self-consistency.

The mangles must fail, or the test measures nothing:
  * perturbing one ghost cell by one ULP -- the smallest defect the criterion must catch;
  * displacing a fetched field by one period, which is the coordinates-versus-fields confusion
    the two separate methods exist to prevent, expressed at the level where it would actually
    do damage. It is large, smooth, and entirely plausible-looking in a plot.

A THIRD MANGLE WAS TRIED AND DISCARDED, and the reason is worth recording. Routing
`fetch_padded_coords` through `fetch_padded_field` changes NOTHING at this gate, because the two
methods have identical bodies here -- both just call the caller's recursion. The distinction they
guard lives in `_ghost_coords` and `_ghost_field`, not in `Comm`. They are separate so that
Gate 2 CAN implement them differently, which means no behavioural mangle can tell them apart
yet. The property that does hold now is structural, so `check_routing` asserts it structurally:
each ghost helper calls its own method, and no third cross-block read exists.
"""
import hashlib
import json
import os
import sys

import numpy as np

from src import checkpoint
from src.comm import Comm

REF = "reference/gate0/cylinder_re100.digests.json"


def _build(comm=None, distributed=False):
    from cylinder_grid import cylinder_domain
    from src.piso_multiblock import MultiBlockPISO
    d, _, _ = cylinder_domain(nz=4)
    if distributed:
        from src.comm_mpi import MPIComm
        d.comm = MPIComm(d)
    if comm is not None:
        d.comm = comm
    d.prepare_geometry()
    m = MultiBlockPISO(d, 1.0 / 100.0, 0.005, 2, 1e-6, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False)
    checkpoint.load(m, "results/fields/cyl_shed_mac.npz")
    return d, m


def _digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).hexdigest()


def _trajectory(m, d, nsteps=10):
    out = {}
    nb = len(d.blocks)
    for i in range(nsteps):
        m.step()
        for b in range(nb):
            for f, arr in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p)):
                out[f"s{i+1}_{f}_{b}"] = _digest(arr[b])
    return out


def check_ownership():
    """Gate 1 criterion: every block owned by rank 0, and no message sent."""
    d, _ = _build()
    c = d.comm
    ok = (c.size == 1 and c.rank == 0
          and all(c.owner(b) == 0 for b in range(len(d.blocks)))
          and all(c.is_local(b) for b in range(len(d.blocks)))
          and c.local_blocks() == tuple(range(len(d.blocks)))
          and c.messages == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] serial ownership: {len(d.blocks)} blocks all on "
          f"rank 0 of {c.size}, {c.messages} messages sent")
    return ok


def check_bitwise(mangle=None):
    """Ten steps against the Gate 0 digests."""
    if not os.path.exists(REF):
        print(f"  [FAIL] reference {REF} missing — run gate0_baseline.py")
        return False
    ref = json.load(open(REF))
    comm = None
    if mangle == "ulp":
        class M(Comm):
            def fetch_field_slab(self, b, k, oaxis, oside, width, local):
                lay, olo, ohi = super().fetch_field_slab(b, k, oaxis, oside, width, local)
                lay = lay.copy()
                lay.flat[0] = np.nextafter(lay.flat[0], np.inf)   # ONE ULP, ONE CELL
                return lay, olo, ohi
        comm = M(16)
    elif mangle == "period":
        class M(Comm):
            def fetch_field_slab(self, b, k, oaxis, oside, width, local):
                lay, olo, ohi = super().fetch_field_slab(b, k, oaxis, oside, width, local)
                # a period-sized displacement: what applying the COORDINATE shift to a field
                # would do. Smooth, large, and plausible -- the failure mode that motivated
                # keeping the two paths apart in the first place.
                return lay + 1.0, olo, ohi
        comm = M(16)
    d, m = _build(comm)
    # A mangle that survives three steps will not be revealed by ten; the clean comparison is
    # the one that needs the full trajectory.
    got = _trajectory(m, d, nsteps=10 if mangle is None else 3)
    bad = [k for k, v in got.items() if ref.get(k) != v]
    ok = (len(bad) == 0) if mangle is None else (len(bad) > 0)
    if mangle is None:
        print(f"  [{'PASS' if ok else 'FAIL'}] 10 steps bitwise identical to the Gate 0 "
              f"reference: {len(got)-len(bad)}/{len(got)} digests match")
        if bad:
            print(f"          first divergence at step {min(int(k[1:].split('_')[0]) for k in bad)}")
    else:
        first = min(int(k[1:].split('_')[0]) for k in bad) if bad else None
        print(f"  [{'PASS' if ok else 'FAIL'}] the '{mangle}' mangle is DETECTED: "
              f"{len(bad)}/{len(got)} digests differ, first at step {first}")
    return ok


def check_routing():
    """Structural: each ghost helper calls its OWN Comm method, and there is no third read.

    This is the plan's "seam call sites reduced to routing through 2 methods, verified by
    grep" criterion, made executable. It is checked on the source because at this gate the two
    methods are behaviourally identical and no runtime test can separate them.
    """
    src = open("src/multiblock.py").read()
    body = {}
    for name in ("_ghost_coords", "_ghost_field"):
        i = src.index(f"def {name}(")
        j = src.find("\n    def ", i + 1)
        body[name] = src[i:j if j > 0 else len(src)]
    ok = ("fetch_coords_slab" in body["_ghost_coords"]
          and "fetch_field_slab" not in body["_ghost_coords"]
          and "fetch_field_slab" in body["_ghost_field"]
          and "fetch_coords_slab" not in body["_ghost_field"])
    # and no cross-block read has escaped the abstraction
    strays = [ln.strip() for ln in src.splitlines()
              if "upto(ob" in ln or "upto(nb" in ln]
    ok = ok and not strays
    print(f"  [{'PASS' if ok else 'FAIL'}] routing: _ghost_coords -> fetch_coords_slab, "
          f"_ghost_field -> fetch_field_slab, {len(strays)} unrouted cross-block reads")
    if strays:
        for t in strays:
            print(f"          stray: {t}")
    return ok


def check_distributed():
    """GATE 2's CENTRAL CRITERION: ten real solver steps, distributed, against the Gate 0
    digests.

    Everything distributed so far has been the analytic halo test -- correct, and not the same
    thing. This runs the actual PISO trajectory with blocks owned by different ranks, which is
    the first time the exchange, the gather and the collective ordering are all exercised
    together by the code that will use them. A collective-ordering error shows up here as a
    HANG, not a wrong number, which is why it has to be run under a timeout.
    """
    from mpi4py import MPI
    ref = json.load(open(REF))
    d, m = _build(distributed=True)
    got = _trajectory(m, d)
    bad = [k for k, v in got.items() if ref.get(k) != v]
    nbad = MPI.COMM_WORLD.allreduce(len(bad), op=MPI.MAX)
    ok = nbad == 0
    if MPI.COMM_WORLD.rank == 0:
        print(f"  [{'PASS' if ok else 'FAIL'}] {MPI.COMM_WORLD.size} ranks, 10 solver steps "
              f"bitwise identical to the Gate 0 reference: {len(got)-nbad}/{len(got)} digests "
              f"match  (owners {d.comm.owners[:4]}..., {d.comm.messages} messages)")
    return ok


def main():
    from mpi4py import MPI
    if MPI.COMM_WORLD.size > 1:
        ok = check_distributed()
        return 0 if ok else 1
    print("=" * 78)
    print("  Gate 1 — the Comm abstraction is behaviourally inert")
    print("=" * 78)
    r = [check_ownership(), check_routing(), check_bitwise(),
         check_bitwise("ulp"), check_bitwise("period")]
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
