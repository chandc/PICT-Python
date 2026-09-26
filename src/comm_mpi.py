"""Gate 2: blocks distributed across ranks, halos exchanged over MPI.

THE SHAPE OF THE PROBLEM. Padding is RECURSIVE: padding axis 1 needs the neighbour already
padded along axis 0, or the corner ghosts are missing. Distributed, that means computing a
neighbour's level-1 data may require ITS neighbour's level-0 data, which may live on a third
rank. A naive depth-first implementation would chase that chain across ranks and deadlock.

THE FIX IS TO EXCHANGE LEVEL BY LEVEL. The recursion has depth three, over axes 0, 1, 2:

    round 1 (axis 0):  neighbours' level-0 slabs   -> every rank can now build its own level 1
    round 2 (axis 1):  neighbours' level-1 slabs   -> ... level 2
    round 3 (axis 2):  neighbours' level-2 slabs   -> ... level 3

Each rank only ever computes `upto` for blocks IT OWNS. Remote data always arrives as an
already-extracted face slab, never as a block to be padded locally. There is no cross-rank
recursion left, so there is nothing to deadlock on.

THE SECOND HAZARD IS COLLECTIVE MISMATCH, and it is the one that would actually bite. If the
exchange were triggered lazily from inside `pad_field`, a rank owning three blocks would enter
it three times and a rank owning two would enter it twice -- and MPI would hang, in a way that
looks like a solver stall rather than a protocol error. So the exchange is a COLLECTIVE EPOCH:
every rank calls `exchange_fields`/`exchange_coords` exactly once per padding pass, before any
`pad_field` call, and `pad_field` itself performs no communication at all. The rule is
mechanical and therefore checkable: no MPI call may appear below `pad_field`.

THE SCHEDULE IS STATIC, derived from the connection list, which every rank holds in full.
Both ends of every message compute the same schedule from the same data in the same order, so
no negotiation round is needed and send/receive counts match by construction rather than by
agreement.

WHAT THIS GATE DOES NOT DO. The implicit solves still gather to rank 0; only the explicit
operators run distributed. That is Gate 2's stated scope, and Gate 3 replaces the gather.
"""
import numpy as np

from src.comm import Comm, _extract_slab


def contiguous_owners(nblocks, size):
    """Assign blocks to ranks in contiguous runs.

    Contiguous rather than round-robin because block index order follows mesh locality in every
    domain this repo builds -- the cylinder's 16 blocks march around the annulus, the channel's
    split along x. Round-robin would give every rank a maximally remote neighbour set and turn
    a two-message exchange into an all-to-all.
    """
    base, extra = divmod(nblocks, size)
    owners, b = [], 0
    for r in range(size):
        n = base + (1 if r < extra else 0)
        owners.extend([r] * n)
        b += n
    return tuple(owners)


class MPIComm(Comm):
    """Distributed ownership with a level-synchronous, slab-based halo exchange."""

    def __init__(self, domain, mpi=None, owners=None):
        if mpi is None:
            from mpi4py import MPI
            mpi = MPI.COMM_WORLD
        self.mpi = mpi
        nb = len(domain.blocks)
        if owners is None:
            owners = contiguous_owners(nb, mpi.size)
        super().__init__(nb, owners, mpi.rank, mpi.size)
        self.d = domain
        # SEPARATE CACHES, and not for tidiness. The two exchanges key on the same
        # (block, level, axis, side, width) tuple, so a single dict lets the coordinate pass
        # clear and then overwrite the field pass -- after which pad_field is handed COORDINATE
        # data under a field key. It does not fail cleanly either: coordinates carry three
        # components where a field carries one, so the error surfaces deep in an orientation
        # transpose as "axes don't match array", pointing at the seam logic rather than at the
        # cache. Had the component counts happened to agree it would not have surfaced at all.
        self._slabs = {"field": {}, "coords": {}}
        self.bytes_moved = 0
        self.halo_cells = 0
        # COMMUNICATION TIME, split from the gather. Gate 6 requires the communication fraction
        # to be KNOWN -- it sets no target for it, only that it be measured -- and until now it
        # was buried in an "other" bucket alongside assembly. The two are separated because they
        # scale oppositely: assembly work per rank FALLS with rank count while exchange volume
        # RISES, so a single bucket hides which one is responsible when "other" stops improving.
        self.t_exchange = 0.0
        self.t_gather = 0.0
        self.n_gather = 0

    # ------------------------------------------------------------------ schedule
    def _pairs(self, axis):
        """(receiver, sender, sender_axis, sender_side) for connections received on `axis`.

        Deterministic order -- blocks ascending, side ascending -- so both ends of every
        message enumerate identically. Faces whose two blocks share an owner are skipped:
        those are served by the local recursion and must not generate a message.
        """
        from src.multiblock import face_axis_side, face_id
        out = []
        for b in range(self.nblocks):
            for side in (0, 1):
                nb = self.d._neighbour_of(b, face_id(axis, side))
                if nb is None:
                    continue
                ob, ofid, _, _ = nb
                if self.owner(b) == self.owner(ob):
                    continue
                oaxis, oside = face_axis_side(ofid)
                out.append((b, ob, oaxis, oside))
        return out

    # ------------------------------------------------------------------ the epoch
    def _run(self, upto, width, ncomp, kind):
        """Three rounds of level-synchronous slab exchange. Collective; call once per pass."""
        import time as _t
        _t0 = _t.perf_counter()
        cache = self._slabs[kind]
        cache.clear()
        for r in (1, 2, 3):
            axis = r - 1
            k = r - 1
            sends, want = {}, {}
            for b, ob, oaxis, oside in self._pairs(axis):
                key = (ob, k, oaxis, oside, width)
                if self.owner(b) == self.rank:
                    want.setdefault(self.owner(ob), []).append(key)
                if self.owner(ob) == self.rank:
                    arrs, olo, ohi = upto(ob, k)
                    if ncomp == 1:
                        arrs = [arrs]
                    lay = _extract_slab(arrs, olo, ohi, oaxis, oside, width)
                    sends.setdefault(self.owner(b), []).append(
                        (key, [np.ascontiguousarray(l) for l in lay], list(olo), list(ohi)))
            reqs = [self.mpi.isend(v, dest=dst, tag=100 + r) for dst, v in sorted(sends.items())]
            for src in sorted(want):
                for key, lay, olo, ohi in self.mpi.recv(source=src, tag=100 + r):
                    cache[key] = (lay if ncomp > 1 else lay[0], olo, ohi)
                    self.messages += 1
                    self.bytes_moved += sum(l.nbytes for l in lay)
                    self.halo_cells += sum(l.size for l in lay)
            if reqs:
                from mpi4py import MPI
                MPI.Request.waitall(reqs)
        self.t_exchange += _t.perf_counter() - _t0

    def exchange_fields(self, upto, width):
        """COLLECTIVE. Prime every remote field slab this rank will need this pass."""
        self._run(upto, width, ncomp=1, kind="field")

    def exchange_coords(self, upto, width):
        """No-op: coordinates are replicated on every rank. See `fetch_coords_slab`."""
        return

    # ------------------------------------------------------------------ the fetches
    def fetch_field_slab(self, b, k, oaxis, oside, width, local):
        """Local when this rank HAS block b's field, exchanged when it does not.

        Ownership is the wrong test here, and using it was a bug. What matters is whether the
        FIELD DICT BEING PADDED carries block b -- which the global assembly path guarantees,
        because `map_blocks` gathered before calling it. Keying on ownership instead made a rank
        demand an exchanged slab for a block whose data was sitting in the dict it was padding.
        """
        if b in getattr(local, "covers", ()) or self.is_local(b):
            return Comm.fetch_field_slab(self, b, k, oaxis, oside, width, local)
        return self._cached(b, k, oaxis, oside, width, "field")

    def fetch_coords_slab(self, b, k, oaxis, oside, width, local):
        """ALWAYS LOCAL. Coordinates are replicated, so nothing has to move.

        THE MESH IS NOT DISTRIBUTED AT THIS GATE, and once that is said plainly the coordinate
        exchange disappears: every rank constructs the same `Domain` and therefore already holds
        every block's x, y and z. What is distributed is the FIELD data -- u, v, w, p -- which
        exists only where the solver computed it. Exchanging coordinates was moving data every
        rank already had.

        It also removes a real failure rather than just waste. The solver's constructor builds
        metrics for EVERY block, not only local ones, so a coordinate path that served only
        local blocks failed during construction with a cache miss -- before a single step ran.
        Serving them locally is both correct and unconditional.

        Distributing the mesh itself is a separate question, and not one Gate 2 poses.
        """
        return Comm.fetch_coords_slab(self, b, k, oaxis, oside, width, local)

    def _cached(self, b, k, oaxis, oside, width, kind):
        cache = self._slabs[kind]
        key = (b, k, oaxis, oside, width)
        if key not in cache:
            # A WIDER SLAB CONTAINS A NARROWER ONE. Layers are stored nearest-first, so the
            # first `width` of them ARE the width-`width` slab; olo/ohi describe the sender's
            # tangential padding and do not depend on width at all. The audit of one solver
            # step found only widths 1 and 2 in use, so serving 1 from 2 halves the number of
            # exchanges rather than merely avoiding a miss -- one message per field per pass
            # instead of two.
            for w in sorted(w for (bb, kk, ax, sd, w) in cache
                            if (bb, kk, ax, sd) == (b, k, oaxis, oside) and w > width):
                lay, olo, ohi = cache[(b, k, oaxis, oside, w)]
                sub = [l[:width] for l in lay] if isinstance(lay, list) else lay[:width]
                cache[key] = (sub, olo, ohi)
                break
        try:
            lay, olo, ohi = cache[key]
        except KeyError:
            raise RuntimeError(
                f"rank {self.rank}: no exchanged {kind} slab for block {b} level {k} "
                f"face ({oaxis},{oside}) width {width}. The collective exchange must run "
                f"BEFORE any pad_* call in a pass -- a lazy fetch here would deadlock, "
                f"because ranks own different numbers of blocks and would enter it a "
                f"different number of times.") from None
        return lay, olo, ohi

    def gather_blocks(self, local):
        """ALLGATHER rather than gather-to-0, deliberately.

        The plan says the implicit solves "gather to rank 0". Gathering to one rank then
        broadcasting the solution is the same traffic in two phases, and adds a scatter step
        whose indexing is a fresh chance to be wrong. Giving every rank the full field instead
        lets the existing global assembly run unchanged and IDENTICALLY everywhere -- same data,
        same order, so the matrix is bit-for-bit the same on every rank and the solves cannot
        disagree. That is worth more at this gate than the memory it costs, because it makes
        "distributed equals serial" a statement about the halo exchange alone.

        It is emphatically not the end state: holding every block on every rank is exactly what
        Gate 3 exists to remove.
        """
        import time as _t
        _t0 = _t.perf_counter()
        out = {}
        for part in self.mpi.allgather({b: local[b] for b in self.local_blocks()
                                        if b in local}):
            out.update(part)
        self.t_gather += _t.perf_counter() - _t0
        self.n_gather += 1
        return out

    def __repr__(self):
        return (f"MPIComm(nblocks={self.nblocks}, rank={self.rank}/{self.size}, "
                f"local={self.local_blocks()}, messages={self.messages})")
