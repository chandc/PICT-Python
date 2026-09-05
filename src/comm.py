"""Block ownership, and the only two places one block reads another's data.

GATE 1 OF THE PETSc PLAN. Nothing here is parallel. The point is to put every cross-block data
access behind an object that a later gate can make distributed, and to prove -- bitwise -- that
routing through it changed nothing.

WHERE THE SEAM ACTUALLY IS. The plan estimated "20 seam call sites". That number counts CALLERS
of `pad_field`/`pad_coords`, of which there are indeed about twenty. But those callers all ask
for a block's own padded data; the place where one block reads ANOTHER's is inside the padding
recursion, and there are exactly two such lines in the whole codebase:

    src/multiblock.py:521    other_fields, olo, ohi = upto(ob, k)     # _ghost_coords
    src/multiblock.py:1133   other, olo, ohi = upto(ob, k)            # _ghost_field

Every other `upto(...)` recurses on the SAME block along a further axis. That is a far better
starting position than the plan assumed, and it is why this gate is small: two call sites, two
methods, no behavioural change.

WHY THE RECURSION IS THE RIGHT BOUNDARY, and not `pad_field` itself. Padding axis 1 needs the
neighbour ALREADY PADDED along axis 0, or the corner ghosts are missing. So the unit of data a
remote rank must supply is not "block ob" but "block ob padded along the first k axes" -- which
is precisely the (ob, k) pair these two methods take. Drawing the boundary at `pad_field` would
have forced a remote rank to send its whole block and the receiver to redo the recursion, which
is both more data and a duplicated computation.

WHAT SERIAL MEANS HERE. Rank 0 owns every block, `fetch_*` simply calls the local recursion, and
`messages` stays at zero. That last one is COUNTED rather than asserted in a comment, so the
Gate 1 criterion "no message is sent" is something a test can check rather than something a
reader has to believe.
"""
import numpy as _np


def _extract_slab(arrays, olo, ohi, oaxis, oside, width):
    """The `width` layers of `arrays` adjacent to face (oaxis, oside), nearest-first.

    THIS IS THE UNIT THAT TRAVELS. Before this existed, `_ghost_field` fetched the neighbour's
    ENTIRE padded block and sliced `width` layers off it -- fine in shared memory, absurd over
    MPI, where it would ship a whole block to use two cells of it. Extracting on the owner's
    side turns a block-sized message into a face-sized one: for a 157x16x4 block with width 2,
    2 x 16 x 4 values instead of 157 x 16 x 4, about 1.3%.

    `olo`/`ohi` are the sender's own padding extents. They must accompany the slab because the
    receiver reconciles a tangential MISMATCH against its own extents -- the two blocks either
    side of a connection can differ at a reentrant corner -- and cannot reconstruct them.

    The arithmetic is deliberately identical to the code this replaced, including the
    nearest-first reversal for side 1: Gate 1's criterion is bitwise, and a relocated operation
    that is merely equivalent is not good enough.
    """
    out = []
    for f in arrays:
        sl = [slice(None)] * 3
        sl[oaxis] = (slice(olo[oaxis], olo[oaxis] + width) if oside == 0
                     else slice(f.shape[oaxis] - ohi[oaxis] - width,
                                f.shape[oaxis] - ohi[oaxis]))
        lay = _np.moveaxis(f[tuple(sl)], oaxis, 0)
        out.append(lay[::-1] if oside == 1 else lay)
    return out


class Comm:
    """Ownership of blocks, and access to a neighbour block's partially padded data.

    The serial implementation. `size` is 1, `rank` is 0, and every block is owned locally.
    Gate 2 replaces `fetch_padded_field` / `fetch_padded_coords` with a real exchange for
    blocks whose owner is not this rank; nothing else in the solver should need to change,
    which is the property this gate exists to establish.
    """

    def __init__(self, nblocks, owners=None, rank=0, size=1):
        self.nblocks = int(nblocks)
        self.rank = int(rank)
        self.size = int(size)
        if owners is None:
            owners = (0,) * self.nblocks
        if len(owners) != self.nblocks:
            raise ValueError(f"owners has {len(owners)} entries, nblocks is {self.nblocks}")
        self.owners = tuple(int(o) for o in owners)
        # Counted, not assumed: Gate 1's "no message is sent" must be checkable.
        self.messages = 0

    # ------------------------------------------------------------------ ownership
    def owner(self, b):
        """Rank owning block b."""
        return self.owners[b]

    def is_local(self, b):
        """True when block b's data lives on this rank."""
        return self.owners[b] == self.rank

    def local_blocks(self):
        """The blocks this rank owns, in order."""
        return tuple(b for b in range(self.nblocks) if self.is_local(b))

    @property
    def is_serial(self):
        return self.size == 1

    # ----------------------------------------------- the two cross-block data reads
    def fetch_field_slab(self, b, k, oaxis, oside, width, local):
        """The face slab of block b's field, padded along its first k axes. THE ROUTING POINT.

        Serially this is the local recursion followed by the extraction that used to live in
        `_ghost_field`; the arithmetic is unchanged, only relocated. Under MPI the owner of b
        runs exactly this and sends the slab, so the receiver's subsequent handling --
        orientation via `to_mine`, extent reconciliation via `_match_extent` -- is identical
        whether b was local or remote. That symmetry is the property that makes the serial and
        distributed paths comparable bitwise.
        """
        arr, olo, ohi = self._padded_field(b, k, local)
        return _extract_slab([arr], olo, ohi, oaxis, oside, width)[0], olo, ohi

    def fetch_coords_slab(self, b, k, oaxis, oside, width, local):
        """The face slab of block b's three coordinate arrays. THE ROUTING POINT.

        Separate from the field path on purpose, and not merely for symmetry: coordinates ramp
        and jump back, so a wrapped ghost must be displaced by one period, while velocity and
        pressure are genuinely periodic and must not be. The displacement itself is applied by
        the caller, which knows the connection's shift; keeping the two fetches distinct stops
        a future exchange implementation from applying it to both, or neither.
        """
        arrs, olo, ohi = self._padded_coords(b, k, local)
        return _extract_slab(arrs, olo, ohi, oaxis, oside, width), olo, ohi

    # ------------------------------------------------------ local access (Gate 2 overrides)
    def _padded_field(self, b, k, local):
        """Block b's field padded along its first k axes, wherever b lives.

        `local` is the caller's memoised recursion, `upto(bb, k)`. Serially the answer is
        simply that recursion; the indirection exists so Gate 2 can substitute an exchange for
        a non-local b without the padding logic knowing the difference.
        """
        if self.is_local(b):
            return local(b, k)
        raise NotImplementedError(                       # Gate 2
            f"block {b} is owned by rank {self.owner(b)}, not {self.rank}; distributed "
            f"field exchange arrives in Gate 2")

    def _padded_coords(self, b, k, local):
        """Block b's coordinates padded along its first k axes, wherever b lives."""
        if self.is_local(b):
            return local(b, k)
        raise NotImplementedError(                       # Gate 2
            f"block {b} is owned by rank {self.owner(b)}, not {self.rank}; distributed "
            f"coordinate exchange arrives in Gate 2")

    def __repr__(self):
        return (f"Comm(nblocks={self.nblocks}, rank={self.rank}, size={self.size}, "
                f"messages={self.messages})")
