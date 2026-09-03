"""Grid constructors, kept free of torch on purpose.

`periodic_box` lived in `src/mb_adjoint.py`, which imports torch at module scope. That made a
GRID BUILDER unusable anywhere torch is absent -- including the AmgX container, where an attempt
to time the LES driver failed with `ModuleNotFoundError: No module named 'torch'`. A domain is
geometry; it has no business requiring an autograd framework.

`src.mb_adjoint` re-exports the name, so existing imports keep working.
"""
import numpy as np


def periodic_box(ntot=12, n_split=1):
    """A fully periodic cube split into `n_split` blocks along x.

    Periodic in all three directions, so there are no walls and no boundary conditions to get
    wrong: the only thing that differs between n_split = 1 and n_split = 2 is the seam. Nodes
    PARTITION without duplicating the interface, as a periodic axis must.

    Lives here rather than in a test file because two test files need it, and importing one
    test from another runs it.
    """
    from src.multiblock import Block, Connection, Domain, face_id
    assert ntot % n_split == 0
    nxb = ntot // n_split
    ax = np.arange(ntot) / ntot
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    blocks = []
    for b in range(n_split):
        sl = slice(b * nxb, (b + 1) * nxb)
        blk = Block((nxb, ntot, ntot), X[sl], Y[sl], Z[sl], (1.0 / ntot,) * 3)
        for a in range(3):
            blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
        blocks.append(blk)
    conns = []
    if n_split > 1:
        for b in range(n_split):
            nb = (b + 1) % n_split
            sh = (1.0, 0.0, 0.0) if nb == 0 else (0.0, 0.0, 0.0)
            conns.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
    return Domain(blocks, conns)


# --------------------------------------------------------------------------- face operators
# Stage 7 needs a genuine multi-step chain, which needs a gradient and a divergence that cross
# seams. Rather than write new stencils and hope they agree with the operator the solver
# actually inverts, these are built from the SAME face enumeration `build_diffusion_matrix`
# uses, and `check_consistency` below proves the identity
#
#     sum_a  F_a^T diag(w_a) F_a  ==  build_diffusion_matrix
#
# exactly. That turns "my gradient looks right" into "my gradient composes into the operator
# the pressure solve uses", which is the only property the projection actually needs.
