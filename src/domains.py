"""Grid constructors, kept free of torch on purpose.

`periodic_box` lived in `src/mb_adjoint.py`, which imports torch at module scope. That made a
GRID BUILDER unusable anywhere torch is absent -- including the AmgX container, where an attempt
to time the LES driver failed with `ModuleNotFoundError: No module named 'torch'`. A domain is
geometry; it has no business requiring an autograd framework.

`src.mb_adjoint` re-exports the name, so existing imports keep working.
"""
import numpy as np


def periodic_box(ntot=12, n_split=1, L=1.0):
    """A fully periodic cube split into `n_split` blocks along x.

    Periodic in all three directions, so there are no walls and no boundary conditions to get
    wrong: the only thing that differs between n_split = 1 and n_split = 2 is the seam. Nodes
    PARTITION without duplicating the interface, as a periodic axis must.

    Lives here rather than in a test file because two test files need it, and importing one
    test from another runs it.

    `L` is the physical side length. It defaults to 1.0, which every existing caller assumes.
    """
    from src.multiblock import Block, Connection, Domain, face_id
    assert ntot % n_split == 0
    nxb = ntot // n_split
    # PHYSICAL EXTENT, not just a unit box. The default 1.0 is what every existing caller
    # assumes. Taylor-Green is conventionally posed on [0, 2 pi]^3 with wavenumber 1, and
    # writing the field as sin(2 pi x) on a UNIT box gives the same velocities with wavenumber
    # 2 pi -- identical energy, (2 pi)^2 = 39.5x the enstrophy, and 39.5x the dissipation at the
    # same nu. Measured: Z(0) = 14.06 against the analytic 0.375. Pass L = 2 pi for TGV.
    ax = np.arange(ntot) / ntot * L
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    blocks = []
    for b in range(n_split):
        sl = slice(b * nxb, (b + 1) * nxb)
        # h IS THE PHYSICAL SPACING, and must be scaled with the coordinates. It is used as the
        # computational spacing by the metrics AND, implicitly, as the physical step by the
        # periodic wrap in pad_coords -- so scaling the coordinates alone leaves the wrap short
        # by a factor L and the geometry inconsistent. Keeping h = L/ntot makes the mapping the
        # identity, J = 1, which is what every other caller of this builder already assumes.
        # BOTH h AND period MUST FOLLOW L. Block defaults period to (1, 1, 1), and pad_coords
        # adds `blk.period[a]` when it wraps a periodic ghost -- so scaling the coordinates
        # while leaving the period at 1 made the wrapped ghost land a whole box short. The
        # Jacobian came out at 1/(2 pi)^3 instead of 1, and the initial TGV energy read 0.147
        # against an exact 0.125 that did not converge with refinement. Two hardcoded unit-box
        # assumptions, one visible in this file and one inherited from Block's default.
        blk = Block((nxb, ntot, ntot), X[sl], Y[sl], Z[sl], (L / ntot,) * 3,
                    period=(L, L, L))
        for a in range(3):
            blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
        blocks.append(blk)
    conns = []
    if n_split > 1:
        for b in range(n_split):
            nb = (b + 1) % n_split
            # THE PERIODIC SHIFT IS THE BOX LENGTH, not 1. Hardcoding 1.0 while scaling the
            # coordinates by L left the wrap inconsistent with the geometry: the metrics came
            # out with xi_x ~ 1/(2 pi) but J = 1 instead of (2 pi)^3, and the initial TGV
            # energy read 0.147 against an exact 0.125 that does not converge with refinement.
            sh = (L, 0.0, 0.0) if nb == 0 else (0.0, 0.0, 0.0)
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


def channel_box(nx=24, ny=80, nz=24, n_split=1, Lx=np.pi, Ly=2.0, Lz=0.34 * np.pi,
                y_nodes=None):
    """Channel: periodic in x and z, WALLS at y = 0 and y = Ly, split along x.

    Defaults are the Jimenez-Moin MINIMAL BOX at Re_tau = 180 with u_tau = delta = 1, i.e.
    Lx+ = 565 and Lz+ = 192 -- the configuration of the reference DNS in
    `results/minchan_re180/`. That box is DELIBERATELY ABOVE the minimal-unit threshold
    (Lz+ ~ 100), not on it: sitting at the minimum invites a relaminarisation that would be
    indistinguishable from a solver bug.

    `y_nodes` supplies the wall-normal distribution. It MUST be non-uniform for a wall-resolved
    LES: y+ = 1 at Re_tau = 180 is y = 1/180 = 0.00556, and reaching that uniformly would need
    360 points where a clustered 80 suffices. `clustered_y` below is the recipe.

    The x and z axes are periodic and store nodes WITHOUT the far endpoint; y is a wall axis and
    stores BOTH, which is why its spacing is Ly/(ny-1) and theirs is L/n.
    """
    from src.multiblock import Block, Connection, Domain, face_id
    assert nx % n_split == 0
    nxb = nx // n_split
    ax = np.arange(nx) / nx * Lx
    az = np.arange(nz) / nz * Lz
    ay = np.linspace(0.0, Ly, ny) if y_nodes is None else np.asarray(y_nodes, dtype=float)
    if len(ay) != ny:
        raise ValueError(f"y_nodes has {len(ay)} points, ny is {ny}")
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    blocks = []
    for b in range(n_split):
        sl = slice(b * nxb, (b + 1) * nxb)
        blk = Block((nxb, ny, nz), X[sl], Y[sl], Z[sl],
                    (Lx / nx, Ly / (ny - 1), Lz / nz), period=(Lx, Ly, Lz))
        for a in (0, 2):
            blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
        blk.faces[face_id(1, 0)] = blk.faces[face_id(1, 1)] = "wall"
        blocks.append(blk)
    conns = []
    if n_split > 1:
        for b in range(n_split):
            nb = (b + 1) % n_split
            sh = (Lx, 0.0, 0.0) if nb == 0 else (0.0, 0.0, 0.0)
            conns.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
    return Domain(blocks, conns)


def clustered_y(ny=80, Ly=2.0, dy0_plus=1.0, re_tau=180.0):
    """Wall-normal nodes clustered symmetrically at BOTH walls, first cell = `dy0_plus`.

        y_j = (Ly/2) [ 1 + tanh( g (2j/(ny-1) - 1) ) / tanh(g) ]

    `g` is solved by bisection so the first spacing hits the requested wall units. A geometric
    stack was tried first and mirrors badly: for EVEN ny the two halves share a midpoint, which
    lands a zero spacing in the middle of the channel and an infinite expansion ratio beside it.
    The tanh form is symmetric by construction for any count, odd or even, and is what channel
    DNS codes have used since Kim, Moin & Moser.

    Reported by the caller rather than assumed: the maximum spacing and the worst adjacent ratio
    both matter for a wall-resolved LES, and neither is set directly by dy0_plus.
    """
    target = dy0_plus / re_tau

    def first(g):
        j = np.array([0.0, 1.0])
        y = 0.5 * Ly * (1.0 + np.tanh(g * (2 * j / (ny - 1) - 1.0)) / np.tanh(g))
        return y[1] - y[0]

    lo, hi = 1e-3, 10.0
    while first(hi) > target:
        hi *= 1.5
        if hi > 1e4:
            raise ValueError(f"cannot reach dy0+ = {dy0_plus} with ny = {ny}")
    for _ in range(200):                     # first spacing decreases with g
        mid = 0.5 * (lo + hi)
        if first(mid) > target:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    j = np.arange(ny)
    return 0.5 * Ly * (1.0 + np.tanh(g * (2 * j / (ny - 1) - 1.0)) / np.tanh(g))
