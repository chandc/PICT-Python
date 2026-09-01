"""
Vortex street behind a SQUARE cylinder: the 8-block H-grid, at production resolution.

TOPOLOGY. Nine tiles in a 3x3 arrangement with the centre removed; the hole IS the obstacle,
and the four faces bounding it are walls. This is the layout PICT's `vortex_street_sample` uses
and the one `test_obstacle_topology.py` already gates -- that test runs at 944 cells purely to
exercise the connection map, which is why it needed replacing rather than reusing.

    TL | TM | TR
    ---+----+---     the four faces around the hole are the obstacle surface
    ML |####| MR
    ---+----+---
    BL | BM | BR

REENTRANT CORNERS. Unlike the cylinder O-grid, this topology has FOUR of them, at the square's
corners, where the diagonal ghost lies inside solid material. Padded coordinates there are
EXTRAPOLATED, and two blocks extrapolate differently -- which broke the shared face flux on the
five-domain BFS step corner (measured mismatch 4.2e+00, flux divergence 2e-04 instead of 2e-12)
until `face_fluxes` was changed to build contravariant components from each block's OWN metrics
and pad those as fields. This grid leans on that fix; without it the corners would leak.

RESOLUTION AND BLOCKAGE. D = 1 is the square's side. The lateral walls sit at +-10 D, giving a
blockage ratio of 5% -- low enough that confinement does not dominate the shedding frequency,
which is the failure mode of a domain sized for a picture rather than for physics. The wake runs
to 30 D so vortices are resolved well downstream of formation, and the inlet is 10 D upstream.
Cells cluster toward the obstacle on all four sides, since the shear layers separate from its
corners.
"""
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id

D = 1.0                                   # square side -- the reference length
X_IN, X_OUT = -10.0 * D, 30.0 * D
Y_HALF = 10.0 * D
# keys are COLUMN then ROW: "LB" = left column, bottom row. Getting this backwards
# is a silent KeyError at construction, which is the good failure mode.
NAMES = ("LB", "MB", "RB", "LM", "RM", "LT", "MT", "RT")


def _grow_from(x_edge, x_far, dx0, ratio):
    """Nodes from `x_far` to `x_edge`, spacing `dx0` AT THE EDGE growing by `ratio` outward.

    Replaces the tanh clustering, which controlled the shape but not the one number that
    matters: the cell size where it meets the body. `_toward` left 0.05331 against the body's
    own 0.03226 -- a 1.65x step at the leading edge and at both lateral seams.

    Growing geometrically FROM the body outward makes continuity exact by construction and
    bounds every adjacent ratio at `ratio`. The count follows from the spacing rather than being
    prescribed, which is the same trade the wake distribution already makes and for the same
    reason: the spacing is the physically meaningful quantity.
    """
    sgn = 1.0 if x_far > x_edge else -1.0
    xs, dx = [x_edge], dx0
    while abs(xs[-1] - x_edge) < abs(x_far - x_edge):
        xs.append(xs[-1] + sgn * dx)
        dx *= ratio
    xs = np.array(xs)
    # pull the outermost node exactly onto x_far, spreading the correction over the tail so the
    # cell at the body -- the one that has to match -- is untouched
    span = xs[-1] - x_edge
    if abs(span) > 1e-30:
        xs = x_edge + (xs - x_edge) * (x_far - x_edge) / span
    # Ascending order, always. Building DOWNWARD (x_far < x_edge, as for the upstream and
    # lower-lateral strips) produces a descending array, and returning it unreversed inverts
    # the axis: measured min(J) = -3.8e+05, a tangled grid, from getting this one test backwards.
    return xs[::-1] if sgn < 0 else xs


def _ramp_plateau_stretch(x0, x1, dx0, dx_hold, x_hold, ramp_ratio, tail_ratio):
    """Ramp `dx0` -> `dx_hold`, hold it to `x_hold`, then stretch geometrically out to `x1`.

    THE RAMP IS THE POINT. The earlier version started the wake at `dx_hold` immediately, which
    put a 4.65x cell-size jump (0.03226 -> 0.15000) at the body's TRAILING EDGE -- exactly where
    the shear layers separate. Nothing caught it: `validate()` checked node coincidence at seams,
    not physical spacing. The pressure rang at the Nyquist wavelength immediately downstream
    (flip fraction 0.950 in that block), and the whole-domain average diluted it to 0.297, which
    read as clean.

    Starting at the body's own spacing and growing at `ramp_ratio` costs about ten extra cells
    and removes the jump entirely.
    """
    xs, dx = [x0], dx0
    while dx < dx_hold and xs[-1] < x_hold:
        xs.append(xs[-1] + dx)
        dx = min(dx * ramp_ratio, dx_hold)
    while xs[-1] < x_hold:
        xs.append(xs[-1] + dx_hold)
    dx = dx_hold
    while xs[-1] < x1:
        dx *= tail_ratio
        xs.append(xs[-1] + dx)
    xs = np.array(xs)
    k = int(np.searchsorted(xs, x_hold))
    tail = xs[k:] - xs[k]
    if len(tail) > 1 and tail[-1] > 0:
        xs[k:] = xs[k] + tail * (x1 - xs[k]) / tail[-1]
    return xs


def square_domain(n_obs=32, nz=8, span=4.0 * D, ratio=1.10,
                  wake_hold=15.0 * D, wake_dx=0.15 * D, wake_ratio=1.06):
    """8 blocks around a square hole of side D centred on the origin.

    n_obs is the node count ACROSS the obstacle INCLUDING both wall lines; those two lines are
    stored by the neighbouring blocks, so the middle strips hold n_obs - 2 nodes. It is the
    ONLY count specified: n_obs fixes dx_body = D/(n_obs-1), and every other strip grows from
    that at `ratio`, so its cell count follows. Prescribing counts instead is what allowed the
    spacing to jump 4.65x at the trailing edge while every check passed.

    `ratio` = 1.10, not the 1.15 first tried. 1.15 gave the same node count as the tanh it
    replaced and spent it differently: twice as fine at the body (0.0290 against 0.0533) and
    nearly twice as COARSE in the far field (max cell 1.264 against 0.703). Buying near-body
    resolution out of the far field is the wrong trade for a wake instability, whose mode has
    support several diameters out. 1.10 costs ~17,000 cells and puts the far field back where it
    was (0.893) while keeping the near-body gain. n_side is the count in each lateral band. The downstream count is
    NOT specified: it follows from wake_hold / wake_dx / wake_ratio, because the wake spacing is
    the physically meaningful quantity and a fixed count would let it drift.

    The defaults hold dx = 0.15 D out to 15 D -- about 51 cells per shedding wavelength, well
    inside the >= 20 needed and not the 77 that dx = 0.10 bought at the cost of collapsing to
    8 cells per wavelength by x = 20. Spending the same cells on a longer plateau and a gentler
    expansion (1.06 rather than 1.08) keeps the street resolved to roughly 19 D instead of 10.
    """
    half = 0.5 * D
    dx_body = D / (n_obs - 1)            # the body's own spacing -- everything matches THIS
    # WHO OWNS THE OBSTACLE'S NODE LINES. Every seam here is non-duplicating -- a connected axis
    # stores up to but NOT including its neighbour's first node -- so each of the four lines
    # x = +-half, y = +-half belongs to exactly one block, and it must be the block whose face
    # there is the WALL, or the wall lands one spacing off. LM/RM carry the left/right faces, so
    # the LEFT and RIGHT columns are closed on the body; MB/MT carry bottom/top, so the BOTTOM
    # and TOP rows are. The middle strips are therefore strictly interior.
    #
    # AND EVERY STRIP NOW STARTS AT dx_body. Adjacent-cell ratios are bounded by `ratio`
    # everywhere, which `Domain.check_spacing()` gates: previously the trailing edge jumped
    # 4.65x and the leading edge and laterals 1.65x, all of it silent.
    x_up = _grow_from(-half, X_IN, dx_body, ratio)                       # closed at -half
    x_ob = np.linspace(-half, half, n_obs)[1:-1]                         # strictly interior
    x_dn = _ramp_plateau_stretch(half, X_OUT, dx_body, wake_dx, wake_hold,
                                 ratio, wake_ratio)                      # closed at +half
    y_lo = _grow_from(-half, -Y_HALF, dx_body, ratio)                    # closed at -half
    y_ob = np.linspace(-half, half, n_obs)[1:-1]                         # strictly interior
    y_up = _grow_from(half, Y_HALF, dx_body, ratio)                      # closed at +half
    z = np.arange(nz) / nz * span                       # periodic: far endpoint not stored

    XS = (x_up, x_ob, x_dn)
    YS = (y_lo, y_ob, y_up)
    nx_tot = sum(len(a) for a in XS)
    ny_tot = sum(len(a) for a in YS)
    # ONE global spacing per axis: x and y are each a single axis partitioned three ways, so a
    # per-block spacing would give each block a different metric scaling and flip the Jacobian.
    h = (1.0 / (nx_tot - 1), 1.0 / (ny_tot - 1), 1.0 / nz)

    # The BACKGROUND distribution: the whole rectangle including the nodes inside the solid.
    # x_ob / y_ob are exactly the obstacle's interior lines, so concatenating the three strips
    # reconstructs the full tensor grid the blocks were cut from. Handing this to every block is
    # what lets the ones abutting the body use a central metric stencil, which the geometric
    # conservation law needs at the reentrant corners -- see reference/reentrant_corner_gcl.md.
    GX, GY = np.concatenate(XS), np.concatenate(YS)
    xoff = (0, len(XS[0]), len(XS[0]) + len(XS[1]))
    yoff = (0, len(YS[0]), len(YS[0]) + len(YS[1]))

    idx, blocks = {}, []
    for j, ry in enumerate("BMT"):
        for i, rx in enumerate("LMR"):
            if i == 1 and j == 1:
                continue                                 # the hole IS the obstacle
            X, Y, Z = np.meshgrid(XS[i], YS[j], z, indexing="ij")
            b = Block((len(XS[i]), len(YS[j]), nz), X, Y, Z, h,
                      period=(1.0, 1.0, span),           # span is physical, not the default 1
                      background=(GX, GY, z, (xoff[i], yoff[j], 0)))
            b.faces[face_id(2, 0)] = b.faces[face_id(2, 1)] = "periodic"
            idx[rx + ry] = len(blocks)
            blocks.append(b)

    W = "wall"
    for nm in ("LB", "LM", "LT"):
        blocks[idx[nm]].faces[face_id(0, 0)] = W         # inflow (velocity set by the caller)
    for nm in ("RB", "RM", "RT"):
        blocks[idx[nm]].faces[face_id(0, 1)] = W         # outflow
    for nm in ("LB", "MB", "RB"):
        blocks[idx[nm]].faces[face_id(1, 0)] = W         # lower lateral boundary
    for nm in ("LT", "MT", "RT"):
        blocks[idx[nm]].faces[face_id(1, 1)] = W         # upper lateral boundary
    # the obstacle surface: the four faces that would have touched the hole
    blocks[idx["LM"]].faces[face_id(0, 1)] = W
    blocks[idx["RM"]].faces[face_id(0, 0)] = W
    blocks[idx["MB"]].faces[face_id(1, 1)] = W
    blocks[idx["MT"]].faces[face_id(1, 0)] = W

    conns = []
    for j, ry in enumerate("BMT"):                        # x-direction, per row
        for i in range(2):
            a, b = "LMR"[i] + ry, "LMR"[i + 1] + ry
            if a in idx and b in idx:
                conns.append(Connection(idx[a], face_id(0, 1), idx[b], face_id(0, 0)))
    for i, rx in enumerate("LMR"):                        # y-direction, per column
        for j in range(2):
            a, b = rx + "BMT"[j], rx + "BMT"[j + 1]
            if a in idx and b in idx:
                conns.append(Connection(idx[a], face_id(1, 1), idx[b], face_id(1, 0)))
    return Domain(blocks, conns), idx


if __name__ == "__main__":
    d, idx = square_domain()
    print(f"  Square cylinder, D = {D}.  domain x [{X_IN:.0f}, {X_OUT:.0f}] D, "
          f"y +-{Y_HALF:.0f} D,  blockage {100*D/(2*Y_HALF):.1f}%")
    print(f"  {len(d.blocks)} blocks, {d.n_cells:,} cells, {len(d.connections)} connections\n")
    probs = d.validate()
    print(f"  validate(): {len(probs)} problem(s)")
    for p in probs[:4]:
        print(f"      {p[:110]}")
    Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(len(d.blocks)))
    print(f"  min(J) = {Jmin:.4e}   {'valid' if Jmin > 0 else 'TANGLED'}")
    print(f"  wall nodes {d.wall_mask().sum():,} of {d.n_cells:,}")
    for nm in ("LM", "MB"):
        b = d.blocks[idx[nm]]
        dx = np.diff(b.x[:, 0, 0]); dy = np.diff(b.y[0, :, 0])
        print(f"    {nm}: {b.shape}  dx {dx.min():.4f}..{dx.max():.4f}  "
              f"dy {dy.min():.4f}..{dy.max():.4f}")
