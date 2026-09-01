"""
Vortex street behind a circular cylinder: a multi-block O-grid extended to the FAR FIELD.

WHY AN O-GRID. An H-grid around a body has REENTRANT CORNERS, where the diagonal ghost lies
inside the solid and the two blocks either side of a seam extrapolate it differently. That cost
real work to fix in `face_fluxes` (see reference/pressure_checkerboard.md). An O-grid has none:
every block joins its two azimuthal neighbours around a closed ring, and the only boundaries are
the cylinder surface and the far field.

WHY THE FAR FIELD HAS TO BE FAR. The earlier O-grid stopped at R = 6 D_cyl -- 12 diameters
across. Confinement that tight changes the shedding frequency and can suppress the instability
outright; the usual guidance for a clean Strouhal number is a domain radius of 20-50 diameters.
Reaching it without an absurd cell count is what the geometric radial stretching is for: the
wall cell stays fine enough to resolve the boundary layer while the outer cells grow to O(1)
diameter.

BOUNDARY ASSIGNMENT IS PER BLOCK, which is why the azimuthal block count matters beyond load
balance. A face carries ONE condition, so the outer ring can only be split into inflow and
outflow at block boundaries. With the flow along +x, blocks straddling theta = pi face upstream
(free stream) and those straddling theta = 0 face downstream (outflow); the count sets how
faithfully that division follows the true stagnation streamline.

NODE PLACEMENT. The azimuthal direction closes on itself, so it is partitioned WITHOUT
duplicating the seam nodes, exactly like a periodic axis. Radially there are real boundaries at
both ends -- cylinder surface and far field -- so both endpoints are kept. The span is periodic.
"""
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id

D = 1.0                       # cylinder diameter -- the reference length
R_CYL = 0.5 * D


def _radial(nr, r0, r1, first):
    """Pure geometric radial growth. Kept for reference; `_radial_plateau` is what is used."""
    lo, hi = 1.0, 1.6
    for _ in range(200):
        s = 0.5 * (lo + hi)
        tot = first * (nr - 1) if abs(s - 1) < 1e-12 else first * (s ** (nr - 1) - 1) / (s - 1)
        if tot < (r1 - r0):
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    dr = first * s ** np.arange(nr - 1)
    r = np.concatenate([[r0], r0 + np.cumsum(dr)])
    return r * (r1 - r0) / (r[-1] - r0) - r0 * ((r1 - r0) / (r[-1] - r0) - 1.0), s


def _radial_plateau(r0, r1, first, dr_hold, r_hold, ratio):
    """Wall cell `first`, growing at `ratio` to `dr_hold`, held to `r_hold`, then stretched.

    PURE GEOMETRIC GROWTH IS WRONG FOR A WAKE. The previous distribution grew monotonically from
    a 0.006 D wall cell all the way to the far field, which resolves the boundary layer
    beautifully and the wake not at all: measured against the shedding wavelength
    lambda = D/St = 6.10 D at Re = 100, it gave 138 cells per wavelength at 1 D and only 17.6 at
    5 D, 10.7 at 8 D, 4.3 at 20 D. The vortices would be smeared away almost as soon as they
    formed.

    This is the same plateau the square-cylinder wake needed, applied radially: grow out of the
    boundary layer, then HOLD a spacing fine enough for the street through the region where the
    street exists, and only stretch beyond it. The count follows from the spacings rather than
    being prescribed -- the spacing is the physically meaningful quantity.
    """
    rs, dr = [r0], first
    while dr < dr_hold and rs[-1] < r_hold:
        rs.append(rs[-1] + dr)
        dr = min(dr * ratio, dr_hold)
    while rs[-1] < r_hold:
        rs.append(rs[-1] + dr_hold)
    dr = dr_hold
    while rs[-1] < r1:
        dr *= ratio
        rs.append(rs[-1] + dr)
    rs = np.array(rs)
    k = int(np.searchsorted(rs, r_hold))
    tail = rs[k:] - rs[k]
    if len(tail) > 1 and tail[-1] > 0:
        rs[k:] = rs[k] + tail * (r1 - rs[k]) / tail[-1]
    return rs


def cylinder_domain(nblk=8, nth_tot=256, nz=8, r_out=30.0 * D, first=0.006 * D,
                    dr_hold=0.28 * D, r_hold=12.0 * D, ratio=1.12, span=4.0 * D):
    """O-grid ring of `nblk` blocks, cylinder at the centre, far field at `r_out`."""
    if nth_tot % nblk:
        raise ValueError(f"nth_tot={nth_tot} must divide by nblk={nblk}")
    # AZIMUTHAL RESOLUTION IS SET BY THE WAKE, NOT THE BODY. The arc length r*dtheta grows with
    # radius, so the coarsest azimuthal cell in the resolved region sits at r_hold: keeping
    # r_hold*dtheta <= dr_hold is what makes the two directions comparable there.
    r = _radial_plateau(R_CYL, r_out, first, dr_hold, r_hold, ratio)
    nr = len(r)
    # HALF-CELL OFFSET, AND IT IS NOT COSMETIC. With theta_k = k*2pi/nth_tot the reflection
    # y -> -y maps node k to node nth_tot-k, which shifts the BLOCK partition by one node: the
    # far-field inflow/outflow split then lands on [-90, +88.594] degrees instead of a
    # symmetric arc, and two of the 256 outer nodes carry a different role than their mirror.
    # The grid is symmetric, the boundary condition on it is not, and that is a standing
    # asymmetric forcing on a wake that is unstable to exactly that perturbation -- measured
    # 0.75 in u, 68% of max|u|, by t = 20 with no kick applied at all.
    #
    # With the offset, reflection maps k -> nth_tot-1-k, so block b maps to block nblk-1-b and
    # the partition is mirror-symmetric by construction. It also keeps the wake centreline
    # BETWEEN two node lines rather than on one.
    th_all = (np.arange(nth_tot) + 0.5) * (2 * np.pi / nth_tot)     # ring: no duplicate seam
    z = np.arange(nz) / nz * span                                   # periodic: no far endpoint
    nth = nth_tot // nblk
    h = (1.0 / (nr - 1), 1.0 / nth_tot, 1.0 / nz)
    arc = 2.0 * np.pi * r_hold / nth_tot

    blocks = []
    for b in range(nblk):
        th = th_all[b * nth:(b + 1) * nth]
        Rg, Tg, Zg = np.meshgrid(r, th, z, indexing="ij")
        blk = Block((nr, nth, nz), Rg * np.cos(Tg), Rg * np.sin(Tg), Zg, h,
                    period=(1.0, 1.0, span))     # span is physical, NOT the default 1
        blk.faces[face_id(0, 0)] = "wall"        # cylinder surface, no-slip
        blk.faces[face_id(0, 1)] = "wall"        # far field: free stream or outflow, by block
        blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
        blocks.append(blk)

    # the ring closes in PHYSICAL space, so no period shift -- unlike a periodic box, where the
    # coordinate itself jumps by one box length across the wrap
    conns = [Connection(b, face_id(1, 1), (b + 1) % nblk, face_id(1, 0)) for b in range(nblk)]
    return Domain(blocks, conns), r, arc


def outer_role(d, nblk, flow_x=True):
    """Classify each block's far-field face as 'inflow' or 'outflow' by its mean azimuth.

    A face carries ONE condition, so the split can only fall on block boundaries. With the
    stream along +x the downstream half-plane is |theta| < pi/2.
    """
    roles = {}
    for b in range(nblk):
        th = np.arctan2(d.blocks[b].y[-1].mean(), d.blocks[b].x[-1].mean())
        roles[b] = "outflow" if (np.cos(th) > 0) == flow_x else "inflow"
    return roles


if __name__ == "__main__":
    NBLK = 8
    d, r, arc = cylinder_domain(nblk=NBLK, nz=4)
    print(f"  Cylinder vortex street, O-grid to the far field.  D = {D}")
    print(f"  {len(d.blocks)} blocks, {d.n_cells:,} cells, {len(d.connections)} connections\n")
    probs = d.validate()
    print(f"  validate(): {len(probs)} problem(s)")
    for p in probs[:4]:
        print(f"      {p[:110]}")
    Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(len(d.blocks)))
    print(f"  min(J) = {Jmin:.4e}   {'valid' if Jmin > 0 else 'TANGLED'}")
    print(f"\n  radial: {len(r)} points, r = {r[0]:.3f} -> {r[-1]:.1f}  "
          f"({r[-1]/D:.0f} diameters)")
    dr = np.diff(r)
    print(f"    wall cell {dr[0]:.4f} D   plateau {dr.min():.4f}-{np.median(dr):.4f} D   "
          f"outer cell {dr[-1]:.3f} D")
    print(f"    worst adjacent ratio {np.maximum(dr[1:]/dr[:-1], dr[:-1]/dr[1:]).max():.3f}")
    ST = 0.164
    lam = D / ST
    print(f"\n  wake resolution vs lambda = D/St = {lam:.2f} D  (St = {ST}, Re = 100)")
    print(f"    {'x/D':>7}{'dr':>9}{'r*dth':>9}{'cells/lambda':>15}")
    import numpy as _np
    for xt in (1, 2, 3, 5, 8, 12, 16, 25):
        i = int(_np.argmin(_np.abs(r - xt)))
        j = min(i, len(dr) - 1)
        w = max(dr[j], 2 * _np.pi * r[i] / 256)
        print(f"    {r[i]:>7.2f}{dr[j]:>9.4f}{2*_np.pi*r[i]/256:>9.4f}{lam/w:>15.1f}")
    roles = outer_role(d, NBLK)
    ins = [b for b in roles if roles[b] == "inflow"]
    outs = [b for b in roles if roles[b] == "outflow"]
    print(f"\n  far-field faces: inflow on blocks {ins}, outflow on blocks {outs}")
