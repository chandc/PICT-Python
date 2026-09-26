"""Cylinder grid: an O-ring at the body, RECTANGULAR Cartesian blocks everywhere else.

WHY, against the butterfly it replaces. The butterfly fans four trapezoids from the ring to the
rectangle, and those sheared corner cells are why `implicit_cross` is REQUIRED physics there --
the orthogonal-only pressure solve blows up ~2x/step from them (min |J| ~ 3e-5), and the
deferred-correction cure costs 2.17 s/step. A Cartesian background is orthogonal by
construction: no shear, no cross term, better conditioning.

TOPOLOGY -- 12 blocks.
  * 4 O-ring blocks (E, N, W, S), inner boundary the cylinder, outer boundary the CENTRAL
    SQUARE [-L1, L1]^2. Radial projection: every ring node lies on the ray from the origin
    through its outer node, so the ring is a true O-grid and its outer face is a straight
    square edge that a Cartesian block can meet.
  * 8 rectangular blocks, a 3x3 tiling of the domain with the centre tile removed.

    xW = [-x_in, -L1]   xC = [-L1, L1]   xE = [L1, x_out]
    yS = [-y_half, -L1] yC = [-L1, L1]   yN = [L1, y_half]

    WN CN EN
    WC ## EC        ## = the O-ring
    WS CS ES

Shapes are shared along each row and column, so every block-block interface is a whole-face
equal-shape connection.

TARGET (HydroGym `medium.geo`, its DEFAULT_MESH): x in [-5, 15], y in [-5, 5], beta = 0.10;
characteristic length 1/n1 = 1/35 = 0.0286 at the cylinder, 1/n2 = 0.05 at the outlet centre,
1/n3 = 0.667 at the far corners. Gmsh fills that with near-ISOTROPIC triangles, so matching it
means matching the LOCAL spacing, not a global count.
"""
import os
import numpy as np

D = 1.0
R_CYL = 0.5 * D


def _auto_conn(blocks, ba, fa, bb, fb):
    """A Connection whose orientation is DERIVED from node coincidence, not asserted by hand.

    `axes` and `flips` have 8 combinations and the wrong one still validates: the spacings match
    term by term, so validate() passes and the error only shows up as a scrambled seam in the
    solver. Two of the sixteen seams here were laid down end-for-end that way. Trying all eight
    and keeping the one whose face nodes actually coincide removes the whole class of mistake.
    """
    from src.multiblock import Connection, face_slice, face_axis_side
    A, B = blocks[ba], blocks[bb]
    sa, sb = face_slice(fa), face_slice(fb)
    pa = np.stack([A.x[sa], A.y[sa], A.z[sa]], -1)
    raw = np.stack([B.x[sb], B.y[sb], B.z[sb]], -1)
    # PICK THE BEST ORIENTATION, DO NOT DEMAND A PERFECT ONE. Face cells either side of a seam
    # are ADJACENT, not coincident -- separated by one cell along the normal by design -- so an
    # exact-coincidence test is the wrong invariant. It also silently certified a grid in which
    # every seam DID duplicate a cell layer. The correct orientation is the one whose face
    # points line up best; a wrong permutation or flip scrambles the tangential ordering and
    # scores far worse, so the choice is unambiguous and is asserted to be.
    scored = []
    for axes in ((0, 1), (1, 0)):
        for f0 in (False, True):
            for f1 in (False, True):
                c = Connection(ba, fa, bb, fb, axes=axes, flips=(f0, f1))
                try:
                    pb = np.stack([c.align(raw[..., k]) for k in range(3)], -1)
                except Exception:
                    continue
                if pb.shape != pa.shape:
                    continue
                scored.append((float(np.abs(pa - pb).max()), c))
    if not scored:
        raise AssertionError(f"no orientation gives matching face shapes for blocks {ba}/{bb}")
    scored.sort(key=lambda t: t[0])
    best, runner = scored[0], (scored[1] if len(scored) > 1 else (float("inf"), None))
    if len(scored) > 1 and runner[0] < 2.0 * best[0]:
        raise AssertionError(
            f"orientation ambiguous for blocks {ba}/{bb}: best {best[0]:.3e}, "
            f"runner-up {runner[0]:.3e}")
    return best[1]


def _pin(L, n, d_end, at_end=True):
    """`n` cells spanning `L` geometrically, with the cell at one END exactly `d_end`.

    Returns normalised node positions (0..1), inner-to-outer. Pinning ONE end and the total
    length leaves the ratio as the single unknown, solved here by bisection -- which is what
    lets every azimuthal column of a square-framed ring carry a DIFFERENT ray length while
    still presenting one common spacing to the block it joins.
    """
    def total(r):
        return d_end * n if abs(r - 1.0) < 1e-13 else d_end * (1.0 - r ** n) / (1.0 - r)
    lo, hi = 1e-4, 1e4
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if total(mid) < L: lo = mid
        else: hi = mid
    d = d_end * (0.5 * (lo + hi)) ** np.arange(n)      # from the pinned end inward
    if at_end:
        d = d[::-1]                                     # pinned cell is the LAST one
    q = np.concatenate([[0.0], np.cumsum(d)])
    return q / q[-1]


def _pin2(L, n, d0, d1):
    """`n` cells spanning `L` with the FIRST cell exactly `d0` and the LAST exactly `d1`.

    A one-ended geometric has a single free ratio, so pinning both ends and the length is one
    constraint too many -- which is why the ring/transition seam stalled at 1.69x. Carrying a
    quadratic bulge instead, d(t) = d0(1-t) + d1 t + c t(1-t), leaves the endpoints untouched
    at t = 0 and 1 while `c` absorbs whatever length is left over, so BOTH neighbours can be
    matched exactly on every azimuthal column independently.
    """
    t = (np.arange(n) + 0.5) / n
    lin = d0 * (1 - t) + d1 * t
    bulge = t * (1 - t)
    c = (L - lin.sum()) / bulge.sum()
    d = lin + c * bulge
    if d.min() <= 0:
        raise ValueError(f"_pin2: non-positive spacing (L={L:.4f} n={n} d0={d0:.4f} d1={d1:.4f})")
    q = np.concatenate([[0.0], np.cumsum(d)])
    return q / q[-1]


def _stretch(a, b, d0, ratio, dmax=None):
    """Nodes from a to b, first spacing d0, geometric growth, capped at dmax."""
    # Do NOT clamp the final step to b. Clamping leaves a RUNT last cell -- whatever fraction
    # of a full step remains -- sitting next to a full-size neighbour; measured 0.1156 against
    # 0.5002, a 4.33x jump, and reversed distributions carry it to the START of the block where
    # it is least expected. Overshoot instead and let the renormalisation below absorb it, which
    # spreads the correction evenly over every cell.
    pts, x, d = [a], a, d0
    while x < b - 1e-12:
        d = min(d * ratio, dmax) if dmax else d * ratio
        x = x + d
        pts.append(x)
    p = np.array(pts)
    return a + (p - a) * (b - a) / (p[-1] - a)          # renormalise to hit b exactly


def _sym_stretch(a, b, d0, ratio):
    """Symmetric about the midpoint: fine at BOTH ends (for the centre tile)."""
    half = _stretch(0.0, 0.5 * (b - a), d0, ratio)
    full = np.concatenate([half[:-1], (b - a) - half[::-1]])
    return a + full


def cart_ring_domain(n_rad=15, n_tr=16, L2=1.40, ds_seam=0.040, nz=2, span=4.0 * D,
                     x_in=5.0, x_out=15.0, y_half=5.0,
                     ds_body=1.0 / 35.0, ds_azim=0.0280, ds_wake=0.05, ds_far=0.667,
                     grow=1.16):
    """16-block grid: 4 O-ring + 4 transition + 8 RECTANGULAR Cartesian blocks.

    TOPOLOGY. The O-ring's radial direction rotates through 360 degrees, so the blocks it hands
    off to cannot all keep a fixed axis convention -- following the identifications round the
    ring forces one Cartesian block's own axis 0 and axis 1 into the same padding level. Four of
    the sixteen connections therefore join axis 0 to axis 1, which `_match_extent` now supports.

    WHY THE TRANSITION LAYER EXISTS. A Cartesian block presents a straight edge and ONE spacing
    along it, which an O-ring cannot meet directly. The transition carries a spacing pinned at
    BOTH ends per azimuthal column (`_pin2`) -- the ring's outer dr on one side, the tile's
    uniform spacing on the other -- so the seam closes at 1.00x on every column.

    THE RING IS A TRUE ANNULUS, r in [R, R + D/2]. An earlier version framed it on a SQUARE of
    half-width (R + D/2)/sqrt(2) so the corners just touched the half-diameter cap, which left
    the transition mapping square -> square: the radial rays are then oblique to the edges at
    BOTH ends and the shear never relaxes, measured at mean |cos| 0.414 between grid lines and
    INDEPENDENT of L2 -- intrinsic to the geometry, not a tuning miss. Since the cross-diffusion
    is carried by deferred correction, an explicit fixed-point iteration whose contraction
    degrades as skew x dt, that pervasive skew is what forced dt down to 0.002.

    Circle -> square instead: the ring is orthogonal by construction (mean |cos| exactly 0, from
    0.251) and uniformly half a diameter thick (from 0.207-0.500), and the transition drops to
    mean 0.240. For calibration the butterfly's traps sit at 0.21 and run at dt = 0.01.

    L2 = 1.4 RATHER THAN 1.2 because moving the ring's outer boundary out to a uniform r = 1.0
    squeezes the transition: against a tile at 1.2 its rays would run 0.2 at the edge midpoints
    and 0.697 at the corners, a 3.5x spread, worse than the 2.41x it replaces. At 1.4 that is
    2.45x, so the skew improves without handing the grading a harder problem.
    """
    from src.multiblock import Block, Connection, Domain, face_id
    R_OUT = R_CYL + 0.5 * D                  # the annulus outer radius: half a diameter, exactly
    z = np.arange(nz) / nz * span
    blocks, conns, idx = [], [], {}

    def add(name, X2, Y2):
        n0, n1 = X2.shape
        X = np.repeat(X2[:, :, None], nz, axis=2)
        Y = np.repeat(Y2[:, :, None], nz, axis=2)
        Z = np.broadcast_to(z[None, None, :], (n0, n1, nz)).copy()
        b = Block((n0, n1, nz), X, Y, Z, (1.0, 1.0, 1.0), period=(1.0, 1.0, span))
        b.faces[face_id(2, 0)] = b.faces[face_id(2, 1)] = "periodic"
        idx[name] = len(blocks)
        blocks.append(b)
        return b

    # The central tile's spacing sets the AZIMUTHAL resolution: a tile node at (L2, y) projects
    # to the cylinder along its own ray, so arc = R * ds_tile / L2. Solve it for HydroGym's
    # measured 0.0280 (112 nodes on the circle) rather than picking the tile spacing blind.
    n_side = int(round(2.0 * L2 / (ds_azim * L2 / R_CYL))) + 1
    tC = np.linspace(-L2, L2, n_side)
    ds_tile = tC[1] - tC[0]

    def ray(L, th):                       # origin -> square of half-width L, along th
        return L / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))

    # Outer frame == the tile's own face, so ring and tile nodes coincide by construction.
    outer = {"E": np.stack([np.full(n_side, L2), tC], 1),
             "N": np.stack([tC[::-1], np.full(n_side, L2)], 1),
             "W": np.stack([np.full(n_side, -L2), tC[::-1]], 1),
             "S": np.stack([tC, np.full(n_side, -L2)], 1)}

    # W AND S MUST PRESENT A LOWER FACE TO THEIR TILE. A Connection pairs an UPPER face with a
    # LOWER one, and the radial axis points OUTWARD on all four sides, so on the west and south
    # the ring's outer face and its Cartesian neighbour's face are both "upper".
    #
    # Reversing the radial axis alone fixes that and BREAKS THE HANDEDNESS: flipping one of
    # three index axes makes J negative, and every contravariant component then carries an extra
    # sign relative to its right-handed neighbours. Measured as a flux ratio of exactly -1.000
    # across six seams, while validate(), node coincidence, the scalar halo and the metrics were
    # all clean -- |J| hid it. So reverse an EVEN number of axes: reverse the radial AND
    # transpose it onto axis 1. Handedness flips twice and comes back positive, the upper/lower
    # rule is satisfied, and the resulting seams simply become axis-rotating connections.
    REVERSED = ("W", "S")

    def lay(P, rev):
        """(X2, Y2) for a ring/transition block: radial on axis 0, or reversed onto axis 1."""
        if not rev:
            return P[:, :, 0], P[:, :, 1]
        return P[::-1, :, 0].T, P[::-1, :, 1].T

    # face roles, which move with the layout
    RAD_LO = lambda rev: face_id(1, 1) if rev else face_id(0, 0)   # toward the cylinder
    RAD_HI = lambda rev: face_id(1, 0) if rev else face_id(0, 1)   # toward the tile
    TAN_0  = lambda rev: face_id(0, 0) if rev else face_id(1, 0)
    TAN_1  = lambda rev: face_id(0, 1) if rev else face_id(1, 1)

    for nm, out in outer.items():
        rev = nm in REVERSED
        th = np.arctan2(out[:, 1], out[:, 0])
        u = np.stack([np.cos(th), np.sin(th)], 1)
        r0 = np.full(n_side, R_CYL)                 # cylinder
        r1 = np.full(n_side, R_OUT)                 # annulus outer == transition inner
        r2 = ray(L2, th)                            # tile face
        Pr = np.empty((n_rad + 1, n_side, 2))
        Pt = np.empty((n_tr + 1, n_side, 2))
        # ONE UNIFORM RADIAL SPACING AT BOTH SEAMS. The annulus pins the WALL at ds_body and its
        # OUTER edge at ds_seam, and the transition runs ds_seam -> ds_seam, so every seam in
        # the radial direction closes at 1.00x with nothing left to match per column.
        #
        # The transition's rays run 0.400 long at the edge midpoints and 0.980 at the corners.
        # Pinning its outer edge to the tile's TANGENTIAL spacing (0.078) demanded 16 cells
        # summing to ~0.96 on a ray only 0.400 long, and _pin2's bulge swung negative to make
        # up the difference: spacing collapsed to 0.0033 between ends of 0.051 and 0.069, a
        # 20.8x swing inside a single ray, visible as a pinch. The outer spacing does NOT have
        # to equal the tangential one -- it only has to match the neighbouring Cartesian block's
        # first RADIAL step, which is ours to choose. At ds_seam = 0.040 the same 16 cells want
        # 0.64, the short ray no longer over-fills, and the worst within-ray ratio is 2.10.
        qr = _pin2(R_OUT - R_CYL, n_rad, ds_body, ds_seam)     # identical on every column
        for j in range(n_side):
            qt = _pin2(r2[j] - r1[j], n_tr, ds_seam, ds_seam)
            Pr[:, j, :] = u[j] * (r0[j] + qr * (r1[j] - r0[j]))[:, None]
            Pt[:, j, :] = u[j] * (r1[j] + qt * (r2[j] - r1[j]))[:, None]
        # EVERY SHARED LINE BELONGS TO EXACTLY ONE BLOCK. Two blocks that both carry the seam
        # line put two cells at one physical point, coupled across a face of zero width. The
        # validated butterfly avoids this deliberately -- `Xr[:-1]` for its ring, and
        # `xw = xw[1:]  # X_HAND line is owned by the east trap` for its wake -- and 0 of its
        # 13 seams are coincident. All 24 of this grid's were, which painted a spurious
        # vorticity sheet right around the body at r = 1.0 and TRIPLED the operator error on a
        # two-block test (Laplacian of a linear field: 30 duplicated vs 10 adjacent).
        # Ownership here: the transition owns r = R_OUT and the square perimeter at +-L2; the
        # ring and the Cartesian tiles drop those lines.
        # Pr[:-1] in BOTH cases: `lay` applies the reversal afterwards, so trimming the
        # last row here always removes the OUTER one.
        rb = add("ring" + nm, *lay(Pr[:-1], rev))
        rb.faces[RAD_LO(rev)] = "wall"                             # the cylinder
        add("tran" + nm, *lay(Pt, rev))

    # ---- the eight RECTANGULAR blocks ---------------------------------------------------
    # [:-1] / [1:]: the transition owns the +-L2 perimeter, so the tiles start one node out.
    xW = -_stretch(L2, x_in, ds_seam, grow, ds_far)[::-1][:-1]
    xE = _stretch(L2, x_out, ds_seam, grow, ds_wake)[1:]
    yS = -_stretch(L2, y_half, ds_seam, grow, ds_far)[::-1][:-1]
    yN = _stretch(L2, y_half, ds_seam, grow, ds_far)[1:]
    for cn, cx in (("W", xW), ("C", tC), ("E", xE)):
        for rn, ry in (("S", yS), ("C", tC), ("N", yN)):
            if cn == "C" and rn == "C":
                continue                                    # the ring lives here
            if cn == "W":
                # THE WEST COLUMN IS TRANSPOSED: axis 0 = y DESCENDING, axis 1 = x. That makes
                # WC's seam with tranW a LIKE-AXIS join, which leaves tranW carrying ONE
                # rotating seam instead of two. A block with two rotating seams needs a
                # COMPOSED axis map for the corner ghost between them -- seam_axis_map resolves
                # direct neighbours only -- and tranW was the only such block and the only one
                # that diverged (ringS and tranS are transposed with a rotating seam apiece and
                # are stable). Reversing y as well as swapping keeps the block right-handed;
                # swapping alone would make J negative, the bug this grid already hit once.
                Y2, X2 = np.meshgrid(ry[::-1], cx, indexing="ij")
                b = add(cn + rn, X2, Y2)
                b.faces[face_id(1, 0)] = "wall"             # inlet plane x = -x_in
                if rn == "N": b.faces[face_id(0, 0)] = "wall"   # lateral y = +y_half
                if rn == "S": b.faces[face_id(0, 1)] = "wall"   # lateral y = -y_half
            else:
                X2, Y2 = np.meshgrid(cx, ry, indexing="ij")
                b = add(cn + rn, X2, Y2)
                if cn == "E": b.faces[face_id(0, 1)] = "wall"   # outlet plane
                if rn == "S": b.faces[face_id(1, 0)] = "wall"   # lateral y = -y_half
                if rn == "N": b.faces[face_id(1, 1)] = "wall"   # lateral y = +y_half

    # ---- connections -------------------------------------------------------------------
    # Orientation is DERIVED (see _auto_conn), never hand-written: on the west and south the
    # radial axis now lives on axis 1, so those seams join axis 1 to axis 0.
    def CN_(ba, fa, bb, fb):
        conns.append(_auto_conn(blocks, ba, fa, bb, fb))

    SIDES = ("E", "N", "W", "S")
    rv = lambda nm: nm in REVERSED
    for pre in ("ring", "tran"):                            # each layer closes on itself
        for a, b in (("E", "N"), ("N", "W"), ("W", "S"), ("S", "E")):
            CN_(idx[pre + a], TAN_1(rv(a)), idx[pre + b], TAN_0(rv(b)))
    for nm in SIDES:                                        # ring -> transition, radially
        CN_(idx["ring" + nm], RAD_HI(rv(nm)), idx["tran" + nm], RAD_LO(rv(nm)))
    # transition -> tile
    CN_(idx["tranE"], RAD_HI(False), idx["EC"], face_id(0, 0))
    CN_(idx["tranN"], RAD_HI(False), idx["CN"], face_id(1, 0))
    CN_(idx["WC"], face_id(1, 1), idx["tranW"], RAD_HI(True))       # now LIKE-AXIS
    CN_(idx["CS"], face_id(1, 1), idx["tranS"], RAD_HI(True))
    # The west column runs y DESCENDING on axis 0, so its north/south faces are the opposite
    # ends from the other two columns, and its east face is axis 1.
    for cn in ("W", "C", "E"):                                          # column: S-C-N
        for r0_, r1_ in (("S", "C"), ("C", "N")):
            if cn == "C" and "C" in (r0_, r1_):
                continue
            if cn == "W":
                CN_(idx[cn + r1_], face_id(0, 1), idx[cn + r0_], face_id(0, 0))
            else:
                CN_(idx[cn + r0_], face_id(1, 1), idx[cn + r1_], face_id(1, 0))
    for rn in ("S", "C", "N"):                                          # row: W-C-E
        for c0, c1 in (("W", "C"), ("C", "E")):
            if rn == "C" and "C" in (c0, c1):
                continue
            fa = face_id(1, 1) if c0 == "W" else face_id(0, 1)
            CN_(idx[c0 + rn], fa, idx[c1 + rn], face_id(0, 0))
    dom = Domain(blocks, conns), idx
    # SIGNED Jacobian, not |J|. Reversing one index axis makes a block left-handed, every
    # contravariant component then carries an extra sign, and NOTHING else catches it:
    # validate() passed, the seams were node-coincident, the scalar halo was exact and the
    # metrics were finite. It showed up only as a flux ratio of -1.000 across six seams, after
    # a solver run. Reporting min|J| is what hid it for two rounds of debugging.
    d0 = dom[0]
    d0.prepare_geometry()
    left = [nm for nm, b in idx.items() if d0.block_metrics_cached(b)[0].min() <= 0.0]
    if left:
        raise AssertionError(f"left-handed blocks (signed J <= 0): {sorted(left)}")
    return dom
