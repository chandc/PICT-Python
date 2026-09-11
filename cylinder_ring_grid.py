"""Butterfly + wake-block grid for the ROUND cylinder in a rectangle.

Nine blocks: 4 ring quarters (circle -> square frame), 4 trapezoids (frame ->
rectangle, cut by diagonals from frame corners to rectangle corners), 1 tensor
wake block east of X_HAND carrying the validated wake plateau, with the Dong
outflow on its flat far face. reference/oring_rect_design.md holds the design;
the seam-machinery constraints that force this topology (whole-face
equal-count connections, single-owner lines, no corner logic, tensor-only
`background`) are documented there.

OWNERSHIP. E and W quarters/trapezoids store BOTH their corner rays, so the
wake block and the inflow column close the rectangle's full height; N and S
store interior rays only. Every corner ray has exactly one owner, and every
seam interleaves single-owner lines. At radial interfaces the OUTER block owns
the shared line (trap owns the frame line, wake owns the X_HAND line); inner
blocks drop their last radial layer.

COUNTS FOLLOW SPACING (the square_domain lesson): the N/S tangential
distributions ramp geometrically from the east spacing at their east corner to
the working coarse spacing and hold; their node counts are derived, not
prescribed. All blocks share computational h = (1,1,1) -- a gauge the metrics
absorb, and the diffusion assembly refuses seams whose h differ.
"""
import os

import numpy as np

from src.multiblock import Block, Connection, Domain, face_id

D = 1.0
R_CYL = 0.5 * D
# PICT_Y_HALF (in diameters) widens the lateral extent for confinement
# studies: R11 at +-10 D measured St 0.1673 vs open-domain 0.164, the
# documented blockage shift; +-20 D is the attribution check. Everything
# downstream -- trapezoid radial counts, BC classification, seam layout --
# adapts through the geometry, and validate() re-audits the result.
X_IN, X_OUT = 10.0 * D, 30.0 * D
Y_HALF = float(os.environ.get("PICT_Y_HALF", "10")) * D
X_HAND = 7.0 * D                 # butterfly ends; tensor wake block begins


def _geometric_weights(first_frac, ratio):
    """nodes on [0,1]: first spacing ~first_frac, growth `ratio`. The whole
    array is rescaled to end exactly at 1 (a few percent on every spacing)
    rather than snapping a fractional last cell, which validate() flags."""
    w = [0.0]
    dr = first_frac
    while w[-1] < 1.0:
        w.append(w[-1] + dr)
        dr *= ratio
    w = np.array(w)
    return w / w[-1]


def _ramp_hold(d0, d1, ratio):
    """param nodes on [0,1]: spacing ramps d0 -> d1 at `ratio`, then holds;
    rescaled to land exactly on 1 (no snapped fractional last cell)."""
    t, dt = [0.0], d0
    while t[-1] < 1.0:
        t.append(t[-1] + dt)
        dt = min(dt * ratio, d1)
    t = np.array(t)
    return t / t[-1]


def _two_sided_n(L, d0, d1, n):
    """n nodes over [0,1] (ray length L): first spacing d0/L, LAST spacing
    d1/L, interior log-linear to a bulge B solved by bisection so the sum
    lands exactly on L. The Vinokur-style answer to a fan of rays with very
    different lengths that must all END at one prescribed spacing: at
    Y_HALF=20 the east trap's columns span 6-20 D, and pure geometric
    distributions end anywhere from 0.1 to 0.8 against the wake block's
    single dx -- a 3.8x seam FAIL no single-parameter family can fix."""
    m = n - 1                                 # cells
    k = np.arange(m, dtype=float)
    mid = max(1, m // 2)

    def spac(B):
        up = d0 * (B / d0) ** (np.minimum(k, mid) / mid)
        dn = B * (d1 / B) ** (np.maximum(k - mid, 0.0) / max(m - 1 - mid, 1))
        return np.where(k <= mid, up, dn)

    lo, hi = 1e-9 * L, 50.0 * L
    for _ in range(90):
        B = np.sqrt(lo * hi)
        if spac(B).sum() < L:
            lo = B
        else:
            hi = B
    w = spac(np.sqrt(lo * hi))
    w = w * (L / w.sum())                     # close the bisection residual
    return np.concatenate([[0.0], np.cumsum(w)]) / L


def _ramp_hold_sym(d0, d1, ratio):
    """param nodes on [0,1]: BOTH end spacings d0, geometric growth `ratio`
    to a plateau <= d1, symmetric about 0.5. For a wall whose two corners
    both meet finer neighbours: at Y_HALF=20 the west wall (40 D) met the
    north/south walls (17 D) with a 2.35x corner jump under the uniform
    parameterisation, a validate() FAIL."""
    ds, dt = [], d0
    s = 0.0
    while s < 0.5:
        ds.append(dt)
        s += dt
        dt = min(dt * ratio, d1)
    half = np.concatenate([[0.0], np.cumsum(ds)])
    half = half * (0.5 / half[-1])
    return np.concatenate([half, (1.0 - half[::-1])[1:]])


def _ramp_then_uniform(L, d0, d1, ratio):
    """node count estimate for a ray of length L ramping d0 -> d1 then flat."""
    w, dr = [0.0], d0
    while dr < d1 and w[-1] + dr < L:
        w.append(w[-1] + dr)
        dr *= ratio
    return len(w) + max(1, int(np.ceil((L - w[-1]) / d1)))


def _geometric_n(L, d0, n):
    """n nodes over [0,1] (ray length L): first spacing d0/L, geometric ratio
    solved by bisection so the sum lands exactly on L. Smooth everywhere."""
    m = n - 1                                 # cells
    target = L / d0
    if abs(target - m) < 1e-9:
        return np.linspace(0.0, 1.0, n)
    lo, hi = (1.0000001, 1.5) if target > m else (0.5, 0.9999999)
    for _ in range(80):
        r = 0.5 * (lo + hi)
        ssum = (r ** m - 1.0) / (r - 1.0)
        # ssum is monotone increasing in r in BOTH branches
        if ssum < target:
            lo = r
        else:
            hi = r
    r = 0.5 * (lo + hi)
    w = np.concatenate([[0.0], np.cumsum(d0 * r ** np.arange(m))])
    return w / w[-1]


def _coons(edge_b, edge_t, edge_l, edge_r):
    """Coons patch; b/t along axis1 (n nodes), l/r along axis0 (m). -> (m,n,2)."""
    m, n = len(edge_l), len(edge_b)
    u = np.linspace(0.0, 1.0, m)[:, None, None]
    v = np.linspace(0.0, 1.0, n)[None, :, None]
    B, T = edge_b[None, :, :], edge_t[None, :, :]
    L, R = edge_l[:, None, :], edge_r[:, None, :]
    P00, P01, P10, P11 = edge_b[0], edge_b[-1], edge_t[0], edge_t[-1]
    return ((1 - u) * B + u * T + (1 - v) * L + v * R
            - ((1 - u) * (1 - v) * P00 + (1 - u) * v * P01
               + u * (1 - v) * P10 + u * v * P11))


def ring_rect_domain(n_east=97, side_dt=0.025, nz=8, span=4.0 * D, L1=1.0 * D,
                     first=0.010 * D, ring_ratio=1.12, trap_ratio=1.10,
                     wake_dx=0.15 * D, wake_hold=15.0 * D, wake_ratio=1.06):
    """Build the 9-block domain. Returns (Domain, idx dict)."""
    from square_cylinder_grid import _ramp_plateau_stretch

    z = np.arange(nz) / nz * span
    h = (1.0, 1.0, 1.0)
    de = 1.0 / (n_east - 1)                     # east tangential param spacing

    fc = {"se": np.array([L1, -L1]), "ne": np.array([L1, L1]),
          "nw": np.array([-L1, L1]), "sw": np.array([-L1, -L1])}
    rc = {"se": np.array([X_HAND, -Y_HALF]), "ne": np.array([X_HAND, Y_HALF]),
          "nw": np.array([-X_IN, Y_HALF]), "sw": np.array([-X_IN, -Y_HALF])}

    # tangential parameters per side (0 -> 1 from start corner to end corner)
    t_E = np.linspace(0.0, 1.0, n_east)
    t_N = _ramp_hold(0.72 * de, side_dt, 1.12)   # ne -> nw: fine at east corner;
    # 0.72*de: geometric-mean start between the frame-end and outer-end fan
    # scales of the adjacent east block, splitting the seam mismatch both ways
    # West wall: uniform while its corner spacing stays within the 1.2x
    # seam-jump budget of the north/south walls' hold spacing; end-graded
    # symmetric otherwise (the Y_HALF=20 confinement domain, where the side
    # wall is 40 D against the 17 D top wall). The guard keeps the validated
    # Y_HALF=10 grid BIT-IDENTICAL to the R11 configuration.
    _L_top = float(np.linalg.norm(rc["nw"] - rc["ne"]))
    _L_side = float(np.linalg.norm(rc["sw"] - rc["nw"]))
    _d0_W = side_dt * _L_top / _L_side
    if _d0_W < side_dt / 1.2:
        # OUTER wall graded (corner spacing matches the north/south holds);
        # INNER (frame) stays uniform -- the frame sides are equal length, so
        # the frame-corner match needs uniformity while the wall-corner match
        # needs grading, and a quarter's two ends may use different params
        # (the east quarter always has).
        t_W_out = _ramp_hold_sym(_d0_W, side_dt, 1.12)
        t_W = np.linspace(0.0, 1.0, len(t_W_out))
    else:
        t_W = np.linspace(0.0, 1.0, int(round(1.0 / side_dt)) + 1)
        t_W_out = t_W
    t_S = 1.0 - _ramp_hold(0.72 * de, side_dt, 1.12)[::-1]   # sw -> se: mirrored

    # E outer (the wake block's y lines): centreline-clustered tanh blended
    # with uniform so corner spacing stays within the N/S grading's reach
    t0 = np.linspace(-1.0, 1.0, n_east)
    yb = np.tanh(2.0 * t0) / np.tanh(2.0)
    tE_out = 0.5 * (0.65 * yb + 0.35 * t0 + 1.0)
    tE_out = (tE_out - tE_out[0]) / (tE_out[-1] - tE_out[0])

    quarters = (("E", "se", "ne", t_E, tE_out, (0, None)),
                ("N", "ne", "nw", t_N, t_N, (1, -1)),
                ("W", "nw", "sw", t_W, t_W_out, (0, None)),
                ("S", "sw", "se", t_S, t_S, (1, -1)))

    w_ring = _geometric_weights(first / (L1 - R_CYL), ring_ratio)
    ring_last = (w_ring[-1] - w_ring[-2]) * (L1 - R_CYL)   # outermost ring dr

    blocks, idx, conns = [], {}, []

    def add_block(name, X2, Y2):
        n0, n1 = X2.shape
        X = np.repeat(X2[:, :, None], nz, axis=2)
        Y = np.repeat(Y2[:, :, None], nz, axis=2)
        Z = np.broadcast_to(z[None, None, :], (n0, n1, nz)).copy()
        b = Block((n0, n1, nz), X, Y, Z, h, period=(1.0, 1.0, span))
        b.faces[face_id(2, 0)] = b.faces[face_id(2, 1)] = "periodic"
        idx[name] = len(blocks)
        blocks.append(b)
        return b

    # PASS 1: geometry prep. The trap radial count must be identical for all
    # four traps (trap-trap connections are whole-face equal-shape), and the
    # count is set by the LONGEST ray anywhere, so every quarter's per-column
    # weights are computed before any block is built.
    wake_dx0 = 0.20 * D          # the wake block's first x spacing; the east
    prep = []                     # trap's columns are built to END near this
    for name, c0, c1, t_in, t_out, (i0, i1) in quarters:
        Ef = fc[c0][None, :] * (1 - t_in[:, None]) + fc[c1][None, :] * t_in[:, None]
        O = rc[c0][None, :] * (1 - t_out[:, None]) + rc[c1][None, :] * t_out[:, None]
        Lcol = np.linalg.norm(O - Ef, axis=1)
        prep.append((name, c0, c1, Ef, O, Lcol, (i0, i1)))
        if name == "E":
            nrt_e = max(_ramp_then_uniform(L, ring_last, 0.55 * wake_dx0, trap_ratio)
                        for L in Lcol)
    nrt = nrt_e
    for name, c0, c1, Ef, O, Lcol, _ in prep:
        if name != "E":
            nrt = max(nrt, max(len(_geometric_weights(ring_last / L, trap_ratio))
                               for L in Lcol))

    # PASS 2: build. Per-column radial distributions resampled to the shared
    # count; each ray starts at the ring's last dr, so ring->trap spacing is
    # continuous on EVERY ray; adjacent traps share their diagonal column's
    # length, so the seam-line nodes coincide exactly.
    for name, c0, c1, Ef, O, Lcol, (i0, i1) in prep:
        th = np.arctan2(Ef[:, 1], Ef[:, 0])
        C = R_CYL * np.column_stack([np.cos(th), np.sin(th)])
        Xr = C[None, :, 0] * (1 - w_ring[:, None]) + Ef[None, :, 0] * w_ring[:, None]
        Yr = C[None, :, 1] * (1 - w_ring[:, None]) + Ef[None, :, 1] * w_ring[:, None]
        rb = add_block("ring" + name, Xr[:-1, i0:i1], Yr[:-1, i0:i1])
        rb.faces[face_id(0, 0)] = "wall"                    # cylinder, no-slip

        W = np.empty((nrt, len(Ef)))
        # LEGACY (Y_HALF = 10): pure geometric per column, exactly the R11
        # validated configuration, preserved bit-identical. Wider domains use
        # two-sided distributions: every column ends at wake_dx0, so the
        # wake seam and the trap-trap seams all match by construction.
        legacy = abs(Y_HALF - 10.0 * D) < 1e-12
        for jcol, L in enumerate(Lcol):
            W[:, jcol] = (_geometric_n(L, ring_last, nrt) if legacy else
                          _two_sided_n(L, ring_last, wake_dx0, nrt))
        Xt = Ef[None, :, 0] * (1 - W) + O[None, :, 0] * W
        Yt = Ef[None, :, 1] * (1 - W) + O[None, :, 1] * W
        # ALL traps keep their full radial extent: the trap-trap diagonal
        # seams demand one shared radial count, and N/W/S end at physical
        # walls. The E-wake interface ownership therefore flips: E owns the
        # X_HAND line and the wake block drops its first column instead.
        tb = add_block("trap" + name, Xt[:, i0:i1], Yt[:, i0:i1])
        if name != "E":
            tb.faces[face_id(0, 1)] = "wall"                # rectangle side
        else:
            # the E columns' end spacings, for matching the wake's first dx
            e_ends = (W[-1, :] - W[-2, :]) * Lcol

    # --- wake block: pure tensor, owns the X_HAND line and the full height --
    # the E columns end at spacings spanning [min, max]; a tensor block can
    # match only one value, and the geometric mean splits the seam mismatch
    # evenly (sqrt(spread) on each side, warn-level, not FAIL-level)
    dx0_w = float(np.sqrt(e_ends.min() * e_ends.max()))
    xw = _ramp_plateau_stretch(X_HAND, X_OUT, dx0_w, wake_dx,
                               wake_hold, 1.10, wake_ratio)
    yw = (rc["se"][None, :] * (1 - tE_out[:, None])
          + rc["ne"][None, :] * tE_out[:, None])[:, 1]
    xw = xw[1:]                    # X_HAND line is owned by the east trap
    Xw, Yw = np.meshgrid(xw, yw, indexing="ij")
    wb = add_block("wake", Xw, Yw)
    wb.faces[face_id(0, 1)] = "wall"                        # Dong plane (role via BC module)
    wb.faces[face_id(1, 0)] = wb.faces[face_id(1, 1)] = "wall"   # slip laterals

    # NOTE (2026-09-11, measured): interior Laplacian smoothing of the trap
    # blocks was tried and REJECTED. The worst in-plane skew (0.83 / 34 deg,
    # east trap outer corners) sits at PINNED corner nodes an interior
    # smoother cannot move, and the pervasive ~0.69 (46 deg) along every
    # diagonal seam is TOPOLOGICAL -- the ray directions are fixed by the
    # decomposition. 30 Jacobi sweeps changed the worst skew by 0.001; 100
    # sweeps degraded seam spacing to 2 validate() FAILs while the skew
    # stayed 0.83. Non-orthogonality on this grid is handled where it can
    # be: the implicit_cross pressure projection.
    order = ("E", "N", "W", "S")
    for k, name in enumerate(order):
        nxt = order[(k + 1) % 4]
        conns.append(Connection(idx["ring" + name], face_id(1, 1),
                                idx["ring" + nxt], face_id(1, 0)))
        conns.append(Connection(idx["trap" + name], face_id(1, 1),
                                idx["trap" + nxt], face_id(1, 0)))
        conns.append(Connection(idx["ring" + name], face_id(0, 1),
                                idx["trap" + name], face_id(0, 0)))
    conns.append(Connection(idx["trapE"], face_id(0, 1), idx["wake"], face_id(0, 0)))

    return Domain(blocks, conns), idx


if __name__ == "__main__":
    d, idx = ring_rect_domain()
    print(f"  round-cylinder butterfly: {len(d.blocks)} blocks, {d.n_cells:,} cells, "
          f"{len(d.connections)} connections")
    probs = d.validate()
    fails = [p for p in probs if "FAIL" in p]
    print(f"  validate(): {len(probs)} problem(s), {len(fails)} FAIL")
    for p in probs[:8]:
        print(f"    {p[:132]}")
    Jmin = min(d.block_metrics_cached(b)[0].min() for b in range(len(d.blocks)))
    print(f"  min(J) = {Jmin:.4e}   {'valid' if Jmin > 0 else 'TANGLED'}")
    for nm, b in ((k, d.blocks[v]) for k, v in idx.items()):
        print(f"    {nm:6s}: {b.shape}")
