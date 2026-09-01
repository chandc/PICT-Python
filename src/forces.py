"""Pressure and viscous force on a solid surface, integrated over the faces that bound it.

WHY THIS EXISTS. Strouhal number is one scalar, and a wake can reproduce it while getting the
loads wrong. C_D and C_L rms are the next two published quantities for both the square and the
circular cylinder, and neither is reachable without a surface integral of the traction. Nothing
in the repo computed one.

THE INTEGRAL. On a surface with outward normal n (pointing OUT of the solid, into the fluid),

    F = closed integral of  sigma . n dA,      sigma = -p I + nu (grad u + grad u^T)

with p and nu both KINEMATIC here (the solver carries p/rho and nu = mu/rho), so F is a force
per unit density. C_D = F_x / (0.5 U^2 D L) with L the spanwise length.

THE AREA VECTOR IS NOT |dx| |dy|. On a curvilinear grid the face of constant xi has the vector
area

    S_xi = J grad(xi) h_eta h_zeta

and it is the same object the flux routines already use: `face_fluxes` builds J*(u . grad xi)
and `divergence` then divides by h and J, which fixes the convention -- the h_eta h_zeta factor
belongs to the area, not to the flux. Getting this wrong scales every force by a constant and
the error is invisible in a plot, which is exactly why the tests below check against closed-form
integrals rather than against a picture.

THE VELOCITY GRADIENT AT THE WALL, AND ITS ONE ASSUMPTION. In general

    du_i/dx_j = (du_i/dxi) xi_xj + (du_i/deta) eta_xj + (du_i/dzeta) zeta_xj

On a no-slip surface the velocity is identically zero ALONG the wall, so the two tangential
derivatives vanish there exactly and only the wall-normal term survives. That is what makes this
a one-sided difference in a single computational direction rather than a full gradient
reconstruction. `assert_uniform_wall` checks the assumption instead of trusting it: a moving or
non-uniform wall (a rotating cylinder, a suction slot) breaks it and the caller must be told,
not silently given a wrong shear.

The wall-normal derivative is the second-order one-sided formula, NOT the two-point difference.
Wall shear on a stretched grid is where first-order error does the most damage: the first cell
off a cylinder here is 0.006 D, so a first-order estimate of du/dn commits an O(h) error on the
one quantity the whole viscous force is made of.
"""
import numpy as np

from src.multiblock import face_id, face_slice, face_axis_side

_METRIC_KEYS = (("xi_x", "xi_y", "xi_z"),
                ("eta_x", "eta_y", "eta_z"),
                ("zeta_x", "zeta_y", "zeta_z"))


def _quadrature_weights(blk, axis, n):
    """Trapezoid weights along a tangential axis, in units of h.

    THE FIELDS ARE NODE-CENTRED, so a face of N nodes spans N-1 intervals and giving every node
    a full h over-counts the area by N/(N-1) -- 1.406x on a 9x5 face, which is a plausible
    C_D and a wrong one. End nodes get half weight.

    UNLESS THE AXIS CONTINUES. A periodic or connected end is not an end: the ring closes and
    the seam node is not duplicated, so the next node's cell is genuinely adjacent and the full
    weight is correct. That is why the cylinder O-grid integrates exactly with naive weights
    and the box does not -- both tangential directions of its body face wrap.
    """
    w = np.ones(n)
    seam = ("periodic", "connected")
    if blk.faces[face_id(axis, 0)] not in seam:
        w[0] = 0.5
    if blk.faces[face_id(axis, 1)] not in seam:
        w[-1] = 0.5
    return w


def face_area_vectors(d, b, fid):
    """Vector area per node of the (b, fid) face, pointing along INCREASING computational index.

    Sums to the true surface area vector of the face: one node owns one h_t1 x h_t2 patch of
    computational space (trapezoid-weighted at true domain ends), and J grad(xi) converts that
    to physical area.
    """
    blk = d.blocks[b]
    axis, _ = face_axis_side(fid)
    J, met = d.block_metrics_cached(b)
    fs = face_slice(fid)
    t1, t2 = [a for a in range(3) if a != axis]
    w1 = _quadrature_weights(blk, t1, blk.shape[t1]) * blk.h[t1]
    w2 = _quadrature_weights(blk, t2, blk.shape[t2]) * blk.h[t2]
    w = w1[:, None] * w2[None, :]
    keys = _METRIC_KEYS[axis]
    return np.stack([J[fs] * met[k][fs] * w for k in keys], axis=-1)


def wall_normal_derivative(arr, axis, side, h):
    """d(arr)/d(computational coord) at the wall face, second-order one-sided, signed along
    increasing index."""
    take = lambda k: np.take(arr, k, axis=axis)
    if side == 0:
        return (-3.0 * take(0) + 4.0 * take(1) - take(2)) / (2.0 * h)
    n = arr.shape[axis]
    return (3.0 * take(n - 1) - 4.0 * take(n - 2) + take(n - 3)) / (2.0 * h)


def assert_uniform_wall(u, v, w, b, fid, tol=1e-10):
    """The no-slip simplification needs the wall velocity CONSTANT over the face."""
    fs = face_slice(fid)
    for nm, f in (("u", u), ("v", v), ("w", w)):
        vals = f[b][fs]
        if vals.size and float(vals.max() - vals.min()) > tol:
            raise ValueError(
                f"block {b} face {fid}: {nm} varies by "
                f"{float(vals.max() - vals.min()):.3e} over the wall. The traction here assumes "
                f"the tangential derivatives vanish on the surface, which holds only for a "
                f"uniform wall velocity; a moving or blowing wall needs the full gradient.")


def surface_force(d, faces, u, v, w, p, nu, check_wall=True):
    """Force on the solid bounded by `faces`, a list of (block, face_id).

    Returns a dict of length-3 arrays: total, pressure, viscous, and the viscous force split
    into its tangential and normal parts. The normal points out of the SOLID: for a face on the
    low side of an axis the fluid lies at increasing index, so the outward-from-solid direction
    is +axis, and the reverse on the high side.

    WHY THE VISCOUS SPLIT IS REPORTED SEPARATELY. On a no-slip wall the NORMAL viscous stress
    vanishes analytically: 2 nu du_n/dn = 0, because continuity at the wall reads
    du_n/dn = -(du_t1/dt1 + du_t2/dt2) and both tangential derivatives are zero on a wall where
    the velocity is identically zero. Discretely it does not vanish, and on the square cylinder
    it is not small -- the front and rear faces returned -0.0299 and +0.0053 in C_D against a
    genuine friction drag of +0.025 from the two side faces, so the spurious part very nearly
    cancelled the real one and left C_D_viscous = 0.0003. Reporting the normal part makes that
    visible instead of letting it hide inside a plausible total; it is a direct measure of how
    well continuity is satisfied at the wall.
    """
    out = {k: np.zeros(3) for k in
           ("pressure", "viscous", "viscous_tangential", "viscous_normal")}
    for b, fid in faces:
        axis, side = face_axis_side(fid)
        if check_wall:
            assert_uniform_wall(u, v, w, b, fid)
        blk = d.blocks[b]
        fs = face_slice(fid)
        S = face_area_vectors(d, b, fid)                 # along increasing index
        sgn = 1.0 if side == 0 else -1.0                 # ... and out of the solid
        ndA = sgn * S.reshape(-1, 3)
        dA = np.linalg.norm(ndA, axis=1)
        nhat = np.divide(ndA, dA[:, None], out=np.zeros_like(ndA), where=dA[:, None] > 0)

        out["pressure"] += -(p[b][fs].reshape(-1, 1) * ndA).sum(axis=0)

        h = blk.h[axis]
        dudn = np.stack([wall_normal_derivative(f[b], axis, side, h).ravel()
                         for f in (u, v, w)], axis=-1)   # (n, 3): d u_i / d xi
        _, met = d.block_metrics_cached(b)
        gxi = np.stack([met[k][fs].ravel() for k in _METRIC_KEYS[axis]], axis=-1)
        # grad u = outer(du/dxi, grad xi); the tangential terms vanish on a uniform wall.
        # traction = nu [ (grad u) + (grad u)^T ] . nhat
        trac = nu * (dudn * (gxi * nhat).sum(axis=1)[:, None]
                     + gxi * (dudn * nhat).sum(axis=1)[:, None])
        t_n = (trac * nhat).sum(axis=1)[:, None] * nhat  # should be zero on a no-slip wall
        out["viscous_normal"] += (t_n * dA[:, None]).sum(axis=0)
        out["viscous_tangential"] += ((trac - t_n) * dA[:, None]).sum(axis=0)

    out["viscous"] = out["viscous_tangential"] + out["viscous_normal"]
    out["total"] = out["pressure"] + out["viscous"]
    return out


def force_coefficients(d, faces, m, span, D=1.0, U=1.0, drop_normal_stress=False):
    """(C_D, C_L) from a solver state.

    `drop_normal_stress` discards the viscous normal component, which is exactly zero on a
    no-slip wall and purely discretisation error here. It is off by default: the honest number
    is the full traction, and the size of what it would drop is the reason to look.
    """
    F = surface_force(d, faces, m.u, m.v, m.w, m.p, m.nu)
    T = F["total"] - (F["viscous_normal"] if drop_normal_stress else 0.0)
    q = 0.5 * U * U * D * span
    return float(T[0] / q), float(T[1] / q)
