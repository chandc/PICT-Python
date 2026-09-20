"""Boundary conditions for the round cylinder on the butterfly-in-rectangle
grid, classified BY GEOMETRY (the square_cylinder_bc rationale: a hand-written
face list silently leaves faces at the no-slip default).

Face labels are documentation; the u_bc/v_bc/w_bc arrays are the boundary
condition. The outlet is registered as an outflow spec for the Dong machinery
and never prescribed here. Unlike the square case there are no obstacle corner
nodes to pin: the body is smooth and every circle node is stored by a ring
quarter's wall face.
"""
import numpy as np

from src.multiblock import face_slice, FACE_NAMES
from cylinder_ring_grid import D, R_CYL, X_IN, X_OUT, Y_HALF, X_HAND

U_INF = 1.0
TOL = 1e-9


def classify(d):
    """(block, face_id) -> 'inlet' | 'outlet' | 'lateral' | 'body'."""
    out = {}
    for b, blk in enumerate(d.blocks):
        for fid, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected"):
                continue
            fs = face_slice(fid)
            x, y = blk.x[fs], blk.y[fs]
            r = np.hypot(x, y)
            if np.all(np.abs(r - R_CYL) < 1e-7):
                out[(b, fid)] = "body"
            elif np.all(np.abs(x + X_IN) < TOL):
                out[(b, fid)] = "inlet"
            elif np.all(np.abs(x - X_OUT) < TOL):
                out[(b, fid)] = "outlet"
            elif np.all(np.abs(np.abs(y) - Y_HALF) < TOL):
                out[(b, fid)] = "lateral"
            else:
                raise ValueError(
                    f"block {b} face {FACE_NAMES[fid]} is a domain boundary on no known "
                    f"surface: x [{x.min():.3f},{x.max():.3f}] y [{y.min():.3f},{y.max():.3f}] "
                    f"r [{r.min():.3f},{r.max():.3f}]. Left alone it becomes a no-slip wall.")
    return out


def seam_endpoint_columns(d):
    """Boundary nodes that NO face classifies: the trapezoid diagonal seam
    rays end exactly at (X_HAND, +-Y_HALF) and (-X_IN, +-Y_HALF). Their owner
    block (the E/W trapezoid, which owns both corner rays) has all four
    in-plane faces CONNECTED, so wall_mask -- which skips connected faces --
    never marks these nodes and the freestream BC never reaches them. Left
    alone, each such column settles at a fixed point of the edge-replicated
    corner discretisation: measured u = 1.41 at (7, +-10) for the entire R11
    run, frozen over 300 time units, and it was the far-field metric's 0.4157
    floor all along."""
    cols = []
    for b, blk in enumerate(d.blocks):
        x, y = blk.x[:, :, 0], blk.y[:, :, 0]
        on = ((np.abs(np.abs(y) - Y_HALF) < 1e-9) &
              ((np.abs(x - X_HAND) < 1e-9) | (np.abs(x + X_IN) < 1e-9)))
        for i, j in zip(*np.where(on)):
            cols.append((b, int(i), int(j)))
    return cols


def apply(m, d, kind="dong"):
    """Write boundary values into the solver; register the Dong outlet.

    PICT_LATERAL=slip: the lateral walls become free-slip instead of
    prescribed freestream. The solver has no Neumann velocity faces, so slip
    is imposed as LAGGED DIRICHLET: the runner copies the adjacent interior
    row's tangential velocity into the wall row's bc arrays before every
    step (v stays 0). Discretely du/dn = 0, v = 0 -- no wall shear sheet,
    and the seam-endpoint corner columns follow the interior instead of
    clamping freestream against the confinement overspeed (the visible
    "dot" at (X_HAND, +-Y_HALF) under the freestream laterals).

    SLIP IS NOW THE DEFAULT, because it is what HydroGym specifies. Its cylinder env sets
    `DirichletBC(V.sub(1), Constant(0.0), FREESTREAM)  # Symmetry BCs` -- the y-component ONLY,
    with u left free -- and the full-vector version, `DirichletBC(V, U_inf, FREESTREAM)`, sits
    COMMENTED OUT on the line directly above it in their source. We were running the line they
    discarded: measured u = 1.0000 and v = 0.00000 pinned on every lateral face. Pinning u
    forbids the blockage acceleration that beta = 0.10 demands -- continuity alone wants ~11%
    speed-up in the gap -- so it stiffens the confinement and inflates drag.

    The two lateral conditions are DIFFERENT physical configurations with different blockage
    laws, so this changes the numbers: R11/R12 were measured under freestream and are not
    comparable across the switch. PICT_LATERAL=freestream restores the old behaviour."""
    import os as _os
    slip = _os.environ.get("PICT_LATERAL", "slip") == "slip"
    roles = classify(d)
    outflow = []
    m.slip_faces = []
    m.slip_corners = []
    for (b, fid), role in roles.items():
        fs = face_slice(fid)
        if role == "body":
            u, v = 0.0, 0.0
        elif role in ("inlet", "lateral"):
            # freestream on the laterals (square_cylinder_bc's measured
            # rationale: walls there squeeze the wake; prescribed freestream
            # passes the displacement flux)
            u, v = U_INF, 0.0
        else:
            outflow.append((b, fid, U_INF, kind))
            continue
        for arr, bc, val in ((m.u, m.u_bc, u), (m.v, m.v_bc, v), (m.w, m.w_bc, 0.0)):
            bc[b][fs] = val
            arr[b][fs] = val
        if slip and role == "lateral":
            m.slip_faces.append((b, fid))
    # Pin the seam-endpoint corner columns and ENROLL them in the solver's
    # Dirichlet set -- writing the bc arrays alone does nothing for a node
    # wall_mask never marked.
    # PICT_NO_CORNER_PIN=1: diagnostic kill-switch for A/B isolation of the
    # seam-endpoint treatment (the R11 frozen-corner cure).
    import os as _os
    corners = ([] if _os.environ.get("PICT_NO_CORNER_PIN") == "1"
               else seam_endpoint_columns(d))
    for b, i, j in corners:
        for arr, bc, val in ((m.u, m.u_bc, U_INF), (m.v, m.v_bc, 0.0),
                             (m.w, m.w_bc, 0.0)):
            bc[b][i, j, :] = val
            arr[b][i, j, :] = val
        if slip:
            # under slip the X_HAND corner columns follow the interior like
            # every other wall node (copy source = one node inboard in j);
            # the inlet corners stay pinned freestream (u=1 is physical
            # there, one cell from the prescribed inflow).
            blk = d.blocks[b]
            if abs(float(blk.x[i, j, 0]) - X_HAND) < 1e-9:
                jsrc = j - 1 if float(blk.y[i, j, 0]) > 0 else j + 1
                m.slip_corners.append((b, i, j, jsrc))
    if corners and hasattr(m, "wall"):
        for b, i, j in corners:
            m.wall[d.global_ids(b)[i, j, :]] = True
        m.interior = np.where(~m.wall)[0]
        m.bnd = np.where(m.wall)[0]
    m.outflow = outflow
    return roles


def probe_index(d, x_probe=2.0 * D, y_probe=0.5 * D):
    """Block and index of the node nearest (x_probe, y_probe)."""
    best, pb, pk = np.inf, None, None
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        r = (blk.x[:, :, 0] - x_probe) ** 2 + (blk.y[:, :, 0] - y_probe) ** 2
        k = np.unravel_index(r.argmin(), r.shape)
        if r[k] < best:
            best, pb, pk = r[k], b, k
    return pb, pk


# ---------------------------------------------------------------------------------------------
# HydroGym's jet actuator, transcribed from hydrogym/firedrake/envs/cylinder/flow.py
# (class Cylinder, `cyl_velocity_field`), verified identical in 0.1.2.1 and 1.0.0:
#
#     omega = pi/18 ; theta_up = 0.5*pi ; theta_lo = -0.5*pi
#     A_*   = conditional(abs(theta - theta_*) < omega/2,
#                         pi/(2*omega*rad**2) * cos((pi/omega)*(theta - theta_*)), 0)
#     u_ctrl = as_tensor((x, y)) * (A_up + A_lo)
#
# THREE THINGS THAT ARE EASY TO GET WRONG, ALL TAKEN FROM THE SOURCE:
#
# NORMAL, NOT TANGENTIAL. (x, y) on the cylinder is rad * n_hat, so the actuation blows along
# the surface normal. The ROTARY actuator is the tangential one -- a different class.
#
# NOT ZERO-NET-MASS-FLUX. A_up and A_lo are both POSITIVE cosine lobes and they are ADDED, so
# both jets blow (or both suck) together. One scalar drives the pair. A ZNMF pair would be
# antisymmetric and would have real lift authority; this one has almost none, which is the
# whole point of the distinction. HydroGym's MAIA backend IS ZNMF -- `[action, -action]` per
# pair -- so the two backends are genuinely different actuators. This is the Firedrake one.
#
# THE NORMALISATION IS A UNIT FLUX PER JET. Integrating u_n over one jet:
#     int_{-w/2}^{w/2} (rad * A) * rad dtheta = rad^2 * pi/(2*omega*rad^2) * (2*omega/pi) = 1
# so each jet carries unit volumetric flux per unit control, and the pair injects 2*control.
# Peak amplitude is pi/(2*omega*rad^2) = 36, so the peak surface speed is rad*36 = 18 per unit
# control -- 1.8 at MAX_CONTROL, well above U_inf. That is HydroGym's scaling, not a typo.
MAX_CONTROL = 0.1
JET_OMEGA = np.pi / 18.0                     # 10 degrees TOTAL width
JET_CENTRES = (0.5 * np.pi, -0.5 * np.pi)    # top and bottom


def jet_amplitude(theta, rad=R_CYL):
    """HydroGym's A_up + A_lo at the given surface angles (radians)."""
    A = np.zeros_like(np.asarray(theta, dtype=float))
    for tc in JET_CENTRES:
        dth = np.arctan2(np.sin(theta - tc), np.cos(theta - tc))   # wrap to (-pi, pi]
        inside = np.abs(dth) < 0.5 * JET_OMEGA
        A = A + np.where(inside,
                         np.pi / (2.0 * JET_OMEGA * rad ** 2)
                         * np.cos((np.pi / JET_OMEGA) * dth), 0.0)
    return A


def apply_jets(m, d, control):
    """Write the jet velocity onto the cylinder faces. `control` is HydroGym's scalar action.

    Overwrites the no-slip body condition, exactly as HydroGym's `set_control` replaces the
    actuation BC each step. Returns the discrete volumetric flux actually imposed, so a caller
    can check it against the analytic 2*control.
    """
    roles = classify(d)
    flux = 0.0
    for (b, fid), role in roles.items():
        if role != "body":
            continue
        fs = face_slice(fid)
        blk = d.blocks[b]
        x, y = blk.x[fs], blk.y[fs]
        r = np.hypot(x, y)
        A = jet_amplitude(np.arctan2(y, x)) * float(control)
        # u = (x, y) * A -- the source's own form; |(x,y)| = rad on the surface
        for arr, bc, val in ((m.u, m.u_bc, x * A), (m.v, m.v_bc, y * A),
                             (m.w, m.w_bc, np.zeros_like(A))):
            bc[b][fs] = val
            arr[b][fs] = val
        # Arc-length weighted normal flux, for the conservation check. ONE z-PLANE ONLY: the
        # face carries nz planes and ravelling them together interleaves the planes, so the
        # unwrap runs over a sawtooth and the integral came out a factor nz/... wrong (measured
        # 0.104 against an analytic 0.200).
        th2 = np.arctan2(y, x)[..., 0]
        r2 = r[..., 0]
        A2 = A[..., 0] if A.ndim == th2.ndim + 1 else A.reshape(th2.shape + (-1,))[..., 0]
        if th2.size > 1:
            dth = np.gradient(np.unwrap(th2.ravel()))
            flux += float(np.sum(r2.ravel() * A2.ravel() * r2.ravel() * dth))
    return flux
