"""
Boundary conditions for the square cylinder, classified BY GEOMETRY rather than by hand.

WHY GEOMETRY AND NOT A HARDCODED LIST. Eight blocks have 48 faces; 20 of them are real
boundaries and the rest are seams. Writing that list by hand is how you get a face silently
left at its default -- and the default is a NO-SLIP WALL, which is a physically plausible,
completely wrong boundary that no assertion would catch. Here every non-seam face must fall
into exactly one geometric class or `classify` raises.

THE FACE-TYPE STRING DOES NOT SET THE PHYSICS. `Domain.wall_mask()` tests only for
'periodic'/'connected'; every other face type -- 'wall', 'inflow', 'outflow' -- becomes the
same thing: a Dirichlet node whose value is read from `u_bc/v_bc/w_bc`. So an inlet is not a
face labelled 'inflow', it is a face where you wrote the inlet profile into `u_bc`. The
labels are documentation. The arrays are the boundary condition.
"""
import numpy as np

from src.multiblock import face_slice, FACE_NAMES
from square_cylinder_grid import D, X_IN, X_OUT, Y_HALF

U_INF = 1.0
TOL = 1e-9


def classify(d):
    """(block, face_id) -> 'inlet' | 'outlet' | 'lateral' | 'body' for every non-seam face."""
    out = {}
    for b, blk in enumerate(d.blocks):
        for fid, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected"):
                continue
            fs = face_slice(fid)
            x, y = blk.x[fs], blk.y[fs]
            if np.all(np.abs(x - X_IN) < TOL):
                out[(b, fid)] = "inlet"
            elif np.all(np.abs(x - X_OUT) < TOL):
                out[(b, fid)] = "outlet"
            elif np.all(np.abs(np.abs(y) - Y_HALF) < TOL):
                out[(b, fid)] = "lateral"
            elif np.all(np.abs(np.abs(x) - 0.5 * D) < TOL) or \
                 np.all(np.abs(np.abs(y) - 0.5 * D) < TOL):
                out[(b, fid)] = "body"
            else:
                raise ValueError(
                    f"block {b} face {FACE_NAMES[fid]} is a domain boundary but sits on no "
                    f"known surface: x in [{x.min():.3f},{x.max():.3f}], "
                    f"y in [{y.min():.3f},{y.max():.3f}]. Left at its default it would "
                    f"silently become a no-slip wall.")
    return out


def pin_obstacle_corners(m, d):
    """No-slip the four obstacle CORNER nodes, which the face registry cannot reach.

    Each corner of the square is the meeting point of two of its faces, and in this
    non-duplicating partition it is stored by a CORNER block (LB/RB/LT/RT) whose two faces
    there are both seams -- LB's '+x' runs to MB, its '+y' to LM. So `wall_mask()`, which
    derives purely from face types, leaves it out: measured 32 free nodes (4 corners x 8
    spanwise planes) sitting exactly on solid material, where the flow would be free to pass
    through the sharpest feature of the body. These are precisely the points the shear layers
    separate from, so leaving them unconstrained corrupts the quantity being measured.

    Returns the number of nodes pinned.
    """
    h = 0.5 * D
    add = np.zeros_like(m.wall)
    for b, blk in enumerate(d.blocks):
        on = ((np.abs(np.abs(blk.x) - h) < TOL) & (np.abs(np.abs(blk.y) - h) < TOL))
        if not on.any():
            continue
        add[d.global_ids(b)[on]] = True
        for arr, bc in ((m.u, m.u_bc), (m.v, m.v_bc), (m.w, m.w_bc)):
            bc[b][on] = 0.0
            arr[b][on] = 0.0
    n = int((add & ~m.wall).sum())
    m.wall = m.wall | add
    m.interior = np.where(~m.wall)[0]
    m.bnd = np.where(m.wall)[0]
    return n


def apply(m, d, kind="dong"):
    """Write the boundary values into the solver. Returns the classification for reporting."""
    roles = classify(d)
    outflow = []
    for (b, fid), role in roles.items():
        fs = face_slice(fid)
        if role == "body":
            # no-slip on the cylinder: the only place the flow is actually held at zero
            u, v = 0.0, 0.0
        elif role in ("inlet", "lateral"):
            # FREESTREAM on the laterals, not no-slip. The solver has no slip or symmetry
            # face type, so the choice is between a wall and a prescribed value -- and walls
            # at y = +-6D would grow boundary layers that squeeze the wake and corrupt the
            # shedding frequency this case exists to measure. Measured: prescribed freestream
            # gives global flux imbalance 2.0e-06 against 1.9e-04 for no-slip, ~100x better,
            # because a no-slip lateral cannot pass the displacement flux the body induces.
            u, v = U_INF, 0.0
        else:                                   # outlet: Dong sets it, do not prescribe
            outflow.append((b, fid, U_INF, kind))
            continue
        for arr, bc, val in ((m.u, m.u_bc, u), (m.v, m.v_bc, v), (m.w, m.w_bc, 0.0)):
            bc[b][fs] = val
            arr[b][fs] = val
    m.outflow = outflow
    pin_obstacle_corners(m, d)
    return roles
