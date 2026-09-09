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
from cylinder_ring_grid import D, R_CYL, X_IN, X_OUT, Y_HALF

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


def apply(m, d, kind="dong"):
    """Write boundary values into the solver; register the Dong outlet."""
    roles = classify(d)
    outflow = []
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
