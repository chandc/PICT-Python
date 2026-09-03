"""
Boundary conditions for the circular cylinder, classified BY GEOMETRY rather than by hand.

SAME DISCIPLINE AS THE SQUARE CASE, AND FOR THE SAME REASON. Face types default to "wall", so a
face you forget becomes a silently plausible no-slip surface that no assertion catches. Here every
non-seam face must fall into exactly one geometric class or `classify` raises. On the square that
discipline caught a grid whose obstacle was 1.203 x 1.107 instead of 1 x 1, with `validate()`
reporting nothing wrong.

THE FACE-TYPE STRING DOES NOT SET THE PHYSICS. `Domain.wall_mask()` tests only for
'periodic'/'connected'; every other label is the same thing -- a Dirichlet node whose value comes
from `u_bc/v_bc/w_bc`. An inflow is a face where you wrote the free stream into `u_bc`, not a face
labelled 'inflow'. The labels are documentation; the arrays are the boundary condition.

WHY THE FAR FIELD SPLITS AT BLOCK BOUNDARIES. A face carries ONE condition, so the outer ring can
only be divided into free stream and outflow where blocks meet. With the stream along +x the
downstream half is |theta| < pi/2; `outer_role` puts the division there.
"""
import numpy as np

from src.multiblock import face_slice, face_id, FACE_NAMES
from cylinder_grid import D, R_CYL, outer_role

U_INF = 1.0
TOL = 1e-6          # geometric tolerance for "is this face on that surface"


def classify(d, nblk=None, slip_sector=None):
    """(block, face) -> 'cylinder' | 'inflow' | 'slip' | 'outflow' for every non-seam face."""
    nblk = nblk or len(d.blocks)
    roles = outer_role(d, nblk, slip_sector=slip_sector)
    out = {}
    for b, blk in enumerate(d.blocks):
        for fid, kind in enumerate(blk.faces):
            if kind in ("periodic", "connected"):
                continue
            fs = face_slice(fid)
            r = np.sqrt(blk.x[fs] ** 2 + blk.y[fs] ** 2)
            if np.all(np.abs(r - R_CYL) < 1e-9):
                out[(b, fid)] = "cylinder"
            elif r.min() > 2.0 * R_CYL and (r.max() - r.min()) < TOL * max(r.max(), 1.0):
                out[(b, fid)] = roles[b]        # 'inflow', 'slip' or 'outflow'
            else:
                raise ValueError(
                    f"block {b} face {FACE_NAMES[fid]} is a domain boundary but lies on no known "
                    f"surface: r in [{r.min():.4f}, {r.max():.4f}], cylinder at {R_CYL}. Left "
                    f"alone it would silently become a no-slip wall.")
    return out


def apply(m, d, nblk=None, kind="dong", slip_sector=None):
    """Write the boundary values into the solver. Returns the classification for reporting."""
    roles = classify(d, nblk, slip_sector=slip_sector)
    outflow = []
    for (b, fid), role in roles.items():
        fs = face_slice(fid)
        if role == "cylinder":
            u, v = 0.0, 0.0                              # no-slip on the body
        elif role == "inflow":
            # FREE STREAM, prescribed. The far field is at 30 D, where the induced velocity is
            # small but not zero; prescribing (U, 0, 0) is the standard choice and is what the
            # square case measured as 100x better on global flux balance than a no-slip wall.
            u, v = U_INF, 0.0
        elif role == "slip":
            # Normal component prescribed from the free stream, tangential free. One constraint
            # per node instead of two, which is the point: see outer_role's docstring.
            outflow.append((b, fid, U_INF, "slip"))
            continue
        else:
            outflow.append((b, fid, U_INF, kind))        # Dong sets it; do not prescribe
            continue
        for arr, bc, val in ((m.u, m.u_bc, u), (m.v, m.v_bc, v), (m.w, m.w_bc, 0.0)):
            bc[b][fs] = val
            arr[b][fs] = val
    m.outflow = outflow
    return roles


def probe_index(d, x_probe=2.0 * D, y_probe=0.5 * D):
    """Block and index of the node nearest (x_probe, y_probe) -- the shedding signal."""
    best, pb, pk = np.inf, None, None
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        r = (blk.x[:, :, 0] - x_probe) ** 2 + (blk.y[:, :, 0] - y_probe) ** 2
        k = np.unravel_index(r.argmin(), r.shape)
        if r[k] < best:
            best, pb, pk = r[k], b, k
    return pb, pk
