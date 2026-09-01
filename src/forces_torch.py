"""`src/forces.py` in torch: C_D and C_L as differentiable functions of the fields.

Stage 8 of `reference/nn_multiblock_plan.md`. Every loss in Stages 1-7 is a field difference on a
periodic box; the objectives that make a multi-block gradient worth having live on solid
surfaces. This makes the traction integral differentiable so drag and lift can BE the loss.

WHY A MIRROR AND NOT A MATRIX. The flux operators became sparse matrices because their stencils
are awkward and their verification needed a composition identity. The force is the opposite: a
short, flat piece of arithmetic over one face, so the direct port is shorter than its own
assembly would be, and it is verified the same way -- against the NumPy original on random
fields, to machine precision.

GEOMETRY IS CONSTANT AND CARRIES NO GRADIENT. Area vectors, metrics and the quadrature weights
come straight from `src.forces` and are converted once. Only u, v, w and p carry gradients, which
is exactly the split `torch.sparse.mm` relies on elsewhere in this port.

THE UNIFORM-WALL ASSUMPTION IS INHERITED, NOT RE-DERIVED. `src.forces` drops the two tangential
derivatives because the velocity is identically zero along a no-slip wall. This port makes the
same assumption, so it is only the traction on a stationary uniform wall -- and
`assert_uniform_wall` from the NumPy module is the thing that checks it. A moving or blowing wall
needs the full gradient in both.
"""
import numpy as np
import torch

from src.forces import _METRIC_KEYS, face_area_vectors
from src.multiblock import face_axis_side, face_slice


def face_geometry(d, faces):
    """Everything constant about a face set, converted once: ndA, |dA|, n_hat, grad(xi), h."""
    out = []
    for b, fid in faces:
        axis, side = face_axis_side(fid)
        blk = d.blocks[b]
        fs = face_slice(fid)
        S = face_area_vectors(d, b, fid).reshape(-1, 3)
        ndA = (1.0 if side == 0 else -1.0) * S
        dA = np.linalg.norm(ndA, axis=1)
        nhat = np.divide(ndA, dA[:, None], out=np.zeros_like(ndA), where=dA[:, None] > 0)
        _, met = d.block_metrics_cached(b)
        gxi = np.stack([met[k][fs].ravel() for k in _METRIC_KEYS[axis]], axis=-1)
        out.append({
            "b": b, "axis": axis, "side": side, "h": blk.h[axis], "fs": fs,
            "ndA": torch.as_tensor(ndA), "dA": torch.as_tensor(dA),
            "nhat": torch.as_tensor(nhat), "gxi": torch.as_tensor(gxi),
        })
    return out


def _wall_normal_derivative(t, axis, side, h):
    """Second-order one-sided derivative at the wall face, signed along increasing index."""
    take = lambda k: torch.index_select(t, axis, torch.tensor([k])).reshape(-1)
    if side == 0:
        return (-3.0 * take(0) + 4.0 * take(1) - take(2)) / (2.0 * h)
    n = t.shape[axis]
    return (3.0 * take(n - 1) - 4.0 * take(n - 2) + take(n - 3)) / (2.0 * h)


def surface_force(geom, u, v, w, p, nu):
    """Force on the solid bounded by `geom`, from per-block torch tensors.

    Returns a dict of length-3 tensors: total, pressure, viscous, and the viscous split into
    tangential and normal. The normal part vanishes analytically on a no-slip wall, so it is
    reported separately -- on the square cylinder it is the same size as the genuine friction
    drag and of opposite sign, and folding it in hides that.
    """
    zero = torch.zeros(3, dtype=torch.float64)
    out = {k: zero.clone() for k in
           ("pressure", "viscous", "viscous_tangential", "viscous_normal")}
    for g in geom:
        b, axis, side, fs = g["b"], g["axis"], g["side"], g["fs"]
        ndA, dA, nhat, gxi = g["ndA"], g["dA"], g["nhat"], g["gxi"]

        out["pressure"] = out["pressure"] - (p[b][fs].reshape(-1, 1) * ndA).sum(dim=0)

        dudn = torch.stack([_wall_normal_derivative(f[b], axis, side, g["h"])
                            for f in (u, v, w)], dim=-1)
        trac = nu * (dudn * (gxi * nhat).sum(dim=1, keepdim=True)
                     + gxi * (dudn * nhat).sum(dim=1, keepdim=True))
        t_n = (trac * nhat).sum(dim=1, keepdim=True) * nhat
        out["viscous_normal"] = out["viscous_normal"] + (t_n * dA[:, None]).sum(dim=0)
        out["viscous_tangential"] = (out["viscous_tangential"]
                                     + ((trac - t_n) * dA[:, None]).sum(dim=0))

    out["viscous"] = out["viscous_tangential"] + out["viscous_normal"]
    out["total"] = out["pressure"] + out["viscous"]
    return out


def coefficients(geom, u, v, w, p, nu, span, D=1.0, U=1.0, tangential_only=False):
    """(C_D, C_L) as differentiable scalars.

    `tangential_only` drops the viscous NORMAL component, which is exactly zero on a no-slip
    wall and is discretisation error here. It is off by default because the honest number is the
    full traction -- but a network told to reduce C_D can reduce that error instead of the drag,
    so the training loss is the place to turn it on. 8.3 checks the two gradients differ.
    """
    F = surface_force(geom, u, v, w, p, nu)
    T = F["total"] - (F["viscous_normal"] if tangential_only else 0.0)
    q = 0.5 * U * U * D * span
    return T[0] / q, T[1] / q
