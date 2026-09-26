"""Near- and far-field spanwise vorticity on the butterfly-in-rectangle grid.

CURVILINEAR-CORRECT: omega_z = dv/dx - du/dy is evaluated per block through
the inverse of the per-node coordinate Jacobian,

    [d/dx]   1  [ y_eta  -y_xi ] [d/dxi ]
    [d/dy] = -- [-x_eta   x_xi ] [d/deta],   J2 = x_xi*y_eta - x_eta*y_xi
             J2

with index-space derivatives from np.gradient. The first version of this
picture used axis-aligned gradients, which are garbage on the ring and
trapezoid blocks (right only on the tensor wake block) and painted a
saturated false-vorticity pattern over a perfectly healthy field.
"""
import argparse
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cylinder_ring_grid import ring_rect_domain, D, X_IN, X_OUT, Y_HALF
from src.multiblock import face_slice, face_axis_side
from src import checkpoint

U_INF = 1.0


def omega_z_block(x, y, u, v):
    """metric-aware dv/dx - du/dy on one 2-D curvilinear slice."""
    x_xi, x_eta = np.gradient(x, axis=0), np.gradient(x, axis=1)
    y_xi, y_eta = np.gradient(y, axis=0), np.gradient(y, axis=1)
    J2 = x_xi * y_eta - x_eta * y_xi
    J2 = np.where(np.abs(J2) < 1e-300, 1e-300, J2)
    u_xi, u_eta = np.gradient(u, axis=0), np.gradient(u, axis=1)
    v_xi, v_eta = np.gradient(v, axis=0), np.gradient(v, axis=1)
    dvdx = (y_eta * v_xi - y_xi * v_eta) / J2
    dudy = (-x_eta * u_xi + x_xi * u_eta) / J2
    return dvdx - dudy


def pad_seams(d, U, V, kmid):
    """Per-block padded (x, y, u, v) 2-D slices: each seam face gains the
    neighbour's first node line.

    Seam nodes are single-owner (validate() REJECTS coincident interface
    nodes), so a block's stored array stops one node short of the seam and a
    naive per-block pcolormesh leaves a one-cell white strip there. Painting
    and differentiating across the seam needs the neighbour's owned line.
    Only the fa-side block is padded: that covers each strip exactly once.
    Corners are edge-replicated (a one-pixel compromise, like the solver's
    own corner treatment).
    """
    pads = {}                                   # (block, axis, side) -> {name: line}
    for c in d.connections:
        axis, side = face_axis_side(c.fa)
        if axis > 1:
            continue                            # spanwise seam: not in-plane
        if c.axes != (0, 1):
            raise NotImplementedError(f"{c}: z-swapped seam orientation")
        entry = {}
        for name, arr3, kz in (("x", d.blocks[c.bb].x, 0), ("y", d.blocks[c.bb].y, 0),
                               ("u", U[c.bb], kmid), ("v", V[c.bb], kmid)):
            entry[name] = np.asarray(c.align(arr3[face_slice(c.fb)])[:, kz], dtype=float)
        pads[(c.ba, axis, side)] = entry

    out = []
    for b, blk in enumerate(d.blocks):
        arrs = {"x": blk.x[:, :, 0].astype(float), "y": blk.y[:, :, 0].astype(float),
                "u": U[b][:, :, kmid].astype(float), "v": V[b][:, :, kmid].astype(float)}
        for axis in (0, 1):
            for side in (0, 1):
                p = pads.get((b, axis, side))
                if p is None:
                    continue
                for name, a in arrs.items():
                    line = p[name]
                    grow = a.shape[1 - axis] - line.shape[0]
                    if grow:                    # the other axis was padded first
                        pre = 1 if (b, 1 - axis, 0) in pads else 0
                        line = np.pad(line, (pre, grow - pre), mode="edge")
                    line = line.reshape((1, -1) if axis == 0 else (-1, 1))
                    arrs[name] = np.concatenate(
                        [line, a] if side == 0 else [a, line], axis=axis)
        out.append((arrs["x"], arrs["y"], arrs["u"], arrs["v"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("field", help="results/fields/<name>.npz, or a bare tag")
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", action="store_true", help="overlay the block mesh lines")
    a = ap.parse_args()
    path = a.field if a.field.endswith(".npz") else f"results/fields/{a.field}.npz"
    name = _os.path.basename(path).replace(".npz", "")

    d, idx = ring_rect_domain(nz=a.nz)
    f, meta = checkpoint.load_fields(path)
    t = float(meta.get("time", 0.0))
    kmid = f["u"][0].shape[2] // 2

    slabs = pad_seams(d, f["u"], f["v"], kmid)

    oz, farmax = [], 0.0
    for x, y, u, v in slabs:
        oz.append(omega_z_block(x, y, u, v))
        far = (np.hypot(x, y) > 10.0) & (np.abs(y) > 4.0)
        if far.any():
            farmax = max(farmax, float(np.hypot(u - U_INF, v)[far].max()))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 2.1]})
    views = ((axes[0], (-2.5, 5.0, -2.8, 2.8), 0.35), (axes[1], (-X_IN, X_OUT, -Y_HALF, Y_HALF), 0.10))
    peak = max(np.abs(w).max() for w in oz)
    for ax, (xl, xr, yb, yt), frac in views:
        lim = frac * peak
        for b, (x2, y2, _, _) in enumerate(slabs):
            ax.pcolormesh(x2, y2, oz[b], cmap="RdBu_r",
                          vmin=-lim, vmax=lim, shading="gouraud")
            if a.grid:
                kw = dict(color="0.45", lw=0.15, alpha=0.7, zorder=4)
                ax.plot(x2, y2, **kw)
                ax.plot(x2.T, y2.T, **kw)
        ax.add_patch(plt.Circle((0, 0), 0.5 * D, fc="0.3", ec="k", zorder=5))
        ax.set(xlim=(xl, xr), ylim=(yb, yt), aspect="equal", xlabel="x/D",
               title=f"|omega_z| scale {lim:.2g}")
    axes[1].axvline(X_OUT, color="g", ls="--", lw=1)
    fig.suptitle(f"{name}  t = {t:.2f}  |omega| peak {peak:.2f}  "
                 f"far-field max|u-U| {farmax:.4f}   (green dashed = Dong plane)")
    fig.tight_layout()
    out = a.out or f"figures/rect_vorticity_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved -> {out}")
    print(f"  |omega| peak {peak:.3f}   far-field max|u-U| {farmax:.4f}")


if __name__ == "__main__":
    main()
