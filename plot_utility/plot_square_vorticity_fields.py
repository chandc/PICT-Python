"""Near- and far-field spanwise vorticity of a square-cylinder checkpoint.

Written for the R7 before/after: the pre-fix settled bases carry the far-field
question (the Rhie-Chow boundary defect showed up as far-field pressure/vorticity
oscillation on the cylinders), and R7's own base is the post-fix answer.

omega_z = dv/dx - du/dy per block on the block's own (nonuniform) coordinates,
mid-span slice; blocks drawn individually so no interpolation touches the data.
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

from square_cylinder_grid import square_domain, D
from src import checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("field", help="results/fields/<name>.npz")
    ap.add_argument("--ratio", type=float, default=1.10)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d, idx = square_domain(nz=a.nz, ratio=a.ratio)
    f, meta = checkpoint.load_fields(a.field)
    t = float(meta.get("time", 0.0))
    kmid = f["u"][0].shape[2] // 2

    name = a.label or _os.path.basename(a.field).replace(".npz", "")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2),
                             gridspec_kw={"width_ratios": [1.0, 2.2]})
    views = ((axes[0], (-2.5, 5.0, -2.5, 2.5), "near field"),
             (axes[1], (-5.0, 25.0, -6.0, 6.0), "far field"))

    # one shared scale per view, from the data inside that view
    oz = {}
    for b, blk in enumerate(d.blocks):
        x = blk.x[:, 0, 0]
        y = blk.y[0, :, 0]
        u = f["u"][b][:, :, kmid]
        v = f["v"][b][:, :, kmid]
        dvdx = np.gradient(v, x, axis=0, edge_order=2)
        dudy = np.gradient(u, y, axis=1, edge_order=2)
        oz[b] = (x, y, dvdx - dudy)

    for ax, (xl, xr, yb, yt), title in views:
        vals = []
        for b in oz:
            x, y, w = oz[b]
            mx = (x / D >= xl) & (x / D <= xr)
            my = (y / D >= yb) & (y / D <= yt)
            if mx.any() and my.any():
                vals.append(np.abs(w[np.ix_(mx, my)]).max())
        lim = 0.35 * max(vals) if title == "near field" else 0.10 * max(vals)
        for b in oz:
            x, y, w = oz[b]
            ax.pcolormesh(x / D, y / D, w.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="gouraud")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fc="0.3", ec="k"))
        ax.set(xlim=(xl, xr), ylim=(yb, yt), xlabel="x/D",
               ylabel="y/D", title=f"{title}  (|omega_z| scale {lim:.2g})",
               aspect="equal")

    fig.suptitle(f"{name}: spanwise vorticity, mid-span, t = {t:.1f}")
    fig.tight_layout()
    out = a.out or f"figures/{name}_vorticity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
