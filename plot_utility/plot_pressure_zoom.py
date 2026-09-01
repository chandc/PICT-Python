"""Tight zoom on the square: pressure at node resolution, with the grid spacing beneath it.

The wiggle and its cause are plotted together on the same x axis, because they were the same
story: on the first grid the cell size jumped 4.65x at the trailing edge and the pressure rang
at the Nyquist wavelength immediately downstream of it.

Every number in the annotations is now MEASURED from the grid the checkpoint was written on,
not written into the labels. The first version hard-coded "4.65x jump" and "dx 0.032 -> 0.150",
which stayed on the figure after the grid was rebuilt and the jump became a smooth ramp -- a
plot asserting a defect the data underneath it no longer showed.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from square_cylinder_grid import square_domain, D
from src import checkpoint


def trailing_edge_spacing(d, idx, xmax=3.0):
    """(dx just past the trailing edge, dx at the edge of the plotted window, worst ratio).

    The window matters: RM runs to x = 12 and reaches dx ~ 1, which is not the number the
    reader is looking at in a panel that stops at x = 3.
    """
    xs = d.blocks[idx["RM"]].x[:, 0, 0]
    dx = np.diff(xs)
    ratio = float(np.max(dx[1:] / dx[:-1])) if dx.size > 1 else 1.0
    inwin = xs[:-1] < xmax
    return float(dx[0]), float(dx[inwin][-1]), ratio


def main(tag="sqcyl_spark"):
    d, idx = square_domain(nz=4)
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    p = f["p"]
    lim = max(np.abs(p[b]).max() for b in range(8))
    te_dx, te_far, te_ratio = trailing_edge_spacing(d, idx)
    print(f"  trailing edge dx {te_dx:.4f} -> {te_far:.4f}, "
          f"worst adjacent ratio {te_ratio:.3f}")

    fig = plt.figure(figsize=(15, 11))

    # --- very tight zoom, every node a cell, edges drawn ----------------------
    ax = fig.add_axes([0.05, 0.55, 0.44, 0.40])
    for b in range(8):
        blk = d.blocks[b]
        ax.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0], p[b][:, :, 0], cmap="RdBu_r",
                      vmin=-0.5*lim, vmax=0.5*lim, shading="nearest",
                      edgecolors="0.6", linewidth=0.15)
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-1.1, 2.2); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"pressure, one cell per node — cell size at the trailing edge "
                 f"grows {te_ratio:.2f}x per cell", fontsize=11)

    # --- same region, contours ----------------------------------------------
    ax = fig.add_axes([0.55, 0.55, 0.42, 0.40])
    for b in range(8):
        blk = d.blocks[b]
        ax.contour(blk.x[:, :, 0], blk.y[:, :, 0], p[b][:, :, 0],
                   levels=np.linspace(-0.9, 0.7, 33), linewidths=0.7, cmap="RdBu_r")
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.axvline(0.5, color="crimson", ls="--", lw=1.0)
    ax.text(0.55, 1.0, f"trailing edge:\ndx {te_dx:.3f} -> {te_far:.3f}",
            color="crimson", fontsize=8.5)
    ax.set_xlim(-1.1, 2.2); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("iso-contours", fontsize=11)

    # --- p along the centreline, raw nodes ----------------------------------
    ax = fig.add_axes([0.05, 0.30, 0.92, 0.19])
    for nm, col in (("LM", "#5b6c8f"), ("RM", "#e4572e")):
        b = idx[nm]; blk = d.blocks[b]; j = blk.shape[1] // 2
        m = np.abs(blk.x[:, j, 0]) < 3.0
        ax.plot(blk.x[m, j, 0], p[b][m, j, 0], "o-", ms=4, lw=1.1, color=col, label=nm)
    ax.axvspan(-0.5, 0.5, color="0.85", zorder=0)
    ax.text(0, ax.get_ylim()[0]*0.9, "body", ha="center", fontsize=9, color="0.35")
    ax.axvline(0.5, color="crimson", ls="--", lw=1.0)
    ax.set_xlim(-3, 3); ax.set_xlabel("x / D"); ax.set_ylabel("p")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title("pressure along the centreline, raw node values — "
                 "a sawtooth downstream of x = 0.5 is the wiggle", fontsize=11)

    # --- the grid spacing that causes it ------------------------------------
    ax = fig.add_axes([0.05, 0.05, 0.92, 0.17])
    for nm, col in (("LM", "#5b6c8f"), ("MB", "#76b041"), ("RM", "#e4572e")):
        blk = d.blocks[idx[nm]]
        xs = blk.x[:, 0, 0]; dx = np.diff(xs)
        m = np.abs(xs[:-1]) < 3.0
        ax.step(xs[:-1][m], dx[m], where="post", lw=1.6, color=col, label=nm)
    ax.axvspan(-0.5, 0.5, color="0.85", zorder=0)
    ax.axvline(0.5, color="crimson", ls="--", lw=1.0)
    ax.annotate(f"worst adjacent ratio {te_ratio:.2f}x",
                xy=(0.5, te_dx), xytext=(1.1, 0.5 * (te_dx + te_far)), color="crimson",
                fontsize=10, arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.set_xlim(-3, 3); ax.set_xlabel("x / D"); ax.set_ylabel("cell size dx / D")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title("cell size along the same line — a smooth grid keeps adjacent ratios < 1.2",
                 fontsize=11)

    fig.suptitle(f"Square cylinder Re = 100, t = {meta['time']:.0f} — the near-body pressure "
                 f"wiggle and its cause", fontsize=13)
    out = f"figures/{tag}_pressure_zoom.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_spark")
