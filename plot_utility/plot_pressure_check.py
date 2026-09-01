"""Pressure contours at NODE resolution, to look for checkerboarding.

DO NOT INTERPOLATE. Every other plot here rasterises the eight blocks onto a uniform grid with
LinearNDInterpolator, which averages neighbouring nodes -- and averaging neighbouring nodes is
precisely what annihilates a node-to-node mode. A checkerboard would be smoothed out of the
picture by the plotting, and the figure would prove nothing.

So each block is drawn on its own mesh with `shading="nearest"`, one coloured cell per node, and
the cut through the wake plots the raw node values as markers. If there is a sawtooth, it shows.
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
from diag_checkerboard import checkerboard
from src import checkpoint


def main(tag="sqcyl_spark"):
    d, idx = square_domain(nz=4)
    inv = {v: k for k, v in idx.items()}
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    p = f["p"]
    lim = max(np.abs(p[b]).max() for b in range(8))

    wf = max(max(checkerboard(p[b], axis=a)[1] for a in (0, 1)) for b in range(8))
    wa = max(max(checkerboard(p[b], axis=a)[0] for a in (0, 1)) for b in range(8))

    fig = plt.figure(figsize=(15, 11))

    # --- full field, raw nodes -------------------------------------------------
    ax = fig.add_axes([0.05, 0.68, 0.92, 0.27])
    for b in range(8):
        blk = d.blocks[b]
        ax.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0], p[b][:, :, 0],
                      cmap="RdBu_r", vmin=-0.6*lim, vmax=0.6*lim, shading="nearest")
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-4, 20); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("pressure, one cell per node — no interpolation", fontsize=11)

    # --- near body, every node visible ----------------------------------------
    ax = fig.add_axes([0.05, 0.36, 0.43, 0.26])
    for b in range(8):
        blk = d.blocks[b]
        ax.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0], p[b][:, :, 0],
                      cmap="RdBu_r", vmin=-0.6*lim, vmax=0.6*lim, shading="nearest",
                      edgecolors="0.75", linewidth=0.08)
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-1.6, 3.2); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("near the body, cell edges drawn", fontsize=11)

    # --- contour lines: a checkerboard makes these ragged ----------------------
    ax = fig.add_axes([0.55, 0.36, 0.42, 0.26])
    for b in range(8):
        blk = d.blocks[b]
        ax.contour(blk.x[:, :, 0], blk.y[:, :, 0], p[b][:, :, 0],
                   levels=np.linspace(-0.5*lim, 0.5*lim, 25), linewidths=0.6, cmap="RdBu_r")
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-2, 8); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("iso-contours — ragged lines would mean oscillation", fontsize=11)

    # --- raw node values along two cuts ---------------------------------------
    ax = fig.add_axes([0.05, 0.05, 0.92, 0.22])
    b = idx["RM"]; blk = d.blocks[b]
    j = blk.shape[1] // 2
    x = blk.x[:, j, 0]; pv = p[b][:, j, 0]
    m = x < 12
    ax.plot(x[m], pv[m], "o-", ms=3.0, lw=1.0, color="#5b6c8f",
            label=f"wake centreline, block RM ({m.sum()} nodes)")
    b2 = idx["MT"]; blk2 = d.blocks[b2]
    y2 = blk2.y[blk2.shape[0]//2, :, 0]; p2 = p[b2][blk2.shape[0]//2, :, 0]
    m2 = y2 < 3
    ax.plot(y2[m2], p2[m2], "s--", ms=3.0, lw=1.0, color="#e4572e", alpha=.8,
            label="above the body, block MT (vs y)")
    ax.set_xlabel("x / D  (or y / D for the dashed line)"); ax.set_ylabel("p")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title("raw node values — a checkerboard is a sawtooth between adjacent markers",
                 fontsize=11)

    verdict = "CLEAN" if wf < 0.35 else "CHECKERBOARD"
    fig.suptitle(f"Square cylinder Re = 100, t = {meta['time']:.0f} — pressure check: "
                 f"{verdict}   (worst flip fraction {wf:.3f}, amplitude {wa:.2e} = "
                 f"{100*wa/lim:.2f}% of |p|)", fontsize=13)
    out = f"figures/{tag}_pressure_check.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  worst flip fraction {wf:.3f}  amplitude {wa:.3e}  ({100*wa/lim:.2f}% of |p|)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_spark")
