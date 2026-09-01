"""Streamlines at several phases of the kicked wake, to show whether it actually oscillates.

A single snapshot cannot distinguish a steady wake from an oscillating one caught mid-cycle.
Four panels spanning the cycle can: if the wake is shedding, the near-wake streamlines deflect
alternately above and below the centreline, and the closed bubble of the steady solution breaks.
"""
import glob
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from square_cylinder_grid import square_domain, D
from src import checkpoint

XL, XR, YB, YT = -2.0, 10.0, -3.0, 3.0
NX, NY = 620, 320


def raster(d, f):
    P, U, V = [], [], []
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        P.append(np.column_stack([blk.x[:, :, 0].ravel(), blk.y[:, :, 0].ravel()]))
        U.append(f["u"][b][:, :, 0].ravel()); V.append(f["v"][b][:, :, 0].ravel())
    P = np.vstack(P); U = np.concatenate(U); V = np.concatenate(V)
    gx = np.linspace(XL, XR, NX); gy = np.linspace(YB, YT, NY)
    GX, GY = np.meshgrid(gx, gy)
    gu = LinearNDInterpolator(P, U)(GX, GY)
    gv = LinearNDInterpolator(P, V)(GX, GY)
    solid = (np.abs(GX) <= .5*D) & (np.abs(GY) <= .5*D)
    gu[solid] = np.nan; gv[solid] = np.nan
    return gx, gy, GX, GY, gu, gv


def main():
    d, idx = square_domain(nz=4)
    # The unkicked steady state is panel 0 -- without it there is no reference for "deflected".
    files = ["results/fields/sqcyl_Re100_rc.npz"] + sorted(
        glob.glob("results/fields/sqcyl_phase*.npz"))
    if len(files) < 2:
        print("  no phase files yet"); return
    MAXP = 6
    pick = files if len(files) <= MAXP else [files[0]] + [files[i] for i in
                                          np.linspace(1, len(files)-1, MAXP-1).astype(int)]
    hist = np.load("results/sqcyl_kick_history.npy")

    rows = (len(pick) + 1) // 2
    fig = plt.figure(figsize=(15, 4 * rows + 3))
    gs = fig.add_gridspec(rows + 1, 2, height_ratios=[1] * rows + [0.6])
    for n, path in enumerate(pick):
        f, meta = checkpoint.load_fields(path)
        gx, gy, GX, GY, gu, gv = raster(d, f)
        ax = fig.add_subplot(gs[n // 2, n % 2])
        w = np.gradient(gv, gx, axis=1) - np.gradient(gu, gy, axis=0)
        ax.pcolormesh(GX, GY, w, cmap="RdBu_r", shading="auto", vmin=-4, vmax=4)
        ax.streamplot(gx, gy, gu, gv, density=2.0, linewidth=0.5, color="0.15", arrowsize=0.6)
        ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
        ax.axhline(0, color="0.5", lw=0.5, ls=":")
        ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
        ax.set_title(f"t = {meta['time']:.1f}" + ("  (steady, before the kick)" if n == 0
                                                   else ""), fontsize=10)
        ax.set_xlabel("x / D"); ax.set_ylabel("y / D")

    ax = fig.add_subplot(gs[-1, :])
    ax.plot(hist[:, 0], hist[:, 1], lw=0.8, color="#5b6c8f")
    for path in pick:
        _, meta = checkpoint.load_fields(path)
        ax.axvline(meta["time"], color="#e4572e", ls="--", lw=1.0)
    ax.set_xlabel("t U / D"); ax.set_ylabel("v at (2D, 0.5D)")
    ax.grid(alpha=.3); ax.set_title("probe trace — dashed lines mark the panels above",
                                    fontsize=10)

    fig.suptitle("Square cylinder Re = 100 — wake kicked from the steady state, "
                 "vorticity with streamlines", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "figures/sqcyl_phases_streamlines.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"  {len(files)} phase files, plotted {len(pick)}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
