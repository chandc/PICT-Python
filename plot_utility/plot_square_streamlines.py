"""Streamlines around the square cylinder, from a checkpoint.

matplotlib's streamplot needs a UNIFORM grid, and this mesh is anything but -- eight blocks,
strongly clustered on the body. So the block fields are scattered into one point cloud and
interpolated onto a uniform raster. The body is masked afterwards: interpolation across the
hole would otherwise invent flow through solid material.
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
from scipy.interpolate import LinearNDInterpolator

from square_cylinder_grid import square_domain, D
from src import checkpoint

XL, XR, YB, YT = -3.0, 12.0, -4.0, 4.0
NX, NY = 900, 480


def raster(tag, nz=4):
    d, idx = square_domain(nz=nz)
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    P, U, V = [], [], []
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        P.append(np.column_stack([blk.x[:, :, 0].ravel(), blk.y[:, :, 0].ravel()]))
        U.append(f["u"][b][:, :, 0].ravel())
        V.append(f["v"][b][:, :, 0].ravel())
    P = np.vstack(P); U = np.concatenate(U); V = np.concatenate(V)
    gx = np.linspace(XL, XR, NX); gy = np.linspace(YB, YT, NY)
    GX, GY = np.meshgrid(gx, gy)
    iu = LinearNDInterpolator(P, U); iv = LinearNDInterpolator(P, V)
    gu, gv = iu(GX, GY), iv(GX, GY)
    solid = (np.abs(GX) <= 0.5 * D) & (np.abs(GY) <= 0.5 * D)
    gu[solid] = np.nan; gv[solid] = np.nan
    return gx, gy, GX, GY, gu, gv, meta


def main(tag="sqcyl_Re100_rc"):
    gx, gy, GX, GY, gu, gv, meta = raster(tag)
    spd = np.sqrt(gu**2 + gv**2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    im = ax.pcolormesh(GX, GY, spd, cmap="viridis", shading="auto", vmin=0, vmax=1.6)
    ax.streamplot(gx, gy, gu, gv, density=2.4, linewidth=0.6, color="w", arrowsize=0.7)
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("streamlines over speed", fontsize=11)
    fig.colorbar(im, ax=ax, label="|u| / U", fraction=0.025, pad=0.01)

    ax = axes[1]
    dvdx = np.gradient(gv, gx, axis=1)
    dudy = np.gradient(gu, gy, axis=0)
    im = ax.pcolormesh(GX, GY, dvdx - dudy, cmap="RdBu_r", shading="auto", vmin=-4, vmax=4)
    ax.streamplot(gx, gy, gu, gv, density=1.6, linewidth=0.5, color="0.25", arrowsize=0.6)
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-2, 6); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("near wake — vorticity with streamlines", fontsize=11)
    fig.colorbar(im, ax=ax, label=r"$\omega_z D / U$", fraction=0.025, pad=0.01)

    # recirculation length: where u on the centreline changes sign back to positive
    j = np.argmin(np.abs(gy)); i0 = np.argmin(np.abs(gx - 0.5))
    line = gu[j, i0:]
    neg = np.where(line < 0)[0]
    lr = (gx[i0 + neg[-1]] - 0.5 * D) if len(neg) else float("nan")

    fig.suptitle(f"Square cylinder, Re = 100 — t = {meta['time']:.1f} "
                 f"(STEADY, not shedding) — recirculation L_r/D = {lr:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"figures/{tag}_streamlines.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  t = {meta['time']:.1f}, step {meta['nstep']}")
    print(f"  recirculation length L_r/D = {lr:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_Re100_rc")
