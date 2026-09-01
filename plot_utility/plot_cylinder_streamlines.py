"""Streamlines around the CIRCULAR cylinder, from a checkpoint.

Same construction as `plot_square_streamlines.py`, and the same reason for it: streamplot needs
a uniform grid and this is an O-grid of blocks clustered hard on the body, so the fields
are scattered into one point cloud and interpolated onto a raster.

TWO DIFFERENCES FROM THE SQUARE CASE, both of which will bite if copied over carelessly:

  The mask is a DISC, not a box. Interpolating across the hole invents flow through solid
  material, and on a circle a bounding-box mask would leave four corners of invented flow
  attached to the body -- exactly where the shear layers separate and exactly where the
  picture is being read.

  The grid runs to r = 30 D. Rasterising the whole thing at wake resolution would spend most of
  its points on undisturbed freestream, so the interpolation is built only from the points
  inside the plotted window plus a margin.
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

from cylinder_grid import cylinder_domain, D, R_CYL
from src import checkpoint

XL, XR, YB, YT = -3.0, 12.0, -4.0, 4.0
NX, NY = 900, 480
MARGIN = 1.0


def raster(tag, nz=4, nblk=None):
    # The block count comes from the CHECKPOINT unless overridden. It is recorded there, and
    # hard-coding it means every plot silently rebuilds the wrong domain the day the
    # decomposition changes -- which it did, 8 -> 16, to cut a narrower far-field outflow arc.
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    d, r, arc = cylinder_domain(nblk=nblk or meta["nblocks"], nz=nz)
    P, U, V = [], [], []
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        x, y = blk.x[:, :, 0].ravel(), blk.y[:, :, 0].ravel()
        keep = ((x > XL - MARGIN) & (x < XR + MARGIN) &
                (y > YB - MARGIN) & (y < YT + MARGIN))
        P.append(np.column_stack([x[keep], y[keep]]))
        U.append(f["u"][b][:, :, 0].ravel()[keep])
        V.append(f["v"][b][:, :, 0].ravel()[keep])
    P = np.vstack(P); U = np.concatenate(U); V = np.concatenate(V)
    gx = np.linspace(XL, XR, NX); gy = np.linspace(YB, YT, NY)
    GX, GY = np.meshgrid(gx, gy)
    gu = LinearNDInterpolator(P, U)(GX, GY)
    gv = LinearNDInterpolator(P, V)(GX, GY)
    solid = (GX**2 + GY**2) <= R_CYL**2
    gu[solid] = np.nan; gv[solid] = np.nan
    return gx, gy, GX, GY, gu, gv, meta, len(P)


def main(tag="cyl_Re100", nz=4, nblk=None):
    gx, gy, GX, GY, gu, gv, meta, npts = raster(tag, nz, nblk)
    spd = np.sqrt(gu**2 + gv**2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    im = ax.pcolormesh(GX, GY, spd, cmap="viridis", shading="auto", vmin=0, vmax=1.6)
    ax.streamplot(gx, gy, gu, gv, density=2.4, linewidth=0.6, color="w", arrowsize=0.7)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("streamlines over speed", fontsize=11)
    fig.colorbar(im, ax=ax, label="|u| / U", fraction=0.025, pad=0.01)

    ax = axes[1]
    dvdx = np.gradient(gv, gx, axis=1)
    dudy = np.gradient(gu, gy, axis=0)
    im = ax.pcolormesh(GX, GY, dvdx - dudy, cmap="RdBu_r", shading="auto", vmin=-4, vmax=4)
    ax.streamplot(gx, gy, gu, gv, density=1.6, linewidth=0.5, color="0.25", arrowsize=0.6)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    ax.set_xlim(-2, 6); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("near wake — vorticity with streamlines", fontsize=11)
    fig.colorbar(im, ax=ax, label=r"$\omega_z D / U$", fraction=0.025, pad=0.01)

    # recirculation length, measured from the REAR STAGNATION POINT at x = R_CYL, so it is
    # comparable with the literature's L_r/D for a circular cylinder (~0.9 at Re = 100 once
    # shedding, longer for the unstable steady base flow this converges to first).
    j = np.argmin(np.abs(gy)); i0 = np.searchsorted(gx, R_CYL)
    line = gu[j, i0:]
    neg = np.where(line < 0)[0]
    lr = (gx[i0 + neg[-1]] - R_CYL) if len(neg) else float("nan")

    fig.suptitle(f"Circular cylinder, Re = 100 — t = {meta['time']:.1f}, step {meta['nstep']} "
                 f"— recirculation L_r/D = {lr:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"figures/{tag}_streamlines.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  t = {meta['time']:.1f}, step {meta['nstep']}, {npts:,} points in the window")
    print(f"  recirculation length L_r/D = {lr:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "cyl_Re100")
