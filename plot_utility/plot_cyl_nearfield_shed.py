"""Cylinder near-field vorticity, and the part of it that is the shedding.

A SYMMETRY SPLIT MAKES THE MODE VISIBLE LONG BEFORE THE EYE CAN SEE IT. The pre-kick base flow
is mirror-symmetric about y = 0, which for the velocity means u(x,-y) = u(x,y) and
v(x,-y) = -v(x,y). Vorticity is omega = dv/dx - du/dy, so under y -> -y both terms flip and

    base flow:      omega(x,-y) = -omega(x,y)        ANTIsymmetric

The shedding mode breaks exactly that. So the SYMMETRIC part of the vorticity,

    omega_s(x,y) = 1/2 [ omega(x,y) + omega(x,-y) ]

is identically zero for the base flow and nonzero only where the flow is shedding. It is the
growing mode with the base flow subtracted, which is why it is legible at an amplitude where the
raw field still looks symmetric: a colour map of omega shows the SUM, and the mode is a small
perturbation on a much larger symmetric background.

Right panel is on its own colour scale for that reason -- printed in the title, so the two are
not mistaken for the same units.
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

from src import checkpoint
from cylinder_grid import cylinder_domain, D, R_CYL

TAG = _sys.argv[1] if len(_sys.argv) > 1 else "cyl_shed_mac"
XL, XR, YB, YT = -2.0, 9.0, -3.0, 3.0
N = 760


def main():
    f, meta = checkpoint.load_fields(f"results/fields/{TAG}.npz")
    nb = meta["nblocks"]
    d, r, _ = cylinder_domain(nblk=nb, nz=4)
    U = {b: f["u"][b] for b in range(nb)}
    V = {b: f["v"][b] for b in range(nb)}
    W = [d.gradient(b, V)[0] - d.gradient(b, U)[1] for b in range(nb)]
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    Wv = np.concatenate([w[:, :, 0].ravel() for w in W])

    gx = np.linspace(XL, XR, N)
    gy = np.linspace(YB, YT, int(N * (YT - YB) / (XR - XL)))
    GX, GY = np.meshgrid(gx, gy)
    itp = LinearNDInterpolator(P, Wv)
    Wg = itp(GX, GY)
    Wm = itp(GX, -GY)                       # the mirrored field
    Ws = 0.5 * (Wg + Wm)                    # symmetric part -- zero for the base flow
    for A in (Wg, Ws):
        A[np.hypot(GX, GY) < R_CYL] = np.nan

    lim_s = float(np.nanmax(np.abs(Ws)))
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 8.8), sharex=True, sharey=True)

    im0 = axes[0].pcolormesh(GX, GY, Wg, cmap="RdBu_r", vmin=-5, vmax=5, shading="auto")
    axes[0].set_title(r"vorticity $\omega_z D/U$ — the full field, scale $\pm5$", fontsize=11)
    fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.01)

    im1 = axes[1].pcolormesh(GX, GY, Ws, cmap="PuOr_r", vmin=-lim_s, vmax=lim_s, shading="auto")
    axes[1].set_title(r"the SHEDDING alone: $\frac{1}{2}[\omega(y)+\omega(-y)]$, which vanishes "
                      rf"for the symmetric base flow — scale $\pm${lim_s:.2f}, "
                      rf"{100*lim_s/5:.0f}% of the panel above", fontsize=11)
    fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.01)

    for ax in axes:
        ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
        ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
        ax.set_ylabel("y / D")
    axes[1].set_xlabel("x / D")
    fig.suptitle(f"circular cylinder, Re = 100, near field at t = {meta['time']:.1f} "
                 f"— {(meta['time']-85):.0f} time units after the kick", fontsize=12.5)
    out = f"figures/{TAG}_nearfield.png"
    fig.savefig(out, dpi=118, bbox_inches="tight")
    print(f"  wrote {out}")
    print(f"  |omega| peak {np.nanmax(np.abs(Wg)):.2f}")
    print(f"  shedding part peak {lim_s:.4f}, i.e. {100*lim_s/np.nanmax(np.abs(Wg)):.2f}% "
          f"of the full field")
    j = np.unravel_index(np.nanargmax(np.abs(Ws)), Ws.shape)
    print(f"  it is largest at x = {GX[j]:.2f}, y = {GY[j]:+.2f}")


if __name__ == "__main__":
    main()
