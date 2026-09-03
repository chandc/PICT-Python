"""Where the vertical stripes in the far wake come from -- the plot, not the flow.

`plot_farfield.py` interpolates u and v onto a fine Cartesian grid and then takes np.gradient of
THAT to get vorticity. LinearNDInterpolator is piecewise linear, so its derivative is piecewise
CONSTANT and jumps at every mesh cell edge. Differentiating it therefore paints one band per
mesh cell, and since the square's wake plateau ends at x = 14 and dx then grows 0.15 -> 0.94,
the bands appear past x ~ 18 and widen downstream. That is exactly what the figure shows.

The flow is not doing it. Native vorticity along y = 1 for x > 16, computed on the mesh with
the block metrics, has a node-to-node alternating component of 1.7% of its own rms -- smooth.

The right order is: differentiate on the MESH, then interpolate the result for display.
Interpolating a scalar for a picture is fine; differentiating an interpolant is not.
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
from square_cylinder_grid import square_domain, D, X_IN, X_OUT, Y_HALF

TAG = _sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_v3_forces"
NX, NY = 1100, 620


def native_vorticity(d, f, nb):
    """omega_z per block, differentiated on the mesh with the real metrics."""
    # pad_field reads across seams, so the gradient needs the field for EVERY block
    U = {b: f["u"][b] for b in range(nb)}
    V = {b: f["v"][b] for b in range(nb)}
    return [d.gradient(b, V)[0] - d.gradient(b, U)[1] for b in range(nb)]


def main():
    f, meta = checkpoint.load_fields(f"results/fields/{TAG}.npz")
    d, _ = square_domain(nz=4)
    nb = len(d.blocks)
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    U = np.concatenate([f["u"][b][:, :, 0].ravel() for b in range(nb)])
    V = np.concatenate([f["v"][b][:, :, 0].ravel() for b in range(nb)])
    W = np.concatenate([w[:, :, 0].ravel() for w in native_vorticity(d, f, nb)])

    gx = np.linspace(X_IN, X_OUT, NX)
    gy = np.linspace(-Y_HALF, Y_HALF, NY)
    Xg, Yg = np.meshgrid(gx, gy)
    ui = LinearNDInterpolator(P, U)(Xg, Yg)
    vi = LinearNDInterpolator(P, V)(Xg, Yg)
    hx, hy = gx[1] - gx[0], gy[1] - gy[0]
    wz_bad = np.gradient(vi, hx, axis=1) - np.gradient(ui, hy, axis=0)
    wz_good = LinearNDInterpolator(P, W)(Xg, Yg)
    for a in (wz_bad, wz_good):
        a[(np.abs(Xg) <= 0.5 * D) & (np.abs(Yg) <= 0.5 * D)] = np.nan

    LIM = 1.0
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.2), sharex=True, sharey=True)
    for ax, w, ttl in ((axes[0], wz_bad,
                        "interpolate u and v, THEN differentiate  —  what plot_farfield.py did"),
                       (axes[1], wz_good,
                        "differentiate on the mesh, THEN interpolate  —  the same instant")):
        im = ax.pcolormesh(Xg, Yg, w, cmap="RdBu_r", vmin=-LIM, vmax=LIM, shading="auto")
        ax.add_patch(plt.Rectangle((-.5 * D, -.5 * D), D, D, color="k", zorder=9))
        ax.axvline(14.0, color="#00a000", ls="--", lw=1.0)
        ax.set_ylim(-6, 6); ax.set_aspect("equal"); ax.set_ylabel("y / D")
        ax.set_title(ttl, fontsize=11)
    axes[0].text(14.4, 4.6, "wake plateau ends here;\ndx grows 0.15 → 0.94",
                 color="#00a000", fontsize=9)
    axes[1].set_xlabel("x / D")
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\omega_z\,D/U$")
    fig.suptitle(f"square cylinder {TAG}, t = {meta['time']:.0f} — the far-wake stripes are the "
                 f"PLOT, not the flow", fontsize=12)
    out = "figures/sqcyl_vorticity_pipeline.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"  wrote {out}")

    # the number behind the picture
    far = (Xg > 18) & (np.abs(Yg) < 3)
    for nm, w in (("interpolate-then-differentiate", wz_bad),
                  ("differentiate-then-interpolate", wz_good)):
        col = w[:, (gx > 18) & (gx < 30)]
        alt = np.nanmean(np.abs(np.diff(col, axis=1)))
        print(f"  {nm:<32} mean |d(omega)/d(pixel)| past x=18: {alt:.5f}")


if __name__ == "__main__":
    main()
