"""Near-field vorticity and pressure of the square cylinder -- the shedding itself.

VORTICITY IS DIFFERENTIATED ON THE MESH and only then interpolated for display. Taking
np.gradient of an interpolated velocity field differentiates a piecewise-LINEAR function, whose
derivative is piecewise constant and jumps at every cell edge; that painted one band per mesh
cell in the far wake and was mistaken for a flow feature. See plot_vorticity_pipeline.py.

The two fields answer different questions and are shown together for that reason. Vorticity
shows where the shear layers roll up and how the cores are shed alternately. Pressure shows WHY
the body feels a force: each shed vortex is a low-pressure core, and the alternation of those
cores across the wake centreline is what C_L oscillating at St and C_D at 2 St actually is.
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
from square_cylinder_grid import square_domain, D

XL, XR, YB, YT = -2.5, 11.0, -3.5, 3.5
N = 900


def load(tag, nz=4):
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    d, _ = square_domain(nz=nz)
    nb = len(d.blocks)
    U = {b: f["u"][b] for b in range(nb)}
    V = {b: f["v"][b] for b in range(nb)}
    Wb = [d.gradient(b, V)[0] - d.gradient(b, U)[1] for b in range(nb)]
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    keys = dict(
        w=np.concatenate([w[:, :, 0].ravel() for w in Wb]),
        p=np.concatenate([f["p"][b][:, :, 0].ravel() for b in range(nb)]),
        u=np.concatenate([f["u"][b][:, :, 0].ravel() for b in range(nb)]),
        v=np.concatenate([f["v"][b][:, :, 0].ravel() for b in range(nb)]))
    return P, keys, float(meta["time"])


def grid_of(P, val):
    gx = np.linspace(XL, XR, N)
    gy = np.linspace(YB, YT, int(N * (YT - YB) / (XR - XL)))
    GX, GY = np.meshgrid(gx, gy)
    Z = LinearNDInterpolator(P, val)(GX, GY)
    Z[(np.abs(GX) <= 0.5 * D) & (np.abs(GY) <= 0.5 * D)] = np.nan
    return GX, GY, Z


def body(ax):
    ax.add_patch(plt.Rectangle((-.5 * D, -.5 * D), D, D, facecolor="k", edgecolor="k", zorder=9))


def main(tag="sqcyl_v3_forces"):
    P, k, t = load(tag)
    GX, GY, W = grid_of(P, k["w"])
    _, _, PR = grid_of(P, k["p"])
    _, _, Uu = grid_of(P, k["u"])
    _, _, Vv = grid_of(P, k["v"])
    # p is defined to an additive constant; referencing it to the far upstream value makes the
    # contour labels mean C_p = (p - p_inf)/(0.5 U^2) rather than an arbitrary offset
    p_inf = np.nanmedian(PR[:, :12])
    CP = (PR - p_inf) / 0.5

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 11.2), sharex=True, sharey=True)

    lim = 5.0
    im0 = axes[0].pcolormesh(GX, GY, W, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    axes[0].contour(GX, GY, W, levels=[-2.0, -0.7, 0.7, 2.0], colors="k",
                    linewidths=0.4, alpha=0.5)
    body(axes[0])
    axes[0].set_title(r"vorticity $\omega_z D/U$ — shear layers roll up and shed alternately",
                      fontsize=11)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.01)
    cb0.set_label(r"$\omega_z D/U$")

    # DIVERGING AND CENTRED ON ZERO, so suction and stagnation read as opposites rather than
    # as two shades of the same colour. The front stagnation point must sit at C_p = +1 exactly,
    # which is the one value in this picture with an analytic answer -- see the print below.
    im1 = axes[1].pcolormesh(GX, GY, CP, cmap="RdYlBu_r", vmin=-1.4, vmax=1.0, shading="auto")
    c = axes[1].contour(GX, GY, CP, levels=np.arange(-1.25, 1.01, 0.25), colors="k",
                        linewidths=0.45, alpha=0.55)
    axes[1].clabel(c, fmt="%.2f", fontsize=6)
    axes[1].streamplot(GX, GY, Uu, Vv, color="k", linewidth=0.4, density=1.1, arrowsize=0.6)
    body(axes[1])
    axes[1].set_title(r"pressure $C_p = (p-p_\infty)/(U^2/2)$ — every shed vortex is a "
                      r"low-pressure core; their alternation IS the lift oscillation",
                      fontsize=11)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.01)
    cb1.set_label(r"$C_p$")

    for ax in axes:
        ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
        ax.set_ylabel("y / D")
    axes[1].set_xlabel("x / D")
    fig.suptitle(f"square cylinder at Re = 100, near field, t = {t:.1f}   —   "
                 f"St = 0.1488, $C_D$ = 1.4529, $C_L$ rms = 0.1722", fontsize=12.5)
    out = f"figures/{tag}_nearfield.png"
    fig.savefig(out, dpi=118, bbox_inches="tight")
    print(f"  wrote {out}")
    print(f"  |omega| peak {np.nanmax(np.abs(W)):.2f};  C_p range "
          f"{np.nanmin(CP):+.3f} to {np.nanmax(CP):+.3f}")
    # THE ONE ANALYTIC VALUE IN THE PICTURE. At the front stagnation point the flow is brought
    # to rest, so C_p = 1 exactly, whatever the Reynolds number. Anything else is an error in
    # the pressure field or in the reference used to normalise it.
    Pn, kn, _ = P, k, t
    xs = Pn[:, 0]; ys = Pn[:, 1]
    face = (np.abs(xs + 0.5 * D) < 1e-9) & (np.abs(ys) < 0.02)
    if face.any():
        cp_stag = (kn["p"][face] - p_inf) / 0.5
        print(f"  front stagnation C_p = {cp_stag.max():+.4f}   (exact answer: +1.0000, "
              f"error {abs(cp_stag.max()-1.0)*100:.2f}%)")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_v3_forces")
