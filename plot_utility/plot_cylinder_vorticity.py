"""Circular cylinder vorticity, near field and whole domain, at two time steps.

The two fields are the SAME case at the SAME instant, t = 85, started from the same t = 60
state, differing only in dt. At dt = 0.010 the far-field disturbance grows without bound --
0.129 at t = 62 rising to 0.208 at t = 84 with accelerating increments. At dt = 0.005 it peaks
at 0.1656 (t = 72) and then decays to 0.1483, still falling. That is a time-step stability
limit, and it is what this figure shows.

Vorticity is differentiated on the MESH with the real metrics and interpolated only for display.
Differentiating an interpolant instead paints one band per mesh cell and manufactures far-field
structure that is not in the solution -- see plot_vorticity_pipeline.py.

The cylinder is NOT SHEDDING here: these are pre-kick base flows, deliberately symmetric, so the
near-field panels should show two attached, mirror-symmetric shear layers and nothing unsteady.
Any asymmetry in them is a defect, not a vortex street.
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
from cylinder_bc import U_INF, outer_role

DEFAULT = [("cyl_expt_ctrl_long", "dt = 0.010  —  disturbance GROWING, 0.208 and rising"),
           ("cyl_expt_dt005long", "dt = 0.005  —  disturbance DECAYING, 0.148 and falling")]
# tags may be given on the command line; a single tag is drawn on one row
CASES = ([(a, a) for a in _sys.argv[1:]] or DEFAULT)
ARC = 21.8


def field(tag, nz=4):
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    nb = meta["nblocks"]
    d, r, _ = cylinder_domain(nblk=nb, nz=nz)
    U = {b: f["u"][b] for b in range(nb)}
    V = {b: f["v"][b] for b in range(nb)}
    W = [d.gradient(b, V)[0] - d.gradient(b, U)[1] for b in range(nb)]
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    Wv = np.concatenate([w[:, :, 0].ravel() for w in W])
    Uv = np.concatenate([f["u"][b][:, :, 0].ravel() for b in range(nb)])
    Vv = np.concatenate([f["v"][b][:, :, 0].ravel() for b in range(nb)])
    dev = np.hypot(Uv - U_INF, Vv)
    return P, Wv, dev, float(r[-1]), float(meta["time"])


def panel(ax, P, val, R, lim, cmap, n=620):
    g = np.linspace(-R, R, n)
    GX, GY = np.meshgrid(g, g)
    Z = LinearNDInterpolator(P, val)(GX, GY)
    rr = np.hypot(GX, GY)
    Z[(rr > R * 0.999) | (rr < R_CYL)] = np.nan
    im = ax.pcolormesh(GX, GY, Z, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    ax.set_aspect("equal")
    return im


def asymmetry(tag, nz=4):
    """Mirror antisymmetry of v about y = 0 in the near wake.

    The pre-kick base flow is symmetric, so v(x,-y) = -v(x,y) exactly and this reads zero.
    Shedding BREAKS that symmetry, so a growing value here is the clearest single number for
    "is it shedding yet" -- clearer than the probe, which also carries the symmetric base value.
    """
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    nb = meta["nblocks"]
    d, r, _ = cylinder_domain(nblk=nb, nz=nz)
    X = np.concatenate([b.x[:, :, 0].ravel() for b in d.blocks])
    Y = np.concatenate([b.y[:, :, 0].ravel() for b in d.blocks])
    V = np.concatenate([f["v"][b][:, :, 0].ravel() for b in range(nb)])
    m = (X > 0.5) & (X < 6.0) & (np.abs(Y) < 2.0)
    from scipy.interpolate import LinearNDInterpolator
    itp = LinearNDInterpolator(np.column_stack([X, Y]), V)
    gx = np.linspace(0.6, 6.0, 120); gy = np.linspace(0.05, 2.0, 60)
    GX, GY = np.meshgrid(gx, gy)
    vp = itp(GX, GY); vm = itp(GX, -GY)
    ok = np.isfinite(vp) & np.isfinite(vm)
    sym = np.abs(vp[ok] + vm[ok])            # zero if antisymmetric
    scale = max(np.abs(vp[ok]).max(), 1e-30)
    return float(sym.max()), float(scale), float(meta["time"])


def main():
    nrow = len(CASES)
    fig, axes = plt.subplots(nrow, 2, figsize=(14.6, 6.3 * nrow), squeeze=False)
    for row, (tag, label) in enumerate(CASES):
        P, W, dev, R, t = field(tag)
        im0 = panel(axes[row, 0], P, W, R, 5.0, "RdBu_r")
        axes[row, 0].set_xlim(-3, 6); axes[row, 0].set_ylim(-3.5, 3.5)
        axes[row, 0].set_title(f"near field — {label.split('—')[0].strip()}", fontsize=10)
        im1 = panel(axes[row, 1], P, W, R, 0.35, "RdBu_r")
        for s in (+1, -1):
            a = np.radians(s * ARC)
            axes[row, 1].plot([0, R*np.cos(a)], [0, R*np.sin(a)], color="#00a000",
                              lw=1.1, ls="--")
        axes[row, 1].set_xlim(-R, R); axes[row, 1].set_ylim(-R, R)
        tail = label.split("—")[1].strip() if "—" in label else ""
        axes[row, 1].set_title(f"whole domain, t = {t:.0f}  {tail}", fontsize=10)
        for ax in axes[row]:
            ax.set_ylabel("y / D")
    for ax in axes[-1]:
        ax.set_xlabel("x / D")
    fig.colorbar(im0, ax=axes[:, 0], shrink=0.55, label=r"$\omega_z D/U$  (near field, ±5)")
    fig.colorbar(im1, ax=axes[:, 1], shrink=0.55, label=r"$\omega_z D/U$  (far field, ±0.35)")
    ttl = ("circular cylinder, Re = 100 — vorticity.  Green dashed = the Dong outflow arc."
           if len(CASES) == 1 else
           "circular cylinder, Re = 100, pre-kick base flow — the same instant at two time "
           "steps.\nGreen dashed = the Dong outflow arc.")
    fig.suptitle(ttl, fontsize=12.5)
    out = ("figures/cyl_vorticity_%s.png" % CASES[0][0]) if len(CASES) == 1 \
        else "figures/cyl_vorticity_dt.png"
    fig.savefig(out, dpi=112, bbox_inches="tight")
    print(f"  wrote {out}")
    for tag, label in CASES:
        P, W, dev, R, t = field(tag)
        far = (np.hypot(P[:, 0], P[:, 1]) > 10) & (np.abs(P[:, 1]) > 4)
        a, sc, tt = asymmetry(tag)
        print(f"  {tag:<22} t={tt:.1f}  |omega| peak {np.abs(W).max():.2f}   "
              f"far-field max|u-U| {dev[far].max():.4f}")
        print(f"  {'':<22} wake mirror asymmetry max|v(y)+v(-y)| = {a:.4e}  "
              f"({100*a/sc:.2f}% of max|v| there)")


if __name__ == "__main__":
    main()
