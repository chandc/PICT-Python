"""Far-field vorticity, the grid that died against the grid running now.

A single far-field picture is hard to read: some structure near the outer boundary is the wake
leaving, which is correct, and some is the outer grid going unstable, which is fatal, and at a
glance they look alike. Both fields are drawn here on the SAME colour scale, with the Dong
outflow arc marked, because the discriminating feature is WHERE the disturbance sits:

  inside |theta| <= 21.8    the wake exiting through the outflow -- what should happen
  outside it                nothing is meant to be leaving there

The old field had its maximum at 22-45 deg, outside the arc, at 4.08 in |u - U_inf|. The new one
peaks at 0.17 inside the arc and falls monotonically outward.
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
from cylinder_bc import U_INF

N = 900
ARC = 21.8


def field(path, dom_kw):
    f, meta = checkpoint.load_fields(path)
    d, r, _ = cylinder_domain(nblk=meta["nblocks"], nz=4, **dom_kw)
    nb = len(d.blocks)
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    U = np.concatenate([f["u"][b][:, :, 0].ravel() for b in range(nb)])
    V = np.concatenate([f["v"][b][:, :, 0].ravel() for b in range(nb)])
    R = float(r[-1])
    g = np.linspace(-R, R, N)
    Xg, Yg = np.meshgrid(g, g)
    ui = LinearNDInterpolator(P, U)(Xg, Yg)
    vi = LinearNDInterpolator(P, V)(Xg, Yg)
    h = g[1] - g[0]
    wz = np.gradient(vi, h, axis=1) - np.gradient(ui, h, axis=0)
    wz[np.hypot(Xg, Yg) > R * 0.999] = np.nan
    wz[np.hypot(Xg, Yg) < R_CYL] = np.nan
    return Xg, Yg, wz, R, float(meta["time"]), np.hypot(ui - U_INF, vi)


cases = [("results/fields/cyl_Re100.npz",
          dict(r_out=30.0 * D, dr_hold=0.28 * D, r_hold=12.0 * D, ratio=1.12),
          "OLD grid — max dr 2.18, cell Peclet 218"),
         ("results/fields/cyl_v2.npz", {},
          "NEW grid — max dr 0.285, cell Peclet 29")]

fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2))
LIM = 0.5
for ax, (path, kw, title) in zip(axes, cases):
    Xg, Yg, wz, R, t, dist = field(path, kw)
    im = ax.pcolormesh(Xg, Yg, wz, cmap="RdBu_r", vmin=-LIM, vmax=LIM, shading="auto")
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    for s in (+1, -1):
        a = np.radians(s * ARC)
        ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], color="#00a000", lw=1.2, ls="--")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="k", lw=0.8))
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    far = np.hypot(Xg, Yg) > 0.85 * R
    th = np.degrees(np.arctan2(Yg, Xg))
    ins = far & (np.abs(th) <= ARC)
    out = far & (np.abs(th) > ARC)
    ax.set_title(f"{title}\nt = {t:.0f}, domain {R:.0f} D\n"
                 f"max |u-U$_\\infty$| in the outer 15%:\n"
                 f"{np.nanmax(dist[ins]):.3f} inside the arc,  "
                 f"{np.nanmax(dist[out]):.3f} outside", fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.82, label=r"$\omega_z\,D/U$")
fig.suptitle("far-field vorticity on a common scale;  dashed green = the Dong outflow arc, "
             "the only place the flow is meant to leave", fontsize=12)
out = "figures/cyl_farfield_compare.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print(f"  wrote {out}")
