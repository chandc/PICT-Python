"""Pressure around the circular cylinder: field contours and the SURFACE distribution.

The surface pressure coefficient is the point of this figure. C_p(theta) is what the literature
tabulates for a cylinder, and two numbers off it are directly comparable:

  * the FRONT stagnation value, which must be +1 for an inviscid free stream and is a check on
    the normalisation rather than on the physics -- if it is not near 1, the reference pressure
    or U_inf is wrong and nothing else on the plot means anything;
  * the BASE pressure at theta = 180, about -0.7 at Re = 100, which is a real measurement and
    the quantity most sensitive to the wake being resolved.

The field panel uses node values with `shading="nearest"` for the same reason
`plot_pressure_check.py` does: interpolating would average away a node-to-node mode, which is
precisely what a pressure plot is supposed to be able to show.
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

from cylinder_grid import cylinder_domain, D, R_CYL
from src import checkpoint

CP_BASE_REF = (-0.75, -0.65)          # published base C_p at Re = 100


def main(tag="cyl_Re100", nz=4):
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    nb = meta["nblocks"]
    d, r, _ = cylinder_domain(nblk=nb, nz=nz)
    span = float(d.blocks[0].period[2])

    # surface values: the r-min node line of every block
    th, cp = [], []
    p_far = []
    for b in range(nb):
        blk = d.blocks[b]
        x0, y0 = blk.x[0, :, 0], blk.y[0, :, 0]
        th.append(np.degrees(np.arctan2(y0, x0)))
        cp.append(f["p"][b][0, :, 0])
        p_far.append(f["p"][b][-1, :, 0])           # far field, for the reference level
    th = np.concatenate(th); cp = np.concatenate(cp)
    p_inf = float(np.mean(np.concatenate(p_far)))
    cp = (cp - p_inf) / (0.5 * 1.0 ** 2)            # p is kinematic, U_inf = 1
    # THETA IS MEASURED FROM THE FRONT STAGNATION POINT, which for a stream along +x sits at
    # x = -R, i.e. atan2(y, x) = 180 deg. Sorting on |atan2| directly puts the REAR first and
    # silently swaps the two numbers this figure exists to report -- it read front = -0.34 and
    # base = +1.02, which is the answer upside down.
    ang = 180.0 - np.abs(th)
    o = np.argsort(ang)
    th_s, cp_s = ang[o], cp[o]

    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_axes([0.06, 0.55, 0.40, 0.40])
    lim = float(np.nanpercentile(np.abs(np.concatenate(
        [f["p"][b][:, :, 0].ravel() for b in range(nb)]) - p_inf), 99.5))
    for b in range(nb):
        blk = d.blocks[b]
        ax.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0], f["p"][b][:, :, 0] - p_inf,
                      cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    ax.set_xlim(-3, 8); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("pressure, one cell per node (no interpolation)", fontsize=11)

    ax = fig.add_axes([0.55, 0.55, 0.40, 0.40])
    for b in range(nb):
        blk = d.blocks[b]
        ax.contour(blk.x[:, :, 0], blk.y[:, :, 0], f["p"][b][:, :, 0] - p_inf,
                   levels=np.linspace(-lim, lim, 25), linewidths=0.7, cmap="RdBu_r")
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))
    ax.set_xlim(-2, 4); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("iso-contours near the body", fontsize=11)

    ax = fig.add_axes([0.08, 0.08, 0.86, 0.36])
    ax.plot(th_s, cp_s, "o", ms=3, color="#5b6c8f", label="solver, both sides")
    ax.axhspan(*CP_BASE_REF, color="#76b041", alpha=0.25, zorder=0,
               label=f"published base $C_p$ {CP_BASE_REF[0]} to {CP_BASE_REF[1]}")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.0, label="inviscid stagnation $C_p$ = 1")
    ax.axhline(0.0, color="0.6", lw=0.8)
    i0 = int(np.argmin(th_s)); i180 = int(np.argmax(th_s))
    ax.set_xlim(0, 180); ax.set_xlabel(r"$\theta$ from the front stagnation point (deg)")
    ax.set_ylabel("$C_p$"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"surface pressure: front {cp_s[i0]:+.3f} (inviscid 1.0), "
                 f"base {cp_s[i180]:+.3f} (published -0.75 to -0.65)", fontsize=11)

    fig.suptitle(f"Circular cylinder, Re = 100 — {tag}, t = {meta['time']:.1f}", fontsize=13)
    out = f"figures/{tag}_pressure.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  t = {meta['time']:.1f}, reference p_inf = {p_inf:+.5f}")
    print(f"  front stagnation C_p = {cp_s[i0]:+.4f}   (inviscid 1.0)")
    print(f"  base C_p             = {cp_s[i180]:+.4f}   (published -0.75 to -0.65)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "cyl_Re100")
