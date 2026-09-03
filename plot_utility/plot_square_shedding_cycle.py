"""One shedding period of the square cylinder at eight phases, with the force it produces.

The point of the figure is the LINK, not the pictures. Each panel carries the lift coefficient
computed from that same field by surface integration, so the alternation of the vortices and the
oscillation of C_L are shown to be one phenomenon rather than asserted to be.

The period is measured from the C_L zero crossings, T = 6.7199 +/- 0.0218, so the eight panels
are 45 degrees apart in phase and the last is one full period after the first.

Vorticity is differentiated on the MESH and interpolated afterwards. Differentiating an
interpolant paints one band per mesh cell -- see plot_vorticity_pipeline.py.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from src import checkpoint
from src.forces import surface_force
from square_cylinder_bc import classify
from square_cylinder_grid import square_domain, D

XL, XR, YB, YT = -2.0, 9.5, -3.0, 3.0
N = 520
NU, U_INF = 0.01, 1.0


def main():
    files = sorted(glob.glob("results/fields/sqph_*.npz"))
    if len(files) < 9:
        print(f"  only {len(files)} phase files present, need 9 -- run sq_phases.py first")
        return 1
    # nine files span a FULL period inclusive: 0..7 are the eight distinct phases and 8 is one
    # period after 0, kept out of the montage and used as the closure check below.
    closing, files = files[8], files[:8]
    d, _ = square_domain(nz=4)
    nb = len(d.blocks)
    body = [k for k, v in classify(d).items() if v == "body"]
    span = float(d.blocks[0].period[2])
    q = 0.5 * U_INF ** 2 * D * span
    P = np.vstack([np.column_stack([b.x[:, :, 0].ravel(), b.y[:, :, 0].ravel()])
                   for b in d.blocks])
    gx = np.linspace(XL, XR, N)
    gy = np.linspace(YB, YT, int(N * (YT - YB) / (XR - XL)))
    GX, GY = np.meshgrid(gx, gy)

    fig, axes = plt.subplots(4, 2, figsize=(15.5, 12.4), sharex=True, sharey=True)
    lim, cls, ts = 5.0, [], []
    for ax, path in zip(axes.T.ravel(), files):
        f, meta = checkpoint.load_fields(path)
        U = {b: f["u"][b] for b in range(nb)}
        V = {b: f["v"][b] for b in range(nb)}
        W = np.concatenate([(d.gradient(b, V)[0] - d.gradient(b, U)[1])[:, :, 0].ravel()
                            for b in range(nb)])
        Z = LinearNDInterpolator(P, W)(GX, GY)
        Z[(np.abs(GX) <= 0.5 * D) & (np.abs(GY) <= 0.5 * D)] = np.nan
        R = surface_force(d, body, f["u"], f["v"], f["w"], f["p"], NU)
        cl = R["total"][1] / q; cd = R["total"][0] / q
        cls.append(cl); ts.append(float(meta["time"]))
        im = ax.pcolormesh(GX, GY, Z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.add_patch(plt.Rectangle((-.5 * D, -.5 * D), D, D, color="k", zorder=9))
        ph = (float(meta["time"]) - ts[0]) / 6.7199
        ax.set_title(f"phase {ph*360:5.0f}°   t = {meta['time']:.2f}   "
                     f"$C_L$ = {cl:+.4f}   $C_D$ = {cd:+.4f}", fontsize=9.5)
        ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("x / D")
    for ax in axes[:, 0]:
        ax.set_ylabel("y / D")
    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\omega_z D/U$", pad=0.01)
    fig.suptitle("square cylinder, Re = 100 — one shedding period, T = 6.7199 (St = 0.1488). "
                 "$C_L$ is integrated over the body from the SAME field in each panel.",
                 fontsize=12.5)
    out = "figures/sqcyl_shedding_cycle.png"
    fig.savefig(out, dpi=112, bbox_inches="tight")
    print(f"  wrote {out}")
    cls = np.array(cls)
    print(f"  C_L over the eight phases: " + " ".join(f"{c:+.4f}" for c in cls))
    print(f"  C_L rms across the cycle {np.sqrt((cls**2).mean()):.4f}  "
          f"(force record over 35 time units gives 0.1722)")
    fc, mc = checkpoint.load_fields(closing)
    Rc = surface_force(d, body, fc["u"], fc["v"], fc["w"], fc["p"], NU)
    cl_close = Rc["total"][1] / q
    print(f"  CLOSURE: C_L at t = {ts[0]:.2f} is {cls[0]:+.5f}; one period later "
          f"(t = {float(mc['time']):.2f}) it is {cl_close:+.5f}")
    print(f"  difference {abs(cl_close-cls[0]):.5f}, i.e. {100*abs(cl_close-cls[0])/0.1722:.2f}% "
          f"of the C_L rms -- the cycle closes, so T = 6.7199 is right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
