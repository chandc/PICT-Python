"""The vortex street: instantaneous vorticity with streamlines, probe trace, and spectrum."""
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

# Computational studies at ~5% blockage cluster here; the 0.13 often quoted comes from higher
# blockage and from experiment. Both are shown rather than picking the flattering one.
ST_CFD = (0.145, 0.150)
ST_EXP = 0.13
XL, XR, YB, YT = -3.0, 20.0, -5.0, 5.0


def main(tag="sqcyl_spark"):
    d, idx = square_domain(nz=4)
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    h = np.load(f"results/{tag}_history.npy")
    t, v = h[:, 0], h[:, 1]

    k = len(t) // 2
    w = v[k:] - v[k:].mean()
    dt = np.mean(np.diff(t[k:]))
    fr = np.fft.rfftfreq(len(w), dt)
    A = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    A[fr < 0.02] = 0
    St = fr[A.argmax()]
    df = fr[1] - fr[0]

    P, U, V = [], [], []
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        P.append(np.column_stack([blk.x[:, :, 0].ravel(), blk.y[:, :, 0].ravel()]))
        U.append(f["u"][b][:, :, 0].ravel()); V.append(f["v"][b][:, :, 0].ravel())
    P = np.vstack(P); U = np.concatenate(U); V = np.concatenate(V)
    gx = np.linspace(XL, XR, 1100); gy = np.linspace(YB, YT, 480)
    GX, GY = np.meshgrid(gx, gy)
    gu = LinearNDInterpolator(P, U)(GX, GY)
    gv = LinearNDInterpolator(P, V)(GX, GY)
    solid = (np.abs(GX) <= .5*D) & (np.abs(GY) <= .5*D)
    gu[solid] = np.nan; gv[solid] = np.nan
    wz = np.gradient(gv, gx, axis=1) - np.gradient(gu, gy, axis=0)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.05, 0.60, 0.92, 0.33])
    im = ax.pcolormesh(GX, GY, wz, cmap="RdBu_r", shading="gouraud", vmin=-2.5, vmax=2.5)
    ax.streamplot(gx, gy, gu, gv, density=2.6, linewidth=0.45, color="0.2", arrowsize=0.55)
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"spanwise vorticity with streamlines, t = {meta['time']:.0f}", fontsize=11)
    fig.colorbar(im, ax=ax, label=r"$\omega_z D/U$", fraction=0.02, pad=0.008)

    ax = fig.add_axes([0.05, 0.32, 0.60, 0.20])
    ax.plot(t, v, lw=0.6, color="#5b6c8f")
    ax.axvspan(t[0], t[k], color="0.88", zorder=0)
    ax.text(t[len(t)//4], v.max()*0.85, "transient (discarded)", fontsize=8, ha="center",
            color="0.35")
    ax.set_xlabel("t U / D"); ax.set_ylabel("v at (2D, 0.5D)"); ax.grid(alpha=.3)
    ax.set_title("probe history — saturated limit cycle", fontsize=11)

    ax = fig.add_axes([0.72, 0.32, 0.25, 0.20])
    ax.plot(fr, A / A.max(), lw=1.1, color="#e4572e")
    ax.axvspan(*ST_CFD, color="#76b041", alpha=.25, label="CFD, 5% blockage")
    ax.axvline(ST_EXP, ls="--", color="0.45", lw=1.1, label=f"experiment {ST_EXP}")
    ax.axvline(St, ls=":", color="#e4572e", lw=1.5, label=f"measured {St:.4f}")
    ax.set_xlim(0, 0.45); ax.set_xlabel("St = f D / U"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3); ax.set_title("spectrum", fontsize=11)

    ax = fig.add_axes([0.05, 0.04, 0.92, 0.19]); ax.axis("off")
    q = v[3*len(v)//4:]
    rows = [("Strouhal", f"{St:.4f}  +/- {df:.4f} (FFT bin)"),
            ("reference", f"{ST_CFD[0]}-{ST_CFD[1]} CFD at 5% blockage; {ST_EXP} experiment"),
            ("period", f"{1/St:.2f} D/U  =  {1/St/0.01:.0f} steps"),
            ("saturated amplitude", f"{q.max()-q.min():.4f}  [{q.min():+.4f}, {q.max():+.4f}]"),
            ("solver", "tol 1e-6, AmgX on GB10, Rhie-Chow on, ddt_corr off, Dong outflow"),
            ("grid", f"{d.n_cells:,} cells, 8 blocks, blockage 5%")]
    for i, (a_, b_) in enumerate(rows):
        ax.text(0.02, 0.88 - i*0.16, a_, fontsize=10, weight="bold", va="top")
        ax.text(0.22, 0.88 - i*0.16, b_, fontsize=10, va="top", family="monospace")

    fig.suptitle(f"Square cylinder, Re = 100 — vortex street, St = {St:.4f}", fontsize=14)
    out = f"figures/{tag}_vortex_street.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  St = {St:.4f} +/- {df:.4f}")
    print(f"  amplitude {q.max()-q.min():.4f}, t to {meta['time']:.0f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_spark")
