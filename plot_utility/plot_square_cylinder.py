"""Vortex street diagnostics: probe history, its spectrum, and the vorticity field."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from square_cylinder_grid import square_domain, D, X_IN, X_OUT, Y_HALF
from src import checkpoint

ST_REF = 0.13          # accepted value for a square cylinder at Re = 100


def strouhal(t, v, skip_frac=0.4):
    """Peak of the FFT of v(t) after discarding the startup transient.

    The transient must go: it is broadband and its low-frequency content sits right where the
    shedding peak is, so including it biases the estimate toward zero.
    """
    k = int(len(t) * skip_frac)
    t, v = t[k:], v[k:] - np.mean(v[k:])
    dt = np.mean(np.diff(t))
    f = np.fft.rfftfreq(len(v), dt)
    A = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    A[f < 0.02] = 0.0                      # kill the residual DC lobe
    return f[A.argmax()], f, A


def main(tag):
    hist = np.load(f"results/{tag}_history.npy")
    t, v = hist[:, 0], hist[:, 1]
    St, f, A = strouhal(t, v)

    d, idx = square_domain(nz=4)
    fld = checkpoint.load_fields(f"results/fields/{tag}.npz")

    fig = plt.figure(figsize=(15, 9))

    ax = fig.add_axes([0.06, 0.70, 0.60, 0.24])
    ax.plot(t, v, lw=0.7, color="#5b6c8f")
    ax.axvspan(t[0], t[int(len(t)*0.4)], color="0.85", zorder=0)
    ax.text(t[int(len(t)*0.2)], ax.get_ylim()[1]*0.8, "transient\n(discarded)",
            ha="center", fontsize=8, color="0.35")
    ax.set_xlabel("t U / D"); ax.set_ylabel("v at (2D, 0.5D)")
    ax.set_title("probe history", fontsize=11); ax.grid(alpha=.3)

    ax = fig.add_axes([0.72, 0.70, 0.25, 0.24])
    ax.plot(f, A / max(A.max(), 1e-30), lw=1.1, color="#e4572e")
    ax.axvline(ST_REF, ls="--", color="0.4", lw=1.2, label=f"reference {ST_REF}")
    ax.axvline(St, ls=":", color="#e4572e", lw=1.4, label=f"measured {St:.4f}")
    ax.set_xlim(0, 0.5); ax.set_xlabel("St = f D / U"); ax.set_ylabel("amplitude")
    ax.set_title("spectrum", fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_axes([0.06, 0.07, 0.91, 0.55])
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        u, v2 = fld["u"][b][:, :, 0], fld["v"][b][:, :, 0]
        # vorticity from the physical coordinates directly -- adequate for a picture
        dvdx = np.gradient(v2, X[:, 0], axis=0)
        dudy = np.gradient(u, Y[0, :], axis=1)
        ax.pcolormesh(X, Y, dvdx - dudy, cmap="RdBu_r", vmin=-3, vmax=3, shading="gouraud")
    ax.add_patch(plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9))
    ax.set_xlim(-3, 22); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"spanwise vorticity at t = {t[-1]:.1f}", fontsize=11)

    err = 100 * abs(St - ST_REF) / ST_REF
    fig.suptitle(f"Square cylinder, Re = 100 — St = {St:.4f} "
                 f"(reference {ST_REF}, {err:.1f}% off)", fontsize=13)
    out = f"figures/{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  St = {St:.4f}   reference {ST_REF}   {err:.1f}% off")
    print(f"  {len(t):,} samples, t up to {t[-1]:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_Re100_rc_n63280")
