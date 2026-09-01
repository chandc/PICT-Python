"""C_D and C_L against time, and the lift spectrum.

WHY THE LIFT SPECTRUM MATTERS MORE THAN THE PROBE. Strouhal number is defined on the shedding
frequency, and the standard experimental and computational route to it is the LIFT signal, not a
velocity probe in the wake. They should agree, and where they do not the probe is the suspect:
it sits at one point, so it sees the local passage of vortices and any drift of the wake past
that point, while lift integrates the whole surface. This plot puts both numbers on the same
figure so the agreement is checked rather than assumed.

C_D OSCILLATES AT TWICE THE LIFT FREQUENCY. Each shed vortex, top or bottom, pulls the body
sideways once but reduces the base pressure once per vortex regardless of which side it came
from, so drag completes two cycles per shedding period. It is a good check that the force
integral is picking up the physics rather than noise: if the drag spectrum does not peak at 2 St
something is wrong upstream of the plot.

THE FOURTH COLUMN IS AN ERROR BAR, NOT A FORCE. `results/<tag>_forces.npy` carries the part of
C_D contributed by the viscous NORMAL stress, which vanishes analytically on a no-slip wall (see
src/forces.py). It is plotted alongside because on the square cylinder it is the same size as
the genuine friction drag and of opposite sign, so a reader who does not see it will conclude
the friction contribution is negligible when in fact two errors cancelled.
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

REF = {"sqcyl": ("square cylinder", (1.45, 1.50), (0.145, 0.150)),
       "cyl": ("circular cylinder", (1.32, 1.35), (0.164, 0.165))}


def reference_for(tag):
    return REF["sqcyl"] if tag.startswith("sqcyl") else REF["cyl"]


def spectrum(t, y):
    y = y - y.mean()
    dt = float(np.mean(np.diff(t)))
    fr = np.fft.rfftfreq(len(y), dt)
    A = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    A[fr < 0.02] = 0.0
    i = int(A.argmax())
    df = fr[1] - fr[0]
    if 0 < i < len(A) - 1:                      # parabolic peak refinement
        c = 0.5 * (A[i-1] - A[i+1]) / (A[i-1] - 2*A[i] + A[i+1])
    else:
        c = 0.0
    return fr, A, float(fr[i] + c * df), float(df)


def main(tag="sqcyl_v3_forces", t_lo=None):
    f = np.load(f"results/{tag}_forces.npy")
    t, cd, cl = f[:, 0], f[:, 1], f[:, 2]
    cdn = f[:, 3] if f.shape[1] > 3 else np.zeros_like(t)
    if t_lo is not None:
        m = t >= t_lo
        t, cd, cl, cdn = t[m], cd[m], cl[m], cdn[m]
    name, cd_ref, st_ref = reference_for(tag)

    frl, Al, St_l, df = spectrum(t, cl)
    frd, Ad, St_d, _ = spectrum(t, cd)

    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_axes([0.07, 0.70, 0.88, 0.24])
    ax.plot(t, cd, lw=1.2, color="#5b6c8f")
    ax.axhline(cd.mean(), color="#5b6c8f", ls="--", lw=1.0)
    ax.axhspan(*cd_ref, color="#76b041", alpha=0.18, zorder=0)
    ax.set_ylabel("$C_D$"); ax.grid(alpha=.3)
    ax.set_title(f"drag: mean {cd.mean():.4f}, peak-to-peak {cd.max()-cd.min():.4f}"
                 f"   (published {cd_ref[0]}-{cd_ref[1]}, shaded)", fontsize=11)

    ax = fig.add_axes([0.07, 0.40, 0.88, 0.24])
    ax.plot(t, cl, lw=1.2, color="#e4572e")
    ax.axhline(0.0, color="0.5", lw=0.8)
    rms = float(np.sqrt((cl**2).mean()))
    ax.set_ylabel("$C_L$"); ax.set_xlabel("t U / D"); ax.grid(alpha=.3)
    ax.set_title(f"lift: rms {rms:.4f}, amplitude {0.5*(cl.max()-cl.min()):.4f}, "
                 f"mean {cl.mean():+.5f}  (mean should vanish by symmetry)", fontsize=11)

    ax = fig.add_axes([0.07, 0.06, 0.40, 0.24])
    ax.semilogy(frl, Al / Al.max(), lw=1.2, color="#e4572e", label="lift")
    ax.semilogy(frd, Ad / Ad.max(), lw=1.0, color="#5b6c8f", alpha=.8, label="drag")
    ax.axvspan(*st_ref, color="#76b041", alpha=0.2, zorder=0)
    ax.axvline(St_l, color="#e4572e", ls=":", lw=1.2)
    ax.axvline(2 * St_l, color="#5b6c8f", ls=":", lw=1.2)
    ax.set_xlim(0, max(6 * St_l, 0.6)); ax.set_ylim(1e-4, 2)
    ax.set_xlabel("St = f D / U"); ax.set_ylabel("normalised amplitude")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
    ax.set_title(f"St from lift {St_l:.4f} +/- {df:.4f};  drag peaks at {St_d:.4f} "
                 f"= {St_d/St_l:.2f} x", fontsize=10)

    ax = fig.add_axes([0.55, 0.06, 0.40, 0.24])
    ax.plot(t, cdn, lw=1.1, color="#a26769")
    ax.axhline(cdn.mean(), color="#a26769", ls="--", lw=1.0)
    ax.set_xlabel("t U / D"); ax.set_ylabel("$C_D$ from viscous normal stress")
    ax.grid(alpha=.3)
    ax.set_title(f"discretisation error, zero analytically: mean {cdn.mean():+.5f}\n"
                 f"({100*abs(cdn.mean())/max(abs(cd.mean()), 1e-30):.2f}% of C_D)", fontsize=10)

    fig.suptitle(f"{name}, {tag} — loads over t = {t[0]:.1f} to {t[-1]:.1f} "
                 f"({(t[-1]-t[0])*St_l:.1f} shedding periods)", fontsize=13)
    out = f"figures/{tag}_loads.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  C_D {cd.mean():.4f} +/- {cd.std():.4f}   C_L rms {rms:.4f}   "
          f"C_L mean {cl.mean():+.5f}")
    print(f"  St from lift {St_l:.4f}, drag peak {St_d:.4f} ({St_d/St_l:.2f}x)")
    print(f"  spurious viscous normal stress in C_D: {cdn.mean():+.5f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_v3_forces",
         float(_sys.argv[2]) if len(_sys.argv) > 2 else None)
