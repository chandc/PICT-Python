"""Time-averaged surface C_p with its cycle rms, and the C_L / C_D spectra.

Four panels, because four different questions are being asked:

  top left     mean C_p around the perimeter with a +/- rms band. The front stagnation must be
               +1 exactly and the base value is what the literature tabulates.
  top right    the same C_p as a map on the body outline, so "which face" needs no translation
  bottom left  the force records themselves, C_L and C_D against time
  bottom right the spectra. This is where the drag peak sitting at exactly twice the lift
               frequency is visible, which is the invariant that no tuning can fake.

The FFT bin is drawn as a horizontal bar on the spectrum, because the peak location cannot be
read more finely than that and the figure should not pretend otherwise. The zero-crossing value
is marked separately with its own, much smaller, uncertainty.
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

D = 1.0
cp = np.load("results/sq_surface_cp.npz")
sp = np.load("results/sq_force_spectra.npz")
S, CPm, CPr = cp["s"], cp["cp_mean"], cp["cp_rms"]
fr, FL, FD = sp["fr"], sp["FL"], sp["FD"]
t, cd, cl, T = sp["t"], sp["cd"], sp["cl"], float(sp["T"])
St_zc = 1.0 / T
kL = int(np.argmax(FL[1:]) + 1); kD = int(np.argmax(FD[1:]) + 1)
bin_w = fr[1]

fig = plt.figure(figsize=(15.5, 10.4))

ax = fig.add_axes([0.055, 0.565, 0.42, 0.36])
if CPr.size:
    ax.fill_between(S, CPm - CPr, CPm + CPr, color="#5b6c8f", alpha=0.28,
                    label=r"$\pm$ rms over one period")
ax.plot(S, CPm, color="#1f3b73", lw=1.7, label="time mean, 3500 samples")
ax.axhline(0, color="k", lw=0.5)
ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":", label=r"$C_p=+1$, exact at stagnation")
for x, nm in ((0.5, "front|top"), (1.5, "top|base"), (2.5, "base|bottom"), (3.5, "bottom|front")):
    ax.axvline(x, color="k", lw=0.4, alpha=0.35)
ax.text(2.0, CPm[np.argmin(np.abs(S-2.0))]-0.28, "base", ha="center", fontsize=9)
ax.set_xlabel("perimeter coordinate s (anticlockwise from front stagnation)")
ax.set_ylabel(r"$C_p$"); ax.set_xlim(0, 4)
ax.legend(fontsize=8, loc="lower left")
ax.set_title(r"time-averaged surface $C_p$", fontsize=11)

ax = fig.add_axes([0.55, 0.565, 0.40, 0.36])
th = {}
def xy(s):
    h = 0.5*D
    if s < 0.5:   return -h, s
    if s < 1.5:   return -h + (s-0.5), h
    if s < 2.5:   return h, h - (s-1.5)
    if s < 3.5:   return h - (s-2.5), -h
    return -h, s-4.0
XY = np.array([xy(s) for s in S])
sc = ax.scatter(XY[:,0], XY[:,1], c=CPm, cmap="RdYlBu_r", s=26, vmin=-1.6, vmax=1.0)
ax.plot([-.5,.5,.5,-.5,-.5], [-.5,-.5,.5,.5,-.5], "k-", lw=0.8)
ax.annotate("flow", xy=(-1.35, 0), xytext=(-2.1, 0),
            arrowprops=dict(arrowstyle="->", lw=1.2), va="center", fontsize=9)
ax.set_aspect("equal"); ax.set_xlim(-2.3, 1.3); ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
fig.colorbar(sc, ax=ax, shrink=0.8, label=r"mean $C_p$")
ax.set_title("the same values on the body", fontsize=11)

ax = fig.add_axes([0.055, 0.075, 0.42, 0.37])
ax.plot(t, cl, color="#c0392b", lw=0.9, label=rf"$C_L$, rms {np.sqrt((cl**2).mean()):.4f}")
ax.plot(t, cd, color="#1f3b73", lw=0.9, label=rf"$C_D$, mean {cd.mean():.4f}, "
                                             rf"rms' {cd.std():.4f}")
ax.set_xlabel("t"); ax.set_ylabel("force coefficient")
ax.legend(fontsize=8, loc="center right"); ax.set_title("force records", fontsize=11)

ax = fig.add_axes([0.55, 0.075, 0.40, 0.37])
ax.semilogy(fr, FL, color="#c0392b", lw=1.2, label=r"$C_L$")
ax.semilogy(fr, FD, color="#1f3b73", lw=1.2, label=r"$C_D$")
ax.axvline(St_zc, color="k", ls="--", lw=1.0)
ax.axvline(2*St_zc, color="k", ls=":", lw=1.0)
ax.text(St_zc, FL.max()*1.4, f"  St = {St_zc:.4f}\n  (zero crossings,\n   ±0.0005)",
        fontsize=8, va="top")
ax.text(2*St_zc, FD[kD]*3.0, f"  2 St", fontsize=8, va="top")
ax.errorbar([fr[kL]], [FL[kL]*2.2], xerr=[bin_w/2], fmt="o", ms=4, color="#c0392b",
            capsize=3, label=f"FFT peak ± half a bin ({bin_w:.4f})")
ax.set_xlim(0, 0.65); ax.set_ylim(FL.max()*3e-4, FL.max()*6)
ax.set_xlabel("St = f D / U"); ax.set_ylabel("amplitude")
ax.legend(fontsize=8); ax.set_title("spectra — the drag peak sits at twice the lift peak",
                                    fontsize=11)

fig.suptitle("square cylinder, Re = 100 — time-averaged surface pressure and force spectra",
             fontsize=13)
out = "figures/sqcyl_cp_spectra.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print(f"  wrote {out}")
