"""Why the drag runs at twice the lift frequency, shown rather than asserted.

A vortex street sheds one vortex per half period, alternating sign. The flow at t + T/2 is
therefore the MIRROR of the flow at t, reflected in y. Under that reflection the lift reverses
and the drag does not, so

    C_L(t + T/2) = -C_L(t)        C_D(t + T/2) = +C_D(t)

and a signal satisfying the second has period T/2, i.e. frequency 2 St. The "drag peak sits at
twice the lift peak" is a CONSEQUENCE of this, not independent evidence for it -- and reading it
off a spectrum is far weaker, because the bin width here is 0.0286, 19% of St.

This plots the identity directly. The shift is T/2 = 3.359957, which is 335.9957 samples at
dt = 0.01, so the shifted record is INTERPOLATED rather than rolled by an integer number of
samples; rolling by 336 would introduce a 0.0043-sample phase error and put a spurious floor
under the residual.
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

sp = np.load("results/sq_force_spectra.npz")
t, cd, cl, T = sp["t"], sp["cd"], sp["cl"], float(sp["T"])
half = 0.5 * T

# interpolate onto t + T/2, keeping only where that lands inside the record
m = t + half <= t[-1]
ts = t[m]
cl_s = np.interp(ts + half, t, cl)
cd_s = np.interp(ts + half, t, cd)
cl_0 = cl[m]
cd_0 = cd[m]
res_L = cl_0 + cl_s          # zero if C_L is antisymmetric
res_D = cd_0 - cd_s          # zero if C_D is symmetric
rms_L = np.sqrt((cl ** 2).mean())

# The eight phase FIELDS live at t = 415.0 to 421.72, past the end of this force record
# (which stops at 415.0, and the shifted copy stops half a period earlier still). They cannot
# be marked on these axes, so they are not claimed in the legend -- see
# figures/sqcyl_shedding_cycle.png for those, where the same identity holds field by field.

fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.6), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 1, 0.85]))

ax = axes[0]
ax.plot(ts, cl_0, color="#c0392b", lw=1.5, label=r"$C_L(t)$")
ax.plot(ts, -cl_s, color="#1f3b73", lw=1.5, ls="--", label=r"$-C_L(t+T/2)$")
ax.set_ylabel(r"$C_L$")
ax.legend(fontsize=9, ncol=2, loc="upper right")
ax.set_title(r"lift REVERSES under a half-period shift:  $C_L(t+T/2) = -C_L(t)$", fontsize=11)

ax = axes[1]
ax.plot(ts, cd_0, color="#c0392b", lw=1.5, label=r"$C_D(t)$")
ax.plot(ts, cd_s, color="#1f3b73", lw=1.5, ls="--", label=r"$C_D(t+T/2)$")
ax.set_ylabel(r"$C_D$")
ax.legend(fontsize=9, loc="upper right")
ax.set_title(r"drag does NOT:  $C_D(t+T/2) = +C_D(t)$, so its period is $T/2$ and its "
             r"frequency is $2\,St$", fontsize=11)

ax = axes[2]
ax.semilogy(ts, np.abs(res_L) / rms_L, color="#c0392b", lw=1.1,
            label=r"$|C_L(t)+C_L(t+T/2)|\ /\ C_L^{rms}$")
ax.semilogy(ts, np.abs(res_D) / cd.mean(), color="#1f3b73", lw=1.1,
            label=r"$|C_D(t)-C_D(t+T/2)|\ /\ \overline{C_D}$")
ax.axhline(1e-2, color="k", lw=0.5, ls=":")
ax.text(ts[0] + 0.3, 1.15e-2, "1%", fontsize=8)
ax.set_ylabel("relative residual")
ax.set_xlabel("t")
ax.legend(fontsize=9, loc="upper right")
ax.set_title(f"relative residuals: median {np.median(np.abs(res_L))/rms_L:.2e} in lift "
             f"(of its rms), {np.median(np.abs(res_D))/(cd.max()-cd.min()):.2e} in drag "
             f"(of its own peak-to-peak)", fontsize=11)

fig.suptitle("square cylinder, Re = 100 — the half-period symmetry that puts the drag at twice "
             f"the lift frequency\nT = {T:.4f} from C_L zero crossings; the shift is "
             f"interpolated, not rolled by whole samples", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "figures/sqcyl_half_period_symmetry.png"
fig.savefig(out, dpi=118)
print(f"  wrote {out}")
print(f"  lift residual   median {np.median(np.abs(res_L)):.2e}  "
      f"max {np.abs(res_L).max():.2e}   ({100*np.median(np.abs(res_L))/rms_L:.3f}% of rms)")
print(f"  drag residual   median {np.median(np.abs(res_D)):.2e}  "
      f"max {np.abs(res_D).max():.2e}   ({100*np.median(np.abs(res_D))/cd.mean():.4f}% of mean)")
print(f"  C_D peak-to-peak {cd.max()-cd.min():.4f}, so the drag residual is "
      f"{100*np.median(np.abs(res_D))/(cd.max()-cd.min()):.2f}% of its own oscillation")
