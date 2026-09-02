"""What the solver PREDICTED: the axial velocity, and the eddy viscosity recovered from it.

Two panels, and only one of them is a prediction in the ordinary sense.

LEFT -- the axial velocity U+(y+). This IS predicted: the solver is given nu_eff(y) and a
uniform body force and starts from rest; the profile is what comes out of the momentum balance.
It is compared against the van Driest integration that the same nu_eff was built from, which is
the manufactured solution.

RIGHT -- the eddy viscosity, two ways. `nu_t prescribed` is what went IN, so plotting it alone
would be circular. `nu_t recovered` is the inverse: given the predicted profile and the known
total stress tau+ = 1 - y/delta,

    nu_t+ = tau+ / (dU+/dy+) - 1

which is exactly the operation used to extract an eddy viscosity from DNS. If the solver is
consistent the two curves coincide, and where they cannot coincide is the interesting part: at
the CENTRELINE dU+/dy+ -> 0 and tau+ -> 0 together, so nu_t is a 0/0 there and unrecoverable
from mean-flow data however good the solver is. That is the same identifiability limit Stage 2
hit with the solenoidal part of a source field, and any network asked to learn nu_t from
velocity data inherits it.
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

KAPPA, APLUS = 0.41, 26.0

d = np.load("results/van_driest_profile.npz")
y, U, nu_eff, NU, RE_TAU = d["y"], d["U"], d["nu_eff"], float(d["NU"]), float(d["RE_TAU"])
yp = d["yp"]
half = len(y) // 2 + 1

ypr = np.linspace(0.0, RE_TAU, 400001)
Dv = 1.0 - np.exp(-ypr / APLUS)
lm = KAPPA * ypr * Dv
tau = 1.0 - ypr / RE_TAU
dUr = 2.0 * tau / (1.0 + np.sqrt(1.0 + 4.0 * lm ** 2 * tau))
Ur = np.concatenate([[0.0], np.cumsum(0.5 * (dUr[1:] + dUr[:-1]) * np.diff(ypr))])
nutr = lm ** 2 * dUr

# recovered eddy viscosity: nu_t+ = tau+/(dU+/dy+) - 1, from the PREDICTED profile
ys, Us = y[:half] * RE_TAU, U[:half]
dU_dy = np.gradient(Us, ys)
tau_s = 1.0 - ys / RE_TAU
with np.errstate(divide="ignore", invalid="ignore"):
    nut_rec = np.where(np.abs(dU_dy) > 1e-9, tau_s / dU_dy - 1.0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.semilogx(ypr[1:], Ur[1:], lw=2.0, color="#5b6c8f", label="van Driest (manufactured answer)")
ax.semilogx(ys[1:], Us[1:], "o", ms=4.5, mfc="none", color="#e4572e",
            label="PISO solver, predicted")
sub = np.logspace(-1, np.log10(12), 40)
ax.semilogx(sub, sub, "--", lw=1.1, color="0.35", label="$U^+ = y^+$")
lg = np.logspace(np.log10(30), np.log10(RE_TAU), 40)
ax.semilogx(lg, np.log(lg)/KAPPA + 4.23, ":", lw=1.5, color="crimson",
            label=r"$\frac{1}{\kappa}\ln y^+ + 4.23$")
ax.set_xlim(0.5, RE_TAU); ax.set_ylim(0, 18)
ax.set_xlabel("$y^+$ (distance from the wall)"); ax.set_ylabel("$U^+$")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"Predicted axial velocity, $Re_\\tau$ = {RE_TAU:.0f}   "
             f"(max diff {np.abs(Us - np.interp(ys, ypr, Ur)).max():.4f})", fontsize=11)

ax = axes[1]
ax.semilogx(ypr[1:], nutr[1:], lw=2.0, color="#5b6c8f", label=r"$\nu_t^+$ prescribed (input)")
ax.semilogx(ys[1:], nut_rec[1:], "o", ms=4.5, mfc="none", color="#e4572e",
            label=r"$\nu_t^+$ recovered from the predicted profile")
ax.semilogx(ypr[1:], KAPPA*ypr[1:], ":", lw=1.4, color="crimson", label=r"$\kappa y^+$")
ax.axvspan(0.7*RE_TAU, RE_TAU, color="0.85", alpha=0.7, zorder=0)
ax.text(0.72*RE_TAU, 1.5, "unrecoverable:\n$dU/dy \\to 0$", fontsize=8.5, color="0.3")
ax.set_xlim(0.5, RE_TAU); ax.set_ylim(0, 32)
ax.set_xlabel("$y^+$ (distance from the wall)"); ax.set_ylabel(r"$\nu_t/\nu$")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9, loc="upper left")
ax.set_title("Eddy viscosity: prescribed vs recovered from the prediction", fontsize=11)

fig.suptitle("Channel at $Re_\\tau$ = 180 — solver prediction against the manufactured solution",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "figures/channel_prediction_Re180.png"
fig.savefig(out, dpi=140, bbox_inches="tight")

good = (ys > 5) & (ys < 0.6*RE_TAU)
err = np.nanmax(np.abs(nut_rec[good] - np.interp(ys[good], ypr, nutr)))
print(f"  predicted U_c+ = {Us.max():.4f}, van Driest {Ur.max():.4f}")
print(f"  max |U+ diff| = {np.abs(Us - np.interp(ys, ypr, Ur)).max():.5f}")
print(f"  nu_t+ recovered vs prescribed, 5 < y+ < {0.6*RE_TAU:.0f}: max diff {err:.4f} "
      f"(peak nu_t+ = {nutr.max():.1f})")
print(f"  wrote {out}")
