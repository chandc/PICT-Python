"""The turbulent mean velocity profile at Re_tau = 180, and the eddy viscosity behind it.

Three curves that should agree and one asymptote that should not be assumed:

  * van Driest integrated finely -- the manufactured solution;
  * the same problem on the 97-point clustered grid, with the face-averaged operator the solver
    uses, which is the resolution study's recommendation;
  * the solver's own 3-D result, if `results/van_driest_profile.npz` exists.

The log law is drawn as a reference rather than fitted. van Driest was constructed to reproduce
it, so agreement there is a consistency check on the integration, not evidence about the
physics; the informative part is the SUBLAYER, where U+ = y+ is exact and independent of the
model, and the y^4 vs y^3 discrepancy in nu_t that no mean-profile plot can show.
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

KAPPA, APLUS, RE_TAU = 0.41, 26.0, 180.0
B_LOG = 5.2


def reference(n=2000001):
    yp = np.linspace(0.0, RE_TAU, n)
    D = 1.0 - np.exp(-yp / APLUS)
    lm = KAPPA * yp * D
    tau = 1.0 - yp / RE_TAU
    dU = 2.0 * tau / (1.0 + np.sqrt(1.0 + 4.0 * lm ** 2 * tau))
    U = np.concatenate([[0.0], np.cumsum(0.5 * (dU[1:] + dU[:-1]) * np.diff(yp))])
    return yp, U, lm ** 2 * dU


def clustered(dy0_plus=1.0, ratio=1.05):
    dy0 = dy0_plus / RE_TAU
    ys, dy = [0.0], dy0
    while ys[-1] < 1.0:
        ys.append(ys[-1] + dy); dy *= ratio
    ys = np.array(ys) / ys[-1]
    return np.concatenate([ys[:-1], 2.0 - ys[::-1]])


def solve_1d(y, yp_ref, nut_ref):
    nu = 1.0 + np.interp(np.minimum(y, 2 - y) * RE_TAU, yp_ref, nut_ref)
    n = len(y)
    A = np.zeros((n, n)); b = -np.ones(n) / RE_TAU
    A[0, 0] = A[-1, -1] = 1.0; b[0] = b[-1] = 0.0
    for i in range(1, n - 1):
        hm, hp = (y[i] - y[i-1]) * RE_TAU, (y[i+1] - y[i]) * RE_TAU
        cm, cp, hc = 0.5*(nu[i-1]+nu[i])/hm, 0.5*(nu[i]+nu[i+1])/hp, 0.5*(hm+hp)
        A[i, i-1], A[i, i+1], A[i, i] = cm/hc, cp/hc, -(cm+cp)/hc
    return np.linalg.solve(A, b)


yp, U, nut = reference()
y = clustered()
U97 = solve_1d(y, yp, nut)
ypg = np.minimum(y, 2 - y) * RE_TAU

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.semilogx(yp[1:], U[1:], lw=2.0, color="#5b6c8f", label="van Driest, finely integrated")
ax.semilogx(ypg[1:len(y)//2], U97[1:len(y)//2], "o", ms=4, mfc="none",
            color="#e4572e", label=f"97-point clustered grid ($\\Delta y^+_w$ = 1, r = 1.05)")
sub = np.logspace(-1, np.log10(12), 50)
ax.semilogx(sub, sub, "--", lw=1.2, color="0.35", label="$U^+ = y^+$ (viscous sublayer)")
lg = np.logspace(np.log10(25), np.log10(RE_TAU), 50)
ax.semilogx(lg, np.log(lg) / KAPPA + B_LOG, ":", lw=1.6, color="crimson",
            label=fr"$\frac{{1}}{{\kappa}}\ln y^+ + B$, $\kappa$ = {KAPPA}, B = {B_LOG}")
try:
    dat = np.load("results/van_driest_profile.npz")
    ax.semilogx(dat["yp"], dat["U"], "s", ms=4, color="#76b041",
                label="PISO solver, 3-D")
except FileNotFoundError:
    pass
ax.set_xlim(0.1, RE_TAU); ax.set_ylim(0, 20)
ax.set_xlabel("$y^+$"); ax.set_ylabel("$U^+$"); ax.grid(alpha=.3, which="both")
ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"Mean velocity, $Re_\\tau$ = {RE_TAU:.0f}", fontsize=12)

ax = axes[1]
ax.loglog(yp[1:], nut[1:], lw=2.0, color="#5b6c8f", label=r"$\nu_t^+$, van Driest")
ax.loglog(yp[1:], KAPPA * yp[1:], ":", lw=1.6, color="crimson",
          label=r"$\kappa y^+$ (log-layer asymptote)")
sm = yp[(yp > 0.2) & (yp < 2)]
ax.loglog(sm, 3e-5 * sm ** 4, "--", lw=1.2, color="0.35", label=r"$\propto (y^+)^4$")
ax.loglog(sm, 3e-5 * sm ** 3, "-.", lw=1.2, color="#76b041",
          label=r"$\propto (y^+)^3$ -- the EXACT asymptote")
ax.set_xlim(0.1, RE_TAU); ax.set_ylim(1e-6, 200)
ax.set_xlabel("$y^+$"); ax.set_ylabel(r"$\nu_t/\nu$"); ax.grid(alpha=.3, which="both")
ax.legend(fontsize=9, loc="upper left")
ax.set_title(r"Eddy viscosity: van Driest gives $y^4$, the truth is $y^3$", fontsize=12)

fig.suptitle("Turbulent channel, $Re_\\tau$ = 180 — van Driest as a manufactured solution",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "figures/channel_profile_Re180.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
Ub = np.trapezoid(U, yp) / RE_TAU
print(f"  U_b+ = {Ub:.4f}, Re_bulk = {Ub*RE_TAU:.0f}, U_c+ = {U[-1]:.3f}")
print(f"  97-point grid: U_b+ = {np.trapezoid(U97, y*RE_TAU)/(2*RE_TAU):.4f}")
print(f"  wrote {out}")
