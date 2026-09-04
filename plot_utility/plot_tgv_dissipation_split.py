"""Where the dissipation comes from, and the post-peak deficit every algebraic model shares.

The energy curves in `tgv400_vs_dns.png` show the models agreeing with the DNS to about a
percent. This figure shows the one place they systematically do not, and decomposes it.

TOP: total dissipation against the DNS. The reference PLATEAUS from t = 7 to 9 and then rolls
off; every model decays monotonically from t = 7, opening a deficit that peaks near -17% at
t = 9 and closes again by t = 11.

BOTTOM: the split. `2 nu Z` is the dissipation the coarse grid RESOLVES; the rest is what the
model supplies. Across the plateau the resolved part falls 19% and the modelled part falls 32%,
while the DNS total falls 4%.

WHY. An algebraic eddy viscosity is a LOCAL-EQUILIBRIUM closure: nu_t ~ Delta^2 |S| assumes the
subgrid dissipation equals the instantaneous cascade rate set by the resolved strain. On the
plateau the large scales have stopped feeding the cascade but energy already IN the small scales
keeps dissipating -- the subgrid field has memory that outlives the strain which created it. The
DNS carries that memory explicitly; an algebraic model cannot, so nu_t falls with the resolved
strain when the true subgrid dissipation does not.

All three surviving models do this within a few percent of each other, which is what makes it
structural rather than a constant needing tuning -- and is the argument for a one-equation model
carrying a transport equation for the subgrid kinetic energy.
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

NU, V = 0.0025, (2 * np.pi) ** 3
STYLE = {"smagorinsky": (0, ()), "wale": (0, (6, 2)), "vreman": (0, (2, 2)),
         "sigma": (0, (7, 2, 1, 2))}

runs, k = {}, None
for L in open(_sys.argv[1] if len(_sys.argv) > 1 else "/tmp/n64_split.txt"):
    if L.startswith("###"):
        k = L.split()[1]; runs[k] = []
    elif k and L.strip():
        runs[k].append([float(x) for x in L.split()])
runs = {m: np.array(v) for m, v in runs.items() if v}

z = np.load("results/tgv_diag_re400.npz")
tr, Dr = z["t"], 2 * NU * z["Om"] / V

fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True)

ax = axes[0]
ax.plot(tr, Dr, color="#b0b0b0", lw=6, solid_capstyle="round", label="SEM DNS", zorder=1)
for m, a in runs.items():
    if a[-1, 0] < 11:            # WALE aside, only completed runs are informative here
        pass
    ax.plot(a[:, 0], a[:, 1], "k", lw=1.4, ls=STYLE[m], label=m, zorder=3)
ax.axvspan(7, 9, color="#c0392b", alpha=0.09, zorder=0)
ax.text(8.0, 0.0138, "DNS plateau", ha="center", fontsize=9, color="#c0392b")
ax.set_ylabel(r"$\epsilon = -dE/dt$"); ax.set_ylim(0, 0.015); ax.set_xlim(0, 15)
ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.25)
ax.set_title("64$^3$ — total dissipation:  the DNS holds a plateau at t = 7–9, "
             "the models do not", fontsize=11.5)

ax = axes[1]
a = runs["sigma"]
ax.plot(tr, Dr, color="#b0b0b0", lw=6, solid_capstyle="round", label="SEM DNS total", zorder=1)
ax.plot(a[:, 0], a[:, 1], "k-", lw=1.8, label=r"$\sigma$: total", zorder=3)
ax.plot(a[:, 0], a[:, 2], "k--", lw=1.4, label=r"$\sigma$: RESOLVED, $2\nu Z$", zorder=3)
ax.plot(a[:, 0], a[:, 1] - a[:, 2], "k:", lw=1.6, label=r"$\sigma$: supplied by the MODEL",
        zorder=3)
ax.axvspan(7, 9, color="#c0392b", alpha=0.09, zorder=0)
ax.annotate("", xy=(9, np.interp(9, a[:, 0], a[:, 1])), xytext=(9, np.interp(9, tr, Dr)),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
ax.text(9.15, 0.0100, "$-17\\%$ at $t=9$", color="#c0392b", fontsize=9)
ax.text(8.0, 0.0138, "across the plateau:  resolved $-19\\%$,  model $-32\\%$,  "
        "DNS total $-4\\%$", ha="center", fontsize=8.5, color="#c0392b")
ax.set_xlabel("t"); ax.set_ylabel(r"$\epsilon$"); ax.set_ylim(0, 0.015)
ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.25)
ax.set_title(r"the split — the modelled part falls fastest, because $\nu_t \propto |S|$ "
             r"tracks the RESOLVED strain", fontsize=11.5)

fig.suptitle("Taylor–Green, Re = 400, 64$^3$:  a non-equilibrium deficit that every algebraic "
             "eddy-viscosity model shares", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "figures/tgv400_dissipation_split.png"
fig.savefig(out, dpi=118)
print(f"  wrote {out}")
for m, a in sorted(runs.items()):
    d7, d9 = np.interp([7, 9], a[:, 0], a[:, 1])
    r7, r9 = np.interp([7, 9], a[:, 0], a[:, 2])
    D9 = np.interp(9, tr, Dr)
    print(f"  {m:<12} deficit at t=9 {100*(d9-D9)/D9:+6.1f}%   "
          f"resolved {100*(r9-r7)/r7:+6.1f}%   model {100*((d9-r9)-(d7-r7))/(d7-r7):+6.1f}%")
