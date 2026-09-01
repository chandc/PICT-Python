"""The cylinder O-grid, coloured by block, with the radial distribution beside it.

The mesh and the wake-resolution curve are plotted together because they are the same decision:
a pure geometric radial stretch resolves the boundary layer beautifully and the wake not at all.
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
from matplotlib.lines import Line2D

from cylinder_grid import cylinder_domain, D, R_CYL, outer_role

NBLK = 16
ST = 0.164
# One colour per block, and they must stay VISIBLE as well as distinct. With 16 blocks a
# palette of 8 repeats, and two same-coloured blocks opposite each other read as one block
# wrapping the ring. tab20 has 20 hues but alternates dark/light, so every other block came out
# nearly white on these thin mesh lines. Evenly spaced hues at fixed saturation, darkened a
# little, keep all NBLK equally legible.
COL = [matplotlib.colors.to_hex(0.85 * np.array(c[:3]))
       for c in plt.get_cmap("hsv")(np.linspace(0, 1, NBLK, endpoint=False))]


def draw(ax, d, step_r=1, step_t=1, lw=0.3):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        c = COL[b % len(COL)]
        for i in range(0, X.shape[0], step_r):
            ax.plot(X[i, :], Y[i, :], color=c, lw=lw, alpha=.9)
        for j in range(0, X.shape[1], step_t):
            ax.plot(X[:, j], Y[:, j], color=c, lw=lw, alpha=.9)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color="k", zorder=9))


d, r, arc = cylinder_domain(nblk=NBLK, nz=4)
dr = np.diff(r)
roles = outer_role(d, NBLK)
lam = D / ST
# the actual outflow arc, measured off the grid rather than assumed -- it is what the far-field
# leak was about, so it belongs on the figure
_th = np.concatenate([np.degrees(np.arctan2(d.blocks[b].y[-1], d.blocks[b].x[-1])).ravel()
                      for b in roles if roles[b] == "outflow"])
arc_half = np.abs((_th + 180) % 360 - 180).max()

fig = plt.figure(figsize=(15, 10))

ax = fig.add_axes([0.04, 0.53, 0.44, 0.43])
draw(ax, d, step_r=2, step_t=4)
ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title(f"full domain, far field at {r[-1]:.0f} D  ({d.n_cells:,} cells)", fontsize=11)

ax = fig.add_axes([0.53, 0.53, 0.44, 0.43])
draw(ax, d, step_r=1, step_t=1, lw=0.25)
ax.set_xlim(-2.2, 4.5); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title("near the cylinder — every mesh line", fontsize=11)

ax = fig.add_axes([0.06, 0.29, 0.41, 0.17])
draw(ax, d, step_r=1, step_t=1, lw=0.3)
ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_aspect("equal")
ax.set_title(f"boundary layer: wall cell {dr[0]:.4f} D, "
             f"{(D/10)/dr[0]:.0f} cells across $\\delta$", fontsize=10)
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")

ax = fig.add_axes([0.55, 0.29, 0.42, 0.17])
ax.semilogy(r[:-1], dr, "o-", ms=2.5, lw=1.1, color="#5b6c8f", label="radial dr")
ax.semilogy(r, 2*np.pi*r/256, "-", lw=1.1, color="#76b041", label=r"azimuthal $r\,d\theta$")
ax.axhline(lam/20, color="crimson", ls="--", lw=1.4,
           label=f"$\\lambda$/20 = {lam/20:.2f} D")
ok = r[:-1][np.maximum(dr, 2*np.pi*r[:-1]/256) <= lam/20]
if len(ok):
    ax.axvline(ok.max(), color="crimson", ls=":", lw=1.2)
    ax.text(ok.max()*1.05, dr.min()*2, f"resolved to {ok.max():.1f} D", color="crimson",
            fontsize=8.5)
ax.set_xlabel("r / D"); ax.set_ylabel("cell size / D")
ax.set_xlim(0.4, 30); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"spacing vs the shedding wavelength $\\lambda$ = {lam:.2f} D", fontsize=10)

ax = fig.add_axes([0.06, 0.03, 0.91, 0.19]); ax.axis("off")
worst = max(np.maximum(dr[1:]/dr[:-1], dr[:-1]/dr[1:]).max(), 1.0)
ins = sorted(b for b in roles if roles[b] == "inflow")
outs = sorted(b for b in roles if roles[b] == "outflow")
rows = [("topology", f"{NBLK}-block O-grid ring, no reentrant corners (unlike an H-grid)"),
        ("cells", f"{d.n_cells:,}   radial {len(r)}  azimuthal 256  spanwise 4"),
        ("validate()", f"{len(d.validate())} problems;  min(J) = "
                       f"{min(d.block_metrics_cached(b)[0].min() for b in range(NBLK)):.3e}"),
        ("spacing", f"wall {dr[0]:.4f} D, plateau {np.median(dr):.3f} D, outer {dr[-1]:.2f} D; "
                    f"worst adjacent ratio {worst:.3f}  (limit 1.20)"),
        ("far field", f"free stream on {len(ins)} blocks;  Dong outflow on blocks {outs}, "
                      f"|theta| <= {arc_half:.1f} deg"),
        ("blockage", f"D / 2R_out = {100*D/(2*r[-1]):.1f}%")]
for i, (a_, b_) in enumerate(rows):
    ax.text(0.01, 0.92 - i*0.16, a_, fontsize=10, weight="bold", va="top")
    ax.text(0.15, 0.92 - i*0.16, b_, fontsize=10, va="top", family="monospace")

fig.legend(handles=[Line2D([], [], color=COL[b % len(COL)], lw=3, label=f"blk {b}")
                    for b in range(NBLK)],
           fontsize=8.5, ncol=min(NBLK, 8), loc="lower center", bbox_to_anchor=(0.5, -0.035),
           frameon=False)
fig.suptitle("Circular cylinder — O-grid, Re = 100 case", fontsize=13, y=1.01)
out = "figures/cylinder_grid.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"  {d.n_cells:,} cells, validate() {len(d.validate())} problems")
print(f"  wake resolved to {ok.max():.1f} D at >=20 cells/wavelength")
print(f"  wrote {out}")
