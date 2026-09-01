"""The square-cylinder H-grid, coloured by block, with the wake resolution made visible."""
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
from square_cylinder_grid import square_domain, D, X_IN, X_OUT, Y_HALF

COL = {"LB":"#e4572e","MB":"#17bebb","RB":"#ffc914","LM":"#5b6c8f","RM":"#76b041",
       "LT":"#a26769","MT":"#8e6bbf","RT":"#d1495b"}
d, idx = square_domain()
inv = {v: k for k, v in idx.items()}

def draw(ax, step, lw=.3, box=None):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:,:,0], blk.y[:,:,0]
        c = COL[inv[b]]
        for i in range(0, X.shape[0], step):
            ax.plot(X[i,:], Y[i,:], color=c, lw=lw, alpha=.9)
        for j in range(0, X.shape[1], step):
            ax.plot(X[:,j], Y[:,j], color=c, lw=lw, alpha=.9)
    ax.add_patch(plt.Rectangle((-.5*D,-.5*D), D, D, color="k", zorder=9))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0.05, 0.55, 0.92, 0.36])
draw(ax, 3)
ax.set_xlim(X_IN, X_OUT); ax.set_ylim(-Y_HALF, Y_HALF); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title(f"full domain — {d.n_cells:,} cells, blockage {100*D/(2*Y_HALF):.0f}%  "
             f"(every 3rd mesh line)", fontsize=11)

ax = fig.add_axes([0.05, 0.07, 0.44, 0.40])
draw(ax, 1, lw=.35)
ax.set_xlim(-2.5, 4); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
ax.set_title("near the square — every mesh line, 32x32 across the body", fontsize=11)

ax = fig.add_axes([0.56, 0.07, 0.41, 0.40])
x = d.blocks[idx["RM"]].x[:,0,0]; dx = np.diff(x); lam = 1.0/0.13
ax.semilogy(x[:-1], dx, "o-", ms=2.5, lw=1.2, color="#5b6c8f", label="dx along the wake")
ax.axhline(lam/20, color="crimson", ls="--", lw=1.5,
           label=f"$\\lambda$/20 = {lam/20:.2f} D  (resolution target)")
res = x[:-1][lam/dx >= 20].max()
ax.axvline(res, color="crimson", ls=":", lw=1.2)
ax.text(res, dx.min()*1.4, f" resolved to {res:.1f} D", color="crimson", fontsize=9)
ax.set_xlabel("x / D"); ax.set_ylabel("cell size dx / D")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title(f"wake spacing vs the shedding wavelength $\\lambda$ = {lam:.1f} D", fontsize=11)

fig.legend(handles=[Line2D([],[],color=COL[k],lw=3,label=k) for k in
                    ("LT","MT","RT","LM","RM","LB","MB","RB")],
           fontsize=9, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.suptitle("Square cylinder — 8-block H-grid, spanwise periodic", fontsize=13)
fig.savefig("figures/square_cylinder_grid.png", dpi=145, bbox_inches="tight")
print(f"  {d.n_cells:,} cells, validate() {len(d.validate())} problems, "
      f"min(J) {min(J.min() for J,_ in (d.block_metrics_cached(b) for b in range(8))):.2e}")
print(f"  wake resolved to x = {res:.1f} D")
print("wrote figures/square_cylinder_grid.png")
