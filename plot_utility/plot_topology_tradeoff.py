"""Why the two cylinder grids differ, and what the choice actually costs."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path: _sys.path.insert(0,_ROOT)
_os.chdir(_ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from cylinder_ring_grid import ring_rect_domain, D, R_CYL
from cylinder_cart_grid import cart_ring_domain

fig = plt.figure(figsize=(16.5, 11))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.18)

def draw(ax, d, idx, title, note, ncolor):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        s0 = max(1, X.shape[0]//55); s1 = max(1, X.shape[1]//55)
        for i in range(0, X.shape[0], s0): ax.plot(X[i,:], Y[i,:], color=ncolor, lw=0.35)
        for j in range(0, X.shape[1], s1): ax.plot(X[:,j], Y[:,j], color=ncolor, lw=0.35)
    th = np.linspace(0, 2*np.pi, 300)
    ax.fill(R_CYL*np.cos(th), R_CYL*np.sin(th), color="0.5", zorder=6)
    ax.add_patch(Rectangle((3,-2), 5, 4, fill=False, ec="#c0392b", lw=2.2, zorder=7))
    ax.text(5.5, 2.25, "wake band\n(sets St, C_L)", color="#c0392b", ha="center",
            fontsize=9, weight="bold")
    ax.set_xlim(-2, 11); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.text(0.01, -0.13, note, transform=ax.transAxes, fontsize=9.3, va="top")

d1, i1 = ring_rect_domain(nz=2)
d2, i2 = cart_ring_domain(nz=2)
ax = fig.add_subplot(gs[0,0])
draw(ax, d1, i1, "A.  BUTTERFLY  -- wake lines come FROM the cylinder",
     "Every horizontal line in the wake is a RAY that started on the cylinder.\n"
     "So wake dy and cylinder azimuthal count are THE SAME NUMBER.\n"
     "To halve dy in the wake you must double the points on the body.", "#3465a4")
ax = fig.add_subplot(gs[0,1])
draw(ax, d2, i2, "B.  CARTESIAN BACKGROUND -- wake lines are its own",
     "The wake block has its own y-distribution, independent of the body.\n"
     "Same wake resolution for less than half the cells.\n"
     "BUT: its outer tiling has a defect -- see panel C.", "#1a7a3a")

# ---- C: the parity obstruction -------------------------------------------------------------
ax = fig.add_subplot(gs[1,0]); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("C.  why the Cartesian grid cannot be finished", fontsize=11, weight="bold")
L = 1.0
ax.add_patch(Rectangle((-L,-L), 2*L, 2*L, fill=False, ec="k", lw=2))
m = 7
e = np.linspace(-L, L, m)
for yy in e: ax.plot([L],[yy],'o',color="#c0392b",ms=7)      # east edge nodes
for xx in e: ax.plot([xx],[L],'s',color="#1a7a3a",ms=7)      # north edge nodes
for yy in e: ax.plot([-L],[yy],'o',color="#c0392b",ms=7)
for xx in e: ax.plot([xx],[-L],'s',color="#1a7a3a",ms=7)
for cx,cy in ((L,L),(-L,L),(-L,-L),(L,-L)):
    ax.add_patch(Circle((cx,cy), 0.17, fill=False, ec="#e67e22", lw=3, zorder=5))
ax.text(0,0, "O-ring\n+ transition", ha="center", va="center", fontsize=10)
ax.text(1.35, 0, "E tile needs\nthese %d nodes" % m, color="#c0392b", fontsize=9, va="center")
ax.text(0, 1.42, "N tile needs these %d nodes" % m, color="#1a7a3a", fontsize=9, ha="center")
ax.text(0, -2.05,
        "The 4 corners (orange) are supplied TWICE -- once by an E/W edge, once by N/S.\n"
        "Perimeter needs 2p+2q-4 distinct nodes; the tiles demand 2p+2q.\n"
        "Shifting ownership turns the 4 duplicates into 4 GAPS. It is a counting\n"
        "identity, not a tuning error -- measured: two cells at ONE point holding\n"
        "velocities 0.20 apart, which floods the ring with spurious vorticity.",
        ha="center", fontsize=9.3)
ax.set_xlim(-2.4, 2.6); ax.set_ylim(-2.6, 1.8)

# ---- D: the cost curve ---------------------------------------------------------------------
ax = fig.add_subplot(gs[1,1])
ne   = [36, 50, 64, 80, 100]
cells= [22664, 30980, 39968, 48896, 61736]
mid  = [1.95, 1.64, 1.44, 1.29, 1.15]
far  = [2.07, 1.74, 1.55, 1.38, 1.23]
ax.plot(cells, mid, "o-", color="#3465a4", label="butterfly, mid wake")
ax.plot(cells, far, "s--", color="#3465a4", alpha=0.6, label="butterfly, far wake")
for c,m_,n in zip(cells, mid, ne):
    ax.annotate(f"n_east={n}", (c,m_), textcoords="offset points", xytext=(4,7), fontsize=8)
ax.plot([27311],[1.02], "*", color="#1a7a3a", ms=20, label="CARTESIAN, mid wake")
ax.plot([27311],[0.84], "*", color="#1a7a3a", ms=14, alpha=0.55, label="CARTESIAN, far wake")
ax.axhline(1.0, color="k", lw=1.2, ls=":")
ax.text(63000, 1.03, "HydroGym", fontsize=9, ha="right")
ax.set_xlabel("cells (2D)  ->  runtime"); ax.set_ylabel("our cell size / HydroGym's   (1.0 = matched)")
ax.set_title("D.  what the choice costs", fontsize=11, weight="bold")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
ax.text(0.01,-0.17, "The Cartesian grid (green star) gets BETTER wake resolution than any\n"
        "butterfly, at half the cells -- but it is the one with the defect in panel C.",
        transform=ax.transAxes, fontsize=9.3, va="top")
fig.suptitle("Two cylinder grids: why one resolves the wake cheaply but cannot be built, "
             "and what the other costs", fontsize=13, weight="bold")
fig.savefig("figures/topology_tradeoff.png", dpi=110, bbox_inches="tight")
print("  wrote figures/topology_tradeoff.png")
