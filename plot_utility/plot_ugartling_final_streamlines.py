"""Final converged Gartling BFS streamlines (fixed fine triangular mesh, section 77), near and far
field, with the separation/reattachment locations marked and labelled directly on the plot."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22

MESH = "meshes/gartling_bfs_tri_matched_fine.msh"
FIELD = "results/ugartling_bfs_tri_matched_fine_fixed/final.npz"
LOWER_REATT, UPPER_SEP, UPPER_REATT = 6.118, 4.876, 10.411
REF_LOWER, REF_UPPER_SEP, REF_UPPER_REATT = 6.1, 4.8, 10.5

d = np.load(FIELD)
nodes, cells_raw, ctag, edges, etag, names = read_gmsh22(MESH)
m = Mesh(nodes, cells_raw, edges, etag, names)
u, v = d["u"], d["v"]

def panel(ax, xlo, xhi, density):
    x = np.linspace(xlo, xhi, int(500 * (xhi - xlo) / 17) + 300); y = np.linspace(-0.5, 0.5, 140)
    X, Y = np.meshgrid(x, y)
    Ui = LinearNDInterpolator(m.centroid, u)(X, Y); Vi = LinearNDInterpolator(m.centroid, v)(X, Y)
    ax.streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="k", density=density, linewidth=0.6, arrowsize=0.7, broken_streamlines=False)
    ax.set_xlim(xlo, xhi); ax.set_ylim(-0.5, 0.5); ax.set_aspect("equal")

fig, axes = plt.subplots(2, 1, figsize=(14, 7.6))

def mark(ax, x, label, ref, color, y=1.14):
    ax.axvline(x, color=color, ls="--", lw=1.3)
    ax.annotate(f"{label}\n{x:.3f} (ref {ref})", xy=(x, 1.0), xytext=(x, y),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                color=color, fontsize=8.5, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

panel(axes[0], 0, 8, 1.6)
mark(axes[0], UPPER_SEP, "upper separation", REF_UPPER_SEP, "C1", y=1.16)
mark(axes[0], LOWER_REATT, "lower reattachment", REF_LOWER, "C0", y=1.16)
axes[0].text(0.01, 1.02, "near field (x = 0-8)", transform=axes[0].transAxes, fontsize=10, va="bottom")

panel(axes[1], 0, 17, 1.0)
mark(axes[1], UPPER_SEP, "upper separation", REF_UPPER_SEP, "C1", y=1.30)
mark(axes[1], LOWER_REATT, "lower reattachment", REF_LOWER, "C0", y=1.16)
mark(axes[1], UPPER_REATT, "upper reattachment", REF_UPPER_REATT, "C2", y=1.16)
axes[1].text(0.01, 1.02, "far field / full domain (x = 0-17)", transform=axes[1].transAxes, fontsize=10, va="bottom")

fig.suptitle("UniFlow Gartling BFS, Re 800 -- converged, fixed fine triangular mesh (56294 cells)", fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4.5)
plt.savefig("figures/ugartling_final_streamlines_marked.png", dpi=155)
print("wrote figures/ugartling_final_streamlines_marked.png")
