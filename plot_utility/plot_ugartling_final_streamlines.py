"""Final converged Gartling BFS streamlines (fixed fine triangular mesh, section 77), near and far
field, with the separation/reattachment locations marked and labelled directly on the plot."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient

MESH = "meshes/gartling_bfs_tri_matched_fine.msh"
FIELD = "results/ugartling_bfs_tri_matched_fine_fixed/final.npz"
LOWER_REATT, UPPER_SEP, UPPER_REATT = 6.118, 4.876, 10.411
REF_LOWER, REF_UPPER_SEP, REF_UPPER_REATT = 6.1, 4.8, 10.5

d = np.load(FIELD)
nodes, cells_raw, ctag, edges, etag, names = read_gmsh22(MESH)
m = Mesh(nodes, cells_raw, edges, etag, names)
u, v = d["u"], d["v"]
cells = [c[:nv] for c, nv in zip(m.cells, m.nvert)]

# vertex-averaged vorticity, same convention as the other field plots in this record
inv = {vv: k for k, vv in names.items()}
bt = m.btag[m.bfaces]
yb = m.fcentre[m.bfaces][:, 1]
uo, vo = u[m.owner[m.bfaces]], v[m.owner[m.bfaces]]
ubv = np.where(bt == inv["Inlet"], np.clip(24.0 * yb * (0.5 - yb), 0.0, None), np.where(bt == inv["Wall"], 0.0, uo))
vbv = np.where(bt == inv["Outlet"], vo, 0.0)
g = Gradient(m); wc = g(v, vbv)[:, 0] - g(u, ubv)[:, 1]
nv_ = len(nodes); ws = np.zeros(nv_); Wv = np.zeros(nv_)
for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
Wv /= ws
import matplotlib.tri as mtri
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=np.array(cells))
LEV = np.linspace(-8, 8, 33)

def panel(ax, xlo, xhi, density):
    cf = ax.tricontourf(T, np.clip(Wv, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both", alpha=0.85)
    x = np.linspace(xlo, xhi, int(500 * (xhi - xlo) / 17) + 300); y = np.linspace(-0.5, 0.5, 140)
    X, Y = np.meshgrid(x, y)
    Ui = LinearNDInterpolator(m.centroid, u)(X, Y); Vi = LinearNDInterpolator(m.centroid, v)(X, Y)
    ax.streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="k", density=density, linewidth=0.55, arrowsize=0.65, broken_streamlines=False)
    ax.set_xlim(xlo, xhi); ax.set_ylim(-0.5, 0.5); ax.set_aspect("equal")
    return cf

fig, axes = plt.subplots(2, 1, figsize=(14, 8.4))

def mark(ax, x, label, ref, color, y=1.14):
    ax.axvline(x, color=color, ls="--", lw=1.3)
    ax.annotate(f"{label}\n{x:.3f} (ref {ref})", xy=(x, 1.0), xytext=(x, y),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                color=color, fontsize=8.5, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

cf0 = panel(axes[0], 0, 8, 0.75)
mark(axes[0], UPPER_SEP, "upper separation", REF_UPPER_SEP, "C1", y=1.16)
mark(axes[0], LOWER_REATT, "lower reattachment", REF_LOWER, "C0", y=1.16)
axes[0].text(0.01, 1.02, "near field (x = 0-8)", transform=axes[0].transAxes, fontsize=10, va="bottom")
plt.colorbar(cf0, ax=axes[0], shrink=0.85, pad=0.01, label="vorticity")

cf1 = panel(axes[1], 0, 17, 0.55)
mark(axes[1], UPPER_SEP, "upper separation", REF_UPPER_SEP, "C1", y=1.30)
mark(axes[1], LOWER_REATT, "lower reattachment", REF_LOWER, "C0", y=1.16)
mark(axes[1], UPPER_REATT, "upper reattachment", REF_UPPER_REATT, "C2", y=1.16)
axes[1].text(0.01, 1.02, "far field / full domain (x = 0-17)", transform=axes[1].transAxes, fontsize=10, va="bottom")
plt.colorbar(cf1, ax=axes[1], shrink=0.85, pad=0.01, label="vorticity")

fig.suptitle("UniFlow Gartling BFS, Re 800 -- converged, fixed fine triangular mesh (56294 cells)", fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4.5)
plt.savefig("figures/ugartling_final_streamlines_marked.png", dpi=155)
print("wrote figures/ugartling_final_streamlines_marked.png")
