"""UniFlow's converged Gartling BFS field: vorticity, streamlines, and the wall-vorticity trace used
for the reattachment/separation zero-crossing read-off, against Chan & Mittal's reference numbers."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient

d = np.load("results/ugartling_bfs/final.npz")
nodes, cells_raw, ctag, edges, etag, names = read_gmsh22("meshes/gartling_bfs.msh")
m = Mesh(nodes, cells_raw, edges, etag, names)
u, v, p = d["u"], d["v"], d["p"]
cells = [c[:nv] for c, nv in zip(m.cells, m.nvert)]

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
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=np.array(cells))

fig, axes = plt.subplots(3, 1, figsize=(14, 7.5))
LEV = np.linspace(-8, 8, 33)
cf = axes[0].tricontourf(T, np.clip(Wv, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
axes[0].set_xlim(0, 17); axes[0].set_ylim(-0.5, 0.5); axes[0].set_aspect("equal")
axes[0].set_title(f"vorticity, converged (t={float(d['time']):.0f})")
plt.colorbar(cf, ax=axes[0], shrink=0.85, pad=0.01)

x = np.linspace(0, 17, 700); y = np.linspace(-0.5, 0.5, 90)
X, Y = np.meshgrid(x, y)
Ui = LinearNDInterpolator(m.centroid, u)(X, Y); Vi = LinearNDInterpolator(m.centroid, v)(X, Y)
axes[1].streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="k", density=2.5, linewidth=0.6, arrowsize=0.7, broken_streamlines=False)
axes[1].axvline(6.1, color="C0", ls="--", lw=1, label="Gartling/Chan lower reatt. 6.1")
axes[1].axvline(10.5, color="C1", ls="--", lw=1, label="Gartling/Chan upper reatt. 10.5")
axes[1].set_xlim(0, 17); axes[1].set_ylim(-0.5, 0.5); axes[1].set_aspect("equal")
axes[1].set_title("streamlines"); axes[1].legend(fontsize=7, loc="upper right")

wall = m.bfaces[bt == inv["Wall"]]; wo2 = m.owner[wall]; Sw = m.normal[wall]
e_in = -Sw / np.hypot(Sw[:, 0], Sw[:, 1])[:, None]
dn = ((m.fcentre[wall] - m.centroid[wo2]) * e_in).sum(axis=1)
xw = m.fcentre[wall][:, 0]; yw = m.fcentre[wall][:, 1]
omega_w = -(u[wo2] / dn) * e_in[:, 1]
bot = yw < -0.49; top = yw > 0.49
order_b = np.argsort(xw[bot]); order_t = np.argsort(xw[top])
axes[2].plot(xw[bot][order_b], omega_w[bot][order_b], label="lower wall")
axes[2].plot(xw[top][order_t], omega_w[top][order_t], label="upper wall")
axes[2].axhline(0, color="k", lw=0.5)
axes[2].axvline(6.1, color="C0", ls="--", lw=1); axes[2].axvline(4.8, color="C1", ls=":", lw=1); axes[2].axvline(10.5, color="C1", ls="--", lw=1)
axes[2].set_xlim(0, 17); axes[2].set_xlabel("x"); axes[2].set_ylabel("wall omega"); axes[2].legend(fontsize=8)
axes[2].set_title("wall vorticity (zero crossing = separation/reattachment); dashed/dotted = Gartling/Chan reference")

fig.suptitle("UniFlow Gartling BFS, Re 800: lower reattachment 6.20 (ref 6.1, +1.6%)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/ugartling_bfs_field.png", dpi=150)
print("wrote figures/ugartling_bfs_field.png")
