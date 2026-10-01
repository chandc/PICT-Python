"""UniFlow's Gartling BFS on the quad, wall-clustered mesh: convergence history, final field, and the
wall-vorticity trace -- the residual history matters here as much as the final numbers, since (unlike
the T=400 unstructured run) this one does not show clean monotone decay to machine precision within
T=150; see the write-up for what that may mean."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient

d = np.load("results/ugartling_bfs_quad/final.npz")
nodes, cells_raw, ctag, edges, etag, names = read_gmsh22("meshes/gartling_bfs_quad.msh")
m = Mesh(nodes, cells_raw, edges, etag, names)
u, v, p = d["u"], d["v"], d["p"]
cells = [c[:nv] for c, nv in zip(m.cells, m.nvert)]
hist = d["hist"]

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
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=np.vstack([
    [c[0], c[1], c[2]] for c in cells] + [[c[0], c[2], c[3]] for c in cells]))

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
axes[0].semilogy(hist[:, 0], hist[:, 1], "o-", label="max|du/dt|")
axes[0].semilogy(hist[:, 0], hist[:, 2], "s-", label="max|dv/dt|")
axes[0].set_xlabel("t"); axes[0].set_ylabel("residual"); axes[0].legend(fontsize=8)
axes[0].set_title("convergence history (NOT monotone past t~90 -- plateaus around 1e-2, does not reach machine precision by T=150)")

LEV = np.linspace(-8, 8, 33)
cf = axes[1].tricontourf(T, np.clip(Wv, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
axes[1].set_xlim(0, 17); axes[1].set_ylim(-0.5, 0.5); axes[1].set_aspect("equal")
axes[1].set_title(f"vorticity at t={float(d['time']):.0f} (quad mesh, wall-clustered)")
plt.colorbar(cf, ax=axes[1], shrink=0.85, pad=0.01)

x = np.linspace(0, 17, 700); y = np.linspace(-0.5, 0.5, 90)
X, Y = np.meshgrid(x, y)
Ui = LinearNDInterpolator(m.centroid, u)(X, Y); Vi = LinearNDInterpolator(m.centroid, v)(X, Y)
axes[2].streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="k", density=1.1, linewidth=0.6, arrowsize=0.7, broken_streamlines=False)
axes[2].axvline(6.1, color="C0", ls="--", lw=1, label="ref lower reatt. 6.1")
axes[2].axvline(10.5, color="C1", ls="--", lw=1, label="ref upper reatt. 10.5")
axes[2].set_xlim(0, 17); axes[2].set_ylim(-0.5, 0.5); axes[2].set_aspect("equal")
axes[2].set_title("streamlines"); axes[2].legend(fontsize=7, loc="upper right")

wall = m.bfaces[bt == inv["Wall"]]; wo2 = m.owner[wall]; Sw = m.normal[wall]
e_in = -Sw / np.hypot(Sw[:, 0], Sw[:, 1])[:, None]
dn = ((m.fcentre[wall] - m.centroid[wo2]) * e_in).sum(axis=1)
xw = m.fcentre[wall][:, 0]; yw = m.fcentre[wall][:, 1]
omega_w = -(u[wo2] / dn) * e_in[:, 1]
bot = yw < -0.49; top = yw > 0.49
ob = np.argsort(xw[bot]); ot = np.argsort(xw[top])
axes[3].plot(xw[bot][ob], omega_w[bot][ob], label="lower wall")
axes[3].plot(xw[top][ot], omega_w[top][ot], label="upper wall")
axes[3].axhline(0, color="k", lw=0.5)
axes[3].axvline(6.1, color="C0", ls="--", lw=1); axes[3].axvline(4.8, color="C1", ls=":", lw=1); axes[3].axvline(10.5, color="C1", ls="--", lw=1)
axes[3].set_xlim(0, 17); axes[3].set_xlabel("x"); axes[3].legend(fontsize=8)
axes[3].set_title("wall vorticity; dashed/dotted = Gartling/Chan reference")

fig.suptitle("UniFlow Gartling BFS, quad wall-clustered mesh: lower reatt. 6.091 (ref 6.1), upper sep. 4.827 (ref 4.8), upper reatt. ~10.41 (ref 10.5)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/ugartling_bfs_quad_field.png", dpi=150)
print("wrote figures/ugartling_bfs_quad_field.png")
