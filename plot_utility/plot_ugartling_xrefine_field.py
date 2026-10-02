"""UniFlow's converged Gartling BFS flow field (x-refined quad mesh, section 74): vorticity,
streamlines, pressure, and the wall-vorticity trace used for the separation/reattachment read-off."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient

d = np.load("results/ugartling_bfs_quad_xrefine/final.npz")
nodes, cells_raw, ctag, edges, etag, names = read_gmsh22("meshes/gartling_bfs_quad_xrefine.msh")
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
nv_ = len(nodes); ws = np.zeros(nv_); Wv = np.zeros(nv_); Pv = np.zeros(nv_); Uv = np.zeros(nv_)
speed = np.hypot(u, v)
for c, vol, ww, pp, uu in zip(cells, m.vol, wc, p, speed):
    ws[c] += vol; Wv[c] += vol * ww; Pv[c] += vol * pp; Uv[c] += vol * uu
Wv /= ws; Pv /= ws; Uv /= ws
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=np.vstack(
    [[c[0], c[1], c[2]] for c in cells] + [[c[0], c[2], c[3]] for c in cells]))

fig, axes = plt.subplots(4, 1, figsize=(14, 11))

LEV = np.linspace(-8, 8, 33)
cf = axes[0].tricontourf(T, np.clip(Wv, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
axes[0].set_xlim(0, 17); axes[0].set_ylim(-0.5, 0.5); axes[0].set_aspect("equal")
axes[0].set_title("vorticity")
plt.colorbar(cf, ax=axes[0], shrink=0.85, pad=0.01)

x = np.linspace(0, 17, 700); y = np.linspace(-0.5, 0.5, 90)
X, Y = np.meshgrid(x, y)
Ui = LinearNDInterpolator(m.centroid, u)(X, Y); Vi = LinearNDInterpolator(m.centroid, v)(X, Y)
axes[1].streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="k", density=1.1, linewidth=0.6, arrowsize=0.7, broken_streamlines=False)
axes[1].axvline(6.1, color="C0", ls="--", lw=1, label="ref lower reatt. 6.1")
axes[1].axvline(4.8, color="C1", ls=":", lw=1, label="ref upper sep. 4.8")
axes[1].axvline(10.5, color="C1", ls="--", lw=1, label="ref upper reatt. 10.5")
axes[1].set_xlim(0, 17); axes[1].set_ylim(-0.5, 0.5); axes[1].set_aspect("equal")
axes[1].set_title("streamlines"); axes[1].legend(fontsize=7, loc="upper right", ncol=3)

LEVP = np.linspace(-0.3, 0.3, 25)
cfp = axes[2].tricontourf(T, np.clip(Pv, LEVP[0], LEVP[-1]), LEVP, cmap="RdBu_r", extend="both")
axes[2].set_xlim(0, 17); axes[2].set_ylim(-0.5, 0.5); axes[2].set_aspect("equal")
axes[2].set_title("pressure (outlet p=0)")
plt.colorbar(cfp, ax=axes[2], shrink=0.85, pad=0.01)

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
axes[3].set_title("wall vorticity (zero crossing = separation/reattachment)")

fig.suptitle(f"UniFlow Gartling BFS, converged (x-refined quad mesh, t={float(d['time']):.0f}): lower reatt. 6.213, upper sep. 4.984, upper reatt. 10.295 (ref 6.1/4.8/10.5)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/ugartling_bfs_xrefine_flowfield.png", dpi=150)
print("wrote figures/ugartling_bfs_xrefine_flowfield.png")
