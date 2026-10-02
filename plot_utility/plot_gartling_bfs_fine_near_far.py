import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient

def vorticity_field(ckpt_path, mesh_path):
    d = np.load(ckpt_path)
    nodes, cells_raw, ctag, edges, etag, names = read_gmsh22(mesh_path)
    m = Mesh(nodes, cells_raw, edges, etag, names)
    u, v = d["u"], d["v"]
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
    if (m.nvert == 3).all():
        tris = np.array(cells)
    else:
        tris = np.vstack([[c[0], c[1], c[2]] for c in cells] + [[c[0], c[2], c[3]] for c in cells])
    T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=tris)
    return T, Wv, float(d["time"])

Tq, Wq, tq = vorticity_field("results/ugartling_bfs_quad_xrefine_fine/ckpt_t0280.npz", "meshes/gartling_bfs_quad_xrefine_fine.msh")
Tt, Wt, tt = vorticity_field("results/ugartling_bfs_tri_matched_fine/ckpt_t0380.npz", "meshes/gartling_bfs_tri_matched_fine.msh")

fig, axes = plt.subplots(4, 1, figsize=(14, 11))
LEV = np.linspace(-8, 8, 33)

cf = axes[0].tricontourf(Tq, np.clip(Wq, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
axes[0].set_xlim(3, 11); axes[0].set_ylim(-0.5, 0.5); axes[0].set_aspect("equal")
axes[0].set_title(f"QUAD fine, NEAR field (t={tq:.0f})")
plt.colorbar(cf, ax=axes[0], shrink=0.85, pad=0.01)

cf = axes[1].tricontourf(Tt, np.clip(Wt, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
axes[1].set_xlim(3, 11); axes[1].set_ylim(-0.5, 0.5); axes[1].set_aspect("equal")
axes[1].set_title(f"TRIANGLE fine, NEAR field (t={tt:.0f}) -- watch x~4.4 and x~6.2 (where the extra wall crossings are)")
plt.colorbar(cf, ax=axes[1], shrink=0.85, pad=0.01)

LEVF = np.linspace(-2, 2, 33)
cf = axes[2].tricontourf(Tq, np.clip(Wq, LEVF[0], LEVF[-1]), LEVF, cmap="RdBu_r", extend="both")
axes[2].set_xlim(11, 17); axes[2].set_ylim(-0.5, 0.5); axes[2].set_aspect("equal")
axes[2].set_title(f"QUAD fine, FAR field (t={tq:.0f})")
plt.colorbar(cf, ax=axes[2], shrink=0.85, pad=0.01)

cf = axes[3].tricontourf(Tt, np.clip(Wt, LEVF[0], LEVF[-1]), LEVF, cmap="RdBu_r", extend="both")
axes[3].set_xlim(11, 17); axes[3].set_ylim(-0.5, 0.5); axes[3].set_aspect("equal")
axes[3].set_title(f"TRIANGLE fine, FAR field (t={tt:.0f})")
plt.colorbar(cf, ax=axes[3], shrink=0.85, pad=0.01)

plt.tight_layout()
plt.savefig("figures/gartling_bfs_fine_near_far_vorticity.png", dpi=150)
print("wrote figures/gartling_bfs_fine_near_far_vorticity.png")
