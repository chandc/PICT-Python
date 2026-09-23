"""Show the NACA0012 C-grid: full domain, near field, leading edge, trailing edge / wake cut."""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from src.umesh import read_gmsh22, Mesh
f = sys.argv[1] if len(sys.argv) > 1 else "meshes/naca0012_a20.msh"; alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
import math; al = math.radians(alpha); tex, tey = math.cos(al), -math.sin(al)
nodes, cells, ctag, edges, etag, names = read_gmsh22(f); m = Mesh(nodes, cells, edges, etag, names)
polys = [nodes[c] for c in cells]; bt = m.btag[m.bfaces]; inv = {v: k for k, v in names.items()}
wall = m.bfaces[bt == inv["Airfoil"]]
i = ~m.boundary; o, n = m.owner[i], m.neigh[i]; ratio = np.maximum(m.vol[o], m.vol[n]) / np.minimum(m.vol[o], m.vol[n])
print(f"{f}: {m.ncell} cells ({int((m.nvert==4).sum())} quads), {len(wall)} wall faces, nbr vol ratio p99 {np.percentile(ratio,99):.2f} max {ratio.max():.2f}, wall-cell sqrt(area) {np.sqrt(m.vol[m.owner[wall]]).min():.4f}..{np.sqrt(m.vol[m.owner[wall]]).max():.4f}")
fig, ax = plt.subplots(2, 3, figsize=(27, 13))
views = [((-9, 25), (-8.5, 8.5), 0.15, "full domain: inlet arc R=8 + top/bottom (Dirichlet u=1), outlet x=24 (p=0)"), ((-0.6, 2.6), (-1.6, 1.0), 0.3, "near field"), ((-1.0, 6.0), (-3.5, 2.5), 0.25, "near wake"),
         ((-0.08, 0.22), (-0.1, 0.13), 0.5, "leading edge"), ((tex - 0.25, tex + 0.55), (tey - 0.4, tey + 0.3), 0.5, "trailing edge and wake cut"), ((tex - 0.08, tex + 0.12), (tey - 0.08, tey + 0.08), 0.7, "trailing edge, close")]
for a_, (xl, yl, lw, ttl) in zip(ax.ravel(), views):
    a_.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=lw)); a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_aspect("equal"); a_.set_title(ttl)
    for fc in m.fcentre[wall]: pass
    W = nodes[np.unique(np.concatenate([edges[k] for k in range(len(edges)) if etag[k] == inv["Airfoil"]]))]
    a_.plot(W[:, 0], W[:, 1], "r.", ms=2)
plt.suptitle(f"NACA0012, alpha = {alpha} deg, Re = 100 (chord): structured C-grid, {m.ncell} quads", fontsize=13); plt.tight_layout()
out = f"figures/naca0012_a{int(alpha)}_grid.png"; plt.savefig(out, dpi=110); print("wrote", out)
