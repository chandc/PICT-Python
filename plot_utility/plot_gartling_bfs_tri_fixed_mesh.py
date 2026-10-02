"""Mesh plot for the fixed fine triangular BFS mesh (sections 76-77): full domain, near-step zoom,
and a close zoom on the former local-outlier defect locations (x~4.4, x~6.2), now clean."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from src.umesh import Mesh, read_gmsh22

nodes, cells_raw, ctag, edges, etag, names = read_gmsh22("meshes/gartling_bfs_tri_matched_fine.msh")
m = Mesh(nodes, cells_raw, edges, etag, names)
cells = [c[:nv] for c, nv in zip(m.cells, m.nvert)]
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=np.array(cells))

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
axes[0].triplot(T, lw=0.08, color="k")
axes[0].set_xlim(0, 17); axes[0].set_ylim(-0.5, 0.5); axes[0].set_aspect("equal")
axes[0].set_title(f"full domain: {m.ncell} cells, {len(nodes)} nodes")

axes[1].triplot(T, lw=0.15, color="k")
axes[1].set_xlim(0, 8); axes[1].set_ylim(-0.5, 0.5); axes[1].set_aspect("equal")
axes[1].set_title("near-step zoom (x = 0-8)")

axes[2].triplot(T, lw=0.3, color="k")
axes[2].set_xlim(3.8, 6.8); axes[2].set_ylim(-0.5, 0.5); axes[2].set_aspect("equal")
axes[2].set_title("closer zoom on x = 3.8-6.8 (former defect locations, now clean)")

fig.suptitle("UniFlow Gartling BFS: fixed fine triangular mesh (section 76-77 grading-mult fix)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/gartling_bfs_tri_fixed_mesh.png", dpi=155)
print("wrote figures/gartling_bfs_tri_fixed_mesh.png")
