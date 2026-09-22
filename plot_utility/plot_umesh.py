"""Draw an unstructured triangular mesh, with a zoom and the quality metrics."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from src.umesh import rect_mesh

fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
for k, pert in enumerate((0.0, 0.25)):
    m = rect_mesh(64, 64, perturb=pert)
    t = Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris)
    ax[k].triplot(t, color="#3465a4", lw=0.25)
    ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1); ax[k].set_aspect("equal")
    i = m.interior
    ax[k].set_title(f"perturb = {pert}   ({m.ncell:,} triangles)\n"
                    f"orthogonality cos(d,S): min {m.orth[i].min():.4f}, mean {m.orth[i].mean():.4f}",
                    fontsize=10)
# zoom on the unperturbed one to show the alternating diagonals
m = rect_mesh(64, 64, perturb=0.0)
t = Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris)
ax[2].triplot(t, color="#3465a4", lw=0.8)
ax[2].plot(m.centroid[:, 0], m.centroid[:, 1], ".", color="#c0392b", ms=3)
ax[2].set_xlim(0.45, 0.58); ax[2].set_ylim(0.45, 0.58); ax[2].set_aspect("equal")
ax[2].set_title("zoom: alternating diagonals\n(red = cell centroids, where u, v, p live)", fontsize=10)
fig.suptitle("Cavity mesh — the Re=1000 run used the LEFT one (64x64, perturb=0)",
             fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig("figures/ucavity_mesh.png", dpi=115)
mm = rect_mesh(64, 64, perturb=0.0)
print(f"  wrote figures/ucavity_mesh.png")
print(f"  run mesh: {mm.ncell:,} triangles, {mm.nface:,} faces, {int(mm.boundary.sum())} boundary")
print(f"  cell area: min {mm.area.min():.3e}  max {mm.area.max():.3e}  (uniform)")
print(f"  audit: {mm.audit() or 'CLEAN'}")
