"""Streamlines for a UniFlow cavity field (interpolated from cell centroids onto a regular grid,
masked to the actual non-convex domain, since matplotlib's streamplot needs a regular grid)."""
import sys, argparse; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from src.umesh import Mesh, read_gmsh22

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/hgcavity_fine.msh")
ap.add_argument("--field", default="results/ucavity_hg_fine/ckpt.npz")
ap.add_argument("--out", default="figures/ucavity_fine_streamlines.png")
ap.add_argument("--n", type=int, default=500, help="grid resolution")
a = ap.parse_args()

nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh)
m = Mesh(nodes, cells, edges, etag, names)
d = np.load(a.field)
u, v, p = d["u"], d["v"], d["p"]
t = float(d["time"]) if "time" in d else None
C = m.centroid

x = np.linspace(-1.2, 2.5, a.n); y = np.linspace(-1.0, 0.5, int(a.n * 1.5 / 3.7))
X, Y = np.meshgrid(x, y)
inside = (Y >= 0) | ((X >= 0) & (X <= 1) & (Y < 0) & (Y >= -1))

Ui = LinearNDInterpolator(C, u)(X, Y)
Vi = LinearNDInterpolator(C, v)(X, Y)
Pi = LinearNDInterpolator(C, p)(X, Y)
speed = np.hypot(Ui, Vi)
Ui = np.where(inside, Ui, np.nan); Vi = np.where(inside, Vi, np.nan); speed = np.where(inside, speed, np.nan)

fig, ax = plt.subplots(figsize=(11, 5))
cf = ax.contourf(X, Y, speed, levels=np.linspace(0, 1.3, 27), cmap="viridis", extend="max")
plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.01, label="|u|")
ax.streamplot(X, Y, np.nan_to_num(Ui), np.nan_to_num(Vi), color="white", density=2.0, linewidth=0.7, arrowsize=0.8,
              broken_streamlines=False)
# paint the solid region (outside the fluid domain) over the streamlines so none leak through walls
mask = np.where(inside, np.nan, 1.0)
ax.contourf(X, Y, mask, levels=[0.5, 1.5], colors=["0.35"])
ax.set_xlim(-1.2, 2.5); ax.set_ylim(-1.0, 0.5); ax.set_aspect("equal")
ttl = f"UniFlow, fine mesh, streamlines" + (f" at t={t:.1f}" if t is not None else "")
ax.set_title(ttl)
plt.tight_layout()
plt.savefig(a.out, dpi=150)
print(f"wrote {a.out}")
