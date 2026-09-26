"""T9: vorticity at the final time with the mesh drawn on top, for every results/t9/*_re100.npz.
Two rows per mesh: the wake (-2..12) and a near-body zoom. Cell-flat colouring on the actual
polygons, so what is shown is exactly the cell data on the cells that carry it."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient
files = sorted(glob.glob("results/t9/*_re100.npz"))
fig, ax = plt.subplots(2, len(files), figsize=(11 * len(files), 12), squeeze=False)
for j, f in enumerate(files):
    d = np.load(f); nm = f.split("/")[-1].replace("_re100.npz", "")
    cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]
    m = Mesh(d["nodes"], cells, span=1.0); g = Gradient(m)
    bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ubv = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    w = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
    polys = [d["nodes"][c] for c in cells]
    for i, (xl, yl, lw) in enumerate([((-2, 12), (-3, 3), 0.12), ((-1.5, 3.0), (-1.5, 1.5), 0.25)]):
        a = ax[i, j]
        pc = PolyCollection(polys, array=np.clip(w, -3, 3), cmap="RdBu_r", edgecolor="k", linewidth=lw)
        pc.set_clim(-3, 3); a.add_collection(pc)
        th = np.linspace(0, 2 * np.pi, 200); a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.3", zorder=3)
        a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        a.set_title(f"{nm} ({m.ncell} cells): vorticity at t={d['hist'][-1, 0]:.0f} with mesh", fontsize=11)
        plt.colorbar(pc, ax=a, shrink=0.8)
plt.tight_layout(); plt.savefig("figures/t9_vorticity_grid.png", dpi=110); print("wrote figures/t9_vorticity_grid.png")
