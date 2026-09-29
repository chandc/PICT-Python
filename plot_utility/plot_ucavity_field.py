"""UniFlow's own open-cavity field at the end of run_ucavity_hg.py (results/ucavity_hg_medium/final.npz):
vorticity, pressure, |u| over the whole domain (x in [-1.2, 2.5], y in [-1, 0.5]), vertex-averaged from
the cell-centred solution the same way plot_hydrogym_vs_ours_fields.py does for the cylinder."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient, DIRICHLET, NEUMANN

d = np.load("results/ucavity_hg_medium/final.npz")
cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]
m = Mesh(d["nodes"], cells, span=1.0)

names = {2: "Inlet", 3: "Freestream", 4: "Outlet", 5: "Slip", 6: "Wall", 7: "Control", 8: "Sensor"}
inv = {v: k for k, v in names.items()}
bt = d["btag"][m.bfaces]
uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
ubv = np.where(bt == inv["Inlet"], 1.0, np.where(np.isin(bt, [inv["Wall"], inv["Control"], inv["Sensor"]]), 0.0, uo))
vbv = np.where(bt == inv["Outlet"], vo, 0.0)
g = Gradient(m)
wc = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]

nv = len(d["nodes"]); ws = np.zeros(nv); Wv = np.zeros(nv); Pv = np.zeros(nv); Uv = np.zeros(nv)
speed = np.hypot(d["u"], d["v"])
for c, vol, ww, pp, uu in zip(cells, m.vol, wc, d["p"], speed):
    ws[c] += vol; Wv[c] += vol * ww; Pv[c] += vol * pp; Uv[c] += vol * uu
Wv /= ws; Pv /= ws; Uv /= ws

X = d["nodes"]
T = mtri.Triangulation(X[:, 0], X[:, 1], triangles=np.array(cells))
polys = [X[c] for c in cells]

fig, axes = plt.subplots(3, 1, figsize=(11, 10.5))
rows = [("vorticity", Wv, np.linspace(-15, 15, 31), "RdBu_r"),
        ("pressure (outlet p = 0)", Pv, np.linspace(-1.0, 1.0, 25), "RdBu_r"),
        ("|u|", Uv, np.linspace(0, 1.3, 27), "viridis")]
for ax, (ttl, val, lev, cmap) in zip(axes, rows):
    cf = ax.tricontourf(T, np.clip(val, lev[0], lev[-1]), lev, cmap=cmap, extend="both")
    ax.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=0.05, alpha=0.4, zorder=2))
    ax.set_xlim(-1.2, 2.5); ax.set_ylim(-1.0, 0.5); ax.set_aspect("equal")
    ax.set_title(ttl); plt.colorbar(cf, ax=ax, shrink=0.9, pad=0.01)

fig.suptitle(f"UniFlow open cavity, Re 7500, medium mesh (65536 cells), t = {float(d['time']):.1f}", fontsize=12)
plt.tight_layout()
plt.savefig("figures/ucavity_hg_medium_field.png", dpi=150)
print("wrote figures/ucavity_hg_medium_field.png")
