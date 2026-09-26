"""Fluidic pinball: HydroGym's Firedrake solution vs ours on the same mesh, near and far field.
Vorticity is computed for BOTH with our least-squares cell gradient on the shared mesh (their DG0 centroid
velocities matched to our cells by centroid), then vertex-averaged, so the estimator is identical.
    python plot_utility/plot_pinball_fields.py 30 results/hydrogym_cmp/pinball_re30_fields.npz results/pinball/re30_medium.npz [label_fd] [label_ours]"""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
from src.umesh import Mesh; from src.uops import Gradient
Re = sys_Re = float(_sys.argv[1]); fd_file, our_file = _sys.argv[2], _sys.argv[3]
lab_fd = _sys.argv[4] if len(_sys.argv) > 4 else "HydroGym Firedrake P2-P1"; lab_ours = _sys.argv[5] if len(_sys.argv) > 5 else "ours, same mesh"
d = np.load(our_file); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0); nodes = d["nodes"]
polys = [nodes[c] for c in cells]; g = Gradient(m); bt = d["btag"][m.bfaces]
def bvals(u, v):
    uo = u[m.owner[m.bfaces]]; vo = v[m.owner[m.bfaces]]
    ub = np.where(bt == 2, 1.0, np.where(np.isin(bt, [5, 6, 7]), 0.0, uo)); vb = np.where(np.isin(bt, [2, 3, 5, 6, 7]), 0.0, vo); return ub, vb
def vert_avg(wc):
    nv = len(nodes); ws = np.zeros(nv); Wv = np.zeros(nv)
    for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
    return Wv / ws
def fields(u, v, p, name):
    ub, vb = bvals(u, v); wc = g(v, vb)[:, 0] - g(u, ub)[:, 1]
    return dict(name=name, W=vert_avg(wc), P=p - p.mean(), U=np.hypot(u, v))
ours = fields(d["u"], d["v"], d["p"], f"{lab_ours} (t={d['hist'][-1,0]:.0f})")
f = np.load(fd_file); idx = cKDTree(f["centroid"]).query(m.centroid)[1]; dist = np.hypot(*(f["centroid"][idx] - m.centroid).T)
print(f"cell matching: max centroid distance {dist.max():.2e}, cells {m.ncell} vs {len(f['centroid'])}")
fdk = fields(f["uv"][idx, 0], f["uv"][idx, 1], f["p"][idx], f"{lab_fd} (t={float(f['t']):.0f})")
runs = [fdk, ours]
T = mtri.Triangulation(nodes[:, 0], nodes[:, 1]); tm = nodes[T.triangles].mean(axis=1)
mask = np.zeros(len(T.triangles), bool)
for cx, cy in ((0, 0), (1.299, 0.75), (1.299, -0.75)): mask |= np.hypot(tm[:, 0] - cx, tm[:, 1] - cy) < 0.5
T.set_mask(mask)
wl = 4.0 if Re < 50 else 6.0
rows = [("vorticity, near field", "W", np.linspace(-wl, wl, 25), (-1.5, 5), (-2.5, 2.5), True), ("vorticity, far field", "W", np.linspace(-wl / 3, wl / 3, 25), (-2, 18), (-5, 5), False),
        ("pressure (mean removed), near field", "P", np.linspace(-1.0, 1.0, 25), (-1.5, 5), (-2.5, 2.5), False), ("|u|, far field", "U", np.linspace(0, 1.4, 29), (-2, 18), (-5, 5), False)]
fig, ax = plt.subplots(len(rows), 2, figsize=(19, 4.6 * len(rows))); th = np.linspace(0, 2 * np.pi, 100)
for i, (ttl, key, lev, xl, yl, mesh) in enumerate(rows):
    for j, r in enumerate(runs):
        a = ax[i, j]; cmap = "RdBu_r" if key != "U" else "viridis"
        if key == "W": cf = a.tricontourf(T, np.clip(r["W"], lev[0], lev[-1]), lev, cmap=cmap, extend="both"); a.tricontour(T, r["W"], lev[::2], colors="k", linewidths=0.3, alpha=0.5)
        else: cf = a.tripcolor(mtri.Triangulation(m.centroid[:, 0], m.centroid[:, 1]), np.clip(r[key], lev[0], lev[-1]), shading="gouraud", cmap=cmap, vmin=lev[0], vmax=lev[-1])
        if mesh: a.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=0.15, alpha=0.6, zorder=2))
        for cx, cy in ((0, 0), (1.299, 0.75), (1.299, -0.75)): a.fill(cx + 0.5 * np.cos(th), cy + 0.5 * np.sin(th), "0.35", zorder=3)
        a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"{r['name']}\n{ttl}", fontsize=10); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
plt.suptitle(f"Fluidic pinball Re {Re:.0f}, HydroGym medium mesh ({m.ncell} triangles), same vorticity estimator for both", fontsize=12)
plt.tight_layout(); out = f"figures/pinball_re{Re:.0f}_fields.png"; plt.savefig(out, dpi=110); print("wrote", out)
for r in runs: print(f"  {r['name']:50s} max|w| {np.abs(r['W']).max():.2f}  p range {r['P'].min():+.3f}..{r['P'].max():+.3f}")
