"""Vorticity (cell-flat and vertex-averaged) with the mesh drawn, near and far field, for one run.
   python plot_utility/plot_field_with_mesh.py results/t9/v2/hydrogym_tri_re100.npz figures/out.png [title]"""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient
f, out = sys.argv[1], sys.argv[2]; title = sys.argv[3] if len(sys.argv) > 3 else f
d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
ub = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vb = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
g = Gradient(m); wc = g(d["v"], vb)[:, 0] - g(d["u"], ub)[:, 1]
N = d["nodes"]; nv = len(N); ws = np.zeros(nv); Wv = np.zeros(nv)
for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
Wv /= ws; polys = [N[c] for c in cells]; th = np.linspace(0, 2 * np.pi, 200); t_end = d["hist"][-1, 0]
T = mtri.Triangulation(N[:, 0], N[:, 1]); tm = N[T.triangles].mean(axis=1); T.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
fig, ax = plt.subplots(2, 2, figsize=(24, 13), gridspec_kw={"width_ratios": [1, 2.2]})
for j, (xl, yl, lev, lw) in enumerate([((-1.2, 2.5), (-1.4, 1.4), np.linspace(-6, 6, 25), 0.3), ((-2, 14), (-5, 5), np.linspace(-2, 2, 21), 0.08)]):
    a = ax[0, j]; pc = PolyCollection(polys, array=np.clip(wc, lev[0], lev[-1]), cmap="RdBu_r", edgecolor="k", linewidth=lw, alpha=0.95); pc.set_clim(lev[0], lev[-1]); a.add_collection(pc)
    a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"cell-level vorticity on the cells, mesh drawn ({'near' if j == 0 else 'far'} field)"); plt.colorbar(pc, ax=a, shrink=0.85, pad=0.01)
    a = ax[1, j]; cf = a.tricontourf(T, np.clip(Wv, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, Wv, lev[::2], colors="k", linewidths=0.3, alpha=0.5)
    a.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=lw, alpha=0.6, zorder=2))
    a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"vertex-averaged vorticity contours, mesh drawn ({'near' if j == 0 else 'far'} field)"); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
plt.suptitle(f"{title}  (t = {t_end:.0f}, {m.ncell} cells)", fontsize=12); plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)
