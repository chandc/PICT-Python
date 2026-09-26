"""Is the free-stream vorticity speckle on the triangle mesh in the field or in the post-processing?
Compare cell-gradient vorticity with vertex-averaged (CG1-like, what HydroGym plots) vorticity, and
measure the noise where the vorticity should be zero (upstream, x < -1, r > 1.5)."""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from src.umesh import Mesh
from src.uops import Gradient
def load(f):
    d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
    bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ubv = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    g = Gradient(m); w_cell = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
    # vertex-averaged velocity (area weights), then P1 curl per triangle of the vertex mesh -> back to cells
    nv = len(d["nodes"]); wsum = np.zeros(nv); U = np.zeros(nv); V = np.zeros(nv)
    for c, vol, uu, vv in zip(cells, m.vol, d["u"], d["v"]):
        wsum[c] += vol; U[c] += vol * uu; V[c] += vol * vv
    U /= wsum; V /= wsum
    # vertex-averaged vorticity: average the cell vorticity to vertices (this is what a CG1 projection does)
    Wv = np.zeros(nv)
    for c, vol, ww in zip(cells, m.vol, w_cell): Wv[c] += vol * ww
    Wv /= wsum
    w_vert_cell = np.array([Wv[c].mean() for c in cells])       # back at cells for a like-for-like rms
    return d, m, w_cell, Wv, w_vert_cell
rows = []; fig, ax = plt.subplots(2, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [1, 2.3]})
for i, (f, nm) in enumerate([("results/t9/v2/hydrogym_tri_re100.npz", "HydroGym tris, our solver"), ("results/t9/v2/butterfly_wake_re100.npz", "wake-refined butterfly")]):
    d, m, wc, Wv, wvc = load(f); C = m.centroid; r = np.hypot(C[:, 0], C[:, 1])
    up = (C[:, 0] < -1.0) & (r > 1.5); wk = (C[:, 0] > 3) & (C[:, 0] < 12) & (abs(C[:, 1]) < 2)
    print(f"{nm:28s} free-stream rms vorticity: cell-gradient {np.sqrt((wc[up]**2).mean()):.2e}  vertex-averaged {np.sqrt((wvc[up]**2).mean()):.2e}   | wake rms {np.sqrt((wc[wk]**2).mean()):.3f}  peak |w| {np.abs(wc).max():.1f}")
    if i == 0:
        tc = mtri.Triangulation(C[:, 0], C[:, 1]); tm = C[tc.triangles].mean(axis=1); tc.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
        tv = mtri.Triangulation(d["nodes"][:, 0], d["nodes"][:, 1], np.array([c for c in [d["cells"][k, :3] for k in range(len(d["nvert"]))]]))
        th = np.linspace(0, 2 * np.pi, 200)
        for j, (lev, xl, yl, ttl) in enumerate([(np.linspace(-6, 6, 25), (-1.5, 3), (-1.5, 1.5), "near"), (np.linspace(-2, 2, 21), (-2, 15), (-5, 5), "far")]):
            a = ax[0, j]; cf = a.tricontourf(tc, np.clip(wc, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(tc, wc, lev[::2], colors="k", linewidths=0.3, alpha=0.6)
            a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"{nm}: cell-gradient vorticity, {ttl} field", fontsize=10); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
            a = ax[1, j]; cf = a.tricontourf(tv, np.clip(Wv, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(tv, Wv, lev[::2], colors="k", linewidths=0.3, alpha=0.6)
            a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"{nm}: vertex-averaged vorticity (as a CG1 plot would show it), {ttl} field", fontsize=10); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
plt.tight_layout(); plt.savefig("figures/t9_tri_vorticity_cell_vs_vertex.png", dpi=110); print("wrote figures/t9_tri_vorticity_cell_vs_vertex.png")
