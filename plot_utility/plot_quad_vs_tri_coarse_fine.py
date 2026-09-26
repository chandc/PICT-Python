"""Butterfly quads vs HydroGym-generator triangles, coarse and fine, corrected-flux runs (results/t9/v3):
row 1 grid near the body, row 2 vertex-averaged vorticity with the mesh (near field), row 3 far field."""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient
REF = dict(St=0.1791, Cd=1.4862, Cl=0.3582)
runs = [("results/t9/v3/butterfly_re100.npz", "coarse butterfly (quads)"), ("results/t9/v3/hydrogym_tri_re100.npz", "coarse: HydroGym medium (tris)"),
        ("results/t9/v3/butterfly_fine_re100.npz", "fine butterfly (quads)"), ("results/t9/v3/hydrogym_tri_fine_re100.npz", "fine tris (HydroGym generator, n1=70)")]
def stats(H):
    t, cd, cl = H[:, 0], H[:, 1], H[:, 2]; n2 = int(0.6 * len(t)); clw = cl[n2:] - cl[n2:].mean()
    z = np.flatnonzero(np.diff(np.sign(clw)) > 0); tz = t[n2:][z] - clw[z] * (t[n2:][z + 1] - t[n2:][z]) / (clw[z + 1] - clw[z])
    return 1 / np.diff(tz).mean(), cd[n2:].mean(), 0.5 * (cl[n2:].max() - cl[n2:].min())
fig, ax = plt.subplots(3, 4, figsize=(30, 17), gridspec_kw={"height_ratios": [1, 1, 1.1]}); th = np.linspace(0, 2 * np.pi, 200)
for j, (f, nm) in enumerate(runs):
    d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
    bt = d["btag"][m.bfaces]; wall = m.bfaces[bt == 5]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ub = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vb = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    g = Gradient(m); wc = g(d["v"], vb)[:, 0] - g(d["u"], ub)[:, 1]
    N = d["nodes"]; nv = len(N); ws = np.zeros(nv); Wv = np.zeros(nv)
    for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
    Wv /= ws; polys = [N[c] for c in cells]; T = mtri.Triangulation(N[:, 0], N[:, 1]); tm = N[T.triangles].mean(axis=1); T.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
    St, Cd, Cl = stats(d["hist"]); hw = np.sqrt(m.vol[m.owner[wall]]).mean()
    head = f"{nm}\n{m.ncell} cells, {len(wall)} wall faces, wall cell {hw:.3f}"
    a = ax[0, j]; a.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=0.35)); a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3)
    a.set_xlim(-1.0, 1.6); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.set_title(head, fontsize=11)
    for i, (xl, yl, lev, lw) in enumerate([((-1.0, 3.0), (-1.5, 1.5), np.linspace(-6, 6, 25), 0.2), ((-2, 14), (-5, 5), np.linspace(-2, 2, 21), 0.06)]):
        a = ax[i + 1, j]; cf = a.tricontourf(T, np.clip(Wv, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, Wv, lev[::4], colors="k", linewidths=0.3, alpha=0.5)
        a.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=lw, alpha=0.5, zorder=2)); a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3)
        a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        a.set_title(("vorticity, near field, t=150" if i == 0 else f"far field   St {St:.4f} ({(St/REF['St']-1)*100:+.1f}%)  Cd {Cd:.4f} ({(Cd/REF['Cd']-1)*100:+.1f}%)  Cl amp {Cl:.3f} ({(Cl/REF['Cl']-1)*100:+.1f}%)"), fontsize=10)
        if j == 3: plt.colorbar(cf, ax=ax[i + 1, :].tolist(), shrink=0.7, pad=0.01)
plt.suptitle("Butterfly quads vs triangles, coarse and fine; corrected-flux runs, T=150; percentages vs HydroGym P2-P1 (St 0.1791, Cd 1.4862, Cl amp 0.358)", fontsize=13)
plt.savefig("figures/t9_quad_vs_tri_coarse_fine.png", dpi=95, bbox_inches="tight"); print("wrote figures/t9_quad_vs_tri_coarse_fine.png")
