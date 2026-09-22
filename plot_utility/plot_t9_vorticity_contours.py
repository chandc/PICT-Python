"""Near- and far-field vorticity contours at the final time for the results/t9/v2 runs."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from src.umesh import Mesh
from src.uops import Gradient
files = [("results/t9/v2/butterfly_re100.npz", "butterfly, 6992 quads"), ("results/t9/v2/butterfly_wake_re100.npz", "wake-refined butterfly, 13952 quads"),
         ("results/t9/v2/hydrogym_tri_re100.npz", "HydroGym medium mesh, 17258 tris")]
lev_near = np.linspace(-6, 6, 25); lev_far = np.linspace(-2, 2, 21)
fig, ax = plt.subplots(3, 2, figsize=(22, 13), gridspec_kw={"width_ratios": [1, 2.3]})
for i, (f, nm) in enumerate(files):
    d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]
    m = Mesh(d["nodes"], cells, span=1.0); g = Gradient(m); C = m.centroid
    bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ubv = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    w = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
    # triangulate the centroids; drop triangles that cross the cylinder so the body is not painted over
    tc = mtri.Triangulation(C[:, 0], C[:, 1]); tm = C[tc.triangles].mean(axis=1); tc.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
    th = np.linspace(0, 2 * np.pi, 200)
    for j, (lev, xl, yl, ttl) in enumerate([(lev_near, (-1.5, 3.0), (-1.5, 1.5), "near field"), (lev_far, (-2.0, 15.0), (-5.0, 5.0), "far field")]):
        a = ax[i, j]; cf = a.tricontourf(tc, np.clip(w, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both")
        a.tricontour(tc, w, lev[::2], colors="k", linewidths=0.3, alpha=0.6)
        a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        a.set_title(f"{nm}: vorticity, {ttl}, t=150  (St {1/np.diff(np.flatnonzero(np.diff(np.sign(d['hist'][9000:,2]-d['hist'][9000:,2].mean()))>0)).mean()/0.01:.4f})", fontsize=10)
        plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
plt.suptitle("Re=100 cylinder, HydroGym domain [-5,15]x[-5,5], slip lateral walls: vorticity contours at t=150 (levels +-6 near, +-2 far)", fontsize=12)
plt.tight_layout(); plt.savefig("figures/t9_vorticity_contours.png", dpi=110); print("wrote figures/t9_vorticity_contours.png")
