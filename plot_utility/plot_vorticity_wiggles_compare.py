"""Cell-level ("raw") vorticity, where the two-colour speckle lives, for four solutions of the same
problem: HydroGym P2-P1 and P1-P1 (DG0 = per-cell vorticity of the FE velocity), ours on their mesh
(cell-centre gradient), ours on the wake-refined butterfly. Lower row: the same fields vertex-averaged
(CG1-projected for Firedrake), which is what published plots show. Free-stream rms (x < -1, r > 1.5)
in every title."""
import sys, os, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from src.umesh import Mesh
from src.uops import Gradient
def hg(f, nm):
    d = np.load(f); C = d["centroid"]; return dict(name=nm, C=C, w_cell=d["vort"], Xv=d["vert"], w_vert=d["vort_cg1"], polys=None)
def ours(f, nm):
    d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
    bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ubv = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    g = Gradient(m); wc = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
    nv = len(d["nodes"]); ws = np.zeros(nv); Wv = np.zeros(nv)
    for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
    return dict(name=nm, C=m.centroid, w_cell=wc, Xv=d["nodes"], w_vert=Wv / ws)
runs = [hg("results/hydrogym_cmp/medium_p2_fields.npz", "HydroGym Firedrake P2-P1 (Taylor-Hood), medium mesh")]
if os.path.exists("results/hydrogym_cmp/medium_p1_fields.npz"): runs.append(hg("results/hydrogym_cmp/medium_p1_fields.npz", "HydroGym Firedrake P1-P1 (equal order, no stabilisation), medium mesh"))
runs += [ours("results/t9/v2/hydrogym_tri_re100.npz", "ours (collocated FV), same medium mesh"), ours("results/t9/v2/butterfly_wake_re100.npz", "ours, wake-refined butterfly quads")]
def rms_free(X, w): r = np.hypot(X[:, 0], X[:, 1]); s = (X[:, 0] < -1) & (r > 1.5); return np.sqrt((w[s] ** 2).mean())
lev = np.linspace(-2, 2, 21); th = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(2, len(runs), figsize=(9 * len(runs), 9.5))
for j, r in enumerate(runs):
    for i, (X, w, kind) in enumerate([(r["C"], r["w_cell"], "cell-level vorticity"), (r["Xv"], r["w_vert"], "vertex-averaged / CG1")]):
        T = mtri.Triangulation(X[:, 0], X[:, 1]); tm = X[T.triangles].mean(axis=1); T.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
        a = ax[i, j]; cf = a.tricontourf(T, np.clip(w, -2, 2), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, w, [0.0], colors="k", linewidths=0.35, alpha=0.8)
        a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(-2, 15); a.set_ylim(-5, 5); a.set_aspect("equal")
        a.set_title(f"{r['name']}\n{kind}: free-stream rms {rms_free(X, w):.1e}", fontsize=9.5); plt.colorbar(cf, ax=a, shrink=0.8, pad=0.01)
        print(f"{r['name']:70s} {kind:26s} free-stream rms {rms_free(X, w):.2e}")
plt.suptitle("Where the wiggles live: raw cell-level vorticity (top) vs vertex-averaged (bottom); black line = zero contour. Re=100, HydroGym domain", fontsize=12)
plt.tight_layout(); plt.savefig("figures/vorticity_wiggles_compare.png", dpi=110); print("wrote figures/vorticity_wiggles_compare.png")
