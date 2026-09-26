"""Side-by-side final fields: HydroGym Firedrake P2-P1 (their medium mesh) | ours on their mesh |
ours on the wake-refined butterfly. Vorticity is vertex-averaged for ours (HydroGym's is CG1-projected,
the same operation). Snapshots are at different times; each panel is phase-labelled from C_L, and a
snapshot half a cycle out of phase is mirrored in y (the wake is reflection-symmetric under a half
period, with vorticity changing sign)."""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient
def phase(t, cl):
    n2 = int(0.6 * len(t)); tw, clw = t[n2:], cl[n2:] - cl[n2:].mean()
    z = np.flatnonzero(np.diff(np.sign(clw)) > 0); tz = tw[z] - clw[z] * (tw[z + 1] - tw[z]) / (clw[z + 1] - clw[z])
    per = np.diff(tz).mean(); return ((tw[-1] - tz[-1]) / per) % 1.0, per
# --- HydroGym
h = np.load("results/hydrogym_cmp/medium_p2_fields.npz"); F = np.loadtxt("results/hydrogym_cmp/medium_p2b_forces.dat")
ph_h, per_h = phase(F[:, 0], F[:, 1])
HG = dict(name=f"HydroGym Firedrake P2-P1, medium mesh (17258 tris), t=200", X=h["vert"], W=h["vort_cg1"], C=h["centroid"], P=h["p"], U=np.hypot(h["uv"][:, 0], h["uv"][:, 1]), ph=ph_h)
def ours(f, nm):
    d = np.load(f); cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
    bt = d["btag"][m.bfaces]; uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
    ubv = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, vo)
    g = Gradient(m); wc = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
    nv = len(d["nodes"]); ws = np.zeros(nv); Wv = np.zeros(nv)
    for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
    Wv /= ws; hist = d["hist"]; ph, per = phase(hist[:, 0], hist[:, 2])
    return dict(name=f"{nm}, t={hist[-1,0]:.0f}", X=d["nodes"], W=Wv, C=m.centroid, P=d["p"], U=np.hypot(d["u"], d["v"]), ph=ph, polys=polys(f))
def polys(f):
    d = np.load(f); return [d["nodes"][d["cells"][k, :d["nvert"][k]]] for k in range(len(d["nvert"]))]
HG["polys"] = polys("results/t9/v2/hydrogym_tri_re100.npz")      # HydroGym's medium.msh is byte-identical to ours
runs = [HG, ours("results/t9/v2/hydrogym_tri_re100.npz", "ours, same mesh (17258 tris)"), ours("results/t9/v2/butterfly_wake_re100.npz", "ours, wake-refined butterfly (13952 quads)")]
# phase-align to HydroGym: mirror in y when the phase differs by ~half a cycle
for r in runs:
    dph = (r["ph"] - HG["ph"]) % 1.0; r["flip"] = 0.25 < dph < 0.75
    if r["flip"]: r["X"] = r["X"] * [1, -1]; r["C"] = r["C"] * [1, -1]; r["W"] = -r["W"]; r["polys"] = [q * [1, -1] for q in r["polys"]]
    print(f"{r['name']:60s} C_L phase {r['ph']:.2f} cycles  mirrored: {r['flip']}")
rows = [("vorticity, near field", "W", "X", np.linspace(-6, 6, 25), (-1.5, 3), (-1.5, 1.5)), ("vorticity, far field", "W", "X", np.linspace(-2, 2, 21), (-2, 15), (-5, 5)),
        ("pressure (outlet p = 0)", "P", "C", np.linspace(-0.6, 0.6, 25), (-2, 15), (-5, 5)), ("|u|", "U", "C", np.linspace(0, 1.4, 29), (-2, 15), (-5, 5))]
fig, ax = plt.subplots(len(rows), 3, figsize=(27, 4.2 * len(rows) + 1)); th = np.linspace(0, 2 * np.pi, 200)
for i, (ttl, key, xkey, lev, xl, yl) in enumerate(rows):
    for j, r in enumerate(runs):
        X = r[xkey]; T = mtri.Triangulation(X[:, 0], X[:, 1]); tm = X[T.triangles].mean(axis=1); T.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
        a = ax[i, j]; cmap = "RdBu_r" if key != "U" else "viridis"
        cf = a.tricontourf(T, np.clip(r[key], lev[0], lev[-1]), lev, cmap=cmap, extend="both")
        if key == "W": a.tricontour(T, r[key], lev[::2], colors="k", linewidths=0.3, alpha=0.5)
        a.add_collection(PolyCollection(r["polys"], facecolor="none", edgecolor="k", linewidth=0.25 if i == 0 else 0.07, alpha=0.7, zorder=2))
        a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        a.set_title(f"{r['name']}\n{ttl}" + ("  [mirrored in y for phase]" if r["flip"] and key == "W" and i == 0 else ""), fontsize=10); plt.colorbar(cf, ax=a, shrink=0.9, pad=0.01)
plt.suptitle("Re=100 cylinder, HydroGym domain and BCs: HydroGym's Firedrake solution vs our unstructured PISO (vorticity vertex-averaged / CG1 in all three)", fontsize=12)
plt.tight_layout(); plt.savefig("figures/hydrogym_vs_ours_fields_mesh.png", dpi=150); print("wrote figures/hydrogym_vs_ours_fields_mesh.png")
