"""NACA0012 (Re=100, alpha=20) post-processing: force histories vs HydroGym's unperturbed means, St if
shedding, surface Cp and Cf along the chord, near/far vorticity with the mesh.
   python plot_utility/plot_naca_result.py results/naca/naca0012_a20_re100.npz"""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient
f = sys.argv[1] if len(sys.argv) > 1 else "results/naca/naca0012_a20_re100.npz"; alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
REF = {20.0: dict(Cd=0.577, Cl=0.783), 40.0: dict(Cd=1.081, Cl=1.027)}[alpha]   # HydroGym NACA0012Gust_2D_Re100_AOA{20,40} environment_config.yaml
d = np.load(f); H = d["hist"]; t, cd, cl, cdp = H.T; nu = 0.01; al = np.radians(alpha)
cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
bt = d["btag"][m.bfaces]; wall = m.bfaces[bt == 5]; wo = m.owner[wall]
n2 = int(0.6 * len(t)); w = slice(n2, None); clw = cl[w] - cl[w].mean()
z = np.flatnonzero(np.diff(np.sign(clw)) > 0); per = np.diff(t[w][z]) if len(z) > 2 else np.array([np.nan])
shedding = cl[w].std() > 1e-3 * abs(cl[w].mean()) and len(z) > 2
print(f"window t={t[n2]:.0f}..{t[-1]:.0f}: Cd {cd[w].mean():.4f} (HydroGym {REF['Cd']}, {(cd[w].mean()/REF['Cd']-1)*100:+.1f}%)  Cl {cl[w].mean():.4f} (HydroGym {REF['Cl']}, {(cl[w].mean()/REF['Cl']-1)*100:+.1f}%)  L/D {cl[w].mean()/cd[w].mean():.3f} (HydroGym {REF['Cl']/REF['Cd']:.3f})")
print(f"  Cl rms {cl[w].std():.2e}, Cd rms {cd[w].std():.2e}, Cd_p {cdp[w].mean():.4f}, Cd_v {cd[w].mean()-cdp[w].mean():.4f}; " + (f"shedding: {len(per)} periods, St_c = {1/per.mean():.4f}, St_h = {np.sin(al)/per.mean():.4f}" if shedding else "steady (no zero-crossings / negligible C_L fluctuation)"))
# surface distributions: Cp = 2 p (rho U^2 = 1), Cf = 2 tau_w, along chordwise coordinate s (0 = LE) on upper/lower
fc = m.fcentre[wall]; Sw = m.normal[wall]; Aw = np.hypot(*Sw.T); e_in = -Sw / Aw[:, None]; dn = ((fc - m.centroid[wo]) * (-e_in)).sum(1)
R = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]]); loc = fc @ R.T          # back to chord-aligned coords
tan = np.c_[-e_in[:, 1], e_in[:, 0]]
tan[(tan @ np.array([np.cos(al), -np.sin(al)])) < 0] *= -1                    # orient every tangent LE -> TE (chordwise), so C_f > 0 = attached forward flow
ut = (d["u"][wo] * tan[:, 0] + d["v"][wo] * tan[:, 1]); cf = 2 * nu * ut / dn
cp = 2 * d["p"][wo] - 2 * d["p"][(np.hypot(*m.centroid.T) > 7.5) & (m.centroid[:, 0] < 0)].mean()
up = loc[:, 1] > 0
fig, ax = plt.subplots(2, 3, figsize=(24, 12))
a = ax[0, 0]; a.plot(t, cd, label="C_D"); a.plot(t, cl, label="C_L"); a.axhline(REF["Cd"], color="C0", ls=":", label=f"HydroGym C_D {REF['Cd']}"); a.axhline(REF["Cl"], color="C1", ls=":", label=f"HydroGym C_L {REF['Cl']}")
a.set_ylim(0, 3); a.set_xlabel("t (chords)"); a.set_title(f"forces; last 40%: C_D {cd[w].mean():.3f}, C_L {cl[w].mean():.3f}"); a.legend(fontsize=8); a.grid(alpha=.3)
a = ax[0, 1]; a.plot(t[w], cl[w]); a.set_title(f"C_L, last 40%: rms {cl[w].std():.2e}" + (f", St_c {1/per.mean():.4f}" if shedding else " (steady)")); a.set_xlabel("t"); a.grid(alpha=.3)
a = ax[0, 2]; a.plot(loc[up, 0], cp[up], ".-", ms=3, label="upper (suction) side"); a.plot(loc[~up, 0], cp[~up], ".-", ms=3, label="lower side"); a.invert_yaxis(); a.set_xlabel("x/c"); a.set_title("C_p (referenced to the far field ahead)"); a.legend(fontsize=8); a.grid(alpha=.3)
a2 = a.twinx(); a2.plot(loc[up, 0], cf[up], "C2.", ms=2, label="C_f upper"); a2.plot(loc[~up, 0], cf[~up], "C3.", ms=2, label="C_f lower"); a2.axhline(0, color="k", lw=0.5); a2.set_ylabel("C_f"); a2.legend(fontsize=8, loc="lower right")
sep = loc[up][np.argsort(loc[up, 0])]; cfu = cf[up][np.argsort(loc[up, 0])]; zc = np.flatnonzero(np.diff(np.sign(cfu)) != 0)
print(f"  upper-surface C_f sign changes at x/c = {np.round(sep[zc, 0], 3)}  (separation / reattachment); Cp min {cp.min():.3f} at x/c {loc[np.argmin(cp), 0]:.3f}")
# vorticity
g = Gradient(m); uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
ub = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vb = np.where(np.isin(bt, [2, 5]), 0.0, vo)
wc = g(d["v"], vb)[:, 0] - g(d["u"], ub)[:, 1]; nv = len(d["nodes"]); ws = np.zeros(nv); Wv = np.zeros(nv)
for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
Wv /= ws; T = mtri.Triangulation(d["nodes"][:, 0], d["nodes"][:, 1]); tm = d["nodes"][T.triangles].mean(axis=1)
inside = np.array([False] * len(tm))   # mask triangles inside the airfoil: use the wall polygon
wallnodes = np.unique(np.concatenate([d["cells"][k, :d["nvert"][k]] for k in range(len(cells))]))
from matplotlib.path import Path
wn = d["nodes"][np.unique(np.concatenate([[m.face_nodes[i] for i in wall]] )) ] if hasattr(m, "face_nodes") else None
polys = [d["nodes"][c] for c in cells]
for a, (xl, yl, lev, ttl, lw) in zip(ax[1], [((-0.5, 2.5), (-1.3, 1.0), np.linspace(-8, 8, 33), "vorticity, near field", 0.15), ((-2, 14), (-5, 5), np.linspace(-2, 2, 21), "vorticity, far field", 0.05), ((-0.2, 1.4), (-0.8, 0.4), np.linspace(-8, 8, 33), "vorticity with mesh", 0.25)]):
    cf_ = a.tricontourf(T, np.clip(Wv, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, Wv, lev[::4], colors="k", linewidths=0.3, alpha=0.5)
    if "mesh" in ttl: a.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=lw, alpha=0.6))
    a.fill(*(d["nodes"][np.unique(np.concatenate([d["cells"][k, :d["nvert"][k]] for k in np.unique(wo)]))][:, :2].T if False else ([], [])), "0.3")
    a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"{ttl}, t={t[-1]:.0f}"); plt.colorbar(cf_, ax=a, shrink=0.8, pad=0.01)
plt.suptitle(f"NACA0012, alpha={alpha:.0f} deg, Re=100: {m.ncell} quads; reference HydroGym (m-AIA LBM) C_D {REF['Cd']}, C_L {REF['Cl']}", fontsize=12); plt.tight_layout()
out = f"figures/naca0012_a{int(alpha)}_result.png"; plt.savefig(out, dpi=110); print("wrote", out)
