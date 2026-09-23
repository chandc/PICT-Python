"""Flow field snapshot for the NACA0012 run: vorticity (near/far), |u| with streamlines, pressure.
   python plot_utility/plot_naca_field.py results/naca/naca0012_a20_t10.npz"""
import sys, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.path import Path
from src.umesh import Mesh
from src.uops import Gradient
f = sys.argv[1]; d = np.load(f); H = d["hist"]; t_end = H[-1, 0]
cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0)
bt = d["btag"][m.bfaces]; wall = m.bfaces[bt == 5]
g = Gradient(m); uo = d["u"][m.owner[m.bfaces]]; vo = d["v"][m.owner[m.bfaces]]
ub = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vb = np.where(np.isin(bt, [2, 5]), 0.0, vo)
wc = g(d["v"], vb)[:, 0] - g(d["u"], ub)[:, 1]
N = d["nodes"]; nv = len(N); ws = np.zeros(nv); Wv = np.zeros(nv); Uv = np.zeros(nv); Vv = np.zeros(nv); Pv = np.zeros(nv)
for c, vol, ww, uu, vv, pp in zip(cells, m.vol, wc, d["u"], d["v"], d["p"]): ws[c] += vol; Wv[c] += vol * ww; Uv[c] += vol * uu; Vv[c] += vol * vv; Pv[c] += vol * pp
Wv /= ws; Uv /= ws; Vv /= ws; Pv /= ws
# airfoil polygon (ordered wall nodes) to mask the interior
al = np.radians(20.0); xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 400))); ys = 0.6 * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1036 * xs**4)
foil = np.vstack([np.c_[xs, ys], np.c_[xs[::-1], -ys[::-1]]]) @ np.array([[np.cos(al), np.sin(al)], [-np.sin(al), np.cos(al)]]).T
T = mtri.Triangulation(N[:, 0], N[:, 1]); tm = N[T.triangles].mean(axis=1); T.set_mask(Path(foil).contains_points(tm))
fig, ax = plt.subplots(2, 2, figsize=(22, 13))
def panel(a, F, lev, cmap, ttl, xl, yl, lines=True):
    cf = a.tricontourf(T, np.clip(F, lev[0], lev[-1]), lev, cmap=cmap, extend="both")
    if lines: a.tricontour(T, F, lev[::4], colors="k", linewidths=0.3, alpha=0.5)
    a.fill(foil[:, 0], foil[:, 1], "0.3", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(ttl); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
panel(ax[0, 0], Wv, np.linspace(-8, 8, 33), "RdBu_r", f"vorticity, near field, t = {t_end:.0f}", (-0.5, 3.0), (-1.5, 1.0))
panel(ax[0, 1], Wv, np.linspace(-2, 2, 21), "RdBu_r", f"vorticity, far field, t = {t_end:.0f}", (-2, 14), (-5, 5))
# |u| with streamlines on a regular grid interpolated from vertex velocities
a = ax[1, 0]; panel(a, np.hypot(Uv, Vv), np.linspace(0, 1.6, 33), "viridis", f"|u| and streamlines, t = {t_end:.0f}", (-0.6, 3.0), (-1.6, 1.0), lines=False)
xg, yg = np.meshgrid(np.linspace(-0.6, 3.0, 240), np.linspace(-1.6, 1.0, 180)); Ui = mtri.LinearTriInterpolator(T, Uv)(xg, yg); Vi = mtri.LinearTriInterpolator(T, Vv)(xg, yg)
a.streamplot(xg, yg, np.ma.filled(Ui, 0), np.ma.filled(Vi, 0), density=1.6, color="w", linewidth=0.5, arrowsize=0.6)
panel(ax[1, 1], 2 * (Pv - Pv[(N[:, 0] < -6)].mean()), np.linspace(-1.5, 1.5, 31), "RdBu_r", f"C_p = 2(p - p_inlet), t = {t_end:.0f}", (-0.6, 3.0), (-1.6, 1.0))
n2 = int(0.6 * len(H)); plt.suptitle(f"NACA0012, alpha = 20 deg, Re = 100, t = {t_end:.0f}: C_D {H[-1,1]:.3f}, C_L {H[-1,2]:.3f} (instantaneous); HydroGym means 0.577 / 0.783", fontsize=12)
out = f"figures/naca0012_a20_field_t{int(round(t_end))}.png"; plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)
