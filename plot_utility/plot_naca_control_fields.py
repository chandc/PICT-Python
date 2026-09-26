"""NACA0012 AoA 40 gust episode: jets off (left) vs the trained policy (right), vorticity near the airfoil at
several instants of the same episode (same snapshot phase), with the lift histories on top.
    python plot_utility/plot_naca_control_fields.py results/naca/fields_eval.npz"""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.path import Path
from src.umesh import Mesh; from src.uops import Gradient
d = np.load(_sys.argv[1] if len(_sys.argv) > 1 else "results/naca/fields_eval.npz"); alpha = 40.0
cells = [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))]; m = Mesh(d["nodes"], cells, span=1.0); N = d["nodes"]; bt = d["btag"][m.bfaces]; g = Gradient(m)
al = np.radians(alpha); xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 400))); ys = 0.6 * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1036 * xs**4)
foil = np.vstack([np.c_[xs, ys], np.c_[xs[::-1], -ys[::-1]]]) @ np.array([[np.cos(al), np.sin(al)], [-np.sin(al), np.cos(al)]]).T
T = mtri.Triangulation(N[:, 0], N[:, 1]); tm = N[T.triangles].mean(axis=1); T.set_mask(Path(foil).contains_points(tm))
def vort_v(u, v):
    uo = u[m.owner[m.bfaces]]; vo = v[m.owner[m.bfaces]]; ub = np.where(bt == 2, 1.0, np.where(bt == 5, 0.0, uo)); vb = np.where(np.isin(bt, [2, 5]), 0.0, vo)
    wc = g(v, vb)[:, 0] - g(u, ub)[:, 1]; ws = np.zeros(len(N)); Wv = np.zeros(len(N))
    for c, vol, ww in zip(cells, m.vol, wc): ws[c] += vol; Wv[c] += vol * ww
    return Wv / ws
times = d["zero_t"]; pick = [i for i in range(len(times)) if times[i] > 0.5][:5]
fig = plt.figure(figsize=(16, 4.2 + 4.4 * len(pick))); gs = fig.add_gridspec(len(pick) + 1, 2, height_ratios=[1] + [1.6] * len(pick))
# histories
for j, (mode, col, lab) in enumerate((("zero", "k", "jets off"), ("policy", "C3", "learned policy"))):
    h = d[f"{mode}_hist"]; a = fig.add_subplot(gs[0, j]); a.plot(h[:, 0], h[:, 2], col, lw=1.4, label="$C_L$"); a.plot(h[:, 0], h[:, 1], col, lw=1.0, ls="--", label="$C_D$")
    a.axhline(float(d["cl0"]), color="0.6", lw=0.8); a.axhline(float(d["cd0"]), color="0.6", lw=0.8, ls="--"); a.axvspan(0, 28.9, color="0.92")
    for i in pick: a.axvline(times[i], color="C0", lw=0.6, ls=":")
    a.set(xlim=(0, 108), ylim=(0.3, 3.2), xlabel="t (c/U)"); a.set_title(f"{lab}: force history (grey: the gust; dotted: the snapshots below)", fontsize=9); a.legend(fontsize=8, loc="upper right")
lev = np.linspace(-15, 15, 31); xl, yl = (-0.4, 3.2), (-1.4, 1.0)
for r, i in enumerate(pick):
    for j, (mode, lab) in enumerate((("zero", "jets off"), ("policy", "learned policy"))):
        a = fig.add_subplot(gs[r + 1, j]); W = vort_v(d[f"{mode}_u"][i], d[f"{mode}_v"][i]); h = d[f"{mode}_hist"]; k = int(np.argmin(np.abs(h[:, 0] - times[i])))
        cf = a.tricontourf(T, np.clip(W, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, W, lev[::3], colors="k", linewidths=0.25, alpha=0.5)
        a.fill(foil[:, 0], foil[:, 1], "0.3", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        act = d[f"{mode}_a"][i]; a.set_title(f"{lab}, t = {times[i]:.1f}{'  (gust)' if times[i] < 28.9 else ''}:  $C_L$ {h[k,2]:.2f}  $C_D$ {h[k,1]:.2f}" + (f"   jets ({act[0]:+.2f}, {act[1]:+.2f}, {act[2]:+.2f})" if mode == "policy" else ""), fontsize=10)
        if j == 1: plt.colorbar(cf, ax=a, shrink=0.8, pad=0.01, label="vorticity")
plt.suptitle("NACA0012 Re 100, AoA 40, HydroGym gust task on our solver: inlet gust to 2 U for t < 28.9. Jets off vs the PPO policy (steady suction on the upper-surface jet)", fontsize=11, y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.985)); out = "figures/naca40_gust_control_fields.png"; plt.savefig(out, dpi=105); print("wrote", out)
