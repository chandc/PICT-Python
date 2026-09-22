"""HydroGym's own final fields (tools/hydrogym_cmp --save-fields): DG0 (cell) vorticity vs their CG1
projection, free-stream noise, and near/far contours to set beside ours."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
d = np.load("results/hydrogym_cmp/medium_p2_fields.npz"); C = d["centroid"]; w0 = d["vort"]; Xv = d["vert"]; w1 = d["vort_cg1"]
r = np.hypot(C[:, 0], C[:, 1]); up = (C[:, 0] < -1) & (r > 1.5)
rv = np.hypot(Xv[:, 0], Xv[:, 1]); upv = (Xv[:, 0] < -1) & (rv > 1.5)
print(f"HydroGym P2-P1 t={float(d['t']):.0f}: cells {len(C)}, vertices {len(Xv)}; free-stream rms vorticity: DG0 (cell) {np.sqrt((w0[up]**2).mean()):.2e}   CG1 projection {np.sqrt((w1[upv]**2).mean()):.2e}   peak |w| {np.abs(w0).max():.1f}")
fig, ax = plt.subplots(2, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [1, 2.3]}); th = np.linspace(0, 2 * np.pi, 200)
tc = mtri.Triangulation(C[:, 0], C[:, 1]); tm = C[tc.triangles].mean(axis=1); tc.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
tv = mtri.Triangulation(Xv[:, 0], Xv[:, 1]); tmv = Xv[tv.triangles].mean(axis=1); tv.set_mask(np.hypot(tmv[:, 0], tmv[:, 1]) < 0.5)
for j, (lev, xl, yl, ttl) in enumerate([(np.linspace(-6, 6, 25), (-1.5, 3), (-1.5, 1.5), "near"), (np.linspace(-2, 2, 21), (-2, 15), (-5, 5), "far")]):
    for i, (T, w, nm) in enumerate([(tc, w0, "HydroGym P2-P1: cell (DG0) vorticity"), (tv, w1, "HydroGym P2-P1: CG1-projected vorticity (their plots)")]):
        a = ax[i, j]; cf = a.tricontourf(T, np.clip(w, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, w, lev[::2], colors="k", linewidths=0.3, alpha=0.6)
        a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_title(f"{nm}, {ttl} field, t={float(d['t']):.0f}", fontsize=10); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
plt.tight_layout(); plt.savefig("figures/hydrogym_vorticity_fields.png", dpi=110); print("wrote figures/hydrogym_vorticity_fields.png")
