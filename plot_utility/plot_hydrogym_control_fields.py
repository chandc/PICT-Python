"""Near- and far-field vorticity of HydroGym's cylinder (Firedrake, medium mesh): natural shedding vs the
trained shipped-jet policy (constant suction a = -0.1 at both slots), plus the force histories of the
two rollouts.  python plot_hydrogym_control_fields.py [tag]  (tag: shipped | znmf)"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
tag = sys.argv[1] if len(sys.argv) > 1 else "shipped"
cases = {"shipped": [("results/hydrogym_rl/fields_shipped_uncontrolled.npz", "uncontrolled"), ("results/hydrogym_rl/fields_shipped_suction.npz", "controlled: PPO policy = constant suction a = -0.1 (both slots)")],
         "znmf": [("results/hydrogym_rl/fields_znmf_uncontrolled.npz", "uncontrolled"), ("results/hydrogym_rl/fields_znmf_policy.npz", "controlled: PPO policy, ZNMF opposed jets")]}[tag]
fig, ax = plt.subplots(3, 2, figsize=(22, 14), gridspec_kw={"height_ratios": [1, 1.6, 1]}); th = np.linspace(0, 2 * np.pi, 200)
for j, (f, nm) in enumerate(cases):
    d = np.load(f); X = d["vert"]; w = d["vort_cg1"]; H = d["hist"]
    T = mtri.Triangulation(X[:, 0], X[:, 1]); tm = X[T.triangles].mean(axis=1); T.set_mask(np.hypot(tm[:, 0], tm[:, 1]) < 0.5)
    for i, (lev, xl, yl, ttl) in enumerate([(np.linspace(-6, 6, 25), (-1.5, 3), (-1.5, 1.5), "near field"), (np.linspace(-2, 2, 21), (-2, 15), (-5, 5), "far field")]):
        a = ax[i, j]; cf = a.tricontourf(T, np.clip(w, lev[0], lev[-1]), lev, cmap="RdBu_r", extend="both"); a.tricontour(T, w, lev[::2], colors="k", linewidths=0.3, alpha=0.5)
        a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.35", zorder=3); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal")
        a.set_title(f"{nm}\nvorticity, {ttl}, t = {float(d['t']):.0f} after the checkpoint", fontsize=10); plt.colorbar(cf, ax=a, shrink=0.85, pad=0.01)
    a = ax[2, j]; a.plot(H[:, 0], H[:, 2], label="C_D"); a.plot(H[:, 0], H[:, 1], label="C_L"); a.plot(H[:, 0], H[:, 3] * 10, "k:", lw=1, label="action x10")
    n2 = int(0.4 * len(H)); a.set_title(f"forces: C_D {H[n2:,2].mean():.4f}, C_L rms {H[n2:,1].std():.4f} (t = {H[n2,0]:.0f}..{H[-1,0]:.0f})", fontsize=10); a.set_xlabel("t"); a.grid(alpha=.3); a.legend(fontsize=8); a.set_ylim(min(-1.2, np.percentile(H[:, 1], 1) * 1.1), max(1.7, np.percentile(H[:, 1], 99) * 1.1))
plt.suptitle(f"HydroGym Firedrake cylinder, Re=100, medium mesh: natural shedding vs trained control ({tag} jets)", fontsize=12); plt.tight_layout()
plt.savefig(f"figures/hydrogym_control_fields_{tag}.png", dpi=110); print(f"wrote figures/hydrogym_control_fields_{tag}.png")
