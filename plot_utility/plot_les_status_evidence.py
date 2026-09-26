"""Evidence figures for reference/unstructured_les_status.md. Numbers that were measured earlier and
recorded in skew_unstructured_literature.md are plotted from the record (section cited in each title);
the cylinder histories are read from the result files."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_os.makedirs("figures/les_status", exist_ok=True)

# ---- 1. energy loss per turnover: BDF2-PISO vs RK3 vs RK3 with Rhie-Chow off; the Choi-term floor (sections 49, 50)
dt = np.array([0.01, 0.005, 0.0025])
bdf2_adv = [1.483e-1, 3.869e-2, 1.042e-2]; rk3_adv = [1.218e-3, 4.131e-4, 1.829e-4]; rk3_rcoff = [5.394e-4, 6.764e-5, 8.476e-6]
bdf2_st = [6.158e-2, 1.694e-2, 5.095e-3]; rk3_st = [6.738e-4, 3.449e-4, 1.744e-4]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].loglog(dt, bdf2_adv, "o-", label="BDF2-PISO, advected TGV"); ax[0].loglog(dt, bdf2_st, "o--", color="C0", alpha=0.6, label="BDF2-PISO, steady TGV")
ax[0].loglog(dt, rk3_adv, "s-", color="C3", label="RK3 per-stage projection, advected"); ax[0].loglog(dt, rk3_st, "s--", color="C3", alpha=0.6, label="RK3, steady")
ax[0].loglog(dt, rk3_rcoff, "^-", color="C2", label="RK3, Rhie-Chow off (integrator alone)")
ax[0].loglog(dt, 2e-3 * (dt / 0.005) ** 3, "k:", lw=0.8, label="$\\propto dt^3$"); ax[0].axhline(2e-3, color="0.5", lw=0.8, ls="--"); ax[0].text(0.0026, 2.3e-3, "G1 criterion 0.2%/turnover", fontsize=8, color="0.4")
ax[0].set_xlabel("dt"); ax[0].set_ylabel("energy loss per turnover"); ax[0].set_title("Inviscid Taylor-Green 64$^2$ quads, T = 0.1  [§50]"); ax[0].legend(fontsize=7.5); ax[0].invert_xaxis()
choi = {"Choi term, predictor pair": np.nan, "Choi term, matched pair,\ncurrent stage dt": 1.7e-2, "Choi term, matched pair,\nprevious stage dt": 1.2e-2, "no transient term\n(adopted)": 1.34e-3, "full-step matched pair": 1.50e-3}
names = list(choi)[1:]; vals = [choi[k] for k in names]
ax[1].bar(range(len(names)), vals, color=["C1", "C1", "C3", "C0"]); ax[1].set_yscale("log"); ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([n.replace("Choi term, matched pair,\n", "Choi term,\n") for n in names], fontsize=8)
ax[1].set_ylabel("energy loss per turnover, dt 0.005"); ax[1].set_title("Rhie-Chow inside the RK3 stages, advected TGV 64$^2$  [§50]\n(the predictor-pair variant was unstable)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/les_status/energy_loss_rk3_vs_bdf2.png", dpi=130)

# ---- 2. temporal order (T4) from section 50
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
dts = np.array([0.04, 0.02, 0.01, 0.005, 0.0025])
ax[0].loglog(dts, [9.02e-3, 2.42e-3, 6.78e-4, 2.17e-4, 8.07e-5], "o-", label="BDF2-PISO (order 1.9 -> 1.4)")
ax[0].loglog(dts, [1.32e-4, 8.29e-5, 4.67e-5, 2.38e-5, 1.07e-5], "s-", color="C3", label="RK3, Rhie-Chow on (order ~1: the O(dt) damping)")
ax[0].loglog(dts, [1.09e-5, 2.71e-6, 6.74e-7, 1.66e-7, 3.96e-8], "^-", color="C2", label="RK3, Rhie-Chow off (order 2.0: Crank-Nicolson diffusion)")
ax[0].set_xlabel("dt"); ax[0].set_ylabel("error vs same scheme at dt/4"); ax[0].set_title("T4 decaying Taylor-Green, $\\nu$ 0.01, 64$^2$  [§50]"); ax[0].legend(fontsize=7.5, loc="lower left"); ax[0].invert_xaxis()
dts2 = np.array([0.02, 0.01, 0.005, 0.0025])
ax[1].loglog(dts2, [8.54e-2, 1.09e-2, 2.74e-3, 6.56e-4], "o-", label="BDF2-PISO (order 2.0)")
ax[1].loglog(dts2, [5.40e-2, 1.50e-4, 5.60e-5, 2.29e-5], "s-", color="C3", label="RK3, Rhie-Chow on")
ax[1].loglog(dts2, [5.40e-2, 6.78e-5, 8.70e-6, 1.17e-6], "^-", color="C2", label="RK3, Rhie-Chow off (order 2.9-3.0)")
ax[1].set_xlabel("dt"); ax[1].set_ylabel("error vs same scheme at dt/4"); ax[1].set_title("convective order: inviscid advected TGV, 64$^2$  [§50]"); ax[1].legend(fontsize=7.5); ax[1].invert_xaxis()
plt.tight_layout(); plt.savefig("figures/les_status/temporal_order_t4.png", dpi=130)

# ---- 3. periodic-span cylinder = 2D (section 51), from the result files
d3 = np.load("results/t9/rk3/butterfly_25d_nz4.npz"); d2 = np.load("results/t9/rk3/butterfly_re100_rk3.npz"); h3, h2 = d3["hist"], d2["hist"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(h2[:, 0], h2[:, 2], "k-", lw=2, label="2D RK3"); ax[0].plot(h3[:, 0], h3[:, 2], "C3--", lw=1, label="2.5D, 4 planes, 3D seed"); ax[0].set(xlim=(120, 150), xlabel="t", ylabel="$C_L$", title="lift history, last 30 D/U  [§51]"); ax[0].legend()
ax[1].semilogy(h2[:, 0], np.abs(h3[:, 2] - h2[:, 2]) + 1e-18, "C0", lw=0.8); ax[1].set(xlabel="t", ylabel="$|C_L^{2.5D} - C_L^{2D}|$", title="difference of the two lift histories")
ax[2].semilogy(h3[:, 0], h3[:, 4] + 1e-30, "C2", lw=1); ax[2].set(xlabel="t", ylabel="spanwise-mode energy", title="3D energy: decays as $e^{-0.31 t}$ (Re 100 is below mode-A onset)")
plt.tight_layout(); plt.savefig("figures/les_status/cylinder_25d_vs_2d.png", dpi=130)

# ---- 4. G4: iterations and step cost (section 57)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
N = [1920, 4096, 15360, 61440, 245760]
ax[0].semilogx(N, [8, 7, 13, 17, 18], "o-", label="Ruge-Stuben + block Jacobi (adopted)"); ax[0].semilogx([15360], [55], "s", color="C1", ms=9, label="smoothed aggregation + block Jacobi"); ax[0].semilogx([15360], [35], "^", color="C1", ms=9, label="smoothed aggregation, pyamg Gauss-Seidel")
ax[0].axhline(30, color="0.5", ls="--", lw=0.8); ax[0].text(2200, 31.5, "G4 criterion < 30", fontsize=8, color="0.4"); ax[0].set(xlabel="cells in the plane", ylabel="PCG iterations to 1e-8", title="pressure family, 32 modes, channel operator  [§57]"); ax[0].legend(fontsize=7.5)
sizes = ["64$^2$x32", "128$^2$x64", "256$^2$x64", "384$^2$x64"]; x = np.arange(4)
before = [77, 480, 2720, 6762]; after = [47, 340, 1428, 3275]; cpu = [397, np.nan, np.nan, np.nan]
ax[1].bar(x - 0.2, before, 0.4, label="GB10, first CuPy port"); ax[1].bar(x + 0.2, after, 0.4, color="C3", label="GB10, fused kernels + block solve"); ax[1].plot([0], [397], "kD", label="one CPU core (LU)")
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(sizes); ax[1].set_ylabel("ms per step"); ax[1].set_title("whole step, 3D Taylor-Green  [§57]"); ax[1].legend(fontsize=8)
rate_after = [67, 63, 66, 67]; rate_before = [116, 89, 139, 139]
ax[2].plot(x, rate_before, "o--", color="C0", label="first CuPy port"); ax[2].plot(x, rate_after, "s-", color="C3", label="after fusions"); ax[2].plot([3.3], [140], "kx", ms=10, label="real mesh (fine butterfly, WALE)")
ax[2].axhline(100, color="0.5", ls="--", lw=0.8); ax[2].text(0, 103, "G4 criterion 100 ms per 1e5 cell-modes", fontsize=8, color="0.4"); ax[2].set_xticks(x); ax[2].set_xticklabels(sizes); ax[2].set_ylabel("ms per 1e5 cell-modes"); ax[2].set_ylim(0, 160); ax[2].set_title("normalised step cost on the GB10  [§57]"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/les_status/g4_solver_and_cost.png", dpi=130)

# ---- 5. Taylor-Green peak-time variants (section 58)
fig, ax = plt.subplots(figsize=(9, 4.2))
lab = ["32$^2$x32", "64$^2$x64\n(baseline)", "96$^2$x96", "128$^2$x128", "48$^2$x96", "96$^2$x48", "64$^2$x64\nrotated IC", "64$^2$x64\ndt 0.01", "64$^2$x64\ndt 0.04"]
tp = [8.90, 8.39, 8.25, 8.42, 8.12, 8.34, 7.50, 8.47, 8.38]; col = ["C0"] * 4 + ["C2", "C2", "C3", "C1", "C1"]
ax.axhspan(8.93 * 0.97, 8.93 * 1.03, color="0.9", zorder=0); ax.bar(range(len(lab)), tp, color=col, zorder=2); ax.axhline(8.93, color="k", lw=1.2, ls="--", zorder=3); ax.text(0.1, 8.96, "SEM DNS peak t = 8.93 (grey: the 3% window)", fontsize=8); ax.set_ylim(7, 9.3)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=7.5); ax.set_ylabel("time of the dissipation peak"); ax.set_title("Taylor-Green Re 800, no model: peak time vs resolution (blue), anisotropy (green), orientation (red), dt (orange)  [§58]", fontsize=9.5)
plt.tight_layout(); plt.savefig("figures/les_status/tgv_peak_time_variants.png", dpi=130)
print("wrote figures/les_status/*.png")
