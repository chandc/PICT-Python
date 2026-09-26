"""Taylor-Green Re 1600 on the 2.5D solver (64^2 x 64 modes): dissipation history, no model vs WALE,
against the DNS peak (0.012299 at t = 8.93, Brachet et al. / van Rees et al., as recorded in
reference/les_findings.md). Three curves per run: the total -dE/dt (what the flow loses), the resolved
viscous dissipation eps_d with the scheme's own operators (what viscosity removes), and nu Z from the
least-squares gradient (under-resolves the small scales; shown to make the gap visible)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = (2 * np.pi) ** 3; nu = 1 / 1600
runs = [("no model", "results/tgv3d_re1600_n64_nz64_dt0.02.npz", "C0"), ("WALE", "results/tgv3d_re1600_n64_nz64_dt0.02_wale.npz", "C3")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for name, path, c in runs:
    if not _os.path.exists(path): continue
    d = np.load(path); t, E, Z, eps = d["t"], d["E"], d["Z"], d["eps"]; dEdt = -np.gradient(E, t) / V
    ax[0].plot(t, dEdt, c, lw=2, label=f"{name}: $-dE/dt$ (peak {dEdt.max():.5f} at t={t[np.argmax(dEdt)]:.1f})")
    ax[0].plot(t, eps / V, c, lw=1.2, ls="--", label=f"{name}: resolved $\\epsilon_d$ (peak {eps.max()/V:.5f})")
    ax[0].plot(t, nu * Z / V, c, lw=0.8, ls=":", label=f"{name}: $\\nu Z$, LSQ gradient")
    ax[1].plot(t, E / V, c, lw=2, label=name)
    if "nut_mean" in d.files and d["nut_mean"].max() > 0:
        a2 = ax[1].twinx(); a2.plot(t, d["nut_mean"], c, ls="--", lw=1); a2.set_ylabel(r"$\langle\nu_t\rangle/\nu$ (dashed)", color=c)
ax[0].axhline(0.012299, color="k", lw=0.8, ls="-."); ax[0].axvline(8.93, color="k", lw=0.8, ls="-."); ax[0].text(9.2, 0.0119, "DNS peak 0.012299\nat t = 8.93", fontsize=8)
ax[0].set_ylim(0, 0.021); ax[0].set_xlabel("t"); ax[0].set_ylabel(r"dissipation per unit volume"); ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_title("Re 1600, 64$^2$ quads x 64 Fourier modes, dt 0.02")
ax[1].set_xlabel("t"); ax[1].set_ylabel("E / V"); ax[1].legend(fontsize=8); ax[1].set_title("kinetic energy")
plt.tight_layout(); plt.savefig("figures/utgv1600_dissipation.png", dpi=140); print("wrote figures/utgv1600_dissipation.png")
