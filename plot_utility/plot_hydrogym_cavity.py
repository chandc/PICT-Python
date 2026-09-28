"""Usage: python plot_utility/plot_hydrogym_cavity.py
HydroGym's own Firedrake open-cavity benchmark (Cavity_2D_Re7500_fine_FD), run directly and
uncontrolled as a reference (tools/hydrogym_cmp/hg_cavity.py, record section 65d): kinetic energy and
fluctuation TKE reaching a saturated limit-cycle state, and the Rossiter-tone spectrum of the
trailing-edge wall-shear-stress sensor (Barbagallo et al. 2009's own probe)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

d = np.loadtxt("results/hydrogym_cmp/cavity_re7500_fine/stats.dat"); t, cfl, KE, TKE, sensor = d.T
sel = t > 15; ts, ss = t[sel], sensor[sel]; fs = 1.0 / np.median(np.diff(ts))
f, Pxx = welch(ss - ss.mean(), fs=fs, nperseg=4096, noverlap=3072)
peaks, _ = find_peaks(Pxx, prominence=Pxx.max() * 0.01); order = np.argsort(Pxx[peaks])[::-1][:5]
f0 = f[peaks[order[0]]]
print(f"dominant tone f = {f0:.4f}; harmonics/peaks found: " + ", ".join(f"{f[peaks[i]]:.3f} ({Pxx[peaks[i]]/Pxx[peaks[order[0]]]:.3f})" for i in order))
print(f"saturated KE {KE[-4000:].mean():.4f}, TKE {TKE[-4000:].mean():.5f} +- {TKE[-4000:].std():.5f}, sensor mean {ss[-4000:].mean():.3f} rms {ss[-4000:].std():.3f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(t, KE, lw=0.8); ax[0, 0].set(xlabel="t", ylabel="KE", title="total kinetic energy")
ax[0, 1].plot(t, TKE, lw=0.8); ax[0, 1].set(xlabel="t", ylabel="TKE (fluctuation vs base flow)", title="turbulent kinetic energy: growth and saturation", yscale="log")
ax[1, 0].plot(ts, ss, lw=0.4); ax[1, 0].set(xlabel="t", ylabel="wall-shear-stress sensor", title=f"trailing-edge sensor, t > 15 (mean {ss.mean():.2f}, rms {ss.std():.2f})"); ax[1, 0].set_xlim(15, 20)
ax[1, 1].semilogy(f, Pxx, lw=1); [ax[1, 1].axvline(f[peaks[i]], color="C1", lw=0.7, ls="--") for i in order]
ax[1, 1].annotate(f"$f_0$ = {f0:.3f}", (f0, Pxx[peaks[order[0]]]), xytext=(5, 5), textcoords="offset points", fontsize=9)
ax[1, 1].set(xlabel="f", ylabel="PSD", title="Rossiter-tone spectrum (trailing-edge sensor)", xlim=(0, 10))
plt.suptitle(f"HydroGym Firedrake open cavity, Re 7500, fine mesh (1.02M dof), uncontrolled: {t.max():.0f} time units, dt 2.5e-4, 16 MPI ranks", fontsize=11)
plt.tight_layout(); plt.savefig("figures/hydrogym_cavity_re7500_reference.png", dpi=140); print("wrote figures/hydrogym_cavity_re7500_reference.png")
