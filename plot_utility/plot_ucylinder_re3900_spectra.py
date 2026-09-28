"""Usage: python plot_utility/plot_ucylinder_re3900_spectra.py
The reviewer's two diagnostics (record section 65/65a) from run_ucylinder3900_probes.py's output:
  (A) spanwise two-point correlation R_uu(dz) at four probes, via Wiener-Khinchin on the time-mean
      spanwise power spectrum -- checks the turbulence is genuinely 3D and L_z = pi D is wide enough
      (R_uu should fall to ~0 well before L_z/2).
  (B) temporal power spectrum of u at the same probes -- checks for a -5/3 inertial range before the
      grid-cutoff frequency, the classic diagnostic (Parnaudeau et al. 2008 Figs. 6-7)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch

d = np.load("results/ucyl3900_probes/diag_probes.npz")
P = d["probe_xy"]; nprobe = len(P); nz = int(d["nz"]); Lz = float(d["Lz"]); nsamp = int(d["nsamp"])
t = d["t"]; dt_probe = float(np.median(np.diff(t)))
print(f"probes {P.tolist()}, {nsamp} time samples over t = {t.min():.1f}..{t.max():.1f} (dt ~ {dt_probe:.5f}, {(t.max()-t.min())/5.939:.1f} shedding periods at St 0.21)")

# ---- (A) spanwise correlation from the time-averaged power spectrum (Wiener-Khinchin)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
kz = d["kz"]; dz = np.arange(nz) * Lz / nz
for i in range(nprobe):
    Pk = d["power_sum"][i] / nsamp                                     # time-mean one-sided power per mode (k=0 already ~0, mean removed each step)
    Pfull = Pk.copy(); Pfull[1:-1] /= 2 if nz % 2 == 0 else 1           # undo the one-sided doubling for a correct inverse FFT
    R = np.fft.irfft(Pfull, n=nz) * nz; R = R / R[0]
    half = dz <= Lz / 2
    izero = np.flatnonzero(R[half] < 0)
    z0 = dz[half][izero[0]] if len(izero) else np.nan
    ax[0].plot(dz[half], R[half], "o-", ms=3, lw=1.2, label=f"({P[i,0]:.1f},{P[i,1]:.1f}): zero at $\\Delta z/D$={z0:.2f}")
    print(f"  probe ({P[i,0]:.1f},{P[i,1]:.1f}): R_uu first zero at dz/D = {z0:.3f}  (L_z/2 = {Lz/2:.3f}); R at L_z/2 = {R[half][-1]:+.3f}")
ax[0].axvline(Lz / 2, color="k", ls="--", lw=1, label="$L_z/2$")
ax[0].axhline(0, color="k", lw=0.5); ax[0].set(xlabel="$\\Delta z / D$", ylabel="$R_{uu}(\\Delta z)$", xlim=(0, Lz / 2), title=f"spanwise correlation ($t$ = {t.max()-t.min():.0f} $D/U$ average, Wiener-Khinchin)"); ax[0].legend(fontsize=8)

# ---- (B) temporal spectrum (Welch), -5/3 reference
fs = 1.0 / dt_probe
for i in range(nprobe):
    u = d[f"uvw_{i}"][:, 0]
    f_, Pxx = welch(u - u.mean(), fs=fs, nperseg=min(4096, len(u)))
    ax[1].loglog(f_[1:], Pxx[1:], lw=1, label=f"({P[i,0]:.1f},{P[i,1]:.1f})")
ax1 = ax[1]; fref = np.logspace(-0.3, 1.3, 20)
i0 = 0; u0 = d["uvw_0"][:, 0]; f0, Pxx0 = welch(u0 - u0.mean(), fs=fs, nperseg=min(4096, len(u0)))
iref = np.argmin(np.abs(f0 - 1.5)); Aref = Pxx0[iref] * f0[iref] ** (5 / 3)
ax1.loglog(fref, Aref * fref ** (-5 / 3), "k--", lw=1, label="$f^{-5/3}$")
ax1.set(xlabel="$f D/U$", ylabel="$E_{uu}(f)$", title=f"temporal spectrum of $u$ at each probe (Welch, {fs:.0f} Hz-equivalent sample rate)"); ax1.legend(fontsize=8)
plt.suptitle(f"Re 3900 cylinder: reviewer diagnostics A (spanwise correlation) and B (temporal spectrum), {(t.max()-t.min()):.0f} $D/U$ of probe data ($\\approx${(t.max()-t.min())/5.939:.0f} shedding periods)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_spectra_probes.png", dpi=140); print("wrote figures/ucylinder_re3900_spectra_probes.png")
