"""Standalone, full-size FFT spectrum of the trailing-edge sensor, all four open-cavity runs
(Firedrake x UniFlow, medium x fine). See skew_unstructured_literature.md section 72."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

runs = [
    ("Firedrake, medium (65k cells)",  "results/hydrogym_cmp/cavity_re7500_medium/stats.dat", 15, "C2"),
    ("Firedrake, fine (225k cells)",   "results/hydrogym_cmp/cavity_re7500_fine/stats.dat",   15, "C0"),
    ("UniFlow, medium (65k cells)",    "results/ucavity_hg_medium/stats.dat",                 15, "C1"),
    ("UniFlow, fine (225k cells)",     "results/ucavity_hg_fine/stats.dat",                    8, "C3"),
]

fig, ax = plt.subplots(figsize=(12, 7))
for name, path, tmin, color in runs:
    d = np.loadtxt(path)
    w = d[:, 0] > tmin
    dt = d[1, 0] - d[0, 0]
    f, P = welch(d[w, 4] - d[w, 4].mean(), fs=1 / dt, nperseg=min(4096, int(w.sum())))
    m = (f > 0.3) & (f < 5); k = np.argmax(P[m]); tone = f[m][k]
    ax.semilogy(f, P, label=f"{name}  (f0 = {tone:.3f})", color=color, lw=1.3)
    ax.axvline(tone, color=color, ls=":", lw=1, alpha=0.6)

ax.set_xlim(0, 5); ax.set_ylim(1e-6, 1e3)
ax.set_xlabel("frequency f"); ax.set_ylabel("PSD (Welch)")
ax.set_title("Open cavity, Re 7500: trailing-edge sensor spectrum, all four runs")
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("figures/ucavity_sensor_spectrum_fourway.png", dpi=150)
print("wrote figures/ucavity_sensor_spectrum_fourway.png")
