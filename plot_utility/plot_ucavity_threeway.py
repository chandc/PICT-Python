"""Three-way open-cavity comparison, Re 7500: Firedrake fine, Firedrake medium, UniFlow medium.
Firedrake on their own two meshes isolates the mesh-resolution effect on their method; Firedrake
medium vs UniFlow medium then isolates the method effect at matched resolution. See
reference/skew_unstructured_literature.md section 68."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

fd_f = np.loadtxt("results/hydrogym_cmp/cavity_re7500_fine/stats.dat")
fd_m = np.loadtxt("results/hydrogym_cmp/cavity_re7500_medium/stats.dat")
uf_m = np.loadtxt("results/ucavity_hg_medium/stats.dat")
runs = [("Firedrake, fine (225k cells)", fd_f, "C0"), ("Firedrake, medium (65k cells)", fd_m, "C2"), ("UniFlow, medium (65k cells)", uf_m, "C1")]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Open cavity, Re 7500: three-way comparison (separating mesh resolution from method)")

ax = axes[0, 0]
for name, d, c in runs: ax.plot(d[:, 0], d[:, 2], label=name, color=c)
ax.set_xlabel("t"); ax.set_ylabel("KE"); ax.set_title("total kinetic energy"); ax.legend(fontsize=8)

ax = axes[0, 1]
for name, d, c in runs: ax.semilogy(d[:, 0], d[:, 3], label=name, color=c)
ax.set_xlabel("t"); ax.set_ylabel("TKE"); ax.set_title("turbulent KE growth"); ax.legend(fontsize=8)

ax = axes[1, 0]
for name, d, c in runs:
    w = d[:, 0] > 15
    ax.plot(d[w, 0], d[w, 4], label=name, color=c, alpha=0.85)
ax.set_xlim(15, 22)
ax.set_xlabel("t"); ax.set_ylabel("trailing-edge sensor"); ax.set_title("sensor trace, t=15-22"); ax.legend(fontsize=8)

ax = axes[1, 1]
tones = {}
for name, d, c in runs:
    w = d[:, 0] > 15; dt = d[1, 0] - d[0, 0]
    f, P = welch(d[w, 4] - d[w, 4].mean(), fs=1 / dt, nperseg=min(4096, int(w.sum())))
    ax.semilogy(f, P, label=name, color=c)
    m = (f > 0.3) & (f < 5); k = np.argmax(P[m]); tones[name] = f[m][k]
ax.set_xlim(0, 5); ax.set_xlabel("f"); ax.set_ylabel("PSD"); ax.set_title("sensor spectrum"); ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/ucavity_threeway_comparison.png", dpi=140)
print("saved figures/ucavity_threeway_comparison.png")

for name, d, c in runs:
    w = d[:, 0] > 15
    print(f"{name:32s} KE {d[w,2].mean():.4f}  TKE {d[w,3].mean():.5f}+-{d[w,3].std():.5f}  sensor mean {d[w,4].mean():.3f} rms {d[w,4].std():.3f}  tone {tones[name]:.4f}")
