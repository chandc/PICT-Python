"""Compare UniFlow's own open-cavity run (run_ucavity_hg.py) against HydroGym's Firedrake reference
(tools/hydrogym_cmp/hg_cavity.py), same geometry/BCs/Re, on whatever meshes are available for each.
The first pass here is UniFlow-medium vs Firedrake-fine, an uncontrolled comparison (different mesh
AND method) -- see reference/skew_unstructured_literature.md for the caveat and the matched-mesh
follow-up (Firedrake-medium)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

fd = np.loadtxt("results/hydrogym_cmp/cavity_re7500_fine/stats.dat")
uf = np.loadtxt("results/ucavity_hg_medium/stats.dat")
wfd = fd[:, 0] > 15; wuf = uf[:, 0] > 15

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Open cavity, Re 7500: HydroGym Firedrake (fine, 1.02M dof) vs UniFlow (medium, 65k cells)")

ax = axes[0, 0]
ax.plot(fd[:, 0], fd[:, 2], label="Firedrake, fine")
ax.plot(uf[:, 0], uf[:, 2], label="UniFlow, medium")
ax.set_xlabel("t"); ax.set_ylabel("KE"); ax.set_title("total kinetic energy"); ax.legend()

ax = axes[0, 1]
ax.semilogy(fd[:, 0], fd[:, 3], label="Firedrake, fine")
ax.semilogy(uf[:, 0], uf[:, 3], label="UniFlow, medium")
ax.set_xlabel("t"); ax.set_ylabel("TKE (fluct. vs running/base mean)"); ax.set_title("turbulent KE growth"); ax.legend()

ax = axes[1, 0]
ax.plot(fd[wfd, 0], fd[wfd, 4], label="Firedrake, fine")
ax.plot(uf[wuf, 0], uf[wuf, 4], label="UniFlow, medium", alpha=0.8)
ax.set_xlim(15, 25)
ax.set_xlabel("t"); ax.set_ylabel("trailing-edge sensor"); ax.set_title("sensor trace, t=15-25"); ax.legend()

ax = axes[1, 1]
dt_fd = fd[1, 0] - fd[0, 0]; f_fd, P_fd = welch(fd[wfd, 4] - fd[wfd, 4].mean(), fs=1/dt_fd, nperseg=4096)
dt_uf = uf[1, 0] - uf[0, 0]; f_uf, P_uf = welch(uf[wuf, 4] - uf[wuf, 4].mean(), fs=1/dt_uf, nperseg=min(2048, int(wuf.sum())))
ax.semilogy(f_fd, P_fd, label="Firedrake, fine")
ax.semilogy(f_uf, P_uf, label="UniFlow, medium")
ax.set_xlim(0, 5); ax.set_xlabel("f"); ax.set_ylabel("PSD"); ax.set_title("sensor spectrum"); ax.legend()

fig.text(0.5, 0.005, "Caveat: different mesh resolution AND method (FEM Taylor-Hood vs FV PISO), window t=15-50\n"
         "for both -- TKE has not fully saturated over this window for either (see text). A Firedrake\n"
         "run on their own medium mesh is in progress to separate the resolution and method effects.",
         ha="center", fontsize=9, style="italic")
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig("figures/ucavity_vs_hydrogym_re7500.png", dpi=140)
print("saved figures/ucavity_vs_hydrogym_re7500.png")

for name, d, w in [("Firedrake fine", fd, wfd), ("UniFlow medium", uf, wuf)]:
    print(f"{name}: KE {d[w,2].mean():.4f}  TKE {d[w,3].mean():.5f} +- {d[w,3].std():.5f}  sensor mean {d[w,4].mean():.3f}  rms(fluct) {d[w,4].std():.3f}")
    dt = d[1, 0] - d[0, 0]
    f, P = welch(d[w, 4] - d[w, 4].mean(), fs=1 / dt, nperseg=min(4096, int(w.sum())))
    m = (f > 0.3) & (f < 5); k = np.argmax(P[m])
    print(f"  dominant tone f = {f[m][k]:.4f}")
