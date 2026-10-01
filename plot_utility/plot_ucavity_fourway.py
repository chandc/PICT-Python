"""All four open-cavity runs together: Firedrake x UniFlow, medium x fine -- the full factorial that
separates method (FEM vs FV) from resolution. See reference/skew_unstructured_literature.md section 72."""
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
data = {}
for name, path, tmin, color in runs:
    d = np.loadtxt(path)
    w = d[:, 0] > tmin
    dt = d[1, 0] - d[0, 0]
    f, P = welch(d[w, 4] - d[w, 4].mean(), fs=1 / dt, nperseg=min(4096, int(w.sum())))
    m = (f > 0.3) & (f < 5); k = np.argmax(P[m]); tone = f[m][k]
    data[name] = dict(d=d, w=w, color=color, f=f, P=P, tone=tone,
                       KE=d[w,2].mean(), TKE=d[w,3].mean(), TKEstd=d[w,3].std(),
                       sensor_mean=d[w,4].mean(), sensor_std=d[w,4].std())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Open cavity, Re 7500: all four runs -- method (FEM vs FV) x resolution (medium vs fine)")

ax = axes[0,0]
for name, v in data.items(): ax.plot(v["d"][:,0], v["d"][:,2], label=name, color=v["color"])
ax.set_xlabel("t (own clock)"); ax.set_ylabel("KE"); ax.set_title("total kinetic energy"); ax.legend(fontsize=7.5)

ax = axes[0,1]
for name, v in data.items(): ax.semilogy(v["d"][:,0], v["d"][:,3], label=name, color=v["color"])
ax.set_xlabel("t"); ax.set_ylabel("TKE"); ax.set_title("turbulent KE (windows differ, see text)"); ax.legend(fontsize=7.5)

ax = axes[1,0]
for name, v in data.items():
    d, w = v["d"], v["w"]
    ax.plot(d[w,0], d[w,4], label=name, color=v["color"], alpha=0.85)
ax.set_xlim(15, 22); ax.set_xlabel("t"); ax.set_ylabel("sensor"); ax.set_title("sensor trace, t=15-22"); ax.legend(fontsize=7.5)

ax = axes[1,1]
for name, v in data.items():
    ax.semilogy(v["f"], v["P"], label=f"{name} (f0={v['tone']:.3f})", color=v["color"])
ax.set_xlim(0, 5); ax.set_xlabel("f"); ax.set_ylabel("PSD"); ax.set_title("sensor spectrum"); ax.legend(fontsize=7)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("figures/ucavity_fourway_comparison.png", dpi=140)
print("saved figures/ucavity_fourway_comparison.png")
for name, v in data.items():
    d, w = v["d"], v["w"]
    print(f"{name:32s} KE {v['KE']:.4f}  TKE {v['TKE']:.5f}+-{v['TKEstd']:.5f}  sensor mean {v['sensor_mean']:.3f} std {v['sensor_std']:.3f}  tone {v['tone']:.4f}  t={d[w,0][0]:.1f}-{d[w,0][-1]:.1f}")
