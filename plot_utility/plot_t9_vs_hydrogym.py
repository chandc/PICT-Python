"""Overlay HydroGym's own Firedrake force histories (tools/hydrogym_cmp, run on the Spark) with
ours (results/t9/*_re100.npz): saturated C_L and C_D cycles, and the St / C_D / C_L table."""
import glob, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def stats(t, cl, cd):
    n2 = int(0.6 * len(t)); tw, clw, cdw = t[n2:], cl[n2:] - cl[n2:].mean(), cd[n2:]
    z = np.flatnonzero(np.diff(np.sign(clw)) > 0); tz = tw[z] - clw[z] * (tw[z + 1] - tw[z]) / (clw[z + 1] - clw[z]); per = np.diff(tz)
    return dict(St=1 / per.mean(), Cd=cdw.mean(), Cl_amp=0.5 * (cl[n2:].max() - cl[n2:].min()), Cl_rms=np.sqrt((clw ** 2).mean()), Cd_amp=0.5 * (cdw.max() - cdw.min()), n=len(per), T=t[-1], tz=tz)
runs = []
for f in sorted(glob.glob("results/hydrogym_cmp/*_forces.dat")):
    d = np.loadtxt(f); runs.append((f"HydroGym {f.split('/')[-1].replace('_forces.dat','')}", d[:, 0], d[:, 1], d[:, 2]))
ours = [("results/t9/hydrogym_tri_re100.npz", "ours v1 (cell-grad shear), tris"), ("results/t9/butterfly_re100.npz", "ours v1, butterfly")]
ours += [(f, "ours v2 (wall-flux shear, lagged F^n), " + f.split("/")[-1].replace("_re100.npz", "")) for f in sorted(glob.glob("results/t9/v2/*_re100.npz"))]
ours += [(f, "ours v3 (2F^n-F^(n-1) flux), " + f.split("/")[-1].replace("_re100.npz", "")) for f in sorted(glob.glob("results/t9/v3/*_re100.npz"))]
for f, nm in ours:
    h = np.load(f)["hist"]; runs.append((nm, h[:, 0], h[:, 2], h[:, 1]))
fig, ax = plt.subplots(1, 3, figsize=(20, 5.5)); rows = []
print(f"{'run':60s} {'St':>7} {'Cd':>7} {'Cl_amp':>7} {'Cl_rms':>7} {'Cd_amp':>7} {'periods':>7}")
for nm, t, cl, cd in runs:
    s = stats(t, cl, cd); print(f"{nm:60s} {s['St']:7.4f} {s['Cd']:7.4f} {s['Cl_amp']:7.4f} {s['Cl_rms']:7.4f} {s['Cd_amp']:7.4f} {s['n']:7d}")
    # align on the last upward zero-crossing so the cycles overlay
    t0 = s["tz"][-2]; sel = (t >= t0 - 12) & (t <= t0 + 0.5)
    ax[0].plot(t[sel] - t0, cl[sel], lw=1.3, label=f"{nm}: St {s['St']:.4f}"); ax[1].plot(t[sel] - t0, cd[sel], lw=1.3, label=f"{nm}: Cd {s['Cd']:.4f}")
    ax[2].plot(t, cl, lw=0.6, label=nm)
ax[0].set_title("C_L, two saturated cycles aligned on an upward zero-crossing"); ax[1].set_title("C_D, same window"); ax[2].set_title("C_L full histories")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel("t")
plt.suptitle("Re=100 cylinder on HydroGym's domain (beta=0.10): HydroGym Firedrake vs our unstructured PISO"); plt.tight_layout()
plt.savefig("figures/t9_vs_hydrogym.png", dpi=120); print("wrote figures/t9_vs_hydrogym.png")
