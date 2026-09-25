"""V1 (LES plan), corrected reference: Taylor-Green at Re 800 on the 2.5D solver (64^2x64, 96^2x96,
128^2x128, no model and WALE) against the in-house spectral-element DNS `results/tgv_diag_re800_88.npz`
(11x11 elements order 8 x 88 planes; -dE/dt / 2 nu Omega = 1.0000 throughout). The full dissipation
history is compared, not only the peak. Prints the V1 table (peak time within 3%, value within 5%)."""
import os as _os, sys as _sys, glob; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = (2 * np.pi) ** 3
ref = np.load("results/tgv_diag_re800_88.npz"); tr, Er, Omr, nur = ref["t"], ref["E"], ref["Om"], float(ref["nu"])
eps_ref = 2 * nur * Omr / V; jr = int(np.argmax(eps_ref)); tref, pref = tr[jr], eps_ref[jr]
# smooth -dE/dt of the reference on a 0.1 grid for the same estimator as ours
tg = np.arange(0, tr[-1], 0.1); Eg = np.interp(tg, tr, Er); dEr = -np.gradient(Eg, tg) / V
print(f"reference DNS Re 800: 2 nu Omega peak {pref:.5f} at t = {tref:.2f}; -dE/dt (0.1 sampling) peak {dEr.max():.5f} at t = {tg[np.argmax(dEr)]:.2f}")
runs = []
for n in (64, 96, 128):
    for sgs, ls in (("none", "-"), ("wale", "--")):
        fs = glob.glob(f"results/tgv3d_re800_n{n}_nz{n}_dt0.02_{sgs}*.npz")
        if fs: runs.append((n, sgs, ls, np.load(fs[0])))
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].plot(tr, eps_ref, "k-", lw=2.2, label=f"SEM DNS 88^3: 2$\\nu\\Omega$, peak {pref:.5f} at t={tref:.2f}"); ax[1].plot(tr, Er / V, "k-", lw=2.2, label="SEM DNS")
print(f"{'run':16s} {'-dE/dt peak':>12s} {'t_peak':>7s} {'dvalue':>8s} {'dtime':>7s} {'model share':>12s} {'<nu_t>/nu':>10s} {'rms rel err of -dE/dt, t<12':>28s}")
for n, sgs, ls, d in runs:
    t, E, eps = d["t"], d["E"], d["eps"]; dEdt = -np.gradient(E, t) / V; j = int(np.argmax(dEdt)); c = f"C{[64, 96, 128].index(n)}"
    a, b, cc = dEdt[j - 1], dEdt[j], dEdt[j + 1]; tp = t[j] + 0.5 * (t[1] - t[0]) * (a - cc) / (a - 2 * b + cc)
    share = (dEdt[j] - eps[j] / V) / dEdt[j]; nut = float(d["nut_mean"][j]) if "nut_mean" in d.files else 0.0
    sel = t < 12; err = np.sqrt(np.mean((dEdt[sel] - np.interp(t[sel], tr, eps_ref)) ** 2)) / np.sqrt(np.mean(np.interp(t[sel], tr, eps_ref) ** 2))
    print(f"{n}^2x{n} {sgs:5s}  {dEdt[j]:12.5f} {tp:7.2f} {(dEdt[j]/pref-1)*100:+7.1f}% {(tp/tref-1)*100:+6.1f}% {share*100:11.1f}% {nut:10.3f} {err*100:27.1f}%")
    ax[0].plot(t, dEdt, c, ls=ls, lw=1.8 if sgs == "wale" else 1.2, label=f"{n}$^2\\times${n} {sgs}: peak {dEdt[j]:.5f} at t={tp:.2f}")
    ax[1].plot(t, E / V, c, ls=ls, lw=1.2, label=f"{n}$^2\\times${n} {sgs}")
    ax[2].plot(t, dEdt / np.interp(t, tr, eps_ref), c, ls=ls, lw=1.2, label=f"{n}$^2\\times${n} {sgs}")
ax[0].set_xlabel("t"); ax[0].set_ylabel("dissipation per unit volume"); ax[0].legend(fontsize=7.5); ax[0].set_title("Taylor-Green Re 800, 2.5D solver on the GB10, dt 0.02")
ax[1].set_xlabel("t"); ax[1].set_ylabel("E / V"); ax[1].legend(fontsize=8); ax[1].set_title("kinetic energy")
ax[2].axhline(1, color="k", lw=0.8); ax[2].set_ylim(0.5, 1.5); ax[2].set_xlabel("t"); ax[2].set_ylabel("ours / DNS"); ax[2].legend(fontsize=8); ax[2].set_title("dissipation ratio to the DNS")
plt.tight_layout(); plt.savefig("figures/utgv800_v1.png", dpi=140); print("wrote figures/utgv800_v1.png")
