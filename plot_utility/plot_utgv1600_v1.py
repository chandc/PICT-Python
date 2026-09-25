"""V1 (LES plan): Taylor-Green Re 1600 on the 2.5D solver, resolution matrix (64^2x64, 96^2x96, 128^2x128)
with and without WALE, dissipation histories against the DNS peak (0.012299 at t = 8.93, Brachet et al. /
van Rees et al., as recorded in reference/les_findings.md). Prints the peak table the V1 criterion is judged
on (time within 3%, value within 5%, with WALE; implicit run alongside)."""
import os as _os, sys as _sys, glob; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = (2 * np.pi) ** 3; nu = 1 / 1600; DNS_PEAK, DNS_T = 0.012299, 8.93
runs = []
for n in (64, 96, 128):
    for sgs, ls in (("none", "-"), ("wale", "--")):
        fs = glob.glob(f"results/tgv3d_re1600_n{n}_nz{n}_dt0.02_{sgs}*.npz")
        if fs: runs.append((n, sgs, ls, np.load(fs[0])))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
print(f"{'run':16s} {'-dE/dt peak':>12s} {'t_peak':>7s} {'dvalue':>8s} {'dtime':>7s} {'eps_d peak':>11s} {'model share':>12s} {'<nu_t>/nu':>10s}")
for k, (n, sgs, ls, d) in enumerate(runs):
    t, E, eps = d["t"], d["E"], d["eps"]; dEdt = -np.gradient(E, t) / V; j = int(np.argmax(dEdt)); c = f"C{[64, 96, 128].index(n)}"
    share = (dEdt[j] - eps[j] / V) / dEdt[j]; nut = float(d["nut_mean"][j]) if "nut_mean" in d.files else 0.0
    print(f"{n}^2x{n} {sgs:5s}  {dEdt[j]:12.5f} {t[j]:7.2f} {(dEdt[j]/DNS_PEAK-1)*100:+7.1f}% {(t[j]/DNS_T-1)*100:+6.1f}% {eps[j]/V:11.5f} {share*100:11.1f}% {nut:10.3f}")
    ax[0].plot(t, dEdt, c, ls=ls, lw=1.8 if sgs == "wale" else 1.2, label=f"{n}$^2\\times${n} {sgs}: peak {dEdt[j]:.5f} at t={t[j]:.1f}")
    ax[1].plot(t, E / V, c, ls=ls, lw=1.2, label=f"{n}$^2\\times${n} {sgs}")
ax[0].plot([DNS_T], [DNS_PEAK], "k*", ms=12, label="DNS peak 0.012299 at t = 8.93"); ax[0].axvline(DNS_T, color="k", lw=0.6, ls=":"); ax[0].axhline(DNS_PEAK, color="k", lw=0.6, ls=":")
ax[0].set_xlabel("t"); ax[0].set_ylabel("$-dE/dt$ per unit volume"); ax[0].legend(fontsize=7.5); ax[0].set_title("Taylor-Green Re 1600, 2.5D solver on the GB10, dt 0.02: total dissipation")
ax[1].set_xlabel("t"); ax[1].set_ylabel("E / V"); ax[1].legend(fontsize=8); ax[1].set_title("kinetic energy")
plt.tight_layout(); plt.savefig("figures/utgv1600_v1.png", dpi=140); print("wrote figures/utgv1600_v1.png")
