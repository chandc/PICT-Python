"""Channel Re_tau 180 on the 2.5D solver against the in-house SEM DNS: mean profile, rms, Reynolds shear
stress, plus the u_tau history. Usage: python plot_utility/plot_uchannel_profiles.py [tag ...]"""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DNS = "results/fosls_chan180_stats_t5.2_30.npz"; NU = 1 / 180        # SEM FOSLS DNS run02, window t = 5.2..30
def load_dns(path):
    d = np.load(path); yd, nu, n = d["y"], float(d["nu"]), int(d["nsamp"]); p = d["sums"] / n
    p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]]); ym = yd[-1] - yd; o = np.argsort(ym); q = np.empty_like(p)
    for k in range(4): q[k] = np.interp(yd, ym[o], p[k][o])
    q[4] = -np.interp(yd, ym[o], p[4][o]); f = 0.5 * (p + q); h = yd <= 0.5 * yd[-1] + 1e-12
    ut = float(d["utau_series"][:, 1].mean()) if "utau_series" in d.files and d["utau_series"].ndim == 2 and d["utau_series"].shape[1] > 1 else float(np.sqrt(nu * abs((f[0][1] - f[0][0]) / (yd[1] - yd[0]))))
    return yd[h], f[0][h], np.sqrt(np.maximum(f[1][h], 0)), np.sqrt(np.maximum(f[2][h], 0)), np.sqrt(np.maximum(f[3][h], 0)), -f[4][h], ut
tags = _sys.argv[1:] or ["uchan_24x80x32_wale_cpg"]
yd, Ud, ud, vd, wd, uvd, utd = load_dns(DNS); ypd = yd * utd / NU
fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
ax[0].semilogx(ypd[1:], Ud[1:] / utd, "k-", lw=2.2, label=f"FOSLS DNS run02 (t 5.2..30), $u_\\tau$ {utd:.3f}"); yl = np.logspace(0.9, 2.3, 50); ax[0].semilogx(yl, np.log(yl) / 0.41 + 5.2, "0.5", lw=0.8, ls="--", label="log law 0.41 / 5.2"); ax[0].semilogx(yl[yl < 12], yl[yl < 12], "0.5", lw=0.8, ls=":")
for k, (arr, nm) in enumerate(((ud, "u'"), (vd, "v'"), (wd, "w'"))): ax[1].plot(ypd, arr / utd, "k-" if k == 0 else ("k--" if k == 1 else "k:"), lw=2, label=f"DNS {nm}")
ax[2].plot(ypd, uvd / utd ** 2, "k-", lw=2.2, label="DNS")
for c, tag in enumerate(tags):
    d = np.load(f"results/{tag}_stats.npz"); yp, ut = d["yp"], float(d["ut"]); col = f"C{c}"
    lab = f"{tag.replace('uchan_', '')}: $u_\\tau$ {ut:.3f}, Re$_\\tau$ {float(d['re_tau']):.0f}"
    ax[0].semilogx(yp, d["U"] / ut, col, marker="o", ms=3, lw=1, label=lab)
    for arr, ls in ((d["urms"], "-"), (d["vrms"], "--"), (d["wrms"], ":")): ax[1].plot(yp, arr / ut, col, ls=ls, lw=1.2)
    ax[1].plot([], [], col, label=lab)
    ax[2].plot(yp, -d["uv"] / ut ** 2, col, marker="o", ms=3, lw=1, label=lab)
    h = d["hist"]; ax[3].plot(h[:, 0], np.sqrt(h[:, 1]), col, lw=0.8, label=lab); ax[3].plot(h[:, 0], h[:, 4], col, lw=0.8, ls="--")
ax[0].set_xlabel("$y^+$"); ax[0].set_ylabel("$U^+$"); ax[0].set_xlim(0.8, 200); ax[0].legend(fontsize=8); ax[0].set_title("mean velocity")
ax[1].set_xlabel("$y^+$"); ax[1].set_ylabel("rms / $u_\\tau$"); ax[1].set_xlim(0, 180); ax[1].legend(fontsize=8); ax[1].set_title("u' (solid), v' (dashed), w' (dotted)")
ax[2].set_xlabel("$y^+$"); ax[2].set_ylabel("$-\\langle u'v'\\rangle / u_\\tau^2$"); ax[2].set_xlim(0, 180); ax[2].legend(fontsize=8); ax[2].set_title("Reynolds shear stress")
ax[3].axhline(1.0, color="k", lw=0.6); ax[3].set_xlabel("t"); ax[3].set_ylabel("$u_\\tau$ (solid), $\\langle\\nu_t\\rangle/\\nu$ (dashed)"); ax[3].set_title("wall stress and eddy viscosity history"); ax[3].legend(fontsize=8)
plt.suptitle(_os.environ.get("TITLE", "Turbulent channel Re$_\\tau$ 180, 2.5D unstructured solver: 24x80 wall-clustered quads x 32 Fourier modes, dt 0.002, DNS initial field, statistics t = 10..30"), fontsize=11)
plt.tight_layout(); out = _os.environ.get("OUT", "figures/uchannel_re180_profiles.png"); plt.savefig(out, dpi=140); print("wrote", out)
