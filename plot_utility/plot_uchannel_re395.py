"""Channel Re_tau 395 (2.5D LES on the A100) against Moser-Kim-Mansour 1999: profiles, rms, shear stress,
wall-stress history, and our instantaneous near-wall planes (the DNS has statistics only, no field).
    python plot_utility/plot_uchannel_re395.py results/uchan395/uchan395_96x160x128_wale_cpg"""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
base = _sys.argv[1] if len(_sys.argv) > 1 else "results/uchan395/uchan395_96x160x128_wale_cpg"
sf = base + "_stats.npz"; sf = sf if _os.path.exists(sf) else base + "_stats_partial.npz"; d = np.load(sf); print("statistics:", sf, "samples", int(d["nsamp"]), "t_end", float(d["t_end"]) if "t_end" in d.files else "-")
NU = 1 / 395; A = np.loadtxt("reference/mkm_chan395/chan395.means"); B = np.loadtxt("reference/mkm_chan395/chan395.reystress")
yd, Ud = A[:, 0], A[:, 2]; ypd = yd * 392.24; ud, vd, wd, uvd = np.sqrt(B[:, 2]), np.sqrt(B[:, 3]), np.sqrt(B[:, 4]), -B[:, 5]
yp, ut = d["yp"], float(d["ut"]); Up = d["U"] / ut; h = d["hist"]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
ax[0].semilogx(ypd[1:], Ud[1:], "k-", lw=2.2, label="MKM 1999 DNS, Re_tau 392"); ax[0].semilogx(yp, Up, "C0o-", ms=3, lw=1, label=f"2.5D LES 96x160x128 WALE: u_tau {ut:.3f}, Re_tau {ut/NU:.0f}")
yl = np.logspace(0.9, 2.55, 50); ax[0].semilogx(yl, np.log(yl) / 0.41 + 5.2, "0.5", ls="--", lw=0.8, label="log law 0.41 / 5.2"); ax[0].set(xlabel="$y^+$", ylabel="$U^+$", xlim=(0.8, 420)); ax[0].legend(fontsize=8); ax[0].set_title("mean velocity")
for arr, ls, nm in ((ud, "-", "u'"), (vd, "--", "v'"), (wd, ":", "w'")): ax[1].plot(ypd, arr, "k", ls=ls, lw=2, label=f"DNS {nm}")
for k, ls in (("urms", "-"), ("vrms", "--"), ("wrms", ":")): ax[1].plot(yp, d[k] / ut, "C0", ls=ls, lw=1.2)
ax[1].set(xlabel="$y^+$", ylabel="rms / $u_\\tau$", xlim=(0, 395)); ax[1].legend(fontsize=8); ax[1].set_title("u' (solid), v' (dashed), w' (dotted); blue = LES")
ax[2].plot(ypd, uvd, "k-", lw=2.2, label="DNS"); ax[2].plot(yp, -d["uv"] / ut**2, "C0o-", ms=3, lw=1, label="LES"); ax[2].set(xlabel="$y^+$", ylabel="$-\\langle u'v'\\rangle/u_\\tau^2$", xlim=(0, 395)); ax[2].legend(fontsize=8); ax[2].set_title("Reynolds shear stress")
ax[3].plot(h[:, 0], np.sqrt(h[:, 1]), "C0", lw=0.8, label="$u_\\tau$"); ax[3].plot(h[:, 0], h[:, 4], "C0--", lw=0.8, label="$\\langle\\nu_t\\rangle/\\nu$"); ax[3].axhline(1, color="k", lw=0.6); ax[3].set(xlabel="t", title="wall stress and eddy viscosity history"); ax[3].legend(fontsize=8)
plt.suptitle(f"Turbulent channel Re_tau 395: 2.5D unstructured LES (96x160 wall-clustered quads x 128 Fourier planes, WALE, minimal box Lx = pi, Lz = 0.34 pi) vs MKM 1999 DNS; statistics {int(d['nsamp'])} samples", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uchannel_re395_profiles.png", dpi=140); print("wrote figures/uchannel_re395_profiles.png")
sel = (yp > 30) & (yp < 120); dU = Up[sel] - np.interp(yp[sel], ypd, Ud); wm = h[:, 0] >= 10
print(f"RESULT Re_tau {ut/NU:.1f} ({(ut/NU/392.24-1)*100:+.1f}% vs DNS 392.2)  U+ log region 30<y+<120 vs DNS: mean {dU.mean():+.3f} max {np.abs(dU).max():.3f} u_tau ({dU.mean()/np.interp(60, ypd, Ud)*100:+.1f}%)")
for nm, ours, ref in (("u_rms+ peak", (d["urms"]/ut).max(), ud.max()), ("v_rms+ max", (d["vrms"]/ut).max(), vd.max()), ("w_rms+ max", (d["wrms"]/ut).max(), wd.max()), ("-<uv>+ max", (-d["uv"]/ut**2).max(), uvd.max())):
    print(f"   {nm:12s} LES {ours:.3f}  DNS {ref:.3f}  ({(ours/ref-1)*100:+.1f}%)")
print(f"   U_c+ LES {Up[-1]:.2f} DNS {Ud[-1]:.2f};  u' peak at y+ LES {yp[np.argmax(d['urms'])]:.1f} DNS {ypd[np.argmax(ud)]:.1f};  <nu_t>/nu over the window {h[wm, 4].mean():.3f}")
