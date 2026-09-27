"""Usage: python plot_utility/plot_uadj_dpc_eval6.py
Six-period closed-loop evaluations from the uncontrolled limit cycle of the two-period ZNMF DPC policies on the
coarse Re 100 cylinder (U8 step 1): C_D, C_L and the jet amplitudes, against the uncontrolled wake."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = "results/uadj_dpc"
runs = [("dpc_h240x5_znmf_roll_wa05_wl1_eval6", "H 240, ZNMF, rolling, $w_a$ 0.5, $w_L$ 1 (lift-penalised)"), ("dpc_h240x5_znmf_roll_wa05_eval6", "H 240, ZNMF, rolling, $w_a$ 0.5"), ("dpc_h240x5_znmf_roll_eval6", "H 240, ZNMF, rolling, $w_a$ 0.05"), ("dpc_h40x5_znmf_roll_eval6", "H 40, ZNMF, rolling, $w_a$ 0.05")]
fig, ax = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True)
for i, (tag, lab) in enumerate(runs):
    e = np.load(f"{d}/{tag}_eval.npz"); cc, uu = e["controlled"], e["uncontrolled"]; tt = (cc[:, 0] + 1) * int(e["sub"]) * 0.01 / float(e["period"]); late = tt > 4
    if i == 0: ax[0].plot(tt, uu[:, 1], "k", lw=1.8, label=f"uncontrolled limit cycle: mean {uu[:, 1].mean():.4f}"); ax[1].plot(tt, uu[:, 2], "k", lw=1.8, label="uncontrolled")
    ax[0].plot(tt, cc[:, 1], f"C{i}", lw=1.3, label=f"{lab}: last two periods {cc[late, 1].mean():.4f} ({(cc[late, 1].mean()/uu[:, 1].mean()-1)*100:+.1f}%)")
    ax[1].plot(tt, cc[:, 2], f"C{i}", lw=1.3, label=f"{lab}: mean {cc[late, 2].mean():+.3f}, rms {cc[late, 2].std():.3f} (last two periods)")
    ax[2].plot(tt, cc[:, 3], f"C{i}", lw=1.3, label=f"{lab}: +90 jet (mean |a| {np.abs(cc[late, 3]).mean():.3f} late)")
    print(f"{tag:42s} last-2-period C_D {cc[late, 1].mean():.4f} ({(cc[late, 1].mean()/uu[:, 1].mean()-1)*100:+.1f}%)  C_L mean {cc[late, 2].mean():+.3f} rms {cc[late, 2].std():.4f} (unc {uu[:, 2].std():.4f})  |a| late {np.abs(cc[late, 3]).mean():.3f} max {np.abs(cc[:, 3]).max():.3f}")
ax[0].set(ylabel="$C_D$", title="closed loop from the uncontrolled limit cycle, six shedding periods"); ax[0].legend(fontsize=8); ax[1].set(ylabel="$C_L$"); ax[1].legend(fontsize=8); ax[1].axhline(0, color="k", lw=0.5)
ax[2].set(xlabel="t / period", ylabel="+90$^\\circ$ jet amplitude / $U_\\infty$ (the -90$^\\circ$ jet is its negative)"); ax[2].legend(fontsize=8); ax[2].axhline(0, color="k", lw=0.5)
plt.suptitle("ZNMF DPC policies on the Re 100 coarse-butterfly cylinder through the unstructured replay adjoint, evaluated from the limit cycle.\nThe lift-penalised controller keeps the wake centred and suppresses the shedding with a jet amplitude of ~0.03 once there.", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uadj_dpc_cylinder_eval6.png", dpi=130); print("wrote figures/uadj_dpc_cylinder_eval6.png")
