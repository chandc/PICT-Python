"""Usage: python plot_utility/plot_uadj_dpc.py [results/uadj_dpc] [tags...]
U8 step 1: DPC on the coarse-cylinder jets, horizon sweep. Training curves (window mean C_D against iteration and
against solver steps consumed, actuation), and the closed-loop evaluation over whole shedding periods (C_D, C_L,
jet amplitudes) against the uncontrolled wake, one colour per horizon."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = _sys.argv[1] if len(_sys.argv) > 1 else "results/uadj_dpc"
tags = _sys.argv[2:] or sorted({_os.path.basename(f)[:-len("_curve.npy")] for f in glob.glob(f"{d}/dpc_h*_curve.npy")}, key=lambda t: int(t.split("_h")[1].split("x")[0]))
fig, ax = plt.subplots(2, 3, figsize=(18, 8.5)); rows = []
for i, tag in enumerate(tags):
    c = np.load(f"{d}/{tag}_curve.npy"); H = int(tag.split("_h")[1].split("x")[0]); sub = int(tag.split("x")[1].split("_")[0]) if "x" in tag else 5
    it, ph, L, cd, cd0, act, gn, sec = c.T; rel = (cd / cd0 - 1) * 100; steps = np.cumsum(np.full_like(it, 2 * H * sub))     # forward + replayed backward, in solver steps
    kind = ("ZNMF" if "znmf" in tag else "free jets") + (", rolling" if "roll" in tag else ""); lab = f"H {H} x {sub}, {kind} ({H*sub*0.01:.1f} time units, {sec.mean():.0f} s/it)"
    ax[0, 0].plot(it, rel, f"C{i}", lw=1.2, label=lab); ax[0, 1].plot(steps, rel, f"C{i}", lw=1.2, label=lab); ax[0, 2].plot(it, act, f"C{i}", lw=1.2, label=lab)
    ev = f"{d}/{tag}_eval.npz"
    if _os.path.exists(ev):
        e = np.load(ev); cc, uu = e["controlled"], e["uncontrolled"]; tt = (cc[:, 0] + 1) * int(e["sub"]) * 0.01 / float(e["period"])
        ax[1, 0].plot(tt, cc[:, 1], f"C{i}", lw=1.2, label=f"H {H} {kind}: mean {cc[:, 1].mean():.4f} ({(cc[:, 1].mean()/uu[:, 1].mean()-1)*100:+.2f}%)")
        ax[1, 1].plot(tt, cc[:, 2], f"C{i}", lw=1.2, label=f"H {H} {kind}: rms {cc[:, 2].std():.4f}"); ax[1, 2].plot(tt, cc[:, 3], f"C{i}", lw=1.2, label=f"H {H} {kind} +90"); ax[1, 2].plot(tt, cc[:, 4], f"C{i}", lw=1.2, ls="--", label=f"H {H} {kind} -90")
        if i == 0: ax[1, 0].plot(tt, uu[:, 1], "k", lw=1.8, label=f"uncontrolled: mean {uu[:, 1].mean():.4f}"); ax[1, 1].plot(tt, uu[:, 2], "k", lw=1.8, label=f"uncontrolled: rms {uu[:, 2].std():.4f}")
        rows.append((H, sub, len(it), rel[-5:].mean(), cc[:, 1].mean(), uu[:, 1].mean(), cc[:, 2].std(), uu[:, 2].std(), np.abs(cc[:, 3:]).mean(), sec.sum() / 60))
ax[0, 0].set(xlabel="iteration", ylabel="window mean $C_D$ change (%)", title="training: window C_D vs the uncontrolled window"); ax[0, 0].axhline(0, color="k", lw=0.6); ax[0, 0].legend(fontsize=8)
ax[0, 1].set(xlabel="solver steps consumed (forward + replay)", ylabel="window mean $C_D$ change (%)", title="the same against solver steps: sample efficiency"); ax[0, 1].axhline(0, color="k", lw=0.6); ax[0, 1].legend(fontsize=8)
ax[0, 2].set(xlabel="iteration", ylabel="mean |a| / $U_\\infty$", title="actuation"); ax[0, 2].legend(fontsize=8)
ax[1, 0].set(xlabel="t / period", ylabel="$C_D$", title="closed loop from phase 0"); ax[1, 0].legend(fontsize=8); ax[1, 1].set(xlabel="t / period", ylabel="$C_L$", title="closed loop: lift"); ax[1, 1].legend(fontsize=8); ax[1, 2].set(xlabel="t / period", ylabel="jet amplitude / $U_\\infty$", title="closed loop: actions"); ax[1, 2].legend(fontsize=7)
plt.suptitle("DPC on the Re 100 coarse-butterfly cylinder with $\\pm$90$^\\circ$ jets through the unstructured replay adjoint (CPU float64, Spark): horizon sweep, loss = $C_D$ + 0.05 $\\overline{a^2}$", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uadj_dpc_cylinder_sweep.png", dpi=130); print("wrote figures/uadj_dpc_cylinder_sweep.png")
print(f"{'H':>4} {'sub':>3} {'iters':>5} {'train dCd% (last 5)':>20} {'eval Cd':>8} {'uncontrolled':>12} {'dCd%':>7} {'Cl rms':>7} {'unc.':>6} {'|a|':>6} {'min':>6}")
for r in rows: print(f"{r[0]:4d} {r[1]:3d} {r[2]:5d} {r[3]:20.2f} {r[4]:8.4f} {r[5]:12.4f} {(r[4]/r[5]-1)*100:7.2f} {r[6]:7.4f} {r[7]:6.4f} {r[8]:6.3f} {r[9]:6.1f}")
