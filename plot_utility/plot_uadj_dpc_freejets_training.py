"""Usage: python plot_utility/plot_uadj_dpc_freejets_training.py
Free-jets (unconstrained suction) cylinder benchmark: DPC's training/reward history against PPO's, on
the same actuator authority. DPC's jets are a physically different actuator than HydroGym's (slot vs.
radial, different peak-velocity scaling per unit action), so --amax is calibrated so DPC's converged
constant-suction ceiling matches (amax 0.8) and exceeds (amax 1.0, 1.5) PPO's -29.9% result -- see
uadj_dpc_cylinder.py's constant-suction sweep in the commit message. PPO's own learning curve (record
section 43, `tools/hydrogym_cmp/train_sb3_firedrake.py` on Firedrake's Cylinder, Re 100 medium mesh,
action in [-0.1, 0.1] on both jets, 5000-step episodes): episode 1 mean C_D 1.306 (-12.1%), episode 2
1.054 (-29.1%), converged reference rollout at constant suction -0.1: C_D 1.0416 (-29.9%)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = "results/uadj_dpc"; PPO_BASE_CD = 1.4863    # HydroGym's own uncontrolled baseline (record section 38/43); DPC uses its own per-window baseline (cd0), not this one
tags = [("dpc_freejets_amax0.8", "DPC, $|a|\\leq0.8$ (matched to PPO's ceiling)", "C0", -28.46),
        ("dpc_freejets_amax1.0", "DPC, $|a|\\leq1.0$", "C1", -33.43),
        ("dpc_freejets_amax1.5", "DPC, $|a|\\leq1.5$", "C2", -42.34)]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for tag, lab, c, eval_red in tags:
    f = f"{d}/{tag}_curve.npy"
    if not _os.path.exists(f): continue
    cv = np.load(f); it, ph, L, cd, cd0, act, gn, sec = cv.T
    red = (1 - cd / cd0) * 100                       # each iteration's own uncontrolled-window baseline (the phase it started from)
    steps = np.cumsum(np.full_like(it, 2 * 8 * 5))     # H8 x sub5, fwd+replay, per iteration
    ax[0].plot(it, red, c, lw=1.6, label=f"{lab}: closed-loop eval {eval_red:.1f}%")
    ax[1].plot(steps, red, c, lw=1.6, label=f"{lab}: closed-loop eval {eval_red:.1f}%")
    ax[0].scatter([it[-1] + 5], [eval_red], marker="*", s=180, color=c, edgecolor="k", zorder=6)
    ax[1].scatter([steps[-1]], [eval_red], marker="*", s=180, color=c, edgecolor="k", zorder=6)
# PPO reference (Firedrake, record section 43): episode returns -> mean C_D per episode, then the converged constant-suction rollout
ppo_steps = np.array([0, 5000, 10000, 100000]); ppo_cd = np.array([PPO_BASE_CD, 1.306, 1.054, 1.0316]); ppo_red = (1 - ppo_cd / PPO_BASE_CD) * 100   # last point: PPO's own trained-policy evaluation (record §43), not the open-loop rollout
for a_ in ax:
    a_.plot(ppo_steps, ppo_red, "ks--", lw=1.8, ms=7, label=f"PPO (Firedrake, record §43): episode 1/2, then converged {ppo_red[-1]:.1f}%")
    a_.axhline(ppo_red[-1], color="k", lw=0.6, ls=":")
    a_.axhline(0, color="k", lw=0.6)
ax[0].set(xlabel="DPC training iteration", ylabel="closed-loop drag reduction (%)", title="against DPC's own iteration count", xlim=(0, 60))
ax[1].set(xlabel="solver steps consumed", ylabel="closed-loop drag reduction (%)", title="against solver steps (both methods)", xscale="log", xlim=(50, 2e5))
ax[0].legend(fontsize=8, loc="lower right"); ax[1].legend(fontsize=8, loc="lower right")
plt.suptitle("Re 100 cylinder, free (unconstrained) jets: DPC training/reward history vs. PPO's, same actuator ceiling family\n"
             "DPC finds the same steady-suction optimum PPO does, and reaches it within one horizon's worth of gradients", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uadj_dpc_freejets_training.png", dpi=140); print("wrote figures/uadj_dpc_freejets_training.png")
for tag, lab, c, eval_red in tags:
    f = f"{d}/{tag}_curve.npy"
    if not _os.path.exists(f): continue
    cv = np.load(f); print(f"{tag}: final window C_D {cv[-1,3]:.4f} (baseline {cv[-1,4]:.4f}, last-window {(1-cv[-1,3]/cv[-1,4])*100:+.1f}%, closed-loop eval {eval_red:+.1f}%)")
print(f"PPO (record section 43): episode 1 {ppo_red[1]:+.1f}% ({ppo_steps[1]:.0f} steps), episode 2 {ppo_red[2]:+.1f}% ({ppo_steps[2]:.0f} steps), converged eval {ppo_red[3]:+.1f}% ({ppo_steps[3]:.0f} steps)")
