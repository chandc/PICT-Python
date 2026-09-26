"""PPO on HydroGym's jet Cylinder as shipped: evaluation rollouts vs the constant-actuation references."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/hydrogym_rl/"; U = np.loadtxt(R + "eval_ppo_jets_uncontrolled.dat"); C = np.loadtxt(R + "eval_ppo_jets_controlled.dat"); S = np.loadtxt(R + "baseline_suck.dat")
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].plot(U[:, 0], U[:, 2], label="uncontrolled"); ax[0].plot(C[:, 0], C[:, 2], label="PPO policy (100k steps)"); ax[0].plot(S[:, 0], S[:, 2], "--", label="constant suction a = -0.1")
ax[0].set_title("C_D"); ax[0].set_ylim(0.9, 1.7)
ax[1].plot(U[:, 0], U[:, 1], label="uncontrolled"); ax[1].plot(C[:, 0], C[:, 1], label="PPO policy"); ax[1].plot(S[:, 0], S[:, 1], "--", label="constant suction"); ax[1].set_title("C_L")
ax[2].plot(C[:, 0], C[:, 3], label="PPO action (deterministic)"); ax[2].axhline(-0.1, color="k", ls=":", label="bound -MAX_CONTROL"); ax[2].set_title("action (one scalar, both jets)"); ax[2].set_ylim(-0.12, 0.12)
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel("t")
n2 = int(0.4 * len(U)); plt.suptitle(f"HydroGym jet Cylinder, Re=100, medium mesh, PPO as shipped: C_D {U[n2:,2].mean():.3f} -> {C[n2:,2].mean():.3f} ({(1-C[n2:,2].mean()/U[n2:,2].mean())*100:.1f}%), the policy sits at the suction bound", fontsize=11)
plt.tight_layout(); plt.savefig("figures/hydrogym_rl_eval.png", dpi=110); print("wrote figures/hydrogym_rl_eval.png")
