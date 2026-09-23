"""PPO training history (parsed from the SB3 stdout log) and the jet configuration of HydroGym's Cylinder."""
import re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
txt = open("results/hydrogym_rl/train_ppo_jets.log").read()
blocks = txt.split("-----------------------------")
rows = []
for b in blocks:
    d = dict(re.findall(r"\|\s+([a-z_/]+)\s+\|\s+([-+0-9.e]+)\s+\|", b))
    if "total_timesteps" in d: rows.append({k: float(v) for k, v in d.items()})
keys = ["total_timesteps", "ep_rew_mean", "std", "approx_kl", "value_loss", "entropy_loss", "explained_variance", "clip_fraction"]
T = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}; steps = T["total_timesteps"]
# per-episode returns from the running mean: mean_n = (sum of n episodes)/n  ->  R_n = n*mean_n - (n-1)*mean_{n-1}
m = T["ep_rew_mean"]; ok = ~np.isnan(m); mu = m[ok]; su = steps[ok]
change = np.r_[True, np.abs(np.diff(mu)) > 1e-9]; mu_ep = mu[change]; st_ep = su[change]
n = np.arange(1, len(mu_ep) + 1); R = n * mu_ep - np.r_[0, (n[:-1]) * mu_ep[:-1]]
print("per-episode returns:", np.round(R, 1)); print("=> mean C_D per episode:", np.round(-R / 5000 / 0.01, 3))
fig, ax = plt.subplots(2, 3, figsize=(20, 9))
a = ax[0, 0]; a.plot(st_ep, R, "o-", label="episode return  sum(-dt C_D), 5000 steps"); a.axhline(-74.3, color="k", ls=":", label="uncontrolled (-74.3)"); a.axhline(-52.1, color="r", ls=":", label="constant suction -0.1 (-52.1)")
a.set_title("episode return vs training steps"); a.legend(fontsize=8); a2 = a.twinx(); a2.set_ylim(74.3 / 50, 52.1 / 50 - 0.1); a2.set_ylabel("episode-mean C_D  (right axis, inverted)")
a2.set_yticks([1.486, 1.3, 1.2, 1.1, 1.042]); a2.invert_yaxis()
ax[0, 1].plot(steps, T["std"]); ax[0, 1].set_title("policy std (pre-clip Gaussian, action units are +-0.1 after clip)")
ax[0, 2].plot(steps, T["approx_kl"]); ax[0, 2].set_title("approx KL per update"); ax[0, 2].set_yscale("log")
ax[1, 0].plot(steps, T["value_loss"]); ax[1, 0].set_title("value loss"); ax[1, 0].set_yscale("log")
ax[1, 1].plot(steps, T["explained_variance"]); ax[1, 1].set_title("explained variance"); ax[1, 1].set_ylim(-1, 1.05)
# jet configuration
th = np.linspace(-np.pi, np.pi, 2000); om = np.pi / 18; R_ = 0.5
def A(t0): return np.where(np.abs(th - t0) < om / 2, np.pi / (2 * om * R_**2) * np.cos(np.pi / om * (th - t0)), 0.0)
un = R_ * (A(np.pi / 2) + A(-np.pi / 2))          # wall-normal velocity per unit control a
a = ax[1, 2]; a.plot(np.degrees(th), 0.1 * un, label="a = +0.1 (blowing, both jets)"); a.plot(np.degrees(th), -0.1 * un, label="a = -0.1 (suction, both jets)")
a.set_xlabel("theta [deg] (0 = downstream)"); a.set_ylabel("wall-normal velocity / U_inf"); a.set_title("jet profile: 10-deg cosine slots at +-90 deg, one scalar a, radial, both outward for a>0\nflux per jet = a (max 0.1 per jet, 0.2 total = 20% of U_inf D); peak 1.8 U_inf; actuator lag TAU 0.0556")
a.legend(fontsize=8); a.set_xlim(-180, 180)
for x in ax.ravel(): x.grid(alpha=.3)
plt.suptitle("PPO on HydroGym's jet Cylinder (Re=100, medium mesh, as shipped): training history", fontsize=12); plt.tight_layout()
plt.savefig("figures/hydrogym_rl_training.png", dpi=110); print("wrote figures/hydrogym_rl_training.png")
