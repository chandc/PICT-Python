"""Usage: python plot_utility/plot_naca_ppo_vs_dpc_training.py
PPO's actual training history (rl_logs/, five progressive stages pulled from the Spark's hgrl
container: the abandoned std=1.0 attempt, then std022 -> cont -> cont2 -> cont3, the lineage behind
the "PPO of record" -30.3 return) against the unstructured adjoint's DPC training (results/uadj_dpc/
dpc_naca_h10_s{0,1,2}[_c2][_c3], three seeds x three passes), both on the same metric: return over
the deterministic 200-action gust episode, PPO's own convention (naca_env.py, matching eval_naca_policy.py)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- PPO: concatenate monitor.csv across stages in training order
STAGES = ["naca40_gust_probe", "naca40_gust_probe_std022", "naca40_gust_probe_std022_cont", "naca40_gust_probe_std022_cont2", "naca40_gust_probe_std022_cont3"]
ABANDONED = 1   # naca40_gust_probe (std init 1.0) was abandoned; the successful lineage starts at index 1
N_SUB = 54   # naca_env.py: action_interval 0.54 / dt 0.01 -- each gym env.step() (one row of "l" below) runs this many real CFD solver steps
ep_len = 200; steps = 0; rows = []
for i, s in enumerate(STAGES):
    a = np.genfromtxt(f"rl_logs/{s}/monitor.monitor.csv", delimiter=",", skip_header=2)
    if a.ndim == 1: a = a[None, :]
    for r, l in zip(a[:, 0], a[:, 1]):
        steps += int(l) * N_SUB; rows.append((steps, r, i))
ppo = np.array(rows)                                        # (cumulative_CFD_solver_steps, episode_return, stage_index)
ppo_used = ppo[ppo[:, 2] >= ABANDONED]; ppo_used_steps0 = ppo[ppo[:, 2] == ABANDONED][0, 0] - ep_len * 8 * N_SUB
print(f"PPO: {len(ppo)} episodes total, {int(ppo[-1,0])} cumulative env steps over {len(STAGES)} stages")
print(f"  abandoned attempt ({STAGES[0]}, std_init 1.0): {int(ppo[ppo[:,2]==0][-1,0]) if (ppo[:,2]==0).any() else 0} steps, final ~20-episode mean return {ppo[ppo[:,2]==0][-20:,1].mean():.1f}" if ABANDONED else "")
print(f"  successful lineage ({', '.join(STAGES[ABANDONED:])}): final 20-episode mean return {ppo_used[-20:, 1].mean():.2f}")

# ---- DPC: three seeds, three passes each; the comparable metric is the 200-action closed-loop return
def dpc_points(seed):
    """returns (cumulative_solver_steps, episode_return) at every point we actually evaluated the 200-action episode"""
    pts = []; step0 = 0
    for tag, has_ehist in ((f"dpc_naca_h10_s{seed}", False), (f"dpc_naca_h10_s{seed}_c2", False), (f"dpc_naca_h10_s{seed}_c3", True)):
        cv = np.load(f"results/uadj_dpc/{tag}_curve.npy"); n_it = len(cv); H, sub = 10, 54
        steps_per_it = 2 * H * sub                            # forward + replay
        if has_ehist and _os.path.exists(f"results/uadj_dpc/{tag}_ehist.npy"):
            eh = np.load(f"results/uadj_dpc/{tag}_ehist.npy")  # (it_local, return)
            for it_local, ret in eh: pts.append((step0 + (it_local + 1) * steps_per_it, ret))
        ev = np.load(f"results/uadj_dpc/{tag}_eval.npz"); tp = ev["traj_policy"]
        pts.append((step0 + n_it * steps_per_it, tp[:, -1].sum()))   # the pass's own final 200-action evaluation
        step0 += n_it * steps_per_it
    return np.array(sorted(pts))
dpc = {s: dpc_points(s) for s in (0, 1, 2)}
for s in (0, 1, 2): print(f"DPC seed {s}: {len(dpc[s])} evaluation points, {int(dpc[s][-1,0])} cumulative solver steps, final return {dpc[s][-1,1]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
succ = ppo_used.copy(); succ[:, 0] -= succ[0, 0] - ep_len * 8 * N_SUB   # re-zero steps at the start of the successful lineage
win = 20; roll = np.array([succ[max(0, i - win):i + 1, 1].mean() for i in range(1, len(succ) + 1)])
for a_ in ax:
    a_.plot(succ[:, 0], succ[:, 1], ".", color="0.75", ms=3, label="PPO, per-episode return (training)")
    a_.plot(succ[:, 0], roll, "k", lw=1.8, label=f"PPO, {win}-episode rolling mean")
    a_.axhline(-30.3, color="k", ls=":", lw=1, label="PPO of record (deterministic eval): -30.3")
    a_.axhline(-64.2, color="0.5", ls=":", lw=1, label="zero action: -64.2")
    for s, c in zip((0, 1, 2), ("C0", "C1", "C2")):
        a_.plot(dpc[s][:, 0], dpc[s][:, 1], "-o", color=c, ms=4, lw=1.2, label=f"DPC seed {s}")
ax[0].set(xlabel="cumulative solver steps (PPO: env steps; DPC: forward + replay)", ylabel="200-action gust-episode return", title="linear"); ax[0].legend(fontsize=7.5, ncol=1, loc="lower right")
ax[1].set(xscale="log", xlabel="cumulative solver steps (log scale)", ylabel="return", xlim=(1e3, 1.3e7), title="log scale: sample-efficiency comparison"); ax[1].legend(fontsize=7.5, loc="lower right")
plt.suptitle(f"NACA0012 alpha 40 gust task: PPO's actual training history vs. the unstructured adjoint's DPC (3 seeds each)\n"
             f"PPO: {int(succ[-1,0]):,} env steps to its recorded result (+ {int(ppo[ppo[:,2]==0][-1,0]) if ABANDONED else 0:,} abandoned, std_init 1.0, not counted); "
             f"DPC: {int(np.mean([dpc[s][-1,0] for s in (0,1,2)])):,} solver steps/seed", fontsize=10.5)
plt.tight_layout(); plt.savefig("figures/naca_ppo_vs_dpc_training.png", dpi=140); print("wrote figures/naca_ppo_vs_dpc_training.png")
