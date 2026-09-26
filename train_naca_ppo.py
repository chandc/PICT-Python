"""PPO on the NACA0012 AoA 40 jet environment (naca_env.NACAJetEnv), SB3, parallel environments on the CPU,
policy on the GPU when available. Mirrors tools/hydrogym_cmp/train_sb3_firedrake.py (their cylinder run):
one episode per environment per rollout, MLP policy, all run and history files kept under --logdir.
    python train_naca_ppo.py --task gust --n-envs 8 --total-steps 40000 --logdir rl_logs/naca40_gust
"""
import os, sys, time, argparse, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="gust", choices=["gust", "ld"]); ap.add_argument("--obs", default="probe", choices=["probe", "forces"])
ap.add_argument("--n-envs", type=int, default=8); ap.add_argument("--total-steps", type=int, default=40000); ap.add_argument("--episode-steps", type=int, default=200)
ap.add_argument("--mesh", default="meshes/naca0012_a40_coarse.msh"); ap.add_argument("--snapshots", default="results/naca/a40_coarse_snapshots.npz")
ap.add_argument("--logdir", default="rl_logs/naca40_gust"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--ent", type=float, default=0.0); ap.add_argument("--resume", default=None)
ap.add_argument("--log-std-init", type=float, default=0.0, help="initial log std of the Gaussian policy; 0 (SB3 default) saturates the +-1 jets from the first rollout, -1.5 starts at 0.22")
ap.add_argument("--vmax", type=float, default=0.52, help="jet speed at |a| = 1, in U_inf (HydroGym 0.52)")
a = ap.parse_args()
os.makedirs(a.logdir, exist_ok=True); json.dump(vars(a), open(os.path.join(a.logdir, "config.json"), "w"), indent=1)
import torch; from stable_baselines3 import PPO; from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor; from stable_baselines3.common.callbacks import CheckpointCallback
from naca_env import NACAJetEnv
def make(rank):
    def _f():
        env = NACAJetEnv(mesh=a.mesh, snapshots=a.snapshots, max_episode_steps=a.episode_steps, task=a.task, obs=a.obs, seed=a.seed + rank, vmax=a.vmax,
                         log_path=os.path.join(a.logdir, f"episodes_env{rank}.txt"))
        return env
    return _f
if __name__ == "__main__":
    venv = VecMonitor(SubprocVecEnv([make(r) for r in range(a.n_envs)], start_method="fork"), filename=os.path.join(a.logdir, "monitor"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device {device}, {a.n_envs} envs, task {a.task}, obs {a.obs}, {a.total_steps} steps", flush=True)
    if a.resume: model = PPO.load(a.resume, env=venv, device=device)
    else: model = PPO("MlpPolicy", venv, n_steps=a.episode_steps, batch_size=a.n_envs * a.episode_steps // 4, n_epochs=10, learning_rate=a.lr, gamma=0.99, gae_lambda=0.95,
                      ent_coef=a.ent, clip_range=0.2, policy_kwargs=dict(net_arch=[64, 64], log_std_init=a.log_std_init), verbose=1, seed=a.seed, device=device, tensorboard_log=os.path.join(a.logdir, "tb"))
    t0 = time.time()
    model.learn(total_timesteps=a.total_steps, callback=CheckpointCallback(save_freq=max(a.episode_steps, a.total_steps // (10 * a.n_envs)), save_path=os.path.join(a.logdir, "ckpt"), name_prefix="ppo_naca40"), progress_bar=False)
    model.save(os.path.join(a.logdir, "ppo_naca40_final")); print(f"done in {(time.time()-t0)/3600:.2f} h", flush=True)
