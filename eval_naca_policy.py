"""Deterministic evaluation of a trained NACA jet policy against the do-nothing baseline: same snapshot phase for
both, the gust episode of 200 actions, per-step forces and actions logged.
    python eval_naca_policy.py rl_logs/naca40_gust_probe_std022_cont3/ppo_naca40_final.zip --episodes 3 --out results/naca/eval_cont3.npz"""
import sys, os, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--episodes", type=int, default=3); ap.add_argument("--out", default="results/naca/eval.npz"); ap.add_argument("--seed", type=int, default=100)
a = ap.parse_args()
from stable_baselines3 import PPO
from naca_env import NACAJetEnv
model = PPO.load(a.model, device="cpu"); rows = []
for ep in range(a.episodes):
    for mode in ("policy", "zero"):
        env = NACAJetEnv(seed=a.seed + ep); obs, _ = env.reset(seed=a.seed + ep); R = 0.0; traj = []
        for k in range(200):
            act = model.predict(obs, deterministic=True)[0] if mode == "policy" else np.zeros(3)
            obs, r, done, trunc, info = env.step(act); R += r; traj.append((info["t"], info["cd"], info["cl"], *act))
        traj = np.array(traj); rows.append((ep, mode, R, traj))
        print(f"episode {ep} {mode:6s}: return {R:+.2f}  mean |dCl| during gust {np.abs(traj[traj[:,0] < 28.9, 2] - env.cl0).mean():.3f}  after {np.abs(traj[traj[:,0] >= 28.9, 2] - env.cl0).mean():.3f}  peak cl {traj[:,2].max():.3f}  mean action {traj[:,3:].mean(axis=0).round(3).tolist()}", flush=True)
np.savez(a.out, returns=np.array([(r[0], r[1] == "policy", r[2]) for r in rows], dtype=float), traj_policy=np.array([r[3] for r in rows if r[1] == "policy"]), traj_zero=np.array([r[3] for r in rows if r[1] == "zero"]), cd0=env.cd0, cl0=env.cl0)
P = [r[2] for r in rows if r[1] == "policy"]; Z = [r[2] for r in rows if r[1] == "zero"]
print(f"RESULT policy {np.mean(P):+.2f} +- {np.std(P):.2f}   do-nothing {np.mean(Z):+.2f} +- {np.std(Z):.2f}   improvement {(1 - np.mean(P)/np.mean(Z))*100:+.1f}%")
