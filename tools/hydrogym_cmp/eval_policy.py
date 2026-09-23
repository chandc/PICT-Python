"""Evaluate a trained SB3 policy on HydroGym's jet Cylinder against the uncontrolled flow.

    python3 eval_policy.py --logdir /work/rl_logs/PPO_Firedrake_cylinder_XXXX [--T 100] [--out out/eval]

Builds the same env as train_sb3_firedrake.py (FlowEnv -> Monitor -> DummyVecEnv -> VecNormalize with
the saved statistics, frozen), starts both rollouts from the environment's initial state (the HydroGym
developed-shedding checkpoint), runs T time units uncontrolled and T time units with the deterministic
policy, and reports the mean C_D over the last 60% of each window, C_L rms, and the actuation statistics."""
import argparse, glob, os, time, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--logdir", required=True); ap.add_argument("--T", type=float, default=100.0)
ap.add_argument("--out", default="out/eval"); ap.add_argument("--algo", default="PPO"); ap.add_argument("--env", default="cylinder"); a = ap.parse_args()
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import hydrogym.firedrake as fdk
from hydrogym import FlowEnv
Algo = {"PPO": PPO, "TD3": TD3, "SAC": SAC}[a.algo]
def make_env():
    from hg_znmf import CylinderZNMF
    return Monitor(FlowEnv({"flow": {"cylinder": fdk.Cylinder, "cylinder_znmf": CylinderZNMF}[a.env], "flow_config": {"Re": 100, "mesh": "medium"}, "solver": fdk.SemiImplicitBDF,
                            "solver_config": {"dt": 1e-2, "order": 3, "stabilization": "none"},
                            "actuation_config": {"num_substeps": 1, "reward_aggregation": "mean"}, "callbacks": [], "max_steps": int(1e6)}))
model_path = os.path.join(a.logdir, "model_final.zip"); stats_path = os.path.join(a.logdir, "vec_normalize_final.pkl")
if not os.path.exists(model_path):   # fall back to the latest checkpoint
    cks = sorted(glob.glob(os.path.join(a.logdir, "model_*_steps.zip")), key=lambda s: int(s.split("_")[-2])); model_path = cks[-1]
    stats_path = model_path.replace("model_", "vec_normalize_").replace(".zip", ".pkl")
print(f"model {model_path}\nstats {stats_path}", flush=True)
nsteps = int(round(a.T / 1e-2)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
def rollout(controlled):
    venv = VecNormalize.load(stats_path, DummyVecEnv([make_env])); venv.training = False; venv.norm_reward = False
    model = Algo.load(model_path, env=venv) if controlled else None
    flow = venv.venv.envs[0].unwrapped.flow
    obs = venv.reset(); hist = np.zeros((nsteps, 4)); t0 = time.time()
    for k in range(nsteps):
        act = model.predict(obs, deterministic=True)[0] if controlled else np.zeros((1, 1))
        obs, r, done, info = venv.step(act)
        CL, CD = flow.compute_forces(); hist[k] = ((k + 1) * 1e-2, CL, CD, float(act[0, 0]))
        if (k + 1) % 1000 == 0: print(f"  {'controlled' if controlled else 'uncontrolled'} t={hist[k,0]:6.1f}  CD {hist[max(0,k-999):k+1,2].mean():.4f}  CL rms {hist[max(0,k-999):k+1,1].std():.4f}  |a| mean {np.abs(hist[max(0,k-999):k+1,3]).mean():.4f}  ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
    return hist
H0 = rollout(False); np.savetxt(f"{a.out}_uncontrolled.dat", H0)
H1 = rollout(True); np.savetxt(f"{a.out}_controlled.dat", H1)
n2 = int(0.4 * nsteps)
cd0, cd1 = H0[n2:, 2].mean(), H1[n2:, 2].mean(); cl0, cl1 = H0[n2:, 1].std(), H1[n2:, 1].std()
print(f"EVAL window t={H0[n2,0]:.0f}..{a.T:.0f}: uncontrolled CD {cd0:.4f} CL_rms {cl0:.4f} | controlled CD {cd1:.4f} CL_rms {cl1:.4f} | drag reduction {(1-cd1/cd0)*100:.1f}% | action mean {H1[n2:,3].mean():+.4f} rms {H1[n2:,3].std():.4f} (MAX_CONTROL 0.1)", flush=True)
