"""Fields of the gust episode, jets off vs the trained policy (deterministic), same snapshot phase, saved at chosen
times for plotting.   python eval_naca_fields.py <model.zip> --out results/naca/fields_eval.npz"""
import sys, os, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--out", default="results/naca/fields_eval.npz"); ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--times", default="0,8.1,13.5,21.6,29.7,54,108")
a = ap.parse_args(); T_SAVE = [float(x) for x in a.times.split(",")]
from stable_baselines3 import PPO
from naca_env import NACAJetEnv
model = PPO.load(a.model, device="cpu"); out = {}
for mode in ("zero", "policy"):
    env = NACAJetEnv(seed=a.seed); obs, _ = env.reset(seed=a.seed); saved = []; hist = []
    def snap(t): saved.append((t, env.s.u.copy(), env.s.v.copy(), env.s.p.copy(), env.jets.a.copy()))
    snap(0.0); nxt = 1
    for k in range(200):
        act = model.predict(obs, deterministic=True)[0] if mode == "policy" else np.zeros(3)
        obs, r, done, trunc, info = env.step(act); hist.append((info["t"], info["cd"], info["cl"], *act))
        if nxt < len(T_SAVE) and info["t"] >= T_SAVE[nxt] - 1e-9: snap(info["t"]); nxt += 1
    out[f"{mode}_t"] = np.array([s[0] for s in saved]); out[f"{mode}_u"] = np.array([s[1] for s in saved]); out[f"{mode}_v"] = np.array([s[2] for s in saved]); out[f"{mode}_p"] = np.array([s[3] for s in saved]); out[f"{mode}_a"] = np.array([s[4] for s in saved]); out[f"{mode}_hist"] = np.array(hist)
    print(mode, "saved at t =", np.round(out[f"{mode}_t"], 2).tolist(), flush=True)
m = env.m; np.savez(a.out, nodes=m.nodes, cells=m.cells, nvert=m.nvert, centroid=m.centroid, btag=m.btag, cd0=env.cd0, cl0=env.cl0, **out); print("wrote", a.out)
