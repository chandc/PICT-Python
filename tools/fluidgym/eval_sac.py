"""Evaluate FluidGym's published SAC checkpoint (SB3) in our container.

Their SAC baseline is literally stable_baselines3 SAC(MlpPolicy) with default
hyperparameters (config in training/sarl/<env>/SAC/<seed>/config.yaml of the
fluidgym-experiments dataset), trained through fluidgym.integration.sb3 for
50k env steps.  ckpt_latest.zip is a standard SB3 save; loading it and running
deterministic episodes gives us their model-free reference WITHOUT retraining
(625 episodes avoided).  Their seed-0 test eval: mean drag 3.0996 (-6.87% vs
cd_ref 3.3281555), last-40 3.0650 (-7.91%).

  docker run --rm --gpus all -v $HOME/fluidgym_m0:/w -w /w fluidgym:m0 \
      bash -c "pip install -q stable-baselines3 omegaconf && python3 -u eval_sac.py"
(their checkpoint's cloudpickle payload imports omegaconf -- hydra artifacts)
"""
import argparse

import numpy as np
import torch
from stable_baselines3 import SAC

import fluidgym
from fluidgym.wrappers import FlattenObservation

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="/w/sac_ckpt_latest.zip")
p.add_argument("--env", default="CylinderJet2D-easy-v0")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--tag", default="sac_eval")
a = p.parse_args()

env = FlattenObservation(fluidgym.make(a.env, differentiable=False))
model = SAC.load(a.ckpt, device="cuda")

rs, ds = [], []
for ep in range(a.episodes):
    obs, info = env.reset(seed=1000 + ep)
    total, drags = 0.0, []
    for t in range(80):
        obs_np = (obs.detach().cpu().numpy() if torch.is_tensor(obs)
                  else np.asarray(obs))
        act, _ = model.predict(obs_np, deterministic=True)
        act_t = torch.as_tensor(act, device="cuda", dtype=torch.float32)
        obs, r, term, trunc, info = env.step(act_t)
        total += float(torch.as_tensor(r))
        drags.append(float(torch.as_tensor(info["drag"]).detach().float().mean()))
        if term or trunc:
            break
    rs.append(total); ds.append(float(np.mean(drags)))
    print(f"  ep {ep}: reward {total:9.4f}  drag {ds[-1]:.5f}", flush=True)

print(f"RESULT {a.tag}: reward {np.mean(rs):.4f} +/- {np.std(rs):.4f}  "
      f"drag {np.mean(ds):.5f} ({100 * (1 - np.mean(ds) / 3.3281555):+.2f}% "
      f"vs cd_ref)", flush=True)
np.savez(f"/w/{a.tag}.npz", rewards=rs, drags=ds)
