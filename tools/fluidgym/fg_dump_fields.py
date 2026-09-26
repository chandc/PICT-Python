"""Dump FluidGym CylinderJet2D fields for vorticity plotting: uncontrolled
episode vs the trained H=80 DPC policy, same seed. Saves per-block velocity
+ vertex coordinates at the final step, plus per-step drag/lift/action
traces, to /w/<tag>_fields.npz.

  docker run --rm --gpus all -v $HOME/fluidgym_m0:/w -w /w fluidgym:m0 \
      python3 -u fg_dump_fields.py --policy /w/m0_H80_policy.pt --tag h80
"""
import argparse

import numpy as np
import torch

import fluidgym

p = argparse.ArgumentParser()
p.add_argument("--policy", default="/w/m0_H80_policy.pt")
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--tag", default="h80")
a = p.parse_args()

dev = "cuda"
env = fluidgym.make("CylinderJet2D-easy-v0", differentiable=False)


def flat_obs(obs):
    return torch.cat([torch.as_tensor(v, device=dev).reshape(-1).float()
                      for _, v in sorted(obs.items())])


def dump_domain(env):
    blocks = []
    coords = env._domain.getVertexCoordinates()
    for b, blk in enumerate(env._domain.getBlocks()):
        blocks.append((blk.velocity.detach().cpu().numpy(),
                       coords[b].detach().cpu().numpy()))
    return blocks


def episode(act_fn):
    obs, info = env.reset(seed=a.seed)
    drag, lift, act = [], [], []
    for t in range(80):
        u = act_fn(obs, t)
        obs, r, term, trunc, info = env.step(u)
        drag.append(float(info["drag"]))
        lift.append(float(info["lift"]))
        act.append(float(torch.as_tensor(u).reshape(-1)[0]))
        if term or trunc:
            break
    return dump_domain(env), np.array(drag), np.array(lift), np.array(act)


zero = torch.zeros(env.action_space.shape, dtype=torch.float32, device=dev)
blocks_u, drag_u, lift_u, _ = episode(lambda o, t: zero)
print(f"uncontrolled: mean drag {drag_u.mean():.4f}", flush=True)

obs0, _ = env.reset(seed=a.seed)
nin = flat_obs(obs0).numel()
nact = int(np.prod(env.action_space.shape))
policy = torch.nn.Sequential(
    torch.nn.Linear(nin, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, nact)).to(dev)
policy.load_state_dict(torch.load(a.policy, map_location=dev))
amax = float(torch.as_tensor(env.action_space.high).max())

with torch.no_grad():
    blocks_c, drag_c, lift_c, act_c = episode(
        lambda o, t: (amax * torch.tanh(policy(flat_obs(o))))
        .reshape(env.action_space.shape))
print(f"controlled:   mean drag {drag_c.mean():.4f}  "
      f"({100 * (1 - drag_c.mean() / 3.3281555):+.2f}% vs cd_ref)", flush=True)

out = {"drag_u": drag_u, "lift_u": lift_u, "drag_c": drag_c, "lift_c": lift_c,
       "act_c": act_c, "nblocks": np.array(len(blocks_u))}
for b, (v, c) in enumerate(blocks_u):
    out[f"u_vel{b}"], out[f"u_xy{b}"] = v, c
for b, (v, c) in enumerate(blocks_c):
    out[f"c_vel{b}"], out[f"c_xy{b}"] = v, c
np.savez(f"/w/{a.tag}_fields.npz", **out)
print(f"saved /w/{a.tag}_fields.npz", flush=True)
