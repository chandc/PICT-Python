"""M0: DPC on FluidGym -- truncated-BPTT policy training with horizon H as
the experimental knob. Best-effort reproduction of the paper's DPC (their
trainer is not shipped); reward gradients only, no value function, no
exploration.

Modes:
  --mode policy    closed-loop MLP(obs) -> action, trained with windows of H
                   env steps (env.detach() truncates the graph between
                   windows) -- comparable to the paper's DPC.
  --mode sequence  open-loop action sequence over the whole episode,
                   full-window BPTT -- the clean horizon-physics probe.
"""
import argparse
import time

import numpy as np
import torch

import fluidgym

p = argparse.ArgumentParser()
p.add_argument("--env", default="CylinderJet2D-easy-v0")
p.add_argument("--mode", choices=("policy", "sequence", "baseline"), default="policy")
p.add_argument("--horizon", type=int, default=8, help="env steps per BPTT window")
p.add_argument("--iters", type=int, default=150)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--eval-episodes", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="dpc")
a = p.parse_args()

torch.manual_seed(a.seed)
dev = torch.device("cuda")


def flat_obs(obs):
    return torch.cat([torch.as_tensor(v, device=dev).reshape(-1).float()
                      for _, v in sorted(obs.items())])


def episode_reward(env, act_fn, seed=0, grad=False, log_info=False):
    """One full episode; returns (sum reward tensor/float, mean drag if available)."""
    obs, info = env.reset(seed=seed)
    total, drags = 0.0, []
    T = 80
    for t in range(T):
        act = act_fn(obs, t)
        obs, r, term, trunc, info = env.step(act)
        total = total + r
        for k in ("drag", "mean_drag", "c_d", "cd"):
            if isinstance(info, dict) and k in info:
                drags.append(float(torch.as_tensor(info[k]).detach().float().mean()))
                break
        if log_info and t == 0:
            print("  info keys:", list(info.keys()) if isinstance(info, dict) else type(info),
                  flush=True)
        if term or trunc:
            break
    return total, (float(np.mean(drags)) if drags else float("nan"))


env = fluidgym.make(a.env, differentiable=(a.mode != "baseline"))
act_shape = env.action_space.shape
zero = torch.zeros(act_shape, dtype=torch.float32, device=dev)
print(f"env {a.env}  mode {a.mode}  action shape {tuple(act_shape)}", flush=True)

if a.mode == "baseline":
    rs, ds = [], []
    for ep in range(max(a.eval_episodes, 5)):
        with torch.no_grad():
            r, dmean = episode_reward(env, lambda o, t: zero, seed=1000 + ep, log_info=(ep == 0))
        rs.append(float(torch.as_tensor(r))); ds.append(dmean)
        print(f"  uncontrolled ep {ep}: sum reward {rs[-1]:.4f}  drag {dmean:.5f}", flush=True)
    print(f"BASELINE: reward {np.mean(rs):.4f} +/- {np.std(rs):.4f}  "
          f"drag {np.nanmean(ds):.5f}", flush=True)
    np.savez(f"/w/{a.tag}_baseline.npz", rewards=rs, drags=ds)
    raise SystemExit(0)

if a.mode == "policy":
    obs0, _ = env.reset(seed=a.seed)
    nin = flat_obs(obs0).numel()
    policy = torch.nn.Sequential(
        torch.nn.Linear(nin, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, int(np.prod(act_shape)))).to(dev)
    amax = float(torch.as_tensor(env.action_space.high).max())
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)

    def act_fn(obs, t):
        return (amax * torch.tanh(policy(flat_obs(obs)))).reshape(act_shape)

    hist = []
    t0 = time.time()
    for it in range(a.iters):
        obs, _ = env.reset(seed=a.seed * 10000 + it)
        ep_reward, win_reward, nwin = 0.0, 0.0, 0
        for t in range(80):
            act = act_fn(obs, t)
            obs, r, term, trunc, _ = env.step(act)
            win_reward = win_reward + r
            ep_reward += float(r.detach())
            if (t + 1) % a.horizon == 0 or t == 79 or term or trunc:
                opt.zero_grad()
                (-win_reward).backward()
                opt.step()
                env.detach()
                # the local obs still references the freed window graph;
                # detach it or the NEXT window's backward re-walks this one
                obs = {k: (v.detach() if torch.is_tensor(v) else v)
                       for k, v in obs.items()}
                win_reward, nwin = 0.0, nwin + 1
            if term or trunc:
                break
        hist.append(ep_reward)
        if it % 5 == 0 or it == a.iters - 1:
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  it {it:4d}  ep reward {ep_reward:8.4f}  windows {nwin}  "
                  f"peak mem {mem:.2f} GiB  {time.time()-t0:.0f}s", flush=True)
    torch.save(policy.state_dict(), f"/w/{a.tag}_policy.pt")

else:  # sequence
    seq = torch.zeros((80,) + tuple(act_shape), dtype=torch.float32, device=dev,
                      requires_grad=True)
    amax = float(torch.as_tensor(env.action_space.high).max())
    opt = torch.optim.Adam([seq], lr=a.lr)
    hist = []
    t0 = time.time()
    for it in range(a.iters):
        obs, _ = env.reset(seed=a.seed)   # FIXED init for open-loop
        total = 0.0
        for t in range(80):
            obs, r, term, trunc, _ = env.step(amax * torch.tanh(seq[t]))
            total = total + r
            if (t + 1) % a.horizon == 0 or t == 79:
                # windowed even in sequence mode so horizon stays the knob
                pass
            if term or trunc:
                break
        opt.zero_grad()
        (-total).backward()
        opt.step()
        env.detach()
        hist.append(float(total.detach()))
        if it % 5 == 0 or it == a.iters - 1:
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  it {it:4d}  ep reward {hist[-1]:8.4f}  peak mem {mem:.2f} GiB  "
                  f"{time.time()-t0:.0f}s", flush=True)
    np.save(f"/w/{a.tag}_seq.npy", seq.detach().cpu().numpy())

# ---- evaluation ----
env_eval = fluidgym.make(a.env, differentiable=False)
rs, ds = [], []
with torch.no_grad():
    for ep in range(a.eval_episodes):
        if a.mode == "policy":
            fn = lambda o, t: act_fn(o, t).detach()
        else:
            fn = lambda o, t: (amax * torch.tanh(seq[t])).detach()
        r, dmean = episode_reward(env_eval, fn, seed=1000 + ep)
        rs.append(float(torch.as_tensor(r))); ds.append(dmean)
print(f"RESULT {a.tag}: eval reward {np.mean(rs):.4f} +/- {np.std(rs):.4f}  "
      f"drag {np.nanmean(ds):.5f}", flush=True)
np.savez(f"/w/{a.tag}_curve.npz", hist=hist, eval_rewards=rs, eval_drags=ds)
