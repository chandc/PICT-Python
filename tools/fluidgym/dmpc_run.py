"""Exact-config reproduction of FluidGym's D-MPC baseline (their gradient method).

Their published artifacts (HF fluidgym-experiments, D-MPC/CylinderJet2D-easy-v0/*/
config.yaml) fix the algorithm: rl_mode=sarl, horizon=20, n_iterations=10, lr=0.1,
discount_factor=0.999.  hydra job name `run_d-mpc`; the runner itself is NOT in the
public repo, so this reimplements receding-horizon differentiable MPC against those
hyperparameters: at every control step, clone the env state (env.get_state()),
optimize the next-`horizon` action sequence by BPTT through the differentiable env
(n_iterations gradient steps), restore the state, execute the FIRST action, shift
the plan as the next warm start.

Their own eval artifacts to compare against (per-step CSVs, cd_ref = 3.3281555):
episode-mean drag 3.2030 +/- 0.0082 over 10 seeds (3.76% reduction), ep reward
-7.75 +/- 1.16.  The one hyperparameter their configs do NOT pin is the optimizer;
default Adam, --opt sgd for plain gradient ascent.

  docker run --rm --gpus all -v $HOME/fluidgym_m0:/w -w /w fluidgym:m0 \
      python3 -u dmpc_run.py --seed 0 --tag dmpc_s0
"""
import argparse
import csv
import time

import torch
import fluidgym

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--horizon", type=int, default=20)
p.add_argument("--n-iterations", type=int, default=10)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--discount", type=float, default=0.999)
p.add_argument("--opt", choices=("adam", "sgd"), default="adam")
p.add_argument("--control-steps", type=int, default=80,
               help="control steps to run (80 = full episode; small for smoke)")
p.add_argument("--tag", default="dmpc")
a = p.parse_args()

dev = "cuda"
env = fluidgym.make("CylinderJet2D-easy-v0", differentiable=True)
obs, info = env.reset(seed=a.seed)

rows = []
plan = torch.zeros(a.horizon, device=dev, dtype=torch.float64)
t_start = time.time()

for t in range(a.control_steps):
    st = env.get_state()
    theta = plan.clone().requires_grad_(True)
    opt = (torch.optim.Adam if a.opt == "adam" else torch.optim.SGD)([theta], lr=a.lr)
    for _ in range(a.n_iterations):
        env.set_state(st)
        R = torch.zeros((), device=dev, dtype=torch.float64)
        for h in range(a.horizon):
            act = torch.clamp(theta[h], -1.0, 1.0).reshape(1)
            _, r, term, trunc, _ = env.step(act)
            R = R + (a.discount ** h) * r
            if term or trunc:
                break
        opt.zero_grad()
        (-R).backward()
        opt.step()
    # execute the first planned action from the true state
    env.set_state(st)
    with torch.no_grad():
        act = torch.clamp(theta[0].detach(), -1.0, 1.0).reshape(1)
        _, r, term, trunc, inf = env.step(act)
    rows.append(dict(drag=float(inf["drag"]), lift=float(inf["lift"]),
                     action_0=float(act), reward=float(r), step=t))
    plan = torch.cat([theta[1:].detach(), theta[-1:].detach()])  # shift warm start
    print(f"  t {t:3d}  drag {rows[-1]['drag']:.4f}  lift {rows[-1]['lift']:+.4f}  "
          f"a {rows[-1]['action_0']:+.3f}  r {rows[-1]['reward']:+.4f}  "
          f"{time.time() - t_start:.0f}s", flush=True)
    if term or trunc:
        break

with open(f"/w/{a.tag}_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["drag", "lift", "action_0", "reward", "step"])
    w.writeheader()
    w.writerows(rows)

md = sum(r["drag"] for r in rows) / len(rows)
er = sum(r["reward"] for r in rows)
print(f"RESULT {a.tag}: steps {len(rows)}  mean drag {md:.4f}  "
      f"({100 * (1 - md / 3.3281555):+.2f}% vs cd_ref)  ep reward {er:.4f}")
