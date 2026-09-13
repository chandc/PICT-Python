"""M0 smoke: FluidGym on the GB10 -- env builds, steps, and differentiates."""
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "device ok:", torch.cuda.is_available(), flush=True)
import fluidgym
print("fluidgym imported", flush=True)

env = fluidgym.make("CylinderJet2D-easy-v0", differentiable=False)
obs, info = env.reset(seed=0)
print("env reset ok; obs keys:", list(obs.keys()) if hasattr(obs, "keys") else type(obs), flush=True)
import numpy as np
a0 = torch.zeros(env.action_space.shape, dtype=torch.float32, device="cuda")
for i in range(3):
    obs, r, term, trunc, info = env.step(a0)
    rv = float(r.detach().cpu()) if torch.is_tensor(r) else float(np.asarray(r))
    print(f"  uncontrolled step {i}: reward = {rv:.5f}", flush=True)

# differentiable mode: one action, gradient of reward
env2 = fluidgym.make("CylinderJet2D-easy-v0", differentiable=True)
obs, info = env2.reset(seed=0)
act = torch.zeros(env2.action_space.shape, dtype=torch.float32,
                  device="cuda", requires_grad=True)
obs, r, term, trunc, info = env2.step(act)
loss = -r if torch.is_tensor(r) else torch.as_tensor(r)
print("reward tensor?", torch.is_tensor(r), flush=True)
if torch.is_tensor(r) and r.requires_grad:
    (-r).sum().backward()
    print("d(reward)/d(action) =", act.grad.detach().cpu().numpy(), flush=True)
print("M0 SMOKE COMPLETE", flush=True)
