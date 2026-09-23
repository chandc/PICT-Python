"""Reference rollouts from HydroGym's developed-shedding checkpoint: uncontrolled, or random actions
uniform in [-MAX_CONTROL, MAX_CONTROL] each step (what an untrained PPO policy with std ~1 does after
clipping). python3 baseline_rollout.py {zero|random} T out.dat"""
import sys, time, numpy as np
import hydrogym.firedrake as fdk
from hydrogym import FlowEnv
mode, T, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]; envname = sys.argv[4] if len(sys.argv) > 4 else "cylinder"
from hg_znmf import CylinderZNMF
env = FlowEnv({"flow": {"cylinder": fdk.Cylinder, "cylinder_znmf": CylinderZNMF}[envname], "flow_config": {"Re": 100, "mesh": "medium"}, "solver": fdk.SemiImplicitBDF,
               "solver_config": {"dt": 1e-2, "order": 3, "stabilization": "none"}, "max_steps": int(1e6)})
rng = np.random.default_rng(0); obs, _ = env.reset(); n = int(round(T / 1e-2)); H = np.zeros((n, 4)); t0 = time.time()
for k in range(n):
    a = {"zero": lambda: np.zeros(1), "random": lambda: rng.uniform(-0.1, 0.1, size=1), "bangbang": lambda: 0.1 * rng.choice([-1.0, 1.0], size=1),
         "blow": lambda: np.full(1, 0.1), "suck": lambda: np.full(1, -0.1), "blowhalf": lambda: np.full(1, 0.05)}[mode]()   # bangbang: what a std~1 Gaussian policy gives after clipping
    obs, r, term, trunc, info = env.step(a); CL, CD = env.flow.compute_forces(); H[k] = ((k + 1) * 1e-2, CL, CD, a[0])
    if (k + 1) % 1000 == 0: print(f"  {mode} t={H[k,0]:5.1f} CD(last 10) {H[k-999:k+1,2].mean():.4f} CL rms {H[k-999:k+1,1].std():.4f} ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
np.savetxt(out, H); print(f"BASELINE {mode}: mean CD over t=0..{T:.0f}: {H[:,2].mean():.4f}   last 60%: {H[int(0.4*n):,2].mean():.4f}   CL rms {H[int(0.4*n):,1].std():.4f}   return sum(-dt*CD) {(-0.01*H[:,2]).sum():.2f}", flush=True)
