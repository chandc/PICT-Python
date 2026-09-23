"""Roll HydroGym's cylinder env from its checkpoint under a fixed policy and save the final fields.
   python3 rollout_fields.py {zero|const:<a>|policy:<logdir>} T out.npz [cylinder|cylinder_znmf]
Saves cell-centroid u,v,p, DG0 vorticity, vertex coords + CG1 vorticity, and the force/action history."""
import sys, time, numpy as np, firedrake as fd
import hydrogym.firedrake as fdk
from hydrogym import FlowEnv
from hg_znmf import CylinderZNMF
mode, T, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]; envname = sys.argv[4] if len(sys.argv) > 4 else "cylinder"
env = FlowEnv({"flow": {"cylinder": fdk.Cylinder, "cylinder_znmf": CylinderZNMF}[envname], "flow_config": {"Re": 100, "mesh": "medium"},
               "solver": fdk.SemiImplicitBDF, "solver_config": {"dt": 1e-2, "order": 3, "stabilization": "none"}, "max_steps": int(1e6)})
policy = None
if mode.startswith("policy:"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import glob, os
    ld = mode.split(":", 1)[1]; venv = VecNormalize.load(os.path.join(ld, "vec_normalize_final.pkl"), DummyVecEnv([lambda: env])); venv.training = False
    policy = PPO.load(os.path.join(ld, "model_final.zip"))
obs, _ = env.reset(); n = int(round(T / 1e-2)); H = np.zeros((n, 4)); t0 = time.time()
for k in range(n):
    if mode == "zero": a = np.zeros(1)
    elif mode.startswith("const:"): a = np.full(1, float(mode.split(":")[1]))
    else: a = policy.predict(venv.normalize_obs(obs[None, :]), deterministic=True)[0][0]
    obs, r, term, trunc, info = env.step(a); CL, CD = env.flow.compute_forces(); H[k] = ((k + 1) * 1e-2, CL, CD, float(np.ravel(a)[0]))
    if (k + 1) % 1000 == 0: print(f"  {mode} t={H[k,0]:5.1f} CD {H[k-999:k+1,2].mean():.4f} CL rms {H[k-999:k+1,1].std():.4f} ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
flow = env.flow; DG0 = fd.FunctionSpace(flow.mesh, "DG", 0); VDG0 = fd.VectorFunctionSpace(flow.mesh, "DG", 0)
X = fd.Function(VDG0).interpolate(fd.SpatialCoordinate(flow.mesh)); U = fd.Function(VDG0).interpolate(flow.u); P = fd.Function(DG0).interpolate(flow.p)
W = fd.Function(DG0).interpolate(flow.u[1].dx(0) - flow.u[0].dx(1)); Wc = fd.Function(fd.FunctionSpace(flow.mesh, "CG", 1)).project(flow.u[1].dx(0) - flow.u[0].dx(1))
np.savez(out, centroid=X.dat.data_ro, uv=U.dat.data_ro, p=P.dat.data_ro, vort=W.dat.data_ro, vert=flow.mesh.coordinates.dat.data_ro, vort_cg1=Wc.dat.data_ro, hist=H, t=T)
print(f"FIELDS {mode} {envname}: saved {out}; last-60% CD {H[int(0.4*n):,2].mean():.4f} CL rms {H[int(0.4*n):,1].std():.4f}", flush=True)
