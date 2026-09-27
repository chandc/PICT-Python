"""U8 step 2 of the unstructured adjoint plan: differentiable predictive control of HydroGym's NACA0012 alpha 40
gust task (naca_env.py) through the replay adjoint, on PPO's terms: the same mesh, solver step, snapshot start,
three jets with the tanh ramp, the inlet gust, the probe observation (u, v one chord upstream of the leading edge)
and the reward -|C_L - C_L0| - 0.25 |C_D - C_D0| with the forces averaged over the 54 solver steps of each action.
  Training: rolling windows of H actions along the episode from the snapshot at t = 0 (the gust runs to t = 28.9,
  54 actions); after --ep-actions the episode restarts from the snapshot. One gradient per window (Adam, clipped).
  Evaluation (identical to eval_naca_policy.py): deterministic policy against zero action over the 200-action
  episode from the same snapshot phase; return, |dC_L| during and after the gust, trajectories saved.
    python uadj_dpc_naca.py --H 10 --iters 60 --seed 0 --tag dpc_naca_s0
    python uadj_dpc_naca.py --eval-only results/uadj_dpc/dpc_naca_s0_policy.pt --seed 0"""
import sys, os, time, argparse, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naca_env import NACAJetEnv
from src.uadj_step import TorchUPISO
from src.uadj_control import WallForces, SlotJets
from src.uadj_replay import replay_policy_grad_sub
from src.uadj_ops import _ti
ap = argparse.ArgumentParser()
ap.add_argument("--H", type=int, default=10, help="actions per gradient window"); ap.add_argument("--n-sub", type=int, default=None, help="solver steps per action (default the env's 54)")
ap.add_argument("--iters", type=int, default=60); ap.add_argument("--ep-actions", type=int, default=60, help="actions per training episode before restarting from the snapshot (the gust lasts 54)")
ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--w-act", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=6); ap.add_argument("--init", default=None); ap.add_argument("--eval-only", default=None)
ap.add_argument("--eval-actions", type=int, default=200); ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default="results/uadj_dpc")
a = ap.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); os.makedirs(a.outdir, exist_ok=True)
tag = a.tag or f"dpc_naca_h{a.H}_s{a.seed}"
env = NACAJetEnv(seed=a.seed); env.reset(seed=a.seed)                     # the snapshot phase eval_naca_policy.py uses for this seed
T = TorchUPISO(env.s); st_snap = T.state_from_solver(); m = env.m
jets = SlotJets.from_jetset(T, env.jets); wall = m.bfaces[m.btag[m.bfaces] == env.T_W]; W = WallForces(T, wall)
inl = _ti(env.inlet); y_in = torch.as_tensor(env.y_in); n_sub = a.n_sub or env.n_sub; dt = env.dt; cd0, cl0 = env.cd0, env.cl0; probe = env.probe
def gust(t):
    if t >= env.gust_T: return torch.ones(len(inl))
    return 1.0 + (env.gust_factor - 1.0) * torch.exp(-y_in ** 2) * float(np.sin(np.pi * t / env.gust_T) ** 2)
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(2, a.hidden), torch.nn.Tanh(), torch.nn.Linear(a.hidden, a.hidden), torch.nn.Tanh(), torch.nn.Linear(a.hidden, 3))
        with torch.no_grad(): self.net[4].weight *= 0.1; self.net[4].bias.zero_()
    def forward(self, st): return torch.tanh(self.net(torch.stack([st["u"][probe] - 1.0, st["v"][probe]])))      # action in [-1, 1]^3, as PPO's
pol = Policy(); params = list(pol.parameters())
if a.init: pol.load_state_dict(torch.load(a.init)); print(f"policy initialised from {a.init}", flush=True)
def control_step(st, act, k):
    """one action: tanh ramp of the jets from the previous amplitude, the gust on the inlet, n_sub solver steps, forces averaged"""
    t = float(st["t"]) if "t" in st else 0.0; a_prev = st["a_prev"] if st.get("a_prev") is not None else torch.zeros(3)
    cd_acc = cl_acc = 0.0; core = {kk: v for kk, v in st.items() if kk not in ("t", "a_prev")}
    for j in range(n_sub):
        aj = SlotJets.ramp(a_prev, act, (j + 1) / n_sub); core = jets.apply(core, aj)
        core = {**core, "ub": core["ub"].index_copy(0, inl, gust(t))}
        core = T.step(core); t += dt
        cd, cl = W(core); cd_acc = cd_acc + cd; cl_acc = cl_acc + cl
    cd, cl = cd_acc / n_sub, cl_acc / n_sub
    loss = torch.abs(cl - cl0) + 0.25 * torch.abs(cd - cd0) + a.w_act * (act ** 2).sum()
    return {**core, "t": torch.tensor(t), "a_prev": act}, loss
def rollout(st, n, policy=None):
    rows = []
    with torch.no_grad():
        for k in range(n):
            act = policy(st) if policy is not None else torch.zeros(3); st, lk = control_step(st, act, k)
            cd, cl = W({kk: v for kk, v in st.items() if kk not in ("t", "a_prev")}); rows.append((float(st["t"]), float(cd), float(cl), *act.tolist(), -float(lk)))
    return np.array(rows), st
print(f"{tag}: NACA0012 alpha 40, {m.ncell} cells, Re 100, snapshot seed {a.seed}; action = {n_sub} steps x dt {dt} ({n_sub*dt:.2f} time units), gust to t = {env.gust_T} (54 actions); "
      f"window H {a.H} actions ({a.H*n_sub} solver steps per gradient), episode {a.ep_actions} actions; policy {sum(p.numel() for p in params)} parameters on (u, v) at the probe; C_D0 {cd0:.4f} C_L0 {cl0:.4f}; {a.threads} threads", flush=True)
if a.eval_only is None:
    opt = torch.optim.Adam(params, lr=a.lr); hist = []; st = dict(st_snap, t=torch.tensor(0.0), a_prev=torch.zeros(3)); k_ep = 0
    for it in range(a.iters):
        t0 = time.time()
        if k_ep + a.H > a.ep_actions: st = dict(st_snap, t=torch.tensor(0.0), a_prev=torch.zeros(3)); k_ep = 0
        for p_ in params: p_.grad = None
        L, st_end = replay_policy_grad_sub(T, st, pol, a.H, control_step)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip)); opt.step()
        base, _ = rollout(dict(st), a.H)                                             # zero action over the same window, for the reward comparison
        hist.append((it, k_ep, float(st["t"]), -L, base[:, -1].sum(), gn, time.time() - t0))
        print(f"  it {it:3d}  window actions {k_ep}-{k_ep+a.H} (t {float(st['t']):5.1f}-{float(st_end['t']):5.1f})  return {-L:8.3f}  zero-action {base[:, -1].sum():8.3f}  |grad| {gn:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        st = st_end; k_ep += a.H
        torch.save(pol.state_dict(), f"{a.outdir}/{tag}_policy.pt"); np.save(f"{a.outdir}/{tag}_curve.npy", np.array(hist))
else:
    pol.load_state_dict(torch.load(a.eval_only))
# ---- evaluation on PPO's protocol
st_e = dict(st_snap, t=torch.tensor(0.0), a_prev=torch.zeros(3))
tp, _ = rollout(dict(st_e), a.eval_actions, pol); tz, _ = rollout(dict(st_e), a.eval_actions)
np.savez(f"{a.outdir}/{tag}_eval.npz", traj_policy=tp, traj_zero=tz, cd0=cd0, cl0=cl0, gust_T=env.gust_T, seed=a.seed)
g = tp[:, 0] < env.gust_T
print(f"RESULT {tag}: return policy {tp[:, -1].sum():+.2f}  zero-action {tz[:, -1].sum():+.2f}  improvement {(1 - tp[:, -1].sum()/tz[:, -1].sum())*100:+.1f}%  "
      f"mean |dC_L| during the gust {np.abs(tp[g, 2] - cl0).mean():.3f} (zero {np.abs(tz[g, 2] - cl0).mean():.3f}) after {np.abs(tp[~g, 2] - cl0).mean():.3f} (zero {np.abs(tz[~g, 2] - cl0).mean():.3f})  "
      f"peak C_L {tp[:, 2].max():.2f} (zero {tz[:, 2].max():.2f})  mean action {tp[:, 3:6].mean(axis=0).round(3).tolist()}  [PPO of record: -30.3 vs -64.2, +53%]", flush=True)
