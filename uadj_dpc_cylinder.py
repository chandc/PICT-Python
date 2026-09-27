"""U8 step 1 of the unstructured adjoint plan: differentiable predictive control (DPC) of the Re 100 cylinder
wake on the coarse butterfly with the +-90 degree jets, through the memory-flat policy-in-the-loop replay
(`uadj_replay.replay_policy_grad`, gates A20-A22). CPU float64 (the adjoint has no GPU path before U10).
  Start state: the limit cycle `results/uadj_shed_cylinder_butterfly_coarse.npz` (t = 200, C_L amplitude 0.278,
  period 5.94); four starting phases a quarter period apart are cycled through the iterations.
  Policy: pressure at wake probes -> 2 jet amplitudes (tanh, |a| <= amax).  Loss per control step: C_D + w_a sum a^2.
  Each iteration: one gradient over H control steps x `sub` solver steps from one phase, Adam, clipped.
  Evaluation: closed loop over `--eval-periods` shedding periods from phase 0, against the uncontrolled wake.
    python uadj_dpc_cylinder.py --H 8 --sub 5 --iters 50 --tag dpc_h8
    python uadj_dpc_cylinder.py --H 40 --sub 5 --iters 50 --tag dpc_h40"""
import sys, os, time, argparse, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.uadj_cases import cylinder
from src.uadj_step import TorchUPISO
from src.uadj_control import WallForces, SlotJets
from src.uadj_replay import replay_policy_grad
ap = argparse.ArgumentParser()
ap.add_argument("--H", type=int, default=8, help="control steps per gradient window"); ap.add_argument("--sub", type=int, default=5, help="solver steps per control step (dt 0.01)")
ap.add_argument("--iters", type=int, default=50); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--amax", type=float, default=0.5); ap.add_argument("--w-act", type=float, default=0.05)
ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--nprobe", type=int, default=24); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--phases", type=int, default=4); ap.add_argument("--eval-periods", type=float, default=1.0); ap.add_argument("--eval-only", default=None, help="policy .pt to evaluate")
ap.add_argument("--const-action", type=float, nargs=2, default=None, help="evaluation only: open-loop constant (a1, a2) instead of a policy, to separate the steady-forcing part of a result"); ap.add_argument("--init", default=None, help="start from a saved policy .pt"); ap.add_argument("--rolling", action="store_true", help="after each iteration advance the start state by the window under the new policy (train along the controlled trajectory instead of restarting from the uncontrolled limit cycle)"); ap.add_argument("--clip", type=float, default=0.5); ap.add_argument("--znmf", action="store_true", help="zero-net-mass-flux: one amplitude, applied as (+a, -a) on the +-90 deg slots, so steady suction is not available"); ap.add_argument("--state", default="results/uadj_shed_cylinder_butterfly_coarse.npz"); ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default="results/uadj_dpc")
a = ap.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.outdir, exist_ok=True)
tag = a.tag or f"dpc_h{a.H}x{a.sub}_s{a.seed}" + ("_znmf" if a.znmf else "")
# ---- the limit-cycle state into the torch step
d = np.load(a.state); s = cylinder(mesh="meshes/cylinder_butterfly_coarse.msh", Re=100.0, dt=0.01, nsteps=0)
for k in ("u", "v", "p", "Ff", "Ff_prev", "Ff_old", "Fbar_old", "u_old", "v_old"): setattr(s, k, np.array(d[k]))
s.time = float(d["t"]); s.nstep = 20000; s._flux_init = True
h = d["hist"]; t, cl = h[:, 0], h[:, 2]; w = t > t[-1] - 60; tw, cw = t[w], cl[w] - cl[w].mean()
z = np.flatnonzero(np.diff(np.sign(cw)) > 0); tz = tw[z] - cw[z] * (tw[z + 1] - tw[z]) / (cw[z + 1] - cw[z]); period = float(np.diff(tz).mean()); per_steps = int(round(period / 0.01))
T = TorchUPISO(s); W = WallForces(T, s.wall_faces); jets = SlotJets.cylinder(T, s.wall_faces, (90.0, -90.0), 10.0, vmax=1.0); m = s.m
C = m.centroid; near = np.flatnonzero((C[:, 0] > 0.6) & (C[:, 0] < 3.0) & (np.abs(C[:, 1]) < 1.0)); probes = torch.as_tensor(np.sort(np.random.choice(near, a.nprobe, replace=False)))
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(a.nprobe, a.hidden), torch.nn.Tanh(), torch.nn.Linear(a.hidden, 1 if a.znmf else 2))
        with torch.no_grad(): self.net[2].weight *= 0.1; self.net[2].bias.zero_()
    def forward(self, st):
        out = a.amax * torch.tanh(self.net(st["p"][probes]))
        return torch.cat([out, -out]) if a.znmf else out
pol = Policy(); params = list(pol.parameters())
if a.init: pol.load_state_dict(torch.load(a.init)); print(f"policy initialised from {a.init}", flush=True)
apply = lambda st, act: jets.apply(st, act)
def step_loss(st, k): return W(st)[0] + a.w_act * (jets.last_action ** 2).sum() if hasattr(jets, "last_action") else W(st)[0]
# the jet amplitude is not stored on the state; penalise it through the boundary values it writes (ub, vb on the slot faces)
slot = torch.cat([torch.as_tensor(b) for b in jets.bidx])
def step_loss(st, k): return W(st)[0] + a.w_act * ((st["ub"][slot] ** 2 + st["vb"][slot] ** 2).sum() / max(len(slot), 1))
# ---- starting phases and the uncontrolled reference over one window
st0 = T.state_from_solver(); phases = [dict(st0)]
with torch.no_grad():
    st = dict(st0)
    for ph in range(1, a.phases):
        for _ in range(per_steps // a.phases): st = T.step(st)
        phases.append(dict(st))
def rollout(st, n_ctrl, sub, policy=None):
    """closed loop without gradients: returns per-control-step (t, C_D, C_L, a1, a2)"""
    rows = []
    with torch.no_grad():
        for k in range(n_ctrl):
            act = policy(st) if policy is not None else torch.zeros(2)
            st = apply(st, act)
            for _ in range(sub): st = T.step(st)
            cd, clift = W(st); rows.append((k, float(cd), float(clift), float(act[0]), float(act[1])))
    return np.array(rows), st
base_cd = [rollout(dict(p), a.H, a.sub)[0][:, 1].mean() for p in phases]
print(f"{tag}: coarse butterfly {m.ncell} cells, Re 100, period {period:.3f} ({per_steps} steps), window H {a.H} x {a.sub} steps = {a.H*a.sub*0.01:.2f} time units ({a.H*a.sub*0.01/period:.2f} periods); "
      f"{'ZNMF (+a, -a)' if a.znmf else 'independent jets'}; {'rolling start' if a.rolling else 'fixed start'}; lr {a.lr} clip {a.clip}; {a.phases} phases; uncontrolled window C_D by phase {np.round(base_cd, 4).tolist()}; policy {sum(p.numel() for p in params)} parameters, {a.nprobe} pressure probes, |a| <= {a.amax}; {a.threads} threads", flush=True)
# ---- training
if a.eval_only is None and a.const_action is None:
    opt = torch.optim.Adam(params, lr=a.lr); hist = []
    for it in range(a.iters):
        t0 = time.time(); ph = it % a.phases
        for p_ in params: p_.grad = None
        L = replay_policy_grad(T, phases[ph], pol, a.H, apply, step_loss, substeps=a.sub)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip)); opt.step()
        fwd, st_end = rollout(dict(phases[ph]), a.H, a.sub, pol)                  # the window under the UPDATED policy, forward only
        if a.rolling: phases[ph] = st_end; base_cd[ph] = rollout(dict(st_end), a.H, a.sub)[0][:, 1].mean()   # next window starts where this one ended; its uncontrolled reference from the same start
        hist.append((it, ph, L, fwd[:, 1].mean(), base_cd[ph], np.abs(fwd[:, 3:]).mean(), gn, time.time() - t0))
        print(f"  it {it:3d} ph {ph}  window loss {L:9.5f}  mean C_D {fwd[:, 1].mean():.5f} (uncontrolled {base_cd[ph]:.5f}, {(fwd[:, 1].mean()/base_cd[ph]-1)*100:+.2f}%)  |a| {np.abs(fwd[:, 3:]).mean():.3f}  |grad| {gn:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save(pol.state_dict(), f"{a.outdir}/{tag}_policy.pt"); np.save(f"{a.outdir}/{tag}_curve.npy", np.array(hist))
elif a.const_action is not None:
    ca = torch.tensor(a.const_action); pol = lambda st: ca; print(f"open-loop constant action {a.const_action}", flush=True)
else:
    pol.load_state_dict(torch.load(a.eval_only))
# ---- closed-loop evaluation over whole periods from phase 0
n_eval = int(round(a.eval_periods * per_steps / a.sub))
ev_c, _ = rollout(dict(phases[0]), n_eval, a.sub, pol); ev_u, _ = rollout(dict(phases[0]), n_eval, a.sub)
np.savez(f"{a.outdir}/{tag}_eval.npz", controlled=ev_c, uncontrolled=ev_u, period=period, sub=a.sub, H=a.H, probes=probes.numpy())
print(f"RESULT {tag}: closed loop over {a.eval_periods:g} period(s) from phase 0 ({n_eval} control steps): mean C_D {ev_c[:, 1].mean():.5f} vs uncontrolled {ev_u[:, 1].mean():.5f} ({(ev_c[:, 1].mean()/ev_u[:, 1].mean()-1)*100:+.2f}%)  "
      f"C_L rms {ev_c[:, 2].std():.4f} vs {ev_u[:, 2].std():.4f}  mean |a| {np.abs(ev_c[:, 3:]).mean():.3f}  mean (a1, a2) ({ev_c[:, 3].mean():+.3f}, {ev_c[:, 4].mean():+.3f})", flush=True)
