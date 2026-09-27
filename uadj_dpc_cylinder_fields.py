"""Extract instantaneous u, v, p fields for the DPC cylinder comparison: the uncontrolled limit cycle
(as stored) and the lift-penalised ZNMF policy's closed loop from it, rolled forward to its steady
controlled state. Used by plot_utility/plot_uadj_dpc_streamlines.py.
    python uadj_dpc_cylinder_fields.py --policy results/uadj_dpc/dpc_h240x5_znmf_roll_wa05_wl1_policy.pt --periods 6"""
import sys, os, argparse, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.uadj_cases import cylinder
from src.uadj_step import TorchUPISO
from src.uadj_control import WallForces, SlotJets
ap = argparse.ArgumentParser()
ap.add_argument("--state", default="results/uadj_shed_cylinder_butterfly_coarse.npz")
ap.add_argument("--policy", default="results/uadj_dpc/dpc_h240x5_znmf_roll_wa05_wl1_policy.pt")
ap.add_argument("--periods", type=float, default=6.0); ap.add_argument("--sub", type=int, default=5); ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--nprobe", type=int, default=24)
ap.add_argument("--amax", type=float, default=0.5); ap.add_argument("--out", default="results/uadj_dpc/cylinder_fields.npz"); ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(a.threads)
d = np.load(a.state); s = cylinder(mesh="meshes/cylinder_butterfly_coarse.msh", Re=100.0, dt=0.01, nsteps=0)
for k in ("u", "v", "p", "Ff", "Ff_prev", "Ff_old", "Fbar_old", "u_old", "v_old"): setattr(s, k, np.array(d[k]))
s.time = float(d["t"]); s.nstep = 20000; s._flux_init = True
h = d["hist"]; t, cl = h[:, 0], h[:, 2]; w = t > t[-1] - 60; tw, cw = t[w], cl[w] - cl[w].mean()
z = np.flatnonzero(np.diff(np.sign(cw)) > 0); tz = tw[z] - cw[z] * (tw[z + 1] - tw[z]) / (cw[z + 1] - cw[z]); period = float(np.diff(tz).mean()); per_steps = int(round(period / 0.01))
T = TorchUPISO(s); W = WallForces(T, s.wall_faces); jets = SlotJets.cylinder(T, s.wall_faces, (90.0, -90.0), 10.0, vmax=1.0); m = s.m
np.random.seed(0); C = m.centroid; near = np.flatnonzero((C[:, 0] > 0.6) & (C[:, 0] < 3.0) & (np.abs(C[:, 1]) < 1.0)); probes = torch.as_tensor(np.sort(np.random.choice(near, a.nprobe, replace=False)))
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(a.nprobe, a.hidden), torch.nn.Tanh(), torch.nn.Linear(a.hidden, 1))
    def forward(self, st):
        out = a.amax * torch.tanh(self.net(st["p"][probes])); return torch.cat([out, -out])
pol = Policy(); pol.load_state_dict(torch.load(a.policy)); apply = lambda st, act: jets.apply(st, act)
st0 = T.state_from_solver()
def field(st): return dict(u=st["u"].detach().numpy().copy(), v=st["v"].detach().numpy().copy(), p=st["p"].detach().numpy().copy())
# uncontrolled: the stored limit-cycle state, and a quarter-period-advanced snapshot for the mid-shed streamline picture
with torch.no_grad():
    st_u = dict(st0)
    for _ in range(per_steps // 4): st_u = T.step(st_u)
unc = field(st_u)
# controlled: roll the policy forward `periods` shedding periods from the limit cycle
n_ctrl = int(round(a.periods * per_steps / a.sub))
with torch.no_grad():
    st_c = dict(st0); actions = []
    for k in range(n_ctrl):
        act = pol(st_c); actions.append(act.tolist()); st_c = apply(st_c, act)
        for _ in range(a.sub): st_c = T.step(st_c)
con = field(st_c)
ut, vt = float(W(st_u)[0]), float(W(st_u)[1]); ct, cl_ = float(W(st_c)[0]), float(W(st_c)[1])
print(f"uncontrolled snapshot: C_D {ut:.4f} C_L {vt:+.4f}   controlled after {a.periods:g} periods: C_D {ct:.4f} C_L {cl_:+.4f}  action {actions[-1]}")
np.savez(a.out, centroid=m.centroid, nodes=m.nodes, cells=m.cells, nvert=m.nvert, btag=m.btag, bfaces=m.bfaces,
         u_unc=unc["u"], v_unc=unc["v"], p_unc=unc["p"], cd_unc=ut, cl_unc=vt,
         u_con=con["u"], v_con=con["v"], p_con=con["p"], cd_con=ct, cl_con=cl_, action_con=actions[-1], period=period)
print("wrote", a.out)
