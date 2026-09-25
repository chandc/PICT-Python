"""Gymnasium environment: NACA0012 at Re 100, alpha 40 deg, three leading-edge synthetic jets, on the
unstructured collocated PISO solver -- HydroGym's NACA0012Gust_2D_Re100_AOA40 task rebuilt on our solver
(their m-AIA backend is closed and amd64-only).

Task (their environment_config.yaml / properties_run.toml, decoded in section 56 of the record):
  action  a in [-1, 1]^3, jet velocity a * 0.52 U_inf along the surface normal, tanh-ramped over 95% of the
          action interval; one action per 0.54 convective times (their 75 LBM steps)
  obs     (u, v) at a probe one chord upstream of the leading edge (their pp_probeCoordinates [-1, 0]);
          --obs forces adds (C_D, C_L)
  reward  gust task: -|C_L - C_L0| - 0.25 |C_D - C_D0| with (C_D0, C_L0) the unperturbed means of THIS
          solver on THIS mesh; L/D task: C_L / C_D (-100 if C_L < 0 or C_D ~ 0)
  gust    inlet velocity U_inf (1 + (F - 1) exp(-(y/c)^2) sin^2(pi t/T_g)) with F = 2 for T_g = 28.9
          convective times from the start of the episode (their lbGustFactor 2, lbGustRanges 16 cells = 2c,
          lbGustDuration 4000 LBM steps); 200 actions per episode (108 convective times)
The episode starts from a snapshot of the developed shedding state (a random one of the stored phases).
"""
import sys, os, numpy as np, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
from src.ujets import JetSet


class NACAJetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mesh="meshes/naca0012_a40_coarse.msh", snapshots="results/naca/a40_coarse_snapshots.npz", Re=100.0, alpha=40.0,
                 dt=0.01, action_interval=0.54, max_episode_steps=200, task="gust", gust_factor=2.0, gust_duration=28.9,
                 obs="probe", vmax=0.52, seed=None, log_path=None):
        super().__init__()
        nodes, cells, ctag, edges, etag, names = read_gmsh22(mesh); m = self.m = Mesh(nodes, cells, edges, etag, names)
        inv = {v: k for k, v in names.items()}; self.T_IN, self.T_OUT, self.T_W = inv["Inlet"], inv["Outlet"], inv["Airfoil"]
        bt = m.btag[m.bfaces]; nb = m.nbface; self.nu = 1.0 / Re; self.dt = dt
        self.ku = np.where(np.isin(bt, [self.T_IN, self.T_W]), DIRICHLET, NEUMANN); self.kv = self.ku.copy(); self.kp = np.where(bt == self.T_OUT, DIRICHLET, NEUMANN)
        self.inlet = np.flatnonzero(bt == self.T_IN); self.y_in = m.fcentre[m.bfaces[self.inlet], 1]
        self.jets = JetSet(m, self.T_W, alpha, vmax=vmax)
        wall = m.bfaces[bt == self.T_W]; self.wo = m.owner[wall]; self.Sw = m.normal[wall] * m.span
        Aw = np.hypot(*self.Sw.T); self.e_in = -self.Sw / Aw[:, None]; self.dn = ((m.fcentre[wall] - m.centroid[self.wo]) * (-self.e_in)).sum(axis=1)
        self.probe = int(np.argmin(np.hypot(m.centroid[:, 0] + 1.0, m.centroid[:, 1])))          # one chord upstream of the LE (origin)
        self.n_sub = int(round(action_interval / dt)); self.max_steps = max_episode_steps; self.task = task
        self.gust_factor, self.gust_T = gust_factor, gust_duration
        self.obs_mode = obs
        d = np.load(snapshots); self.snaps = d; self.cd0, self.cl0 = float(d["cd0"]), float(d["cl0"])
        nobs = 2 + (2 if obs == "forces" else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
        self.rng = np.random.default_rng(seed); self.log_path = log_path; self.s = None; self.episode = 0

    # ------------------------------------------------------------------ solver
    def _new_solver(self):
        m = self.m; nb = m.nbface
        s = PISO(m, nu=self.nu, dt=self.dt, bc_u=BC(m, self.ku.copy(), np.where(m.btag[m.bfaces] == self.T_IN, 1.0, 0.0)), bc_v=BC(m, self.kv.copy(), np.zeros(nb)),
                 bc_p=BC(m, self.kp, np.zeros(nb)), n_corr=2, n_nonorth=3, scheme="central")
        return s

    def forces(self):
        s = self.s; gus = s.u[self.wo] / self.dn; gvs = s.v[self.wo] / self.dn; pf = s.p[self.wo]; nu = self.nu; e = self.e_in; S = self.Sw
        txx = 2 * nu * gus * e[:, 0]; tyy = 2 * nu * gvs * e[:, 1]; txy = nu * (gus * e[:, 1] + gvs * e[:, 0])
        return 2 * float((pf * S[:, 0] - (txx * S[:, 0] + txy * S[:, 1])).sum()), 2 * float((pf * S[:, 1] - (txy * S[:, 0] + tyy * S[:, 1])).sum())

    def _gust(self, t):
        if self.task != "gust" or t >= self.gust_T: return 1.0 + 0.0 * self.y_in
        return 1.0 + (self.gust_factor - 1.0) * np.exp(-self.y_in ** 2) * np.sin(np.pi * t / self.gust_T) ** 2

    def _obs(self):
        o = [self.s.u[self.probe], self.s.v[self.probe]]
        if self.obs_mode == "forces": o += list(self.last_forces)
        return np.array(o, dtype=np.float64)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.s = self._new_solver(); k = int(self.rng.integers(self.snaps["u"].shape[0]))
        self.s.u[:] = self.snaps["u"][k]; self.s.v[:] = self.snaps["v"][k]; self.s.p[:] = self.snaps["p"][k]
        self.jets = JetSet(self.m, self.T_W, 40.0, vmax=self.jets.vmax); self.jets.ramp(1.0, self.s.bc_u, self.s.bc_v)
        self.t = 0.0; self.iter = 0; self.episode += 1; self.last_forces = self.forces(); self.ep_reward = 0.0; self.ep_actions = []
        return self._obs(), {}

    def step(self, action):
        self.jets.set_target(action); s = self.s; cd_acc = cl_acc = 0.0
        for k in range(self.n_sub):
            self.jets.ramp((k + 1) / self.n_sub, s.bc_u, s.bc_v)
            s.bc_u.value[self.inlet] = self._gust(self.t)
            s.step(); self.t += self.dt
            cd, cl = self.forces(); cd_acc += cd; cl_acc += cl
        cd, cl = cd_acc / self.n_sub, cl_acc / self.n_sub; self.last_forces = (cd, cl); self.iter += 1
        if not np.isfinite(s.u).all(): return self._obs() * 0, -100.0, True, False, {"diverged": True}
        if self.task == "gust": reward = -abs(cl - self.cl0) - 0.25 * abs(cd - self.cd0)
        else: reward = -100.0 if (cd < 1e-6 or cl < 0) else cl / cd
        self.ep_reward += reward; self.ep_actions.append(np.asarray(action, float))
        done = self.iter >= self.max_steps
        info = {"cd": cd, "cl": cl, "t": self.t, "jet_flux": self.jets.mass_flux()}
        if done and self.log_path:
            A = np.array(self.ep_actions)
            with open(self.log_path, "a") as f: f.write(f"{self.episode} {self.ep_reward:.4f} {cd:.4f} {cl:.4f} {A.mean(axis=0).round(3).tolist()} {np.abs(A).mean():.3f}\n")
        return self._obs(), float(reward), done, False, info
