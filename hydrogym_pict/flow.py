"""PICT-Python as a HydroGym backend: the rotary cylinder on the butterfly.

Implements HydroGym's `PDEBase` contract (see core_shim) around the
PRODUCTION solver -- `MultiBlockPISO` on the R11-validated butterfly-in-
rectangle grid, with the full production configuration (rotational scheme,
Rhie-Chow, persistent flux; PICT_IMPLICIT_CROSS honoured). This mirrors
HydroGym's own `RotaryCylinder` (firedrake backend): one scalar input, the
rotation rate omega; observations (C_L, C_D); objective C_D; MAX_CONTROL
pi/2; first-order damped actuator with their TAU.

Differences that are physics, not interface: HydroGym's firedrake cylinder
is an unbounded-domain benchmark; ours carries the measured confinement of
Y_HALF (C_D ~ 1.32-1.34, St ~ 0.167-0.170 across the R11-R13 matrix, with
quadratic blockage laws to the open-domain values). The rotating wall means
the traction integral's no-slip assumption no longer holds on the body, so
forces are evaluated with check_wall=False and the spurious viscous-normal
part is reported in info rather than silently folded in.

State handling: copy_state/set_state snapshot exactly what src/checkpoint
calls the full restart state (fields + BDF2 history + Rhie-Chow running
state + clock), so an env reset is EXACT, not approximate.
"""
import os

import numpy as np

from cylinder_rect_bc import U_INF, apply as apply_bc, classify, probe_index
from cylinder_ring_grid import D, ring_rect_domain
from src import checkpoint
from src.forces import surface_force
from src.multiblock import face_slice
from src.piso_multiblock import MultiBlockPISO

from hydrogym_pict.core_shim import ActuatorBase, PDEBase

RE = 100.0

MESHES = {
    # the Stage 6.9 / M2 gate build: ~21k cells, minutes-scale episodes
    "coarse": dict(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                   wake_hold=8.0, wake_ratio=1.10),
    # the R11 campaign build (nz=4): the validated physics, hours-scale
    "production": dict(nz=4),
}


class DampedActuator(ActuatorBase):
    """du/dt = (v - u)/tau, integrated exactly; tau = 0 is instantaneous."""

    def __init__(self, tau=0.0, state=0.0):
        super().__init__(state=state)
        self.tau = tau

    def step(self, u, dt):
        if self.tau <= 0.0:
            self.x = float(u)
        else:
            a = 1.0 - np.exp(-dt / self.tau)
            self.x = self.x + a * (float(u) - self.x)
        return self.x


class RotaryCylinder(PDEBase):
    """Re 100 circular cylinder, rotary control, on the PICT-Python butterfly."""

    DEFAULT_MESH = "coarse"
    DEFAULT_DT = 1e-2
    MAX_CONTROL = 0.5 * np.pi          # matches hydrogym.firedrake.RotaryCylinder
    TAU = 0.0556                       # their actuator time constant

    def __init__(self, **config):
        self._dt = config.get("dt", self.DEFAULT_DT)
        self._tol = config.get("tol", 1e-6)
        self._backend = config.get("backend", "scipy")
        super().__init__(**config)

    # ---------------------------------------------------------------- mesh
    def load_mesh(self, name: str):
        if name not in MESHES:
            raise ValueError(f"mesh must be one of {sorted(MESHES)}, got {name!r}")
        self._mesh_name = name
        d, idx = ring_rect_domain(**MESHES[name])
        self._idx = idx
        return d

    # --------------------------------------------------------------- state
    def initialize_state(self):
        d = self.mesh
        # implicit_cross is REQUIRED physics on the butterfly, not an option:
        # the orthogonal-only pressure solve blows up ~2x/step from the
        # sheared trapezoid corners (the R11 growth signature -- reproduced
        # verbatim by this backend's first gate run when the default was 0).
        self.m = MultiBlockPISO(
            d, U_INF * D / RE, self._dt, 2, self._tol,
            time_scheme="bdf2", scheme="rotational", picard_iters=2,
            rhie_chow=True, persistent_flux=True, ddt_corr=False,
            implicit_cross=bool(int(os.environ.get("PICT_IMPLICIT_CROSS", "1"))),
            linear_backend=self._backend)
        for b in range(len(d.blocks)):
            self.m.u[b][:] = U_INF
            self.m.v[b][:] = 0.0
            self.m.w[b][:] = 0.0
        roles = apply_bc(self.m, d)
        self.body_faces = [(b, f) for (b, f), r in roles.items() if r == "body"]
        # per-face body geometry for the rotary Dirichlet write
        self._body = []
        for b, fid in self.body_faces:
            fs = face_slice(fid)
            blk = d.blocks[b]
            self._body.append((b, fs, blk.x[fs].copy(), blk.y[fs].copy()))
        self._span = float(d.blocks[0].period[2])
        self._q = 0.5 * U_INF ** 2 * D * self._span
        self.t = 0.0
        self._q0 = None

    def init_bcs(self):
        apply_bc(self.m, self.mesh)
        self._apply_rotation(self.actuators[0].state if hasattr(self, "actuators")
                             else 0.0)

    def reset_controls(self):
        self.actuators = [DampedActuator(tau=self.TAU)]

    def reset(self, q0=None, t=0.0):
        self.reset_controls()
        if q0 is not None:
            self.set_state(q0)
        self.t = getattr(self.m, "time", t)

    _STATE_KEYS = ("u", "v", "w", "p", "u_prev", "p_flux", "F_prev")

    def _snap(self, obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: self._snap(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._snap(v) for v in obj)
        return np.array(obj, copy=True)

    @property
    def state(self):
        return {k: self._snap(getattr(self.m, k, None)) for k in self._STATE_KEYS} | \
            {"nstep": self.m.nstep, "time": self.m.time}

    def copy_state(self, deepcopy=True):
        return self.state

    def set_state(self, q):
        for k in self._STATE_KEYS:
            cur, new = getattr(self.m, k, None), q.get(k)
            if new is None:
                continue
            if isinstance(cur, dict):
                for b in cur:
                    if isinstance(cur[b], np.ndarray):
                        cur[b][:] = new[b]
                    else:                      # tuple of per-axis arrays (F_prev)
                        for a, arr in enumerate(cur[b]):
                            arr[:] = new[b][a]
            elif isinstance(cur, (list, tuple)):
                for part, npart in zip(cur, new):
                    if isinstance(part, dict):
                        for b in part:
                            part[b][:] = npart[b]
                    else:
                        part[:] = npart
            elif cur is not None:
                setattr(self.m, k, self._snap(new))
        self.m.nstep, self.m.time = q["nstep"], q["time"]
        self.t = self.m.time

    def save_checkpoint(self, filename: str):
        checkpoint.save(self.m, filename)

    def load_checkpoint(self, filename: str):
        checkpoint.load(self.m, filename)
        self.t = self.m.time

    # ------------------------------------------------------------- control
    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 2                        # (C_L, C_D), their convention

    def _apply_rotation(self, omega: float):
        """Tangential Dirichlet u = omega x r on the body wall (CCW positive)."""
        m = self.m
        for b, fs, x, y in self._body:
            m.u_bc[b][fs] = -omega * y
            m.v_bc[b][fs] = +omega * x
            m.u[b][fs] = -omega * y
            m.v[b][fs] = +omega * x

    def set_control(self, act=None):
        super().set_control(act)
        self._apply_rotation(float(self.actuators[0].state))

    # -------------------------------------------------------- observations
    def compute_forces(self):
        R = surface_force(self.mesh, self.body_faces, self.m.u, self.m.v,
                          self.m.w, self.m.p, self.m.nu, check_wall=False)
        CL = R["total"][1] / self._q
        CD = R["total"][0] / self._q
        self._spurious_normal = R["viscous_normal"][0] / self._q
        return CL, CD

    def get_observations(self):
        return np.array(self.compute_forces(), dtype=float)

    def evaluate_objective(self, q=None):
        _, CD = self.compute_forces()
        return CD

    def render(self, **kwargs):
        pass
