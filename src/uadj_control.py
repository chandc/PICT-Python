"""U6 of the unstructured adjoint plan: force objectives and actuation, differentiable.

Forces. `WallForces` is the one-sided wall traction of `run_ucylinder.py` / `naca_env.forces()`,
transcribed: tangential derivatives vanish at a no-slip wall, so grad u = (u_P / d_n) e_in with
e_in the unit normal into the fluid, the wall pressure is the owner value (Neumann), and
C_D = 2 F_x, C_L = 2 F_y with rho = U = D = 1.

Actuation. Every actuator here writes the Dirichlet VALUES of boundary faces, which are state
entries (`ub`, `vb`) of the torch step, so the action reaches the flow through the same boundary
paths that gate A15 certifies:

    SlotJets       wall slots blowing along the local normal: HydroGym's NACA jets (built from the
                   production `JetSet`, so the slots are the production ones) and the cylinder's
                   +-90 degree jets
    Rotation       rigid rotation of a wall, u = omega x r (the rotary cylinder)
    InletProfile   a multiplicative inlet modulation (the NACA gust)
"""
import numpy as np
import torch

from src.uadj_ops import DT, _t, _ti


class WallForces:
    def __init__(self, T, wall_faces):
        m = T.tm.m
        wall = np.asarray(wall_faces)
        self.wo = _ti(m.owner[wall])
        Sw = m.normal[wall] * m.span
        Aw = np.hypot(Sw[:, 0], Sw[:, 1])
        e_in = -Sw / Aw[:, None]
        self.dn = _t(((m.fcentre[wall] - m.centroid[m.owner[wall]]) * (-e_in)).sum(axis=1))
        self.Sw, self.e = _t(Sw), _t(e_in)
        self.T = T

    def __call__(self, st):
        """(C_D, C_L) as torch scalars."""
        nu = self.T.nu_t
        gus = st["u"][self.wo] / self.dn
        gvs = st["v"][self.wo] / self.dn
        pf = st["p"][self.wo]
        e, S = self.e, self.Sw
        txx = 2 * nu * gus * e[:, 0]
        tyy = 2 * nu * gvs * e[:, 1]
        txy = nu * (gus * e[:, 1] + gvs * e[:, 0])
        Fx = (pf * S[:, 0] - (txx * S[:, 0] + txy * S[:, 1])).sum()
        Fy = (pf * S[:, 1] - (txy * S[:, 0] + tyy * S[:, 1])).sum()
        return 2 * Fx, 2 * Fy

    def numpy(self, s):
        """The production formula on the production solver, for gate F1."""
        wo, dn, e, S = self.wo.numpy(), self.dn.numpy(), self.e.numpy(), self.Sw.numpy()
        nu = s.nu
        gus = s.u[wo] / dn; gvs = s.v[wo] / dn; pf = s.p[wo]
        txx = 2 * nu * gus * e[:, 0]; tyy = 2 * nu * gvs * e[:, 1]; txy = nu * (gus * e[:, 1] + gvs * e[:, 0])
        return (2 * float((pf * S[:, 0] - (txx * S[:, 0] + txy * S[:, 1])).sum()),
                2 * float((pf * S[:, 1] - (txy * S[:, 0] + tyy * S[:, 1])).sum()))


class SlotJets:
    """Jets as Dirichlet wall values: face value = base + vmax * a_k * profile * n_out for the faces
    of slot k. `a` is a (n_jets,) tensor."""

    def __init__(self, T, slots, n_out, bidx, profiles=None, vmax=1.0):
        self.bidx = [_ti(bidx[s]) for s in slots]
        self.dirs = [_t(n_out[s]) for s in slots]
        self.prof = [(_t(np.ones(len(s))) if profiles is None else _t(profiles[k])) for k, s in enumerate(slots)]
        self.vmax = float(vmax)
        self.n = len(slots)

    @classmethod
    def from_jetset(cls, T, js):
        """The production `src.ujets.JetSet` (HydroGym's NACA jets): same faces, same normals."""
        return cls(T, js.slots, js.n_out, js.bidx, vmax=js.vmax)

    @classmethod
    def cylinder(cls, T, wall_faces, centres_deg=(90.0, -90.0), half_width_deg=10.0, vmax=1.0):
        """Cylinder slots centred at the given angles, with a cosine profile across each slot."""
        m = T.tm.m
        wall = np.asarray(wall_faces)
        c = m.fcentre[wall]
        th = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
        S = m.normal[wall]
        n_out = -S / np.hypot(S[:, 0], S[:, 1])[:, None]
        slots, prof = [], []
        for c0 in centres_deg:
            d = (th - c0 + 180.0) % 360.0 - 180.0
            sel = np.flatnonzero(np.abs(d) <= half_width_deg)
            slots.append(sel)
            prof.append(np.cos(0.5 * np.pi * d[sel] / half_width_deg))
        nb_idx = m.bface_index[wall]
        return cls(T, slots, n_out, nb_idx, profiles=prof, vmax=vmax)

    def apply(self, st, a):
        ub, vb = st["ub"], st["vb"]
        for k in range(self.n):
            v = self.vmax * a[k] * self.prof[k][:, None] * self.dirs[k]
            ub = ub.index_copy(0, self.bidx[k], v[:, 0])
            vb = vb.index_copy(0, self.bidx[k], v[:, 1])
        return {**st, "ub": ub, "vb": vb}

    @staticmethod
    def ramp(a_prev, a_target, frac, ramp_frac=0.95):
        """`JetSet.ramp`'s tanh amplitude ramp, differentiable in both endpoints."""
        s = 0.5 * (1 + np.tanh(6.0 * (min(frac / ramp_frac, 1.0) - 0.5))) if ramp_frac > 0 else 1.0
        return a_prev + (a_target - a_prev) * s


class Rotation:
    """Rigid rotation of a wall about `centre`: u = -omega (y - yc), v = omega (x - xc)."""

    def __init__(self, T, wall_faces, centre=(0.0, 0.0)):
        m = T.tm.m
        wall = np.asarray(wall_faces)
        r = m.fcentre[wall] - np.asarray(centre)
        self.bidx = _ti(m.bface_index[wall])
        self.tx, self.ty = _t(-r[:, 1]), _t(r[:, 0])

    def apply(self, st, omega):
        return {**st, "ub": st["ub"].index_copy(0, self.bidx, omega * self.tx),
                "vb": st["vb"].index_copy(0, self.bidx, omega * self.ty)}


class InletProfile:
    """Inlet u = base * g on the inlet faces: the NACA gust multiplies the uniform inflow."""

    def __init__(self, T, inlet_slots):
        self.bidx = _ti(inlet_slots)

    def apply(self, st, g):
        return {**st, "ub": st["ub"].index_copy(0, self.bidx, g * torch.ones(len(self.bidx), dtype=DT))}
