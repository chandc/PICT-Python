"""Synthetic-jet actuation on a no-slip surface of the unstructured solver, after HydroGym's NACA0012
m-AIA environment (properties_run.toml of NACA0012Gust_2D_Re100_AOA40): three jets around the leading
edge, each a slot of given width on the upper or lower surface at a chord fraction, blowing/sucking along
the local surface normal with |V| <= vmax (their max_control 0.03 lattice units = 0.52 U_inf), the
amplitude ramped with a tanh between consecutive actions.

Their jets, located from the STL (section 56 of the record):
    jet 1  x/c 0.084  upper (suction side)   width 0.050 c
    jet 2  x/c 0.000  the nose               width 0.037 c
    jet 3  x/c 0.088  lower (pressure side)  width 0.050 c

The jet velocity is imposed as the Dirichlet value of the wall faces inside each slot; the face flux
follows from the boundary velocity, so blowing adds mass through the wall (balanced at the outlet).
"""
import numpy as np
from src.uops import DIRICHLET

HYDROGYM_NACA_JETS = [(0.084, +1, 0.050), (0.000, 0, 0.037), (0.088, -1, 0.050)]     # (x/c, side, width/c); side +1 upper, -1 lower, 0 nose (both)


class JetSet:
    def __init__(self, mesh, wall_tag, alpha_deg, jets=HYDROGYM_NACA_JETS, vmax=0.52, chord=1.0, le=(0.0, 0.0)):
        m = self.m = mesh
        bt = m.btag[m.bfaces]; self.faces = m.bfaces[bt == wall_tag]; f = self.faces
        a = np.radians(alpha_deg); ch = chord * np.array([np.cos(a), -np.sin(a)])       # chord line, nose-up alpha, LE at `le`
        r = m.fcentre[f] - np.asarray(le); t = (r @ ch) / (ch @ ch); side = np.sign(ch[0] * r[:, 1] - ch[1] * r[:, 0])
        S = m.normal[f]; self.n_out = -S / np.hypot(S[:, 0], S[:, 1])[:, None]         # unit normal pointing OUT of the wall into the fluid
        self.S = S * m.span
        self.slots = []
        le_pt = m.fcentre[f][np.argmin(t)]
        for (xc, sd, w) in jets:
            if sd == 0:                                  # the nose slot: by distance from the leading-edge point (a chord-fraction band wraps the whole nose)
                sel = np.hypot(*(m.fcentre[f] - le_pt).T) <= 0.5 * w
            else:
                sel = (np.abs(t - xc) <= 0.5 * w) & (side == sd)
            self.slots.append(np.flatnonzero(sel))
        self.vmax = float(vmax); self.n = len(jets)
        self.a = np.zeros(self.n); self.a_prev = np.zeros(self.n); self.a_target = np.zeros(self.n)
        self.bidx = m.bface_index[f]

    def describe(self):
        return [f"jet {k+1}: {len(s)} faces, slot length {np.hypot(*self.S[s].T).sum():.4f} c, centre {self.m.fcentre[self.faces[s]].mean(axis=0).round(3).tolist()}" for k, s in enumerate(self.slots)]

    def set_target(self, action):
        """New action in [-1, 1]^n; the amplitude ramps from the previous one (call `ramp` each substep)."""
        self.a_prev = self.a.copy(); self.a_target = np.clip(np.asarray(action, float), -1.0, 1.0)

    def ramp(self, frac, bc_u, bc_v, ramp_frac=0.95):
        """Amplitude at fraction `frac` of the action interval: tanh ramp over the first `ramp_frac` of it
        (HydroGym lbUseControlRamping / lbRampPercentage 0.95), then write the Dirichlet wall values."""
        s = 0.5 * (1 + np.tanh(6.0 * (min(frac / ramp_frac, 1.0) - 0.5))) if ramp_frac > 0 else 1.0
        self.a = self.a_prev + (self.a_target - self.a_prev) * s
        for k, sel in enumerate(self.slots):
            v = self.vmax * self.a[k] * self.n_out[sel]                         # positive = blowing out of the surface
            bc_u.value[self.bidx[sel]] = v[:, 0]; bc_v.value[self.bidx[sel]] = v[:, 1]
            bc_u.kind[self.bidx[sel]] = DIRICHLET; bc_v.kind[self.bidx[sel]] = DIRICHLET

    def mass_flux(self):
        """Net volume flux out of the wall through the jets (per unit span)."""
        tot = 0.0
        for k, sel in enumerate(self.slots): tot += float((self.vmax * self.a[k] * self.n_out[sel] * self.S[sel]).sum())
        return -tot   # S points out of the owner cell, i.e. INTO the wall; flip so blowing is positive
