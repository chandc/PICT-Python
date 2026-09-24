"""Subgrid-scale eddy viscosity for the 2.5D solver (LES plan L3): the velocity-gradient tensor on
(ncell, nz) fields -- in-plane from the cell gradient, spanwise spectral -- the cell filter width
Delta = (V dz)^(1/3), and the closures of `src/sgs.py` (Smagorinsky, WALE) on that tensor. The
closure algebra is the structured code's, applied to 3x3 lists of (ncell, nz) arrays.
"""
import numpy as np
from src.sgs import CS_SMAGORINSKY, CW_WALE, A_PLUS


def velocity_gradient(s, u=None, v=None, w=None):
    """g[i][j] = d u_i / d x_j as (ncell, nz) arrays; x, y from the LSQ cell gradient with the
    solver's boundary values, z spectral."""
    u = s.u if u is None else u; v = s.v if v is None else v; w = s.w if w is None else w
    g = []
    for f, bc in ((u, s.bc_u), (v, s.bc_v), (w, s.bc_w)):
        gp = s.grad(f, s.beff(bc, f))                       # (ncell, 2, nz)
        g.append([gp[:, 0], gp[:, 1], s.ddz(f)])
    return g


def filter_width(s):
    """Delta = (cell volume x dz)^(1/3), (ncell,)."""
    return (s.m.vol * s.Lz / s.nz) ** (1.0 / 3.0)


def strain_rate(g):
    return [[0.5 * (g[i][j] + g[j][i]) for j in range(3)] for i in range(3)]


def strain_magnitude(g):
    S = strain_rate(g)
    return np.sqrt(2.0 * sum(S[i][j] ** 2 for i in range(3) for j in range(3)))


def smagorinsky(s, g=None, cs=CS_SMAGORINSKY, damping=None):
    """nu_t = (C_s Delta)^2 |S|; `damping` = y_plus per cell for van Driest, else none."""
    g = velocity_gradient(s) if g is None else g
    D = filter_width(s)
    if damping is not None: D = D * (1.0 - np.exp(-np.asarray(damping) / A_PLUS))
    return (cs * D)[:, None] ** 2 * strain_magnitude(g)


def wale(s, g=None, cw=CW_WALE):
    """Nicoud & Ducros (1999); nu_t ~ y^3 at a wall, responds to rotation by design (see src/sgs.py)."""
    g = velocity_gradient(s) if g is None else g
    g2 = [[sum(g[i][k] * g[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tr = sum(g2[k][k] for k in range(3))
    Sd = [[0.5 * (g2[i][j] + g2[j][i]) - (tr / 3.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    S = strain_rate(g)
    SS = sum(S[i][j] ** 2 for i in range(3) for j in range(3))
    SdSd = sum(Sd[i][j] ** 2 for i in range(3) for j in range(3))
    num = SdSd ** 1.5; den = SS ** 2.5 + SdSd ** 1.25
    D = filter_width(s)
    return (cw * D)[:, None] ** 2 * np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)


def eddy_viscosity(s, model="wale", **kw):
    if model is None or model == "none": return np.zeros_like(s.u)
    return {"smagorinsky": smagorinsky, "wale": wale}[model](s, **kw)
