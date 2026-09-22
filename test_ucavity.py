"""T7: lid-driven cavity at Re = 1000, against Ghia, Ghia & Shin (1982).

Unit square, lid at y=1 with u=1, no-slip elsewhere, nu = 1/Re. Reference profiles come from
src/ghia.py -- named U_RE1000 / V_RE1000 rather than bare, because the source repo they were
transcribed from holds two different arrays under each bare name (Re=100 and Re=1000) and its
own notes flag the hazard.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
import src.ghia as G


def sample_line(m, phi, const_coord, axis, npts=400):
    """Sample a cell field along a line by inverse-distance weighting of nearby cells."""
    t = np.linspace(0.0, 1.0, npts)
    pts = np.stack([np.full(npts, const_coord), t], 1) if axis == 0 else \
          np.stack([t, np.full(npts, const_coord)], 1)
    out = np.empty(npts)
    for k, P in enumerate(pts):
        d2 = ((m.centroid - P) ** 2).sum(axis=1)
        j = np.argpartition(d2, 6)[:6]
        w = 1.0 / np.maximum(d2[j], 1e-14)
        out[k] = float((w * phi[j]).sum() / w.sum())
    return t, out


def run(n=64, Re=1000.0, dt=0.005, t_end=40.0, perturb=0.0, report=500, cluster=0.0):
    m = rect_mesh(n, n, perturb=perturb, cluster=cluster)
    tag = m.btag[m.bfaces]
    ku = np.full(m.nbface, DIRICHLET); vu = np.zeros(m.nbface); vu[tag == 4] = 1.0
    s = PISO(m, nu=1.0 / Re, dt=dt, bc_u=BC(m, ku, vu),
             bc_v=BC(m, np.full(m.nbface, DIRICHLET)),
             bc_p=BC(m, np.full(m.nbface, NEUMANN)),
             n_corr=2, n_nonorth=3, scheme="central")
    nstep = int(t_end / dt); t0 = time.time(); prev = s.u.copy()
    for k in range(nstep):
        s.step()
        if (k + 1) % report == 0:
            ch = float(np.abs(s.u - prev).max()) / max(float(np.abs(s.u).max()), 1e-30)
            print(f"    t={s.time:6.2f}  |u|max {np.abs(s.u).max():.4f}  "
                  f"rel change/{report} steps {ch:.2e}  ({time.time()-t0:.0f}s)", flush=True)
            prev = s.u.copy()
    return m, s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print(f"  Ghia check: {G.check() or 'CLEAN'}")
    m, s = run(n=n)
    y, u = sample_line(m, s.u, 0.5, axis=0)
    x, v = sample_line(m, s.v, 0.5, axis=1)
    eu = np.interp(G.Y_RE1000[::-1], y, u)[::-1] - G.U_RE1000
    ev = np.interp(G.X_RE1000[::-1], x, v)[::-1] - G.V_RE1000
    print(f"\n  vs Ghia Re=1000 ({m.ncell} cells)")
    print(f"    u on x=0.5 : max|err| {np.abs(eu).max():.4f}  rms {np.sqrt((eu**2).mean()):.4f}")
    print(f"    v on y=0.5 : max|err| {np.abs(ev).max():.4f}  rms {np.sqrt((ev**2).mean()):.4f}")
    print(f"    u min {u.min():+.4f} (Ghia {G.U_MIN_RE1000:+.4f})   "
          f"v min {v.min():+.4f} (Ghia {G.V_MIN_RE1000:+.4f})   "
          f"v max {v.max():+.4f} (Ghia {G.V_MAX_RE1000:+.4f})")
    np.savez("results/ucavity_re1000.npz", y=y, u=u, x=x, v=v, ncell=m.ncell)
    print("    saved results/ucavity_re1000.npz")
