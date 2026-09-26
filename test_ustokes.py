"""T5: steady Stokes, manufactured solution. Isolates the pressure-velocity coupling.

No convection, so a failure here is the coupling, the Laplacian, or the projection -- never the
convection scheme. If this passes and Poiseuille or the cavity fails, the fault is convection;
if this fails, nothing downstream is worth debugging.

    u =  sin(pi x) cos(pi y)      div u = 0 exactly
    v = -cos(pi x) sin(pi y)
    p =  cos(pi x) cos(pi y)      zero mean on the unit square, so the singular
                                  all-Neumann pressure problem is well posed
    f = grad p - nu lap u
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC

PI = np.pi
uex = lambda p: np.sin(PI*p[:,0])*np.cos(PI*p[:,1])
vex = lambda p: -np.cos(PI*p[:,0])*np.sin(PI*p[:,1])
pex = lambda p: np.cos(PI*p[:,0])*np.cos(PI*p[:,1])

def forcing(p, nu):
    x, y = p[:,0], p[:,1]
    dpdx = -PI*np.sin(PI*x)*np.cos(PI*y)
    dpdy = -PI*np.cos(PI*x)*np.sin(PI*y)
    lapu = -2*PI**2*uex(p)
    lapv = -2*PI**2*vex(p)
    return dpdx - nu*lapu, dpdy - nu*lapv

def run(n, perturb, nu=1.0, dt=0.005, cap=20000):
    """Run to STAGNATION of the error, not a fixed step count.

    Two lessons from the record (reference/skew_unstructured_literature.md): a fixed 400 steps
    reported "order 1.1" that was an unconverged transient over a flat floor; and dt = 0.005
    reaches the steady state in ~700 steps where dt = 0.05 needs ~3800, because the pseudo-time
    march is a pressure-coupling iteration whose contraction improves as dt shrinks.
    """
    m = rect_mesh(n, n, perturb=perturb, seed=3)
    fb = m.fcentre[m.bfaces]
    bu = BC(m, np.full(m.nbface, DIRICHLET), uex(fb))
    bv = BC(m, np.full(m.nbface, DIRICHLET), vex(fb))
    bp = BC(m, np.full(m.nbface, NEUMANN))
    fx, fy = forcing(m.centroid, nu)
    s = PISO(m, nu=nu, dt=dt, bc_u=bu, bc_v=bv, bc_p=bp,
             n_corr=2, n_nonorth=6, body_force=(fx, fy), convect=False)
    prev = None
    for k in range(1, cap + 1):
        s.step()
        if k % 100 == 0:
            e = float(np.sqrt((((s.u - uex(m.centroid))**2) * m.vol).sum() / m.vol.sum()))
            if prev is not None and abs(e - prev) < 1e-11:
                break
            prev = e
    eu, ev = s.u - uex(m.centroid), s.v - vex(m.centroid)
    pp = s.p - (s.p*m.vol).sum()/m.vol.sum()
    pe = pex(m.centroid) - (pex(m.centroid)*m.vol).sum()/m.vol.sum()
    L2 = lambda e: float(np.sqrt((e**2*m.vol).sum()/m.vol.sum()))
    return L2(eu), L2(ev), L2(pp-pe), m.ncell

if __name__ == "__main__":
    for perturb in (0.0, 0.25):
        print(f"\n  perturb = {perturb}")
        print(f"   {'n':>4} {'cells':>7}   {'L2(u)':>11} {'ord':>5}   {'L2(p)':>11} {'ord':>5}")
        pu = pp_ = None
        for n in (16, 32, 64):
            lu, lv, lp, nc = run(n, perturb)
            ou = "" if pu is None else f"{np.log2(pu/lu):5.2f}"
            op = "" if pp_ is None else f"{np.log2(pp_/lp):5.2f}"
            print(f"   {n:4d} {nc:7d}   {lu:.4e} {ou:>5s}   {lp:.4e} {op:>5s}")
            pu, pp_ = lu, lp
