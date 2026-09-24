"""T6: plane Poiseuille on the unstructured code -- walls in y, PERIODIC in x (make_periodic), body
force. Port of test_poiseuille.py. Case A: constant force -> parabola (any error is the wall
treatment: the wall face flux is the half-cell one-sided stencil, first order, so this is a defect
detector, not a rate). Case B: force nu pi^2 sin(pi y) -> sin(pi y), non-polynomial, so the rate is
meaningful. Quads, so no checkerboard is in play.  python test_upoiseuille.py"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
NU, NX, DT = 0.1, 4, 0.5
def run(ny, case, max_steps=4000, tol=1e-11):
    m = rect_mesh(NX, ny, 0, 1, 0, 1, cells="quad"); m.make_periodic(1, 2, (1.0, 0.0))
    y = m.centroid[:, 1]
    if case == "parabola":
        G = 8 * NU; f = np.full(m.ncell, G); exact = G / (2 * NU) * y * (1 - y)
    else:
        f = NU * np.pi**2 * np.sin(np.pi * y); exact = np.sin(np.pi * y)
    nb = m.nbface; kd = np.full(nb, DIRICHLET); kn = np.full(nb, NEUMANN)
    s = PISO(m, nu=NU, dt=DT, bc_u=BC(m, kd, np.zeros(nb)), bc_v=BC(m, kd, np.zeros(nb)), bc_p=BC(m, kn, np.zeros(nb)),
             n_corr=2, n_nonorth=1, scheme="central", body_force=(f, np.zeros(m.ncell)))
    prev = s.u.copy(); t0 = time.time()
    for it in range(max_steps):
        s.step()
        if np.abs(s.u - prev).max() < tol: break
        prev = s.u.copy()
    err = np.sqrt(((s.u - exact) ** 2).mean())
    return err, float(np.abs(s.u).max()), float(np.abs(exact).max()), it + 1, float(np.abs(s.v).max()), time.time() - t0
print(f"T6 plane Poiseuille, unstructured quads, walls in y, periodic in x ({NX} cells), nu={NU}, dt={DT}")
print("A. constant forcing -> parabola")
for ny in (8, 16, 32):
    e, um, ue, its, vm, wt = run(ny, "parabola"); print(f"   ny={ny:3d}  L2 err {e:.3e}  u_max {um:.6f} (exact {ue:.6f})  |v|max {vm:.1e}  {its} steps {wt:.0f}s")
print("B. forcing nu pi^2 sin(pi y) -> sin(pi y)")
errs = []
for ny in (8, 16, 32, 64):
    e, um, ue, its, vm, wt = run(ny, "sine"); errs.append(e); print(f"   ny={ny:3d}  L2 err {e:.3e}  u_max {um:.6f} (exact {ue:.6f})  |v|max {vm:.1e}  {its} steps {wt:.0f}s")
rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
print("   convergence rates:", ", ".join(f"{r:.2f}" for r in rates))
