"""Inviscid kinetic-energy conservation on the unstructured code -- port of test_energy_conservation.py.

Fully periodic unit box (both seams via Mesh.make_periodic), nu = 0. Two measurements:
  1. OPERATOR: production P = sum_c u_c . (C u_c) of the assembled convection operator (implicit
     upwind part + deferred central/skewness correction, exactly what the solver applies) on a
     solver-consistent field: the analytic field after ONE tiny PISO step, so the face flux is
     discretely divergence-free. Reported as eps = P/E per unit time, and per turnover (1/max|u|).
     Central on quads with w = 1/2 is skew-symmetric by algebra (sum F (u_O^2 - u_N^2)/2 = 0 when
     div F = 0), so eps must be at round-off; on triangles the skewness correction breaks the
     symmetry and eps measures by how much. Upwind is dissipative by construction.
  2. SOLVER: E(T)/E(0) through full inviscid steps, T = 0.1 at dt = 0.005. The 2D Taylor-Green
     vortex is an EXACT STEADY solution of the Euler equations (u.grad u = -grad p), so any change
     in E is numerical dissipation: Rhie-Chow, time integration, and the operator together.
Fields: Taylor-Green (steady) and a random solenoidal field from a stream function (many modes).
   python test_uenergy.py"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import convection
from src.upiso import PISO, BC
K = 2 * np.pi
def tgv(x, y): return np.sin(K * x) * np.cos(K * y), -np.cos(K * x) * np.sin(K * y)
def solenoidal(x, y, seed=0, kmax=3):
    rng = np.random.default_rng(seed); psi_x = np.zeros_like(x); psi_y = np.zeros_like(x)
    for mx in range(1, kmax + 1):
        for my in range(1, kmax + 1):
            a = rng.uniform(-1, 1) / (mx * mx + my * my); phx, phy = rng.uniform(0, 2 * np.pi, 2)
            psi_x += a * K * mx * np.cos(K * mx * x + phx) * np.sin(K * my * y + phy)
            psi_y += a * K * my * np.sin(K * mx * x + phx) * np.cos(K * my * y + phy)
    u, v = psi_y, -psi_x; s = max(np.abs(u).max(), np.abs(v).max()); return u / s, v / s
def build(n, cells, scheme, dt):
    m = rect_mesh(n, n, 0, 1, 0, 1, cells=cells); m.make_periodic(1, 2, (1.0, 0.0)); m.make_periodic(3, 4, (0.0, 1.0))
    s = PISO(m, nu=0.0, dt=dt, bc_u=BC(m), bc_v=BC(m), bc_p=BC(m), n_corr=2, n_nonorth=1, scheme=scheme)
    return m, s
def operator_eps(n, cells, field, scheme):
    m, s = build(n, cells, scheme, 1e-4); x, y = m.centroid.T; u, v = field(x, y); s.u[:] = u; s.v[:] = v
    with contextlib.redirect_stdout(io.StringIO()): s.step()
    u, v, F = s.u, s.v, s.Ff
    C, C_rhs = convection(m, F, s.bc_u.kind, scheme=scheme); pb = np.zeros(0)
    P = sum(float(f @ (C @ f + C_rhs(f, pb, s.grad(f, pb)))) for f in (u, v))
    E = 0.5 * float((m.vol * (u * u + v * v)).sum()); um = float(np.hypot(u, v).max())
    from src.uops import divergence
    return P / E, float(np.abs(divergence(m, F) / m.vol).max()), um
def solver_ratio(n, cells, scheme, T=0.1, dt=0.005):
    m, s = build(n, cells, scheme, dt); x, y = m.centroid.T; u, v = tgv(x, y); s.u[:] = u; s.v[:] = v
    E0 = 0.5 * float((m.vol * (u * u + v * v)).sum())
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(int(round(T / dt))): s.step()
    E1 = 0.5 * float((m.vol * (s.u ** 2 + s.v ** 2)).sum()); return E1 / E0, (E1 / E0 - 1) / T
if __name__ == "__main__":
    print("Inviscid energy conservation, unstructured PISO, fully periodic unit box, nu = 0")
    for fname, field in (("Taylor-Green", tgv), ("random solenoidal", solenoidal)):
        print(f"\n1. OPERATOR, {fname}: eps = P/E per turnover (1/max|u|); 0 = conserving, > 0 = dissipative")
        print(f"   {'cells':>5} {'n':>4} {'max|div F|/V':>12} | {'central':>12} | {'upwind':>12}")
        for cells in ("quad", "tri"):
            for n in (16, 32, 64):
                out = {}
                for scheme in ("central", "upwind"):
                    eps, dv, um = operator_eps(n, cells, field, scheme); out[scheme] = eps / um
                print(f"   {cells:>5} {n:4d} {dv:12.2e} | {out['central']:+12.3e} | {out['upwind']:+12.3e}")
    print("\n2. FULL INVISCID STEPS, Taylor-Green (exact steady Euler solution): E(T)/E(0), T=0.1, dt=0.005")
    print(f"   {'cells':>5} {'n':>4} | {'central: E(T)/E(0)':>18} {'rate/turnover':>14} | {'upwind: E(T)/E(0)':>18} {'rate/turnover':>14}")
    for cells in ("quad", "tri"):
        for n in (16, 32, 64):
            rc, lc = solver_ratio(n, cells, "central"); ru, lu = solver_ratio(n, cells, "upwind")
            print(f"   {cells:>5} {n:4d} | {rc:18.6f} {lc:+14.3e} | {ru:18.6f} {lu:+14.3e}")
