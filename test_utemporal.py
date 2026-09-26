"""T4: temporal order of the time scheme alone (unstructured). Decaying Taylor-Green on the fully
periodic unit box, u = TGV(x,y) exp(-2 K^2 nu t), an exact Navier-Stokes solution (the nonlinear
term is balanced by the pressure). nu = 0.01, K = 2 pi, T = 1 (decay to 0.45). The mesh is fixed
(n x n quads) and dt is refined; the error is measured two ways: against the exact solution (includes
the spatial floor) and against the same scheme at dt_ref = dt_min/4 (temporal error only). Both
time schemes: BDF2-PISO (n_inner=2) and the Le-Moin RK3/CN fractional step (LES plan L1a).
    python test_utemporal.py [n] [scheme ...]      e.g.  python test_utemporal.py 64 bdf2 rk3
"""
import sys, io, contextlib, time, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.upiso import PISO, BC
K, NU, T = 2 * np.pi, 0.01, 1.0
LAM = 2 * K ** 2 * NU
def tgv(x, y): return np.sin(K * x) * np.cos(K * y), -np.cos(K * x) * np.sin(K * y)
def run(n, dt, scheme):
    m = rect_mesh(n, n, 0, 1, 0, 1, cells="quad"); m.make_periodic(1, 2, (1.0, 0.0)); m.make_periodic(3, 4, (0.0, 1.0))
    s = PISO(m, nu=NU, dt=dt, bc_u=BC(m), bc_v=BC(m), bc_p=BC(m), n_corr=2, n_nonorth=1, scheme="central")
    s.time_scheme = scheme; x, y = m.centroid.T; s.u[:], s.v[:] = tgv(x, y)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(int(round(T / dt))): s.step()
    ue, ve = tgv(x, y); d = np.exp(-LAM * T)
    err = np.sqrt(float((m.vol * ((s.u - d * ue) ** 2 + (s.v - d * ve) ** 2)).sum()))
    return s.u.copy(), s.v.copy(), m.vol, err, time.time() - t0
if __name__ == "__main__":
    args = sys.argv[1:]; n = int(args[0]) if args and args[0].isdigit() else 64
    schemes = [a for a in args if not a.isdigit()] or ["bdf2", "rk3"]
    dts = [0.04, 0.02, 0.01, 0.005, 0.0025]
    print(f"T4 decaying Taylor-Green, nu={NU}, K=2pi, T={T} (exp(-lam T) = {np.exp(-LAM*T):.3f}), {n}x{n} quads, fully periodic")
    for sc in schemes:
        uref, vref, vol, eref, _ = run(n, dts[-1] / 4, sc)
        print(f"  {sc}: reference dt={dts[-1]/4:g}  exact-error {eref:.3e} (the spatial floor)")
        print(f"   {'dt':>7} {'steps':>5} {'err vs exact':>13} {'order':>6} {'err vs dt_ref':>14} {'order':>6} {'s':>6}")
        pe = pr = None
        for dt in dts:
            u, v, _, e, wt = run(n, dt, sc); er = np.sqrt(float((vol * ((u - uref) ** 2 + (v - vref) ** 2)).sum()))
            oe = f"{np.log2(pe/e):6.2f}" if pe else "    --"; orr = f"{np.log2(pr/er):6.2f}" if pr else "    --"
            print(f"   {dt:7g} {int(round(T/dt)):5d} {e:13.3e} {oe} {er:14.3e} {orr} {wt:6.1f}", flush=True); pe, pr = e, er
