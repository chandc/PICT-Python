"""T8: Orr-Sommerfeld growth at Re = 7500, alpha = 1, on the unstructured code. Plane Poiseuille
U = 1 - y^2 on y in [-1, 1], periodic in x over [0, 2 pi] (make_periodic), no-slip walls, body force
f_x = 2 nu. The least-stable OS eigenmode (orr_sommerfeld.least_stable) is seeded at 1e-4 and must
GROW at alpha Im(c) = 0.00223497 with phase speed 0.24989154 (Streett / Chan). Measurement as in
test_orr_sommerfeld.py: the first streamwise Fourier mode of the perturbation, a(t) = <A(t),A(0)>/
<A(0),A(0)>; growth = d ln|a|/dt, phase = -d arg(a)/dt / alpha, fitted over t in [30, T].
   python test_uorr_sommerfeld.py [ny ...]     (cells across the channel; default 100 200 400)"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
from orr_sommerfeld import least_stable, cheb
from scipy.interpolate import BarycentricInterpolator
import os
RE, ALPHA, AMP, NX, DT, T = 7500.0, 1.0, 1e-4, int(os.environ.get("OS_NX", 48)), float(os.environ.get("OS_DT", 0.05)), 100.0   # OS_NX / OS_DT override the streamwise cells and the time step
NU = 1.0 / RE; G_REF, C_REF = 0.00223497, 0.24989154; LX = 2 * np.pi / ALPHA
def build(ny):
    m = rect_mesh(NX, ny, 0, LX, -1, 1, cells="quad"); m.make_periodic(1, 2, (LX, 0.0))
    x, y = m.centroid.T
    nb = m.nbface; kd = np.full(nb, DIRICHLET); kn = np.full(nb, NEUMANN)
    s = PISO(m, nu=NU, dt=DT, bc_u=BC(m, kd, np.zeros(nb)), bc_v=BC(m, kd, np.zeros(nb)), bc_p=BC(m, kn, np.zeros(nb)),
             n_corr=2, n_nonorth=1, scheme="central", body_force=(np.full(m.ncell, 2 * NU), np.zeros(m.ncell)))
    s.conv_flux_extrap = os.environ.get("OS_FLUXEXTRAP", "1") == "1"     # default on; OS_FLUXEXTRAP=0 reproduces the lagged-flux (first-order-in-time) runs
    c, phi, yc = least_stable(RE, ALPHA, 120)
    D, _ = cheb(len(yc) - 1)
    Pphi = BarycentricInterpolator(yc, phi); Pdphi = BarycentricInterpolator(yc, D @ phi)
    ph = np.exp(1j * ALPHA * x)
    up = np.real(Pdphi(y) * ph); vp = np.real(-1j * ALPHA * Pphi(y) * ph)
    scale = AMP / max(np.abs(up).max(), np.abs(vp).max())
    s.u[:] = 1 - y**2 + scale * up; s.v[:] = scale * vp
    # (i, j) index of every cell from its centroid, for the streamwise Fourier transform
    i = np.floor(x / (LX / NX)).astype(int); j = np.floor((y + 1) / (2 / ny)).astype(int)
    idx = np.full((NX, ny), -1); idx[i, j] = np.arange(m.ncell); assert (idx >= 0).all()
    return s, m, idx, c
def fourier_amp(s, idx):
    U = s.u[idx]; V = s.v[idx]                          # (NX, ny)
    up = U - U.mean(axis=0, keepdims=True); vp = V - V.mean(axis=0, keepdims=True)
    e = np.exp(-1j * ALPHA * (np.arange(NX) + 0.5) * (LX / NX))[:, None]
    return (up * e).mean(axis=0), (vp * e).mean(axis=0)
def run(ny, sample=100):
    s, m, idx, c = build(ny); A0 = fourier_amp(s, idx)
    den = sum(np.vdot(A0[k], A0[k]) for k in (0, 1))
    n = int(round(T / DT)); every = max(1, n // sample); ts, aa = [0.0], [1.0 + 0j]; t0 = time.time()
    for k in range(n):
        s.step()
        if (k + 1) % every == 0:
            A = fourier_amp(s, idx); ts.append((k + 1) * DT); aa.append(sum(np.vdot(A0[q], A[q]) for q in (0, 1)) / den)
        if (k + 1) % (n // 4) == 0: print(f"      ny={ny} t={(k+1)*DT:5.1f} |a| {abs(aa[-1]):.4f}  ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
    ts = np.array(ts); aa = np.array(aa); sel = ts >= 30
    growth = np.polyfit(ts[sel], np.log(np.abs(aa[sel])), 1)[0]
    phase = -np.polyfit(ts[sel], np.unwrap(np.angle(aa[sel])), 1)[0] / ALPHA
    return growth, phase, m.ncell, time.time() - t0, ts, aa
if __name__ == "__main__":
    nys = [int(a) for a in sys.argv[1:]] or [100, 200, 400]
    print(f"T8 Orr-Sommerfeld Re={RE:.0f} alpha={ALPHA}: {NX} x ny quads, periodic x, dt={DT}, T={T}, fit t>=30")
    print(f"   reference: growth alpha*Im(c) = {G_REF}, phase speed = {C_REF}")
    rows = []
    for ny in nys:
        g, cph, nc, wt, ts, aa = run(ny); rows.append((ny, g, cph))
        np.savez(f"results/t8_os_nx{NX}_ny{ny}_dt{DT}" + ("_fx" if os.environ.get("OS_FLUXEXTRAP", "1") == "1" else "") + ".npz", t=ts, a=aa, growth=g, phase=cph)
        print(f"   {NX}x{ny:<4d} ({nc:6d} cells)  growth {g:.6f} ({(g/G_REF-1)*100:+6.1f}%)   phase {cph:.6f} ({(cph/C_REF-1)*100:+5.2f}%)   {wt/60:.1f} min", flush=True)
    if len(rows) > 1:
        for (n1, g1, _), (n2, g2, _) in zip(rows[:-1], rows[1:]):
            e1, e2 = abs(g1 - G_REF), abs(g2 - G_REF); print(f"   growth-error order {n1}->{n2}: {np.log2(e1/e2):.2f}")
