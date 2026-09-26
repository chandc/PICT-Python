"""G2 (LES plan L2): 3D Taylor-Green on the 2.5D solver (unstructured x-y plane, Fourier z).
Box (2 pi)^3, u = sin x cos y cos z, v = -cos x sin y cos z, w = 0, nu = 1/Re.
  A. energy balance: -dE/dt against nu * int |omega|^2 (exact for periodic incompressible flow),
     the discrete check of "no dissipation other than the viscous one" at every sample time.
  B. spanwise convergence at fixed plane: error vs the finest nz at t = T_short (spectral: geometric).
  C. in-plane convergence at fixed nz: error vs the finest plane (order 2).
    python test_utgv3d.py [A|B|C ...]     (default: all)
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.upiso import BC
from src.upiso25 import PISO25
L = 2 * np.pi
import os
DEVICE = os.environ.get("TGV_DEVICE", "cpu"); SOLVER = os.environ.get("TGV_SOLVER", "lu" if DEVICE == "cpu" else "amg")
def build(n, nz, nu, dt, sgs="none"):
    m = rect_mesh(n, n, 0, L, 0, L, cells="quad"); m.make_periodic(1, 2, (L, 0.0)); m.make_periodic(3, 4, (0.0, L))
    s = PISO25(m, nz, L, nu, dt, BC(m), BC(m), BC(m), BC(m), n_nonorth=1, solver=SOLVER, device=DEVICE, mom_rtol=1e-7); s.sgs_model = sgs
    x, y = m.centroid.T; z = s.z
    s.u[:] = s.asdev(np.sin(x)[:, None] * np.cos(y)[:, None] * np.cos(z)[None, :])
    s.v[:] = s.asdev(-np.cos(x)[:, None] * np.sin(y)[:, None] * np.cos(z)[None, :])
    s.p[:] = s.asdev((np.cos(2 * x) + np.cos(2 * y))[:, None] * (np.cos(2 * z) + 2)[None, :] / 16)
    return m, s
def run_to(s, T, sample=None):
    n = int(round(T / s.dt)); hist = []
    for k in range(n):
        s.step()
        if sample and (k + 1) % sample == 0: hist.append((s.time, s.energy(), s.enstrophy(), s.dissipation(), float(s.nu_t.mean()) / max(s.nu, 1e-300), float(s.nu_t.max()) / max(s.nu, 1e-300)))
        if s.device == "gpu" and (k + 1) % (10 * sample) == 0: import cupy; cupy.cuda.Device().synchronize()
    return hist
if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B", "C"]
    if "A" in parts:
        Re = float(os.environ.get("TGV_RE", 100)); T = float(os.environ.get("TGV_T", 10)); n = int(os.environ.get("TGV_N", 32)); nz = int(os.environ.get("TGV_NZ", 32)); dt = float(os.environ.get("TGV_DT", 0.02))
        sgs = os.environ.get("TGV_SGS", "none"); m, s = build(n, nz, 1 / Re, dt, sgs); t0 = time.time()
        print(f"A. TGV Re={Re:.0f}, {n}^2 x {nz} modes, dt={dt}, T={T}, sgs={sgs}: E0={s.energy():.6f} (exact pi^3 = {np.pi**3:.6f}), Z0={s.enstrophy():.6f} (exact 6 pi^3 = {6*np.pi**3:.6f}), eps_d0={s.dissipation():.6f} (exact {6*np.pi**3/Re:.6f})")
        hist = [(0.0, s.energy(), s.enstrophy(), s.dissipation(), 0.0, 0.0)] + run_to(s, T, sample=int(0.1 / dt))
        h = np.array(hist); t, E, Z, eps, ntm, ntx = h.T; nu = 1 / Re
        dEdt = np.gradient(E, t)                                     # centred over 0.1
        print(f"   {'t':>5} {'E':>10} {'nu*Z':>10} {'eps_d':>10} {'-dE/dt':>10} {'-dE/dt / nu Z':>13} {'-dE/dt / eps_d':>14} {'<nu_t>/nu':>9} {'max':>7}   ({(time.time()-t0)/s.nstep*1e3:.0f} ms/step)")
        for j in range(5, len(t) - 1, 5):
            print(f"   {t[j]:5.1f} {E[j]:10.6f} {nu*Z[j]:10.6f} {eps[j]:10.6f} {-dEdt[j]:10.6f} {-dEdt[j]/(nu*Z[j]):13.4f} {-dEdt[j]/eps[j]:14.4f} {ntm[j]:9.3f} {ntx[j]:7.2f}")
        np.savez(f"results/tgv3d_re{Re:.0f}_n{n}_nz{nz}_dt{dt}_{sgs}{'_gpu' if DEVICE == 'gpu' else ''}.npz", t=t, E=E, Z=Z, eps=eps, nu=nu, nut_mean=ntm, nut_max=ntx)
        print(f"   peak nu*Z {nu*Z.max():.5f} at t={t[np.argmax(Z)]:.1f}")
    if "B" in parts:
        n, nu, dt, T = 32, 0.01, 0.02, 1.0; ref = None
        print(f"B. spanwise convergence, {n}^2 plane, dt={dt}, T={T}: error vs nz=32 (velocity L2 on the common planes)")
        sols = {}
        for nz in (4, 8, 16, 32):
            m, s = build(n, nz, nu, dt); run_to(s, T); sols[nz] = s
        r = sols[32]
        for nz in (4, 8, 16):
            s = sols[nz]; step = 32 // nz
            su, sv, sw, ru, rv, rw = (s.host(a) for a in (s.u, s.v, s.w, r.u, r.v, r.w))
            e = np.sqrt(float((m.vol[:, None] * ((su - ru[:, ::step]) ** 2 + (sv - rv[:, ::step]) ** 2 + (sw - rw[:, ::step]) ** 2)).sum()) * L / nz)
            print(f"   nz={nz:3d}: {e:.3e}")
    if "C" in parts:
        nz, nu, dt, T = 8, 0.01, 0.01, 1.0
        print(f"C. in-plane convergence, nz={nz}, dt={dt}, T={T}: error vs exact-in-time reference n=128 (sampled at coarse centroids by nearest fine cell)")
        sols = {}
        for n in (16, 32, 64, 128):
            m, s = build(n, nz, nu, dt); t0 = time.time(); run_to(s, T); sols[n] = (m, s); print(f"      n={n} done {time.time()-t0:.0f}s", flush=True)
        mr, r = sols[128]
        from scipy.spatial import cKDTree
        tree = cKDTree(mr.centroid); prev = None
        for n in (16, 32, 64):
            m, s = sols[n]; _, idx = tree.query(m.centroid)     # coarse centroid coincides with a fine centroid? no: nearest of the 4 fine cells
            # average the 2x2 block of fine cells inside each coarse cell (exact for cell averages)
            k = 128 // n; cx = ((m.centroid[:, 0] // (L / n))).astype(int); cy = ((m.centroid[:, 1] // (L / n))).astype(int)
            fx = ((mr.centroid[:, 0] // (L / n))).astype(int); fy = ((mr.centroid[:, 1] // (L / n))).astype(int)
            key_c = cx * n + cy; key_f = fx * n + fy; order = np.argsort(key_c); inv = np.empty_like(order); inv[order] = np.arange(len(order))
            acc = lambda a: np.add.at.__self__ and None
            ub = np.zeros((n * n, nz)); vb = np.zeros_like(ub); wb = np.zeros_like(ub); cnt = np.zeros(n * n)
            np.add.at(ub, key_f, r.host(r.u)); np.add.at(vb, key_f, r.host(r.v)); np.add.at(wb, key_f, r.host(r.w)); np.add.at(cnt, key_f, 1)
            ub /= cnt[:, None]; vb /= cnt[:, None]; wb /= cnt[:, None]
            e = np.sqrt(float((m.vol[:, None] * ((s.host(s.u) - ub[key_c]) ** 2 + (s.host(s.v) - vb[key_c]) ** 2 + (s.host(s.w) - wb[key_c]) ** 2)).sum()) * L / nz)
            print(f"   n={n:4d}: {e:.3e}" + (f"  order {np.log2(prev/e):.2f}" if prev else "")); prev = e
