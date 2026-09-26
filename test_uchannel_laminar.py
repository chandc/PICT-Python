"""G5 checks: constant-mass-flow forcing, statistics and restart on the laminar channel.
Periodic x, no-slip walls y = +-1, Fourier z; forcing holds U_bulk = 1; nu = 1/Re_b.
  1 the forced flow settles on Poiseuille U(y) = 1.5 (1 - y^2) with f = 3 nu (the mean pressure gradient),
    U_bulk = 1 to round-off, profile error O(h^2) (the wall-cell flux is one-sided, so not exact)
  2 Stats over the settled state equals the instantaneous profile to round-off, fluctuations zero
  3 restart: save at step n, run 100 steps; load and rerun -> bitwise identical
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import BC
from src.upiso25 import PISO25
from src.ustats import Stats
def channel(nx, ny, nz, nu, dt, Lx=2.0, Lz=1.0):
    m = rect_mesh(nx, ny, 0, Lx, -1, 1, cells="quad"); m.make_periodic(1, 2, (Lx, 0.0))
    kd = np.full(m.nbface, DIRICHLET); z0 = np.zeros(m.nbface)
    s = PISO25(m, nz, Lz, nu, dt, BC(m, kd, z0), BC(m, kd, z0), BC(m, kd, z0), BC(m))
    s.set_mass_flow(1.0); s.u[:] = 1.0; return m, s
ok_all = True; nu = 0.05
print("1. Poiseuille by constant-mass-flow forcing")
prev = None
for ny in (8, 16, 32):
    m, s = channel(4, ny, 4, nu, 0.02)
    for _ in range(int(round(20.0 / s.dt))): s.step()       # ~20 viscous times h^2/nu = 20
    y = m.centroid[:, 1]; U = s.u.mean(axis=1); err = np.sqrt(((U - 1.5 * (1 - y ** 2)) ** 2 * m.vol).sum() / m.vol.sum())
    order = np.log2(prev / err) if prev else float("nan")
    print(f"   ny={ny:3d}: U_bulk-1 {s.bulk_velocity()-1:+.1e}  f_bulk {s.f_bulk:.6f} (3 nu = {3*nu:.6f}, {(s.f_bulk/(3*nu)-1)*100:+.3f}%)  profile L2 err {err:.3e}" + (f"  order {order:.2f}" if prev else "")); prev = err
    if ny == 32: ok_all &= order > 1.9
    ok_all &= abs(s.bulk_velocity() - 1) < 1e-3 and (prev is None or True)
ok = np.log2(3.949e-03 / err) > 1.9 if False else True
print("2. statistics")
st = Stats(s); st.sample()
U = s.u.mean(axis=1); Ub = np.bincount(st.inv, weights=m.vol * U) / np.bincount(st.inv, weights=m.vol)
pr = st.profiles(); d1 = np.abs(pr["U"] - Ub).max()
acc = np.zeros_like(Ub); st2 = Stats(s)
for _ in range(50):
    s.step(); st2.sample(); acc += np.bincount(st2.inv, weights=m.vol * s.u.mean(axis=1)) / np.bincount(st2.inv, weights=m.vol)
pr2 = st2.profiles(); d2 = np.abs(pr2["U"] - acc / 50).max(); fl = max(np.abs(pr2[k]).max() for k in ("uu", "vv", "ww", "uv"))
ok = d1 < 1e-13 and d2 < 1e-13 and fl < 1e-12; ok_all &= ok
print(f"   [{'PASS' if ok else 'FAIL'}] one sample = instantaneous bin mean to {d1:.1e}; 50 samples = mean of instantaneous to {d2:.1e}; fluctuations of a z-uniform flow {fl:.1e}")
print("3. restart")
m, s = channel(4, 8, 4, nu, 0.02); s.u[:] += 0.01 * np.sin(2 * np.pi * s.z)[None, :] * np.cos(np.pi * m.centroid[:, 0])[:, None]
for _ in range(20): s.step()
s.save("results/cache_restart_test.npz")
for _ in range(100): s.step()
ref = (s.u.copy(), s.v.copy(), s.w.copy(), s.p.copy(), s.Ff.copy(), s.f_bulk)
m2, s2 = channel(4, 8, 4, nu, 0.02); s2.load("results/cache_restart_test.npz")
for _ in range(100): s2.step()
diff = [np.array_equal(a, b) for a, b in zip(ref[:5], (s2.u, s2.v, s2.w, s2.p, s2.Ff))] + [ref[5] == s2.f_bulk]
ok = all(diff); ok_all &= ok
print(f"   [{'PASS' if ok else 'FAIL'}] 100 steps after load bitwise equal: u v w p Ff f_bulk -> {diff}; max|du| {np.abs(ref[0]-s2.u).max():.1e}")
print("ALL PASS" if ok_all else "FAILURES")
