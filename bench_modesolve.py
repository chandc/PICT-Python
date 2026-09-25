"""G4 benchmark: the channel pressure family (32 modes) solved as a block by PCG + shared Ruge-Stuben AMG,
CPU (numpy) vs GPU (cupy), against the plan's bar of 100 ms per step per 1e5 cell-modes. One step of the
2.5D solver does 3 stages x (3 momentum + 1 pressure) block solves; the pressure one is the hard one, so
the per-step budget for this solve alone is ~25 ms per 1e5 cell-modes.
    python bench_modesolve.py [cpu|gpu] [nz]"""
import sys, time, numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh; from src.uops import DIRICHLET, laplacian; from src.upiso import BC; from src.upiso25 import PISO25; from src.umodesolve import ModeFamily
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; nz = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LX, LY = np.pi, 2.0
def fam_of(nx, ny):
    m = rect_mesh(nx, ny, 0, LX, 0, LY, cells="quad", cluster_y=1.8); m.make_periodic(1, 2, (LX, 0.0)); kd = np.full(m.nbface, DIRICHLET); z0 = np.zeros(m.nbface)
    s = PISO25(m, nz, 0.34 * np.pi, 1 / 180, 0.002, BC(m, kd, z0), BC(m, kd, z0), BC(m, kd, z0), BC(m))
    al = s.RK3_ALPHA[0]; Dcell = 2 * al * s.dt * np.ones(m.ncell); w = m.wf; i, b = m.interior, m.boundary
    gam = np.empty(m.nface); gam[i] = w[i] * Dcell[m.owner[i]] + (1 - w[i]) * Dcell[m.neigh[i]]; gam[b] = Dcell[m.owner[b]]
    Ap, _ = laplacian(m, gam, s.bc_p.kind); return m.ncell, Ap.tocsr(), Dcell * m.vol, s.kz ** 2
rng = np.random.default_rng(1)
print(f"device {dev}, {nz} modes ({nz // 2 + 1} rfft modes -> {nz + 2} real columns), pressure family, rtol 1e-8")
sizes = ((24, 80), (96, 160), (192, 320), (384, 640)) + (((768, 1280),) if dev == "gpu" else ())
for nx, ny in sizes:
    N, A, d, kz2 = fam_of(nx, ny); nk = len(kz2)
    t0 = time.time(); fam = ModeFamily(A, d, kz2, singular_k0=True, presmooth=1, postsmooth=1, device=dev); tset = time.time() - t0
    Bc = rng.standard_normal((N, nk)) + 1j * rng.standard_normal((N, nk)); Bc[:, 0] = Bc[:, 0].real
    X = fam.solve_complex(Bc, rtol=1e-8)                                    # warm-up (GPU kernels, allocations)
    if dev == "gpu": import cupy; cupy.cuda.Device().synchronize()
    reps = 3; t0 = time.time()
    for _ in range(reps): X = fam.solve_complex(Bc, rtol=1e-8)
    if dev == "gpu": cupy.cuda.Device().synchronize()
    tsol = (time.time() - t0) / reps
    Xh = X.get() if hasattr(X, 'get') else X; Bp = Bc.copy(); Bp[:, 0] -= Bp[:, 0].mean()
    res = np.linalg.norm(-(A @ Xh) + (d[:, None] * kz2[None, :]) * Xh - Bp, axis=0) / np.linalg.norm(Bp, axis=0)
    cm = N * (2 * nk); per1e5 = tsol / cm * 1e5
    print(f"  {nx:4d}x{ny:4d}  N {N:7d}  {fam.iterations:2d} it  setup {tset:5.2f} s  solve {tsol*1e3:8.1f} ms  ->  {per1e5*1e3:6.2f} ms per 1e5 cell-modes  (max rel res {res.max():.1e})", flush=True)
