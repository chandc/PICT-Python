"""Profile one time step of the 2.5D solver on the current GPU and estimate run times.

    python tools/a100/profile_step.py                      # the standard size ladder on a periodic box
    python tools/a100/profile_step.py --mesh meshes/cylinder_butterfly_fine.msh --nz 64   # a real plane

Per size: whole-step time (device-synchronised), the phase split of one RK stage, the memory
bandwidth the step achieves against the device's nominal, and the time per 1e5 cell-modes. Then
the run-time table for the target cases (below) at this device's measured rate, and -- if the
device is not the target -- the same scaled by the bandwidth ratio, which is what a memory-bound
step follows once it is past the launch-latency floor (measured as the smallest case's time).
"""
import sys, os, time, argparse, numpy as np; sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser(); ap.add_argument("--mesh", default=None); ap.add_argument("--nz", type=int, default=None); ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--sizes", default="32x16,64x32,128x64,256x64,384x64"); ap.add_argument("--target-bw", type=float, default=None, help="GB/s of the target device, e.g. 2039 for A100-80GB (1555 for A100-40GB, 3350 for H100)")
a = ap.parse_args()
import cupy as cp
from src.umesh import rect_mesh, Mesh, read_gmsh22; from src.uops import DIRICHLET, NEUMANN; from src.upiso import BC; from src.upiso25 import PISO25
props = cp.cuda.runtime.getDeviceProperties(0); name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
bw_nominal = 2 * props["memoryClockRate"] * 1e3 * (props["memoryBusWidth"] / 8) / 1e9      # GB/s, DDR factor 2
print(f"device: {name}, {props['multiProcessorCount']} SMs, nominal memory bandwidth {bw_nominal:.0f} GB/s, {props['totalGlobalMem']/2**30:.0f} GB")
def sync(): cp.cuda.Device().synchronize()
def timed(f, reps):
    f(); sync(); t0 = time.time()
    for _ in range(reps): f()
    sync(); return (time.time() - t0) / reps
def box(n, nz):
    L = 2 * np.pi; m = rect_mesh(n, n, 0, L, 0, L, cells="quad"); m.make_periodic(1, 2, (L, 0.0)); m.make_periodic(3, 4, (0.0, L))
    s = PISO25(m, nz, L, 1 / 1600, 0.02, BC(m), BC(m), BC(m), BC(m), solver="amg", device="gpu", mom_rtol=1e-7)
    x, y = m.centroid.T; z = s.z; s.u[:] = s.asdev(np.sin(x)[:, None] * np.cos(y)[:, None] * np.cos(z)[None, :]); s.v[:] = s.asdev(-np.cos(x)[:, None] * np.sin(y)[:, None] * np.cos(z)[None, :])
    return m, s
def cyl(path, nz):
    nodes, cells, ctag, edges, etag, names = read_gmsh22(path); m = Mesh(nodes, cells, edges, etag, names); inv = {v: k for k, v in names.items()}
    bt = m.btag[m.bfaces]; nb = m.nbface; T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
    ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0); kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN)
    kw = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN)
    s = PISO25(m, nz, np.pi, 1 / 3900, 0.002, BC(m, ku, vu), BC(m, kv, np.zeros(nb)), BC(m, kw, np.zeros(nb)), BC(m, kp, np.zeros(nb)), n_nonorth=3, solver="amg", device="gpu", mom_rtol=1e-7)
    s.sgs_model = "wale"; C = m.centroid; blob = np.exp(-((C[:, 0] - 1.0) ** 2 + C[:, 1] ** 2)); s.u[:] = 1.0; s.v[:] = s.asdev(0.05 * blob[:, None]); s.w[:] = s.asdev(1e-3 * blob[:, None] * np.sin(2 * np.pi * s.z / np.pi)[None, :])
    return m, s
rows = []
cases = [("mesh", a.mesh, a.nz or 64)] if a.mesh else [("box", int(t.split("x")[0]), int(t.split("x")[1])) for t in a.sizes.split(",")]
for kind, n, nz in cases:
    m, s = cyl(n, nz) if kind == "mesh" else box(n, nz)
    s.step(); sync(); t_step = timed(s.step, a.steps)
    cm = m.ncell * (nz // 2 + 1)
    # phase split of one stage
    u, v, w, F, p = s.u, s.v, s.w, s.Ff, s.p; al = s.RK3_ALPHA[0]
    t_nl = timed(lambda: s.nonlinear(u, v, w, F), 3)
    t_diff = timed(lambda: [s.diffusion(f_, L_, R_, b_) for f_, L_, R_, b_ in ((u, s.Lu, s.Lu_rhs, s.bc_u), (v, s.Lv, s.Lv_rhs, s.bc_v), (w, s.Lw, s.Lw_rhs, s.bc_w))], 3)
    fam, aP, aC = s._mom_family(0, 0, s.Lu_h, al); bh = s.fft(u) * s.dm.vol[:, None] / s.dt
    t_mom = timed(lambda: fam.solve_complex(bh, X0=s.fft(u), rtol=s.mom_rtol), 3); it_mom = fam.iterations
    uh, vh, wh = s.fft(u), s.fft(v), s.fft(w); ph = s.fft(p); gp = s.grad_p(p, s.beff(s.bc_p, p)); gph = s.fft(gp); Drc = 2 * al * s.dm.vol[:, None] / aP
    Fs = s._rhie_chow(uh, vh, ph, gph, Drc); pf, Ap_rhs, gam = s._pois_family(0, Drc[:, 0]); rhs = (s._div(Fs) + 1j * s.kz[None, :] * wh) * s.dm.vol[:, None]
    t_p = timed(lambda: pf.solve_complex(-rhs, rtol=s.amg_rtol), 3); it_p = pf.iterations
    # bytes moved per step, lower bound: every phase reads/writes the (N, nk) blocks a number of times; use the sum of the solve traffic as the estimate
    label = f"{os.path.basename(n)} x {nz}" if kind == "mesh" else f"{n}^2 x {nz}"
    rows.append((label, m.ncell, nz, cm, t_step, t_nl, t_diff, t_mom, it_mom, t_p, it_p))
    print(f"{label:30s} N {m.ncell:7d} cell-modes {cm:9d}  step {t_step*1e3:8.0f} ms = {t_step/cm*1e8:6.1f} ms per 1e5 cell-modes | per stage: nonlinear {t_nl*1e3:.0f}  diffusion {t_diff*1e3:.0f}  momentum {t_mom*1e3:.0f} x6 ({it_mom} it)  pressure {t_p*1e3:.0f} ({it_p} it)", flush=True)
# ---- run-time estimates
floor = rows[0][4] if not a.mesh else 0.0
rate = max((r[4] - floor) / r[3] for r in rows[-2:])                       # s per cell-mode past the latency floor, at the large end
targets = [("V3 cylinder Re 3900, 27968-quad butterfly x 64 modes, dt 0.002, 200 D/U", 27968, 33, 100000),
           ("V3 cylinder Re 3900, 1e5-cell plane x 64 modes, dt 0.002, 200 D/U", 100000, 33, 100000),
           ("V3 cylinder Re 3900, 1e5-cell plane x 128 modes, dt 0.002, 200 D/U", 100000, 65, 100000),
           ("channel Re_tau 180, 24x80 quads x 32 modes, dt 0.002, 30 time units", 1920, 17, 15000),
           ("channel Re_tau 395, 96x160 quads x 128 modes, dt 0.001, 30 time units", 15360, 65, 30000),
           ("TGV Re 1600, 128^2 x 128, dt 0.02, T 20", 16384, 65, 1000)]
print(f"\nrun-time estimates at this device's measured rate ({rate*1e8:.1f} ms per 1e5 cell-modes past a {floor*1e3:.0f} ms launch floor)" + (f", and scaled to a {a.target_bw:.0f} GB/s device (x{a.target_bw/bw_nominal:.1f} on the bandwidth-bound part)" if a.target_bw else ""))
for name_, N, nk, steps in targets:
    t_here = floor + rate * N * nk; line = f"  {name_:72s} {t_here*1e3:7.0f} ms/step  {t_here*steps/3600:7.1f} h"
    if a.target_bw: t_t = floor + rate * N * nk / (a.target_bw / bw_nominal); line += f"   | target {t_t*1e3:6.0f} ms/step  {t_t*steps/3600:6.1f} h"
    print(line)
