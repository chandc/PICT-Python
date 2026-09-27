"""Continuation run collecting the two turbulence diagnostics a reviewer asked for beyond the V3 gate
(record section 65/65a): (A) the spanwise two-point correlation R_uu(dz) at fixed probes, to check the
turbulence is genuinely three-dimensional and the span L_z = pi D is wide enough (R_uu should fall to
~0 well before L_z/2); (B) the temporal power spectrum at the same probes, to check for a -5/3 inertial
range before the grid cutoff. Restarts from the finished V3 run's field (already statistically
stationary; no need to redo the 150 D/U production run), so this is pure diagnostic time on top of it.
  Spanwise correlation: at every step, FFT the field's own spanwise (Fourier) representation at each
  probe cell, accumulate the time-mean power spectrum of the fluctuation modes (k >= 1); R(dz) is its
  inverse FFT at the end (Wiener-Khinchin) -- no extra cost, the solver already holds u(z) spectrally.
  Temporal spectrum: append (t, u, v, w) at z-index 0 of each probe every step; Welch/periodogram PSD
  computed in the analysis script, not here (keeps this driver simple and restartable).
    python run_ucylinder3900_probes.py --restart <final.npz> --T-extra 35 --device gpu --outdir results/ucyl3900_probes"""
import sys, os, time, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import BC
from src.upiso25 import PISO25
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/cylinder_re3900.msh"); ap.add_argument("--Re", type=float, default=3900.0)
ap.add_argument("--nz", type=int, default=64); ap.add_argument("--Lz", type=float, default=np.pi)
ap.add_argument("--cfl-max", type=float, default=0.8); ap.add_argument("--sgs", default="wale"); ap.add_argument("--n-nonorth", type=int, default=3)
ap.add_argument("--restart", required=True, help="the V3 run's _final.npz or _ckpt.npz"); ap.add_argument("--T-extra", type=float, default=35.0, help="additional D/U to run (~7-8 shedding periods at St 0.21)")
ap.add_argument("--probes", type=float, nargs="+", default=[1.5, 0.5, 1.5, 0.0, 3.0, 0.5, 5.0, 0.0], help="flat list x1 y1 x2 y2 ... (x/D, y/D from the cylinder centre)")
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"]); ap.add_argument("--outdir", default="results/ucyl3900_probes"); ap.add_argument("--tag", default="diag")
ap.add_argument("--report", type=int, default=250); ap.add_argument("--flush", type=int, default=2000)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}; T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
bt = m.btag[m.bfaces]; nb = m.nbface; nu = 1.0 / a.Re
ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN); vv = np.zeros(nb)
kw = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vw = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN); vp = np.zeros(nb)
s = PISO25(m, a.nz, a.Lz, nu, 0.004, BC(m, ku, vu), BC(m, kv, vv), BC(m, kw, vw), BC(m, kp, vp), n_nonorth=a.n_nonorth, device=a.device, solver=("amg" if a.device == "gpu" else "lu"), mom_rtol=1e-7)
s.sgs_model = a.sgs; s.cfl_max = a.cfl_max; xp = s.xp
s.load(a.restart); t0_run = s.time
print(f"restarted from {a.restart} at t = {s.time:.3f}, step {s.nstep}; running {a.T_extra} more D/U on {a.device}", flush=True)
C = m.centroid; P = np.array(a.probes).reshape(-1, 2)
probe_cells = [int(np.argmin((C[:, 0] - x) ** 2 + (C[:, 1] - y) ** 2)) for x, y in P]
for k, (x, y) in zip(probe_cells, P):
    print(f"  probe target ({x:.2f},{y:.2f}) -> cell {k} at ({C[k,0]:.3f},{C[k,1]:.3f})", flush=True)
pc_d = s.asdev(np.array(probe_cells))
kz = xp.asarray(2 * np.pi / a.Lz * np.arange(a.nz // 2 + 1))
power_sum = xp.zeros((len(probe_cells), a.nz // 2 + 1)); nsamp = 0
thist = []; probe_ts = {p: [] for p in range(len(probe_cells))}
t0 = time.time(); k = 0; T_end = s.time + a.T_extra
def flush():
    np.savez(f"{a.outdir}/{a.tag}_probes.npz", probe_xy=P, probe_cells=probe_cells, t=np.array(thist),
             **{f"uvw_{p}": np.array(probe_ts[p]) for p in range(len(probe_cells))},
             power_sum=s.host(power_sum), nsamp=nsamp, kz=s.host(kz), Lz=a.Lz, nz=a.nz, t0_run=t0_run, t_end=s.time)
while s.time < T_end - 1e-12:
    s.step(); k += 1
    if not bool(xp.isfinite(s.u).all()): print(f"  DIVERGED at step {s.nstep}, t = {s.time:.4f}", flush=True); break
    uz = s.u[pc_d]                                                     # (nprobe, nz)
    uzh = xp.fft.rfft(uz - uz.mean(axis=1, keepdims=True), axis=1) / a.nz
    power_sum += 2 * xp.abs(uzh) ** 2; nsamp += 1                       # one-sided power, k=0 stays ~0 (mean removed)
    thist.append(float(s.time))
    for p in range(len(probe_cells)):
        probe_ts[p].append((float(s.u[pc_d[p], 0]), float(s.v[pc_d[p], 0]), float(s.w[pc_d[p], 0])))
    if k % a.report == 0:
        print(f"  t={s.time:7.2f}  nsamp {nsamp}  ({(time.time()-t0)/k*1e3:.0f} ms/step)", flush=True)
    if k % a.flush == 0: flush()
flush()
print(f"done: t = {t0_run:.1f} -> {s.time:.1f}, {nsamp} samples, wrote {a.outdir}/{a.tag}_probes.npz", flush=True)
