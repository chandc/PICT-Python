"""Cylinder LES at Re 3900 on the 2.5D solver (V3 of the LES plan): unstructured quad plane x Fourier span
Lz = pi D, WALE, CFL-limited step, checkpoint/restart, time-and-span statistics. Written for the A100
notebook (tools/a100/A100_cylinder_re3900.ipynb) with --outdir on a mounted Drive so a lost session loses
nothing; runs on the CPU for smoke tests.
  BCs: inlet u = 1, v = w = 0; free-stream v = 0 (slip); outlet p = 0; cylinder no-slip. Impulsive start with
  a small y-odd v perturbation and a spanwise-periodic w perturbation in the near wake.
  Forces: span average of the per-plane wall-flux integration (as run_ucylinder25.py). Base pressure
  coefficient from the wall faces within 5 deg of the rear stagnation point, referenced to the inlet mean.
  Statistics (after --t-stats): span-and-time means of u, v, p, uu, vv, uv, ww, nu_t per cell, and the
  time-mean wall pressure per wall face (Cp distribution). Post-processing: plot_utility/plot_ucylinder_re3900.py.
References (Re 3900, span pi D): Parnaudeau et al. 2008 PIV St 0.208, L_r 1.51 D; Kravchenko & Moin 2000 LES
C_D 1.04, C_pb -0.94, L_r 1.35; Lehmkuhl et al. 2013 DNS two states L_r 1.26/1.55, C_D 1.05/0.98; Norberg C_D 0.98.
"""
import sys, os, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import BC
from src.upiso25 import PISO25
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/cylinder_re3900.msh"); ap.add_argument("--Re", type=float, default=3900.0)
ap.add_argument("--nz", type=int, default=64); ap.add_argument("--Lz", type=float, default=np.pi)
ap.add_argument("--dt", type=float, default=0.004, help="dt0; with --cfl-max the step is dt0 / 2^m")
ap.add_argument("--cfl-max", type=float, default=0.8); ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--t-stats", type=float, default=50.0)
ap.add_argument("--sgs", default="wale", choices=["none", "wale", "smagorinsky"]); ap.add_argument("--t-sgs", type=float, default=2.0, help="switch the SGS term on at this time: the impulsive start's wall gradient makes WALE blow up at t = 0"); ap.add_argument("--n-nonorth", type=int, default=3)
ap.add_argument("--eps3d", type=float, default=0.02, help="amplitude of the spanwise-periodic w perturbation at the start")
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"]); ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default="results/ucyl3900")
ap.add_argument("--checkpoint", type=int, default=2000); ap.add_argument("--restart", default=None); ap.add_argument("--report", type=int, default=250)
ap.add_argument("--stat-every", type=int, default=2); ap.add_argument("--nsteps", type=int, default=None)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
tag = a.tag or f"ucyl3900_{os.path.splitext(os.path.basename(a.mesh))[0]}_nz{a.nz}_{a.sgs}"
nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}; T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
bt = m.btag[m.bfaces]; nb = m.nbface
ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN); vv = np.zeros(nb)
kw = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vw = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN); vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO25(m, a.nz, a.Lz, nu, a.dt, BC(m, ku, vu), BC(m, kv, vv), BC(m, kw, vw), BC(m, kp, vp), n_nonorth=a.n_nonorth, device=a.device, solver=("amg" if a.device == "gpu" else "lu"), mom_rtol=1e-7)
s.sgs_model = "none"; s.cfl_max = a.cfl_max; xp = s.xp
C = m.centroid; blob = np.exp(-((C[:, 0] - 1.0) ** 2 + C[:, 1] ** 2))
s.u[:] = 1.0; s.v[:] = s.asdev(0.05 * blob[:, None] * np.ones((1, a.nz)))
s.w[:] = s.asdev(a.eps3d * blob[:, None] * (np.sin(2 * np.pi * s.z / a.Lz) + 0.5 * np.sin(6 * np.pi * s.z / a.Lz + 1.0))[None, :])
# ---- forces and base pressure (device gathers, scalar results)
wall = m.bfaces[bt == T_CYL]; wo = m.owner[wall]; Sw = m.normal[wall] * m.span; Aw = np.hypot(Sw[:, 0], Sw[:, 1]); e_in = -Sw / Aw[:, None]
dn = ((m.fcentre[wall] - m.centroid[wo]) * (-e_in)).sum(axis=1); theta = np.degrees(np.arctan2(m.fcentre[wall, 1], m.fcentre[wall, 0]))     # 0 at the rear stagnation point, +-180 at the front
base = np.abs(theta) < 5.0; inlet_cells = m.owner[m.bfaces[bt == T_IN]]
wo_d, dn_d, ein_d, Sw_d, base_d, inl_d = (s.asdev(x) for x in (wo, dn, e_in, Sw, base, inlet_cells))
def forces():
    uh, vh, ph = s.u[wo_d], s.v[wo_d], s.p[wo_d]
    gus = uh / dn_d[:, None]; gvs = vh / dn_d[:, None]
    gux, guy = gus * ein_d[:, 0, None], gus * ein_d[:, 1, None]; gvx, gvy = gvs * ein_d[:, 0, None], gvs * ein_d[:, 1, None]
    txx = 2 * nu * gux; tyy = 2 * nu * gvy; txy = nu * (guy + gvx)
    Fx = (ph * Sw_d[:, 0, None] - (txx * Sw_d[:, 0, None] + txy * Sw_d[:, 1, None])).sum(axis=0).mean()
    Fy = (ph * Sw_d[:, 1, None] - (txy * Sw_d[:, 0, None] + tyy * Sw_d[:, 1, None])).sum(axis=0).mean()
    Fxp = (ph * Sw_d[:, 0, None]).sum(axis=0).mean(); pref = s.p[inl_d].mean(); pb = ph[base_d].mean()
    return float(2 * Fx), float(2 * Fy), float(2 * Fxp), float(2 * (pb - pref))
def e3d():
    tot = 0.0
    for f in (s.u, s.v, s.w):
        fh = xp.fft.rfft(f, axis=1) / s.nz; tot += float((s.dm.vol[:, None] * 2 * xp.abs(fh[:, 1:]) ** 2).sum())
    return 0.5 * tot
# ---- statistics: span-and-time means per cell
KEYS = ("u", "v", "p", "uu", "vv", "uv", "ww", "nut"); acc = {k: xp.zeros(m.ncell) for k in KEYS}; acc_pw = xp.zeros(len(wall)); nsamp = 0
def sample():
    global nsamp
    u, v, w, p = s.u, s.v, s.w, s.p
    acc["u"] += u.mean(axis=1); acc["v"] += v.mean(axis=1); acc["p"] += p.mean(axis=1); acc["uu"] += (u * u).mean(axis=1); acc["vv"] += (v * v).mean(axis=1)
    acc["uv"] += (u * v).mean(axis=1); acc["ww"] += (w * w).mean(axis=1); acc["nut"] += s.nu_t.mean(axis=1); acc_pw[:] = acc_pw + p[wo_d].mean(axis=1); nsamp += 1
def save_acc(path): np.savez(path, n=nsamp, pw=s.host(acc_pw), **{k: s.host(v) for k, v in acc.items()})
def load_acc(path):
    global nsamp
    d = np.load(path); nsamp = int(d["n"]); acc_pw[:] = s.asdev(d["pw"])
    for k in KEYS: acc[k][:] = s.asdev(d[k])
def write_stats(path, hist):
    h = np.array(hist); win = h[:, 0] >= a.t_stats; n = max(nsamp, 1); mean = {k: s.host(v) / n for k, v in acc.items()}
    mean["uu"] -= mean["u"] ** 2; mean["vv"] -= mean["v"] ** 2; mean["uv"] -= mean["u"] * mean["v"]                   # fluctuation statistics
    pref = mean["p"][inlet_cells].mean(); cp_wall = 2 * (s.host(acc_pw) / n - pref)
    np.savez(path, nsamp=nsamp, t_end=s.time, t_stats=a.t_stats, Re=a.Re, Lz=a.Lz, nz=a.nz, sgs=a.sgs, mesh=a.mesh, nodes=m.nodes, cells=m.cells, nvert=m.nvert, centroid=m.centroid, vol=m.vol,
             theta_wall=theta, cp_wall=cp_wall, hist=h, cd_mean=(h[win, 1].mean() if win.any() else np.nan), cl_rms=(h[win, 2].std() if win.any() else np.nan), cpb_mean=(h[win, 4].mean() if win.any() else np.nan), **mean)
def strouhal(hist):
    h = np.array(hist); win = h[:, 0] >= a.t_stats
    if win.sum() < 64: return float("nan"), 0
    t, cl = h[win, 0], h[win, 2] - h[win, 2].mean(); z = np.flatnonzero(np.diff(np.sign(cl)) > 0)
    if len(z) < 4: return float("nan"), len(z)
    tz = t[z] - cl[z] * (t[z + 1] - t[z]) / (cl[z + 1] - cl[z]); per = np.diff(tz); return 1.0 / per.mean(), len(per)
# ---- run
hist = []
if a.restart:
    s.load(a.restart); print(f"  restart from {a.restart} at t = {s.time:.3f}, step {s.nstep}", flush=True)
    accp = a.restart.replace("_ckpt.npz", "_stats_acc.npz"); hp = a.restart.replace("_ckpt.npz", "_hist.npy")
    if os.path.exists(accp): load_acc(accp); print(f"  statistics accumulator restored: {nsamp} samples", flush=True)
    if os.path.exists(hp): hist = [tuple(r) for r in np.load(hp)]; print(f"  force history restored: {len(hist)} entries", flush=True)
print(f"cylinder Re {a.Re:.0f} on {a.mesh}: {m.ncell} cells ({int((m.nvert == 3).sum())} triangles, orth min {float(m.orth.min()):.3f}, {len(wall)} wall faces, wall cell {2*dn.min():.4f}-{2*dn.max():.4f} D) x {a.nz} modes (Lz {a.Lz:.3f} D, dz {a.Lz/a.nz:.4f} D); "
      f"dt0 {a.dt} cfl_max {a.cfl_max}; sgs {a.sgs}; n_nonorth {a.n_nonorth}; T {a.T}, statistics from t = {a.t_stats}; device {a.device}; outdir {a.outdir}; tag {tag}", flush=True)
t0 = time.time(); k = 0; diverged = False
while True:
    if a.nsteps and k >= a.nsteps: break
    if not a.nsteps and s.time >= a.T - 1e-12: break
    if s.sgs_model != a.sgs and s.time >= a.t_sgs - 1e-12: s.sgs_model = a.sgs; print(f"  SGS term ({a.sgs}) on at t = {s.time:.3f}", flush=True)
    s.step(); k += 1
    if not bool(xp.isfinite(s.u).all()): print(f"  DIVERGED at step {s.nstep}, t = {s.time:.4f}, last CFL {s.cfl_last:.2f}", flush=True); diverged = True; break
    cd, cl, cdp, cpb = forces(); hist.append((s.time, cd, cl, cdp, cpb, s.dt))
    if s.time >= a.t_stats - 1e-12 and s.nstep % a.stat_every == 0: sample()
    if k % a.report == 0:
        w = np.array(hist[-a.report:]); st, nper = strouhal(hist)
        print(f"  t={s.time:7.2f}  Cd {w[:, 1].mean():.4f}  Cl {w[:, 2].mean():+.4f} (amp {0.5*(w[:, 2].max()-w[:, 2].min()):.4f})  Cd_p {w[:, 3].mean():.4f}  Cpb {w[:, 4].mean():+.4f}  St {st:.4f} ({nper} periods)  E3d {e3d():.3e}  "
              f"<nu_t>/nu {float(s.nu_t.mean())/nu:.3f} max {float(s.nu_t.max())/nu:.1f}  CFL {s.cfl_last if a.cfl_max > 0 else s.courant():.2f} dt {s.dt:.5f}  stats {nsamp}  ({(time.time()-t0)/k*1e3:.0f} ms/step)", flush=True)
    if k % a.checkpoint == 0:
        s.save(f"{a.outdir}/{tag}_ckpt.npz"); save_acc(f"{a.outdir}/{tag}_stats_acc.npz"); np.save(f"{a.outdir}/{tag}_hist.npy", np.array(hist))
        if nsamp: write_stats(f"{a.outdir}/{tag}_stats_partial.npz", hist)
if not diverged:
    s.save(f"{a.outdir}/{tag}_final.npz"); np.save(f"{a.outdir}/{tag}_hist.npy", np.array(hist))
    if nsamp: write_stats(f"{a.outdir}/{tag}_stats.npz", hist)
    h = np.array(hist); win = h[:, 0] >= a.t_stats; st, nper = strouhal(hist)
    if win.any(): print(f"\nRESULT {tag}: window t = {a.t_stats}..{s.time:.2f} ({nper} shedding periods, {nsamp} statistics samples)  St {st:.4f}  Cd {h[win, 1].mean():.4f}  Cl_rms {h[win, 2].std():.4f}  Cd_p {h[win, 3].mean():.4f}  Cpb {h[win, 4].mean():+.4f}\n"
                        f"   references: St 0.208 (Parnaudeau PIV) / 0.21;  Cd 0.98 (Norberg) - 1.04 (K&M) - 1.05/0.98 (Lehmkuhl H/L);  Cpb -0.88 (Norberg) / -0.94 (K&M);  L_r from the mean field with plot_utility/plot_ucylinder_re3900.py", flush=True)
