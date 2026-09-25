"""Re=100 cylinder on the 2.5D solver (unstructured plane, Fourier span): the periodic-span laminar case
must reproduce the 2D RK3 forces (G2 of the LES plan). A small spanwise-periodic w perturbation is added
in the near wake so the 3D modes are exercised; at Re=100 they must decay (mode A onsets near Re 190).
BCs as run_ucylinder.py, plus w: Dirichlet 0 at the inlet and the cylinder, free elsewhere. Forces are
the span average of the per-plane wall-flux integration.
"""
import sys, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import BC
from src.upiso25 import PISO25
ap = argparse.ArgumentParser()
ap.add_argument("mesh"); ap.add_argument("--Re", type=float, default=100.0); ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--nz", type=int, default=4); ap.add_argument("--Lz", type=float, default=4.0)
ap.add_argument("--eps3d", type=float, default=1e-3, help="amplitude of the spanwise-periodic w perturbation")
ap.add_argument("--out", default=None); ap.add_argument("--report", type=int, default=2000); ap.add_argument("--nsteps", type=int, default=None)
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"]); ap.add_argument("--sgs", default="none", choices=["none", "wale", "smagorinsky"])
a = ap.parse_args()
nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}; T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
bt = m.btag[m.bfaces]; nb = m.nbface
ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN); vv = np.zeros(nb)
kw = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vw = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN); vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO25(m, a.nz, a.Lz, nu, a.dt, BC(m, ku, vu), BC(m, kv, vv), BC(m, kw, vw), BC(m, kp, vp), n_nonorth=3, device=a.device, solver=("amg" if a.device == "gpu" else "lu"), mom_rtol=1e-7); s.sgs_model = a.sgs
C = m.centroid; blob = np.exp(-((C[:, 0] - 1.0) ** 2 + C[:, 1] ** 2))
s.u[:] = 1.0; s.v[:] = s.asdev(0.05 * blob[:, None] * np.ones((1, a.nz)))
s.w[:] = s.asdev(a.eps3d * blob[:, None] * np.sin(2 * np.pi * s.z / a.Lz)[None, :])
wall = m.bfaces[bt == T_CYL]; wo = m.owner[wall]; Sw = m.normal[wall] * m.span
Aw = np.hypot(Sw[:, 0], Sw[:, 1]); e_in = -Sw / Aw[:, None]; dn = ((m.fcentre[wall] - m.centroid[wo]) * (-e_in)).sum(axis=1)
def forces():
    uh, vh, ph = s.host(s.u), s.host(s.v), s.host(s.p)
    gus = uh[wo] / dn[:, None]; gvs = vh[wo] / dn[:, None]
    gux, guy = gus * e_in[:, 0, None], gus * e_in[:, 1, None]; gvx, gvy = gvs * e_in[:, 0, None], gvs * e_in[:, 1, None]
    pf = ph[wo]; txx = 2 * nu * gux; tyy = 2 * nu * gvy; txy = nu * (guy + gvx)
    Fx = (pf * Sw[:, 0, None] - (txx * Sw[:, 0, None] + txy * Sw[:, 1, None])).sum(axis=0).mean()
    Fy = (pf * Sw[:, 1, None] - (txy * Sw[:, 0, None] + tyy * Sw[:, 1, None])).sum(axis=0).mean()
    return 2 * Fx, 2 * Fy, 2 * (pf * Sw[:, 0, None]).sum(axis=0).mean()
def e3d():
    """kinetic energy in the spanwise modes k >= 1 (per unit span), the 3D content."""
    tot = 0.0
    for f in (s.u, s.v, s.w):
        fh = np.fft.rfft(s.host(f), axis=1) / s.nz; tot += float((m.vol[:, None] * 2 * np.abs(fh[:, 1:]) ** 2).sum())
    return 0.5 * tot
nsteps = a.nsteps or int(round(a.T / a.dt))
print(f"T9-2.5D {a.mesh}: {m.ncell} cells x {a.nz} planes (Lz={a.Lz}), Re={a.Re}, dt={a.dt}, eps3d={a.eps3d}, {nsteps} steps to T={nsteps*a.dt:.1f}", flush=True)
hist = np.zeros((nsteps, 5)); t0 = time.time()
for k in range(nsteps):
    s.step()
    if not bool(s.xp.isfinite(s.u).all()): print(f"  DIVERGED at step {k+1}", flush=True); hist = hist[:k]; break
    cd, cl, cdp = forces(); hist[k] = (s.time, cd, cl, cdp, e3d())
    if (k + 1) % a.report == 0:
        w = hist[max(0, k - a.report + 1):k + 1]
        print(f"  t={s.time:7.2f}  Cd {w[:,1].mean():.4f}  Cl {w[:,2].mean():+.4f} (amp {0.5*(w[:,2].max()-w[:,2].min()):.4f})  Cd_p {w[:,3].mean():.4f}  E3d {hist[k,4]:.3e}  |w|max {float(s.xp.abs(s.w).max()):.2e}  ({(time.time()-t0)/(k+1)*1e3:.1f} ms/step)", flush=True)
n2 = int(0.6 * len(hist)); t, cl = hist[n2:, 0], hist[n2:, 2] - hist[n2:, 2].mean()
if len(t) > 64:
    z = np.flatnonzero(np.diff(np.sign(cl)) > 0); tz = t[z] - cl[z] * (t[z + 1] - t[z]) / (cl[z + 1] - cl[z])
    per = np.diff(tz); f0 = 1.0 / per.mean() if len(per) >= 3 else float("nan")
    print(f"  {len(per)} periods, mean {per.mean():.4f} +- {per.std():.4f}", flush=True)
    print(f"RESULT {a.mesh}: St={f0:.4f}  Cd_mean={hist[n2:,1].mean():.4f}  Cl_rms={np.sqrt((cl**2).mean()):.4f}  Cl_amp={0.5*(hist[n2:,2].max()-hist[n2:,2].min()):.4f}  Cd_p={hist[n2:,3].mean():.4f}  E3d_end={hist[-1,4]:.3e}  (window t={t[0]:.1f}..{t[-1]:.1f})", flush=True)
if a.out: np.savez(a.out, hist=hist, nodes=m.nodes, cells=m.cells, nvert=m.nvert, centroid=m.centroid, vol=m.vol, u=s.host(s.u), v=s.host(s.v), w=s.host(s.w), p=s.host(s.p), btag=m.btag, z=s.z)
