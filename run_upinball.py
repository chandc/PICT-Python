"""Fluidic pinball (three cylinders, equilateral, side 1.5 D, apex upstream) on the unstructured PISO solver,
HydroGym's domain, boundary conditions and meshes (meshes/pinball_medium.msh, pinball_fine.msh: their
Firedrake meshes, LFS-fetched and converted to msh2). Inlet u = 1; Freestream v = 0 (symmetry);
Outlet p = 0; Cyl1 (front), Cyl2 (top), Cyl3 (bottom) no-slip. Forces per cylinder from the solver's own
one-sided wall flux (as run_ucylinder.py). Starts impulsively with a small y-even perturbation.
    python run_upinball.py meshes/pinball_medium.msh --Re 30 --T 150 --dt 0.01 --out results/pinball/re30_medium.npz
"""
import sys, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
ap = argparse.ArgumentParser()
ap.add_argument("mesh"); ap.add_argument("--Re", type=float, default=30.0); ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--out", default=None); ap.add_argument("--report", type=int, default=500)
ap.add_argument("--nsteps", type=int, default=None); ap.add_argument("--time-scheme", default="bdf2", choices=["bdf2", "rk3"])
ap.add_argument("--init", default=None, help="npz with centroid,u,v,p to start from (e.g. a lower-Re result)")
a = ap.parse_args()
nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges, etag, names); inv = {v: k for k, v in names.items()}
T_IN, T_FS, T_OUT = inv["Inlet"], inv["Freestream"], inv["Outlet"]; T_CYL = [inv["Cyl1"], inv["Cyl2"], inv["Cyl3"]]
bt = m.btag[m.bfaces]; nb = m.nbface
ku = np.where(np.isin(bt, [T_IN] + T_CYL), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
kv = np.where(np.isin(bt, [T_IN, T_FS] + T_CYL), DIRICHLET, NEUMANN); vv = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN); vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO(m, nu=nu, dt=a.dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, vv), bc_p=BC(m, kp, vp), n_corr=2, n_nonorth=3, scheme="central", convect=True)
s.time_scheme = a.time_scheme
C = m.centroid; s.u[:] = 1.0; s.v[:] = 0.05 * np.exp(-((C[:, 0] - 3.0) ** 2 + C[:, 1] ** 2))
if a.init:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    d0 = np.load(a.init); P0 = d0["centroid"]
    for name in ("u", "v", "p"):
        val = LinearNDInterpolator(P0, d0[name])(C); miss = np.isnan(val)
        if miss.any(): val[miss] = NearestNDInterpolator(P0, d0[name])(C[miss])
        getattr(s, name)[:] = val
# ---- per-cylinder wall-flux forces
cyl = []
for tag in T_CYL:
    wall = m.bfaces[bt == tag]; wo = m.owner[wall]; Sw = m.normal[wall] * m.span; Aw = np.hypot(Sw[:, 0], Sw[:, 1]); e_in = -Sw / Aw[:, None]
    dn = ((m.fcentre[wall] - m.centroid[wo]) * (-e_in)).sum(axis=1); cyl.append((wo, Sw, e_in, dn))
def forces():
    out = []
    for wo, Sw, e_in, dn in cyl:
        gus = s.u[wo] / dn; gvs = s.v[wo] / dn; pf = s.p[wo]
        txx = 2 * nu * gus * e_in[:, 0]; tyy = 2 * nu * gvs * e_in[:, 1]; txy = nu * (gus * e_in[:, 1] + gvs * e_in[:, 0])
        Fx = (pf * Sw[:, 0] - (txx * Sw[:, 0] + txy * Sw[:, 1])).sum(); Fy = (pf * Sw[:, 1] - (txy * Sw[:, 0] + tyy * Sw[:, 1])).sum()
        out += [2 * Fx, 2 * Fy]
    return out                                                           # [CD1, CL1, CD2, CL2, CD3, CL3]
nsteps = a.nsteps or int(round(a.T / a.dt))
print(f"pinball {a.mesh}: {m.ncell} cells, wall faces {[int((bt == t).sum()) for t in T_CYL]}, Re={a.Re}, dt={a.dt}, scheme {a.time_scheme}, {nsteps} steps to T={nsteps*a.dt:.1f}", flush=True)
hist = np.zeros((nsteps, 7)); t0 = time.time()
for k in range(nsteps):
    s.step()
    if not np.isfinite(s.u).all(): print(f"  DIVERGED at step {k+1}", flush=True); hist = hist[:k]; break
    hist[k, 0] = s.time; hist[k, 1:] = forces()
    if (k + 1) % a.report == 0:
        w = hist[max(0, k - a.report + 1):k + 1]; f = w[:, 1:].mean(axis=0)
        print(f"  t={s.time:7.2f}  CD {f[0]:.4f} {f[2]:.4f} {f[4]:.4f} (sum {f[0]+f[2]+f[4]:.4f})  CL {f[1]:+.4f} {f[3]:+.4f} {f[5]:+.4f} (sum {f[1]+f[3]+f[5]:+.4f})  |u|max {np.abs(s.u).max():.3f}  dCD/dt {(hist[k,1]+hist[k,3]+hist[k,5]-(hist[max(0,k-a.report+1),1]+hist[max(0,k-a.report+1),3]+hist[max(0,k-a.report+1),5]))/(a.report*a.dt):+.2e}  ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
n2 = int(0.6 * len(hist)); w = hist[n2:]; CD = w[:, [1, 3, 5]].mean(axis=0); CL = w[:, [2, 4, 6]].mean(axis=0); CLa = 0.5 * (w[:, [2, 4, 6]].max(axis=0) - w[:, [2, 4, 6]].min(axis=0))
clt = w[:, [2, 4, 6]].sum(axis=1); clt = clt - clt.mean(); z = np.flatnonzero(np.diff(np.sign(clt)) > 0); t = w[:, 0]
St = (len(z) - 1) / (t[z[-1]] - t[z[0]]) if len(z) > 3 and CLa.max() > 1e-4 else float("nan")
print(f"RESULT {a.mesh} Re={a.Re}: CD_total={CD.sum():.4f} CD={np.round(CD, 4).tolist()} CL={np.round(CL, 4).tolist()} CL_amp={np.round(CLa, 4).tolist()} St={St:.4f} (window t={t[0]:.0f}..{t[-1]:.0f})", flush=True)
if a.out:
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, hist=hist, nodes=m.nodes, cells=m.cells, nvert=m.nvert, centroid=m.centroid, vol=m.vol, u=s.u, v=s.v, p=s.p, btag=m.btag)
