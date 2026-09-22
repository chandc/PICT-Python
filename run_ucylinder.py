"""T9: circular cylinder at Re = 100 on an unstructured/polygonal mesh, HydroGym Firedrake specs.

Boundaries (tags from the Gmsh physical groups): Inlet u = 1, v = 0; Freestream v = 0 with u
free (SYMMETRY -- HydroGym's `DirichletBC(V.sub(1), 0, FREESTREAM)`); Outlet p = 0 with u, v free;
Cylinder no-slip. Forces on the body from the fluid: with n the body's outward normal (into the
fluid) = -S_f/|S_f|,  F = sum_f [ p_f S_f - tau_f . S_f ],  tau = nu (grad u + grad u^T), rho = 1.
C_D = 2 F_x / (rho U^2 D) = 2 F_x with U = D = 1; C_L = 2 F_y. St from the lift spectrum.

Usage: python run_ucylinder.py MESH.msh [--Re 100] [--dt 0.01] [--T 150] [--out results/x.npz]
"""
import sys, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import read_gmsh22, Mesh
from src.uops import Gradient, DIRICHLET, NEUMANN
from src.upiso import PISO, BC

ap = argparse.ArgumentParser()
ap.add_argument("mesh"); ap.add_argument("--Re", type=float, default=100.0); ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--out", default=None); ap.add_argument("--report", type=int, default=500)
ap.add_argument("--nsteps", type=int, default=None)
ap.add_argument("--init", default=None, help="npz with centroid,u,v,p to interpolate as the initial state (no perturbation added)")
ap.add_argument("--rc", type=float, default=1.0, help="Rhie-Chow damping scale (PISO.rc_scale)")
ap.add_argument("--steady", action="store_true", help="project out the y-antisymmetric part every step (mirror map about y=0) so the flow converges to the unstable symmetric steady state; no perturbation")
a = ap.parse_args()

nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh)
m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}
T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
bt = m.btag[m.bfaces]
nb = m.nbface
ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN); vv = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN); vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO(m, nu=nu, dt=a.dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, vv), bc_p=BC(m, kp, vp),
         n_corr=2, n_nonorth=3, scheme="central", convect=True)
s.rc_scale = a.rc
# impulsive start with a small asymmetric perturbation so shedding does not wait on roundoff
C = m.centroid
# The perturbation must be EVEN in y to break the reflection symmetry: the base flow has v odd in y,
# so an odd perturbation is in the same symmetry class and leaves C_L identically zero (measured:
# 0.0000 over 300 steps with a sign(y) factor). A y-even blob of v in the near wake does it.
s.u[:] = 1.0; s.v[:] = 0.05 * np.exp(-((C[:, 0] - 1.0) ** 2 + C[:, 1] ** 2))
if a.init:
    # start from another mesh's solution (linear interpolation on centroids, nearest outside the hull);
    # used to read the unstable steady branch's C_D on a mesh that cannot be symmetrised: the
    # antisymmetric seed is then only the mesh asymmetry and C_D plateaus before shedding grows
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    d0 = np.load(a.init); P0 = d0["centroid"]
    for name in ("u", "v", "p"):
        lin = LinearNDInterpolator(P0, d0[name]); val = lin(C); miss = np.isnan(val)
        if miss.any(): val[miss] = NearestNDInterpolator(P0, d0[name])(C[miss])
        getattr(s, name)[:] = val
    print(f"  --init from {a.init}: u range {s.u.min():.3f}..{s.u.max():.3f}", flush=True)
if a.steady:
    # HydroGym's test_cyl.py::test_steady asserts C_D = 1.2840 (C_L = 0) for the Newton steady state at
    # Re = 100 on medium.msh. The steady state is unstable, so it is reached here by removing the
    # y-antisymmetric mode each step: u, p even in y, v odd. Needs a mirror-symmetric mesh.
    from scipy.spatial import cKDTree
    dist, mir = cKDTree(C).query(np.c_[C[:, 0], -C[:, 1]]); h = np.sqrt(m.vol)
    bad = dist > 0.05 * h
    if bad.any(): sys.exit(f"--steady: mesh is not mirror-symmetric ({bad.sum()} cells without a partner)")
    s.v[:] = 0.0
    print(f"  --steady: mirror map built, max partner distance {(dist / h).max():.2e} h", flush=True)
wall = m.bfaces[bt == T_CYL]; wo = m.owner[wall]; Sw = m.normal[wall] * m.span
# Wall shear from the solver's own discrete wall flux: at a no-slip face the diffusive flux the momentum
# equation applies is nu*|S|*(u_P - u_wall)/d_n, i.e. the one-sided normal derivative u_P/d_n. The earlier
# version reconstructed the stress from the wall cell's cell-centre gradient, which reads the shear d_n/2
# from the wall and was 3% low in the steady viscous drag (section 38 of the skew record: HydroGym's
# converged viscous drag 0.3302; cell-gradient 0.3204, one-sided 0.3351 on the fine butterfly). Tangential
# derivatives vanish at a no-slip wall, so grad u = (u_P/d_n) e with e the unit normal into the fluid.
# Pressure at the wall is the owner value (Neumann). rho = U = D = 1: C_D = 2 F_x, C_L = 2 F_y.
Aw = np.hypot(Sw[:, 0], Sw[:, 1]); e_in = -Sw / Aw[:, None]                       # into the fluid
dn = ((m.fcentre[wall] - m.centroid[wo]) * (-e_in)).sum(axis=1)                     # centroid-to-wall distance
def forces():
    gus = s.u[wo] / dn; gvs = s.v[wo] / dn                                          # d/ds of u, v at the wall
    gux, guy = gus * e_in[:, 0], gus * e_in[:, 1]; gvx, gvy = gvs * e_in[:, 0], gvs * e_in[:, 1]
    pf = s.p[wo]
    txx = 2 * nu * gux; tyy = 2 * nu * gvy; txy = nu * (guy + gvx)
    Fx = (pf * Sw[:, 0] - (txx * Sw[:, 0] + txy * Sw[:, 1])).sum()
    Fy = (pf * Sw[:, 1] - (txy * Sw[:, 0] + tyy * Sw[:, 1])).sum()
    Fpx = (pf * Sw[:, 0]).sum()
    return 2 * Fx, 2 * Fy, 2 * Fpx
nsteps = a.nsteps or int(round(a.T / a.dt))
print(f"T9 {a.mesh}: {m.ncell} cells ({int((m.nvert==4).sum())} quads), {len(wall)} wall faces, Re={a.Re}, dt={a.dt}, {nsteps} steps to T={nsteps*a.dt:.1f}", flush=True)
hist = np.zeros((nsteps, 4)); t0 = time.time()
for k in range(nsteps):
    s.step()
    if a.steady:
        s.u[:] = 0.5 * (s.u + s.u[mir]); s.v[:] = 0.5 * (s.v - s.v[mir]); s.p[:] = 0.5 * (s.p + s.p[mir])
    if not np.isfinite(s.u).all():
        print(f"  DIVERGED at step {k+1}", flush=True); hist = hist[:k]; break
    cd, cl, cdp = forces(); hist[k] = (s.time, cd, cl, cdp)
    if (k + 1) % a.report == 0:
        w = hist[max(0, k - a.report + 1):k + 1]
        print(f"  t={s.time:7.2f}  Cd {w[:,1].mean():.4f}  Cl {w[:,2].mean():+.4f} (amp {0.5*(w[:,2].max()-w[:,2].min()):.4f})  Cd_p {w[:,3].mean():.4f}  |u|max {np.abs(s.u).max():.3f}  dCd/dt {(hist[k,1]-hist[max(0,k-a.report+1),1])/(a.report*a.dt):+.2e}  ({(time.time()-t0)/(k+1)*1e3:.1f} ms/step)", flush=True)
# Strouhal from the last 40% of the lift history
n2 = int(0.6 * len(hist)); t, cl = hist[n2:, 0], hist[n2:, 2] - hist[n2:, 2].mean()
if len(t) > 64:
    # St from upward zero-crossings of C_L, linearly interpolated: with ~10 periods in the window the
    # FFT bin is 0.017 wide (measured 0.1833 against a zero-crossing 0.1750 on the same signal), far
    # too coarse to compare meshes at the 1% level.
    z = np.flatnonzero(np.diff(np.sign(cl)) > 0); tz = t[z] - cl[z] * (t[z + 1] - t[z]) / (cl[z + 1] - cl[z])
    per = np.diff(tz); f0 = 1.0 / per.mean() if len(per) >= 3 else float("nan")
    print(f"  {len(per)} periods, mean {per.mean():.4f} +- {per.std():.4f}", flush=True)
    print(f"RESULT {a.mesh}: St={f0:.4f}  Cd_mean={hist[n2:,1].mean():.4f}  Cl_rms={np.sqrt((cl**2).mean()):.4f}  Cl_amp={0.5*(hist[n2:,2].max()-hist[n2:,2].min()):.4f}  Cd_p={hist[n2:,3].mean():.4f}  (window t={t[0]:.1f}..{t[-1]:.1f})", flush=True)
if a.out:
    np.savez(a.out, hist=hist, nodes=m.nodes, cells=m.cells, nvert=m.nvert, centroid=m.centroid, vol=m.vol, u=s.u, v=s.v, p=s.p, btag=m.btag)
    print(f"  saved {a.out}", flush=True)
