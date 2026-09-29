"""HydroGym's own open-cavity flow, rebuilt on UniFlow: same mesh (their fine.msh/medium.msh, read
directly), same BCs, same Re, for a like-for-like comparison against tools/hydrogym_cmp/hg_cavity.py's
Firedrake run (results/hydrogym_cmp/cavity_re7500_fine/stats.dat).

Geometry/BCs (from hydrogym/firedrake/envs/cavity/{flow.py,fine.geo}): recess x in [0,1], y in [-1,0];
upstream plate x in [-1.2,0] at y=0 (mostly Slip, a short no-slip Wall segment, then the Control
actuator slot, all v=0 -- here amplitude 0, uncontrolled); Inlet u=(1,0) at x=-1.2; Freestream v=0 at
y=0.5; Outlet p=0 at x=2.5; downstream plate (Wall) and the Sensor segment (x in [1,1.1], also
no-slip) after the trailing edge. nu = 1/Re directly (hydrogym/firedrake/flow.py: self.nu = 1/Re),
matching the mesh's own coordinate units -- no additional length/velocity rescaling.

TKE here is fluctuation energy about the RUNNING TIME MEAN (there is no Newton steady base flow in
this transient-only driver), not HydroGym's fluctuation-vs-Newton-steady-state; the two agree once the
running mean has converged past the initial transient, which is the window compared.

Usage: python run_ucavity_hg.py --mesh meshes/hgcavity_fine.msh --Re 7500 --dt 0.001 --T 50
"""
import sys, os, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import read_gmsh22, Mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/hgcavity_medium.msh")
ap.add_argument("--Re", type=float, default=7500.0)
ap.add_argument("--dt", type=float, default=0.001)
ap.add_argument("--T", type=float, default=50.0)
ap.add_argument("--t-mean", type=float, default=15.0, help="running time-mean (for TKE) starts accumulating after this time, matching hg_cavity.py's analysis window")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--pert", type=float, default=1e-2, help="amplitude of the initial white-noise-like cell perturbation, matching hg_cavity.py's fd.RandomGenerator(...).normal(0, 1e-2)")
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
ap.add_argument("--report", type=int, default=500)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--nsteps", type=int, default=None, help="stop after this many steps regardless of --T (smoke testing)")
ap.add_argument("--outdir", default="results/ucavity_hg")
ap.add_argument("--checkpoint", type=int, default=5000)
ap.add_argument("--restart", default=None)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh)
m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}
T_IN, T_FS, T_OUT, T_SLIP, T_WALL, T_CTRL, T_SENS = (inv[n] for n in
    ("Inlet", "Freestream", "Outlet", "Slip", "Wall", "Control", "Sensor"))
bt = m.btag[m.bfaces]; nb = m.nbface

# u: Dirichlet 1 on Inlet, 0 on Wall/Control/Sensor (all no-slip, uncontrolled); free (Neumann) on
# Freestream/Slip (symmetry: only v is pinned there) and Outlet.
ku = np.where(np.isin(bt, [T_IN, T_WALL, T_CTRL, T_SENS]), DIRICHLET, NEUMANN)
vu = np.where(bt == T_IN, 1.0, 0.0)
# v: Dirichlet 0 everywhere except Outlet (free), matching flow.py's per-component BCs exactly --
# Inlet, Freestream, Slip, Wall, Control (uncontrolled), Sensor are ALL v = 0 there.
kv = np.where(bt == T_OUT, NEUMANN, DIRICHLET)
vv = np.zeros(nb)
# p: Dirichlet 0 on Outlet only.
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN)
vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO(m, nu=nu, dt=a.dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, vv), bc_p=BC(m, kp, vp),
         n_corr=2, n_nonorth=3, scheme="central", convect=True)

rng = np.random.default_rng(a.seed)
if a.restart:
    d0 = np.load(a.restart)
    s.u[:], s.v[:], s.p[:] = d0["u"], d0["v"], d0["p"]
    t_start, k_start = float(d0["time"]), int(d0["nstep"])
    print(f"  restarted from {a.restart} at t={t_start:.3f}", flush=True)
else:
    # Freestream/inlet region starts at u=1 (outside the recess), recess starts at rest, plus a
    # small perturbation everywhere -- matches hg_cavity.py's "steady base flow + N(0, pert)" in
    # spirit; we have no Newton steady solve, so the base state is this simple initial guess and
    # the shear layer instability is left to grow from the perturbation as the flow develops.
    C = m.centroid
    in_cavity = (C[:, 0] > 0.0) & (C[:, 0] < 1.0) & (C[:, 1] < 0.0)
    s.u[:] = np.where(in_cavity, 0.0, 1.0) + a.pert * rng.standard_normal(m.ncell)
    s.v[:] = a.pert * rng.standard_normal(m.ncell)
    t_start, k_start = 0.0, 0

wall = m.bfaces[np.isin(bt, [T_WALL, T_SENS])]
sens = m.bfaces[bt == T_SENS]
wo_all = m.owner[wall]; Sw_all = m.normal[wall] * m.span
wo_s = m.owner[sens]; Sw_s = m.normal[sens] * m.span
dn_s = ((m.fcentre[sens] - m.centroid[wo_s]) * (-Sw_s / np.hypot(Sw_s[:, 0], Sw_s[:, 1])[:, None])).sum(axis=1)
e_in_s = -Sw_s / np.hypot(Sw_s[:, 0], Sw_s[:, 1])[:, None]

def wall_stress_sensor():
    # -du_x/dy integrated over the Sensor segment (Barbagallo et al. 2009's trailing-edge probe,
    # hydrogym/firedrake/envs/cavity/flow.py: wall_stress_sensor). Our one-sided wall-normal
    # derivative convention: du_x/dn = u_owner_x / d_n along the wall-into-fluid normal; here the
    # Sensor segment is horizontal (wall-normal = -y), so du_x/dy = -du_x/dn.
    gux_dn = s.u[wo_s] / dn_s  # d(u_x)/dn, n into the fluid
    # e_in_s is the unit normal into the fluid (points +y on this horizontal segment, fluid
    # above); du_x/dy = gux_dn * (e_in . e_y), which is just gux_dn * e_in_s[:, 1].
    dudy = gux_dn * e_in_s[:, 1]
    Aw = np.hypot(Sw_s[:, 0], Sw_s[:, 1])
    return -(dudy * Aw).sum()

nsteps = a.nsteps or int(round((a.T - t_start) / a.dt))
print(f"cavity {a.mesh}: {m.ncell} cells, Re={a.Re}, dt={a.dt}, {nsteps} steps to T={t_start+nsteps*a.dt:.1f}", flush=True)

mean_u = np.zeros(m.ncell); mean_v = np.zeros(m.ncell); n_mean = 0
stats = []
t0 = time.time()
for k in range(nsteps):
    s.step()
    if not np.isfinite(s.u).all():
        print(f"  DIVERGED at step {k+1}", flush=True); break
    t = t_start + (k + 1) * a.dt
    if t > a.t_mean:
        mean_u += s.u; mean_v += s.v; n_mean += 1
        fu, fv = s.u - mean_u / n_mean, s.v - mean_v / n_mean
        TKE = 0.5 * (m.vol * (fu * fu + fv * fv)).sum()
    else:
        TKE = np.nan
    KE = 0.5 * (m.vol * (s.u * s.u + s.v * s.v)).sum()
    hmin = np.sqrt(m.vol).min()
    CFL = a.dt * (np.abs(s.u).max() + np.abs(s.v).max()) / hmin
    sensor = wall_stress_sensor()
    if (k + 1) % a.log_every == 0:
        stats.append((t, CFL, KE, TKE, sensor))
    if (k + 1) % a.report == 0:
        print(f"  t={t:7.3f}  CFL {CFL:.3f}  KE {KE:.5e}  TKE {TKE:.5e}  sensor {sensor:+.4e}  ({(time.time()-t0)/(k+1)*1e3:.1f} ms/step)", flush=True)
    if a.checkpoint and (k + 1) % a.checkpoint == 0:
        np.savez(f"{a.outdir}/ckpt.npz", u=s.u, v=s.v, p=s.p, time=t, nstep=k_start + k + 1)

stats = np.array(stats)
np.savetxt(f"{a.outdir}/stats.dat", stats, header="t CFL KE TKE sensor")
np.savez(f"{a.outdir}/final.npz", u=s.u, v=s.v, p=s.p, time=t_start + nsteps * a.dt, nstep=k_start + nsteps,
          centroid=m.centroid, nodes=m.nodes, cells=m.cells, nvert=m.nvert, btag=m.btag)
print(f"wrote {a.outdir}/stats.dat and {a.outdir}/final.npz", flush=True)
