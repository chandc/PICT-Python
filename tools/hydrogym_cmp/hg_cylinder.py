"""Run HydroGym's own Firedrake cylinder at Re = 100 to get the numbers it never published:
steady C_D (Newton, unstable branch) and the saturated shedding St / C_D / C_L on medium.msh.

    python hg_cylinder.py --mesh medium --order 2 --T 200 --dt 0.01 --out out/medium_p2
    mpirun -np 8 python hg_cylinder.py ...

Follows examples/firedrake/advanced/cylinder/unsteady.py: Reynolds-ramped Newton solve, then a
1e-3 random perturbation and SemiImplicitBDF (order 3) at dt. Forces from flow.compute_forces()
-> (C_L, C_D), the same definition as the unit test that asserts C_D = 1.2840.
"""
import argparse, os, sys, time
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="medium"); ap.add_argument("--order", type=int, default=2)
ap.add_argument("--T", type=float, default=200.0); ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--out", default="out/run"); ap.add_argument("--steady-only", action="store_true")
ap.add_argument("--stab", default="none"); ap.add_argument("--save-fields", default=None, help="npz: cell-centroid u, v, p, vorticity (DG0) at the final time")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rank0 = fd.COMM_WORLD.rank == 0
def P(*s):
    if rank0: print(*s, flush=True)

flow = hgym.Cylinder(Re=100, mesh=a.mesh, velocity_order=a.order, use_HF_data_manager=False)
P(f"HydroGym Cylinder mesh={a.mesh} velocity_order={a.order}: cells {flow.mesh.num_cells()}, "
  f"velocity dofs {flow.velocity_space.dim()}, pressure dofs {flow.pressure_space.dim()}, ranks {fd.COMM_WORLD.size}")

# ---- steady branch (Newton, Reynolds ramp as in unsteady.py)
t0 = time.time()
for Re_val in [40, 60, 80, 100]:
    flow.Re.assign(Re_val)
    hgym.NewtonSolver(flow, stabilization=a.stab).solve()
    CL, CD = flow.compute_forces(); P(f"  steady Re={Re_val}: CL={CL:+.6f} CD={CD:.6f}")
CLs, CDs = flow.compute_forces()
P(f"STEADY mesh={a.mesh} order={a.order}: CD={CDs:.6f} CL={CLs:+.2e}  ({time.time()-t0:.0f}s)")
if a.steady_only: sys.exit(0)

# ---- shedding: perturb and integrate, log every step
rng = fd.RandomGenerator(fd.PCG64(seed=42))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-3)
def post(flow):
    CL, CD = flow.compute_forces(); return [CL, CD]
log = hgym.utils.io.LogCallback(postprocess=post, nvals=2, interval=1, filename=f"{a.out}_forces.dat",
                                print_fmt=None)
_t0 = [time.time()]
def progress(it, t, flow):
    CL, CD = flow.compute_forces()
    P(f"  t={t:7.2f}  CL={CL:+.4f}  CD={CD:.4f}  ({(time.time()-_t0[0])/max(it,1)*1e3:.0f} ms/step)")
prog = hgym.utils.io.GenericCallback(progress, interval=int(round(5.0 / a.dt)))
t0 = time.time()
hgym.integrate(flow, t_span=(0, a.T), dt=a.dt, callbacks=[log, prog], stabilization=a.stab)
P(f"  integration done in {time.time()-t0:.0f}s")

# ---- final fields at cell centroids (DG0), gathered to rank 0
if a.save_fields:
    from ufl import curl, as_vector
    DG0 = fd.FunctionSpace(flow.mesh, "DG", 0); VDG0 = fd.VectorFunctionSpace(flow.mesh, "DG", 0)
    X = fd.Function(VDG0).interpolate(fd.SpatialCoordinate(flow.mesh)); U = fd.Function(VDG0).interpolate(flow.u)
    Pc = fd.Function(DG0).interpolate(flow.p)
    W = fd.Function(DG0).interpolate(flow.u[1].dx(0) - flow.u[0].dx(1))
    Wc = fd.Function(fd.FunctionSpace(flow.mesh, "CG", 1)).project(flow.u[1].dx(0) - flow.u[0].dx(1))   # smooth vorticity as they plot it
    Xv = flow.mesh.coordinates
    parts = fd.COMM_WORLD.gather((X.dat.data_ro.copy(), U.dat.data_ro.copy(), Pc.dat.data_ro.copy(), W.dat.data_ro.copy(),
                                  Xv.dat.data_ro.copy(), Wc.dat.data_ro.copy()), root=0)
    if rank0:
        cat = lambda k: np.concatenate([q[k] for q in parts])
        np.savez(a.save_fields, centroid=cat(0), uv=cat(1), p=cat(2), vort=cat(3), vert=cat(4), vort_cg1=cat(5), t=a.T)
        P(f"  saved fields {a.save_fields}: {len(cat(0))} cells, {len(cat(4))} vertices")
# ---- post-process the last 40% of the run: St by interpolated zero-crossings of C_L
if rank0:
    d = np.loadtxt(f"{a.out}_forces.dat"); t, cl, cd = d[:, 0], d[:, 1], d[:, 2]
    n2 = int(0.6 * len(t)); tw, clw, cdw = t[n2:], cl[n2:] - cl[n2:].mean(), cd[n2:]
    z = np.flatnonzero(np.diff(np.sign(clw)) > 0); tz = tw[z] - clw[z] * (tw[z + 1] - tw[z]) / (clw[z + 1] - clw[z])
    per = np.diff(tz)
    print(f"  {len(per)} periods, mean {per.mean():.4f} +- {per.std():.4f}", flush=True)
    print(f"RESULT hydrogym mesh={a.mesh} order={a.order} dt={a.dt}: St={1/per.mean():.4f}  Cd_mean={cdw.mean():.4f}  "
          f"Cl_rms={np.sqrt((clw**2).mean()):.4f}  Cl_amp={0.5*(cl[n2:].max()-cl[n2:].min()):.4f}  "
          f"Cd_amp={0.5*(cdw.max()-cdw.min()):.4f}  (window t={tw[0]:.1f}..{tw[-1]:.1f})  steady CD={CDs:.4f}", flush=True)
