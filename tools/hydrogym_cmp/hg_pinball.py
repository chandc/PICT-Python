"""HydroGym's own Firedrake fluidic pinball (three cylinders): the reference numbers for our unstructured
solver on the same mesh. Steady Newton solve from the symmetric state (the symmetric branch, unstable
above Re ~ 18 so what a Newton solve from a symmetric guess converges to), then a perturbed transient to
the attractor (asymmetric steady state below the Hopf point near Re 68, shedding above).
    python hg_pinball.py --Re 30 --mesh medium --T 150 --dt 0.01 --out out/pinball_re30
Forces per cylinder from flow.compute_forces() -> (CL[3], CD[3]).
"""
import argparse, os, sys, time
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
ap = argparse.ArgumentParser()
ap.add_argument("--Re", type=float, default=30.0); ap.add_argument("--mesh", default="medium"); ap.add_argument("--order", type=int, default=2)
ap.add_argument("--T", type=float, default=150.0); ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--out", default="out/pinball")
ap.add_argument("--steady-only", action="store_true"); ap.add_argument("--save-fields", default=None)
ap.add_argument("--impulsive", action="store_true", help="skip the Newton solve: start from uniform flow plus a y-even blob of v in the near wake (our solver's start), to see which state their solver reaches from there")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rank0 = fd.COMM_WORLD.rank == 0
def P(*s):
    if rank0: print(*s, flush=True)
flow = hgym.Pinball(Re=a.Re, mesh=a.mesh, velocity_order=a.order, use_HF_data_manager=False)
P(f"HydroGym Pinball mesh={a.mesh} order={a.order} Re={a.Re}: cells {flow.mesh.num_cells()}, velocity dofs {flow.velocity_space.dim()}, ranks {fd.COMM_WORLD.size}")
t0 = time.time()
if a.impulsive:
    x, y = fd.SpatialCoordinate(flow.mesh); u0 = fd.as_vector((1.0, 0.05 * fd.exp(-((x - 3.0) ** 2 + y ** 2))))
    flow.q.sub(0).interpolate(u0); flow.q.sub(1).assign(0.0)
for Re_val in ([] if a.impulsive else ([10, 20, a.Re] if a.Re > 20 else [a.Re])):
    flow.Re.assign(Re_val); hgym.NewtonSolver(flow).solve()
    CL, CD = flow.compute_forces(); P(f"  steady Re={Re_val}: CL={[round(float(c), 6) for c in CL]} CD={[round(float(c), 6) for c in CD]}")
CL, CD = flow.compute_forces()
P(f"STEADY mesh={a.mesh} Re={a.Re}: CD_total={sum(float(c) for c in CD):.6f} CL={[float(c) for c in CL]} CD={[float(c) for c in CD]} ({time.time()-t0:.0f}s)")
if a.steady_only: sys.exit(0)
if not a.impulsive:
    rng = fd.RandomGenerator(fd.PCG64(seed=42)); flow.q += rng.normal(flow.mixed_space, 0.0, 1e-3)
def post(flow):
    CL, CD = flow.compute_forces(); return [*map(float, CL), *map(float, CD)]
log = hgym.utils.io.LogCallback(postprocess=post, nvals=6, interval=1, filename=f"{a.out}_forces.dat", print_fmt=None)
_t0 = [time.time()]
def progress(it, t, flow):
    CL, CD = flow.compute_forces(); P(f"  t={t:7.2f}  CL={[round(float(c), 4) for c in CL]}  CD={[round(float(c), 4) for c in CD]}  ({(time.time()-_t0[0])/max(it,1)*1e3:.0f} ms/step)")
prog = hgym.utils.io.GenericCallback(progress, interval=int(round(5.0 / a.dt)))
t0 = time.time(); hgym.integrate(flow, t_span=(0, a.T), dt=a.dt, callbacks=[log, prog]); P(f"  integration done in {time.time()-t0:.0f}s")
if a.save_fields and rank0:
    VDG0 = fd.VectorFunctionSpace(flow.mesh, "DG", 0); DG0 = fd.FunctionSpace(flow.mesh, "DG", 0)
    X = fd.Function(VDG0).interpolate(fd.SpatialCoordinate(flow.mesh)); U = fd.Function(VDG0).interpolate(flow.u); Pp = fd.Function(DG0).interpolate(flow.p)
    np.savez(a.save_fields, centroid=X.dat.data_ro.copy(), uv=U.dat.data_ro.copy(), p=Pp.dat.data_ro.copy(), t=a.T)
if rank0:
    F = np.loadtxt(f"{a.out}_forces.dat"); t = F[:, 0]; n2 = int(0.6 * len(t)); w = F[n2:]
    CLm = w[:, 1:4].mean(axis=0); CDm = w[:, 4:7].mean(axis=0); CLa = 0.5 * (w[:, 1:4].max(axis=0) - w[:, 1:4].min(axis=0))
    clt = w[:, 1:4].sum(axis=1) - w[:, 1:4].sum(axis=1).mean(); z = np.flatnonzero(np.diff(np.sign(clt)) > 0)
    St = (len(z) - 1) / (t[n2:][z[-1]] - t[n2:][z[0]]) if len(z) > 3 else float("nan")
    print(f"RESULT hydrogym pinball mesh={a.mesh} Re={a.Re}: CD_total={CDm.sum():.4f} CD={np.round(CDm, 4).tolist()} CL={np.round(CLm, 4).tolist()} CL_amp={np.round(CLa, 4).tolist()} St={St:.4f} (window t={t[n2]:.0f}..{t[-1]:.0f})", flush=True)
