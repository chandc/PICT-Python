"""HydroGym's own Firedrake open-cavity benchmark (Cavity_2D_Re7500_fine_FD), run directly, uncontrolled,
as a reference case (Barbagallo et al. 2009's closed-loop cavity setup: single leading-edge blowing/
suction actuator, trailing-edge wall-shear-stress sensor -- here the actuator stays at zero). Two stages,
matching HydroGym's own examples/firedrake/advanced/cavity/unsteady.py: Newton steady solve with Reynolds
ramping (500 -> 1000 -> 2000 -> 4000 -> Re), then a perturbed long transient (BDF, dt from the CLI) to
develop and statistics the shear-layer instability. Logs CFL, KE, TKE (fluctuation energy vs the base
flow) and the trailing-edge stress sensor every `--log-every` steps to `<outdir>/stats.dat`.
    python tools/hydrogym_cmp/hg_cavity.py --mesh fine --Re 7500 --Tf 500 --dt 2.5e-4 --outdir results/hydrogym_cmp/cavity_re7500_fine"""
import os, sys, time, argparse
import firedrake as fd
import hydrogym.firedrake as hgym

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="fine", choices=["fine", "medium"]); ap.add_argument("--Re", type=float, default=7500.0)
ap.add_argument("--Tf", type=float, default=500.0); ap.add_argument("--dt", type=float, default=2.5e-4)
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--outdir", default="results/hydrogym_cmp/cavity_re7500_fine")
ap.add_argument("--nsteps", type=int, default=None, help="stop after this many steps regardless of Tf (smoke testing)")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

flow = hgym.Cavity(Re=a.Re, mesh=a.mesh, use_HF_data_manager=False)
dof = flow.mixed_space.dim(); hgym.print(f"mesh {a.mesh}: {dof} dof (rank {fd.COMM_WORLD.rank}/{fd.COMM_WORLD.size})")

hgym.print("=" * 70); hgym.print("Stage 1: steady base flow, Reynolds ramp 500 -> 1000 -> 2000 -> 4000 -> Re"); hgym.print("=" * 70)
t0 = time.time()
for Re_i in (500, 1000, 2000, 4000, a.Re):
    flow.Re.assign(Re_i); hgym.print(f"  steady solve at Re={Re_i}")
    solver = hgym.NewtonSolver(flow, solver_parameters={"snes_monitor": None})
    flow.qB.assign(solver.solve())
flow.save_checkpoint(f"{a.outdir}/{int(a.Re)}_{a.mesh}_steady.h5")
hgym.print(f"steady stage done in {time.time()-t0:.0f}s")

hgym.print("=" * 70); hgym.print(f"Stage 2: perturbed transient, Tf={a.Tf}, dt={a.dt}"); hgym.print("=" * 70)
rng = fd.RandomGenerator(fd.PCG64(seed=a.seed)); flow.q += rng.normal(flow.mixed_space, 0.0, 1e-2)

def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = flow.evaluate_objective()
    CFL = flow.max_cfl(a.dt)
    sensor = flow.wall_stress_sensor()[0]
    return [CFL, KE, TKE, sensor]

print_fmt = "t: {0:0.4f}\tCFL: {1:0.3f}\tKE: {2:0.6e}\tTKE: {3:0.6e}\tsensor: {4:0.6e}"
callbacks = [hgym.io.LogCallback(postprocess=log_postprocess, nvals=4, interval=a.log_every, filename=f"{a.outdir}/stats.dat", print_fmt=print_fmt)]

t_span = (0, a.Tf) if a.nsteps is None else (0, a.nsteps * a.dt)
t1 = time.time(); hgym.print("starting time integration ...")
hgym.integrate(flow, t_span=t_span, dt=a.dt, callbacks=callbacks)
nsteps_done = round((t_span[1] - t_span[0]) / a.dt)
hgym.print(f"transient stage done: {nsteps_done} steps in {time.time()-t1:.0f}s ({(time.time()-t1)/max(nsteps_done,1)*1e3:.1f} ms/step)")
flow.save_checkpoint(f"{a.outdir}/{int(a.Re)}_{a.mesh}_final.h5")
hgym.print(f"wrote {a.outdir}/{int(a.Re)}_{a.mesh}_final.h5 and {a.outdir}/stats.dat")
