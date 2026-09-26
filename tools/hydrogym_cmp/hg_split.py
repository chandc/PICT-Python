"""HydroGym steady Newton solve with the drag split into pressure and viscous parts (same sigma as
compute_forces): python3 hg_split.py --mesh medium --order 2"""
import argparse, firedrake as fd, hydrogym.firedrake as hgym
from ufl import dot, ds
ap = argparse.ArgumentParser(); ap.add_argument("--mesh", default="medium"); ap.add_argument("--order", type=int, default=2); a = ap.parse_args()
flow = hgym.Cylinder(Re=100, mesh=a.mesh, velocity_order=a.order, use_HF_data_manager=False)
for Re_val in [40, 60, 80, 100]:
    flow.Re.assign(Re_val); hgym.NewtonSolver(flow, stabilization="none").solve()
u, p = fd.split(flow.q); n = flow.n
fp = -dot(-p * fd.Identity(2), n); fv = -dot(2 * flow.nu * flow.epsilon(u), n)
CDp = fd.assemble(2 * fp[0] * ds(flow.CYLINDER)); CDv = fd.assemble(2 * fv[0] * ds(flow.CYLINDER))
CL, CD = flow.compute_forces()
print(f"SPLIT mesh={a.mesh} order={a.order}: CD={CD:.6f}  CD_p={CDp:.6f}  CD_v={CDv:.6f}  (sum {CDp+CDv:.6f})", flush=True)
