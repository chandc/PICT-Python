"""Gartling's backward-facing-step benchmark (Gartling 1990, IJNMF 11:953), on UniFlow's unstructured
PISO. Convention taken from ~/Dropbox/Apple_MLX_CFD/sem_demo/GARTLING_VALIDATION.md (an independent
least-squares spectral-element reproduction of Chan & Mittal's CTR 1996 figures 3-6, cross-checked
there against Gartling's own paper): domain [0,17]x[-0.5,0.5] (a single rectangle -- the "step" is
purely the inflow boundary condition, not a meshed geometric step), parabolic inflow u=24y(0.5-y) over
the UPPER half of the left edge only (y in [0,0.5], peak 1.5, mean 1), no-slip over the lower half (the
step face) and top/bottom walls, outlet p=0 with u,v free (our FV analogue of their P+Z condition).
nu = 1/800 (Re built on the inlet hydraulic diameter 2h=1 and mean velocity 1 -- NOT literally on the
step height; that reading is the "viscosity trap" documented in section 1 of the reference note, and is
what reproduces Gartling's quoted reattachment of 6.1).

Reference (Chan & Mittal 1996, reproducing Gartling 1990): lower (primary) reattachment x_r/S = 6.1,
upper separation 4.8, upper reattachment 10.5 (S = step height = 0.5 here).

Runs from rest to steady state (BDF2 time-marching as a pseudo-transient; this is a steady laminar flow
at Re=800 on a well-resolved mesh -- the reference study found a genuine physical steady state on its
finer grid and a mesh-induced spurious limit cycle on a much coarser one, so convergence is checked, not
assumed).

Usage: python run_ugartling_bfs.py --mesh meshes/gartling_bfs.msh --T 300"""
import sys, os, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from src.umesh import read_gmsh22, Mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/gartling_bfs.msh")
ap.add_argument("--Re", type=float, default=800.0)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--report", type=int, default=2000)
ap.add_argument("--nsteps", type=int, default=None, help="stop after this many steps regardless of --T (smoke testing)")
ap.add_argument("--conv-tol", type=float, default=1e-8, help="stop early once max|du/dt|,max|dv/dt| (per unit dt) fall below this, checked every --report steps")
ap.add_argument("--ckpt-dt", type=float, default=None, help="write a separate checkpoint file every this many time units (e.g. 20), in addition to the final result")
ap.add_argument("--outdir", default="results/ugartling_bfs")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh)
m = Mesh(nodes, cells, edges, etag, names)
inv = {v: k for k, v in names.items()}
T_IN, T_OUT, T_WALL = inv["Inlet"], inv["Outlet"], inv["Wall"]
bt = m.btag[m.bfaces]; nb = m.nbface
yb = m.fcentre[m.bfaces][:, 1]

ku = np.where(np.isin(bt, [T_IN, T_WALL]), DIRICHLET, NEUMANN)
vu = np.where(bt == T_IN, np.clip(24.0 * yb * (0.5 - yb), 0.0, None), 0.0)
kv = np.where(bt == T_OUT, NEUMANN, DIRICHLET)
vv = np.zeros(nb)
kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN)
vp = np.zeros(nb)
nu = 1.0 / a.Re
s = PISO(m, nu=nu, dt=a.dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, vv), bc_p=BC(m, kp, vp),
         n_corr=2, n_nonorth=3, scheme="central", convect=True)
# Chan & Mittal's figs 5/6 start "from rest" (u=v=0 everywhere), but that is their VVP spectral
# scheme; on this collocated PISO, zero interior + the full parabolic inlet from step 1 is an
# impulsive shock that blows up within ~50 steps (measured: |u|max 14 -> 116 over t=0.05-0.25).
# Every other case in this project avoids exactly this by matching the interior IC to the inflow
# (run_ucylinder.py: s.u[:] = 1.0), so initialise u to the inflow profile extended uniformly
# downstream (0 below the step) instead of true rest -- still an impulsive guess, not a converged
# base flow, just not a step discontinuity in both space AND the full Dirichlet value at once.
Cy = m.centroid[:, 1]
s.u[:] = np.clip(24.0 * Cy * (0.5 - Cy), 0.0, None)

nsteps = a.nsteps or int(round(a.T / a.dt))
print(f"Gartling BFS {a.mesh}: {m.ncell} cells, Re={a.Re}, dt={a.dt}, {nsteps} steps to T={nsteps*a.dt:.1f}", flush=True)

u_prev, v_prev = s.u.copy(), s.v.copy()
hist = []
t0 = time.time()
next_ckpt = a.ckpt_dt if a.ckpt_dt else None
for k in range(nsteps):
    s.step()
    if not np.isfinite(s.u).all():
        print(f"  DIVERGED at step {k+1}", flush=True); break
    if next_ckpt is not None and s.time >= next_ckpt - 1e-9:
        fn = f"{a.outdir}/ckpt_t{round(next_ckpt):04d}.npz"
        np.savez(fn, u=s.u, v=s.v, p=s.p, time=s.time, nstep=k + 1,
                 centroid=m.centroid, nodes=m.nodes, cells=m.cells, nvert=m.nvert, btag=m.btag)
        print(f"  wrote checkpoint {fn}", flush=True)
        next_ckpt += a.ckpt_dt
    if (k + 1) % a.report == 0:
        du = np.abs(s.u - u_prev).max() / (a.report * a.dt)
        dv = np.abs(s.v - v_prev).max() / (a.report * a.dt)
        u_prev[:], v_prev[:] = s.u, s.v
        hist.append((s.time, du, dv, np.abs(s.u).max(), np.abs(s.v).max()))
        print(f"  t={s.time:7.2f}  max|du/dt| {du:.3e}  max|dv/dt| {dv:.3e}  |u|max {np.abs(s.u).max():.3f}  |v|max {np.abs(s.v).max():.3f}  ({(time.time()-t0)/(k+1)*1e3:.2f} ms/step)", flush=True)
        if du < a.conv_tol and dv < a.conv_tol:
            print(f"  converged (tol {a.conv_tol:.1e}) at t={s.time:.2f}", flush=True); break

# Wall vorticity zero-crossings on the bottom wall (lower/primary reattachment) and top wall
# (upper separation/reattachment), the independent check the reference study uses (section 4 there):
# on a no-slip wall v=0 so omega = dv/dx - du/dy = -du/dy exactly, read from the one-sided wall-normal
# derivative, same convention as run_ucylinder.py / run_ucavity_hg.py's forces().
wall = m.bfaces[bt == T_WALL]; wo = m.owner[wall]; Sw = m.normal[wall]
e_in = -Sw / np.hypot(Sw[:, 0], Sw[:, 1])[:, None]
dn = ((m.fcentre[wall] - m.centroid[wo]) * e_in).sum(axis=1)
xw = m.fcentre[wall][:, 0]; yw = m.fcentre[wall][:, 1]
omega_w = -(s.u[wo] / dn)  # du/dn with n into the fluid; on a horizontal wall this IS -omega sign-adjusted below
# sign convention: omega = -du/dy; on the BOTTOM wall n=+y so du/dn = du/dy -> omega = -du/dn
# on the TOP wall n=-y so du/dn = -du/dy -> omega = +du/dn. Both folded into e_in[:,1] already carrying
# the correct sign of dy, so omega = -(du/dn) * sign(e_in_y)/|e_in_y| reduces, for a purely horizontal
# wall (e_in = (0, +-1)), to omega = -(du/dn) * e_in_y:
omega_w = -(s.u[wo] / dn) * e_in[:, 1]

def zero_crossings(mask, xlo, xhi):
    sel = mask & (xw > xlo) & (xw < xhi)
    xs, om = xw[sel], omega_w[sel]
    order = np.argsort(xs); xs, om = xs[order], om[order]
    z = np.flatnonzero(np.diff(np.sign(om)) != 0)
    return [float(xs[i] - om[i] * (xs[i+1] - xs[i]) / (om[i+1] - om[i])) for i in z]

bot = yw < -0.49; top = yw > 0.49
lower = zero_crossings(bot, 0.0, 17.0)
upper = zero_crossings(top, 0.0, 17.0)
print(f"RESULT lower-wall omega zero crossings (reattachment): {['%.3f'%x for x in lower]}  (Gartling/Chan: 6.1)", flush=True)
print(f"RESULT upper-wall omega zero crossings (sep, reatt): {['%.3f'%x for x in upper]}  (Gartling/Chan: 4.8, 10.5)", flush=True)

np.savez(f"{a.outdir}/final.npz", u=s.u, v=s.v, p=s.p, time=s.time, nstep=nsteps,
         centroid=m.centroid, nodes=m.nodes, cells=m.cells, nvert=m.nvert, btag=m.btag,
         hist=np.array(hist), lower_crossings=np.array(lower), upper_crossings=np.array(upper))
print(f"wrote {a.outdir}/final.npz", flush=True)
