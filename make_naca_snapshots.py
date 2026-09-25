"""Shedding-state snapshots for NACAJetEnv.reset: continue the developed baseline over one shedding period
and store N phases (u, v, p) plus the unperturbed force means (C_D0, C_L0) from the last 40% of the baseline.
    python make_naca_snapshots.py results/naca/a40_coarse_re100.npz meshes/naca0012_a40_coarse.msh results/naca/a40_coarse_snapshots.npz"""
import sys, io, contextlib, numpy as np; sys.path.insert(0, ".")
from src.umesh import Mesh, read_gmsh22
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC
src, mesh, out = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 10; dt = 0.01
d = np.load(src); hist = d["hist"]; n2 = int(0.6 * len(hist)); t, cl = hist[n2:, 0], hist[n2:, 2] - hist[n2:, 2].mean()
z = np.flatnonzero(np.diff(np.sign(cl)) > 0); tz = t[z] - cl[z] * (t[z + 1] - t[z]) / (cl[z + 1] - cl[z]); per = float(np.diff(tz).mean())
cd0, cl0 = float(hist[n2:, 1].mean()), float(hist[n2:, 2].mean()); print(f"baseline window t {t[0]:.0f}..{t[-1]:.0f}: C_D0 {cd0:.4f} C_L0 {cl0:.4f} period {per:.3f} (St_c {1/per:.4f}), Cl amp {0.5*(hist[n2:,2].max()-hist[n2:,2].min()):.4f}")
nodes, cells, ctag, edges, etag, names = read_gmsh22(mesh); m = Mesh(nodes, cells, edges, etag, names); inv = {v: k for k, v in names.items()}
bt = m.btag[m.bfaces]; nb = m.nbface; ku = np.where(np.isin(bt, [inv["Inlet"], inv["Airfoil"]]), DIRICHLET, NEUMANN)
s = PISO(m, nu=0.01, dt=dt, bc_u=BC(m, ku, np.where(bt == inv["Inlet"], 1.0, 0.0)), bc_v=BC(m, ku.copy(), np.zeros(nb)), bc_p=BC(m, np.where(bt == inv["Outlet"], DIRICHLET, NEUMANN), np.zeros(nb)), n_corr=2, n_nonorth=3, scheme="central")
s.u[:] = d["u"]; s.v[:] = d["v"]; s.p[:] = d["p"]
steps = int(round(per / dt)); every = max(1, steps // N); U, V, P = [], [], []
with contextlib.redirect_stdout(io.StringIO()):
    for k in range(steps):
        s.step()
        if (k + 1) % every == 0 and len(U) < N: U.append(s.u.copy()); V.append(s.v.copy()); P.append(s.p.copy())
np.savez(out, u=np.array(U), v=np.array(V), p=np.array(P), cd0=cd0, cl0=cl0, period=per, dt=dt, mesh=mesh)
print(f"wrote {out}: {len(U)} phases over one period ({steps} steps)")
