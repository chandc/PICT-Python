"""Small production cases for the unstructured-adjoint gates (plan section 7.2, the fast suite).

Each builder returns a configured production `PISO`, already advanced a few steps so the state
is developed (non-zero flux, BDF2 and Rhie-Chow history present) and every code path is live.

    cavity_quads    M1  uniform quads, lid-driven, all-Neumann pressure (singular)
    cavity_tris     M3  perturbed alternating triangles, same flow: every deferred term live
    channel_open    M3' perturbed triangles, parabolic inlet, zero-pressure outlet: Dirichlet
                        pressure (non-singular), convective inflow AND outflow boundary faces
    channel_periodic M7 quads, periodic in x, body-force driven with a seeded perturbation
"""
import numpy as np

from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import PISO, BC


def _run(s, nsteps):
    for _ in range(nsteps):
        s.step()
    return s


def cavity_quads(n=10, Re=100.0, dt=0.01, nsteps=5):
    m = rect_mesh(n, n, cells="quad")
    return _cavity(m, Re, dt, nsteps)


def cavity_tris(n=8, Re=100.0, dt=0.01, nsteps=5, perturb=0.4):
    m = rect_mesh(n, n, perturb=perturb, seed=3)
    return _cavity(m, Re, dt, nsteps)


def _cavity(m, Re, dt, nsteps):
    tag = m.btag[m.bfaces]
    vu = np.zeros(m.nbface); vu[tag == 4] = 1.0
    s = PISO(m, nu=1.0 / Re, dt=dt, bc_u=BC(m, np.full(m.nbface, DIRICHLET), vu),
             bc_v=BC(m, np.full(m.nbface, DIRICHLET)), bc_p=BC(m, np.full(m.nbface, NEUMANN)),
             n_corr=2, n_nonorth=3, scheme="central")
    return _run(s, nsteps)


def channel_open(nx=12, ny=6, Re=50.0, dt=0.02, nsteps=5, perturb=0.3):
    """x in [0, 3], y in [0, 1]: inlet left (tag 1), outlet right (tag 2), walls bottom/top."""
    m = rect_mesh(nx, ny, 0.0, 3.0, 0.0, 1.0, perturb=perturb, seed=5)
    tag = m.btag[m.bfaces]; yb = m.fcentre[m.bfaces, 1]
    inl, out = tag == 1, tag == 2
    ku = np.full(m.nbface, DIRICHLET); ku[out] = NEUMANN
    kv = ku.copy()
    kp = np.full(m.nbface, NEUMANN); kp[out] = DIRICHLET
    vu = np.where(inl, 6.0 * yb * (1.0 - yb), 0.0)
    s = PISO(m, nu=1.0 / Re, dt=dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, np.zeros(m.nbface)),
             bc_p=BC(m, kp, np.zeros(m.nbface)), n_corr=2, n_nonorth=3, scheme="central")
    y = m.centroid[:, 1]
    s.u[:] = 6.0 * y * (1.0 - y)
    return _run(s, nsteps)


def channel_periodic(nx=12, ny=8, Re=100.0, dt=0.02, nsteps=5):
    """x in [0, 2 pi], y in [-1, 1], periodic in x, U = 1 - y^2 held by f = 2 nu, seeded."""
    LX = 2 * np.pi
    m = rect_mesh(nx, ny, 0.0, LX, -1.0, 1.0, cells="quad")
    m.make_periodic(1, 2, (LX, 0.0))
    nu = 1.0 / Re
    kd = np.full(m.nbface, DIRICHLET)
    s = PISO(m, nu=nu, dt=dt, bc_u=BC(m, kd), bc_v=BC(m, kd.copy()),
             bc_p=BC(m, np.full(m.nbface, NEUMANN)), n_corr=2, n_nonorth=2, scheme="central",
             body_force=(np.full(m.ncell, 2 * nu), np.zeros(m.ncell)))
    x, y = m.centroid.T
    s.u[:] = 1 - y ** 2 + 0.05 * np.sin(x) * (1 - y ** 2)
    s.v[:] = 0.05 * np.cos(x) * (1 - y ** 2) ** 2
    return _run(s, nsteps)


CASES = {"cavity_quads": cavity_quads, "cavity_tris": cavity_tris,
         "channel_open": channel_open, "channel_periodic": channel_periodic}


def cylinder(mesh="meshes/cylinder_butterfly_coarse.msh", Re=100.0, dt=0.01, nsteps=5):
    """M4/M5: the T9 cylinder on a production Gmsh mesh, run_ucylinder.py's BCs exactly: Inlet
    u = 1; Freestream v = 0 with u free (symmetry); Outlet p = 0; Cylinder no-slip."""
    from src.umesh import read_gmsh22, Mesh
    nodes, cells, ctag, edges, etag, names = read_gmsh22(mesh)
    m = Mesh(nodes, cells, edges, etag, names)
    inv = {v: k for k, v in names.items()}
    T_IN, T_FS, T_OUT, T_CYL = inv["Inlet"], inv["Freestream"], inv["Outlet"], inv["Cylinder"]
    bt = m.btag[m.bfaces]; nb = m.nbface
    ku = np.where(np.isin(bt, [T_IN, T_CYL]), DIRICHLET, NEUMANN); vu = np.where(bt == T_IN, 1.0, 0.0)
    kv = np.where(np.isin(bt, [T_IN, T_FS, T_CYL]), DIRICHLET, NEUMANN)
    kp = np.where(bt == T_OUT, DIRICHLET, NEUMANN)
    s = PISO(m, nu=1.0 / Re, dt=dt, bc_u=BC(m, ku, vu), bc_v=BC(m, kv, np.zeros(nb)),
             bc_p=BC(m, kp, np.zeros(nb)), n_corr=2, n_nonorth=3, scheme="central", convect=True)
    C = m.centroid
    s.u[:] = 1.0; s.v[:] = 0.05 * np.exp(-((C[:, 0] - 1.0) ** 2 + C[:, 1] ** 2))
    s.wall_faces = np.flatnonzero(m.boundary & (m.btag == T_CYL))
    return _run(s, nsteps)


def cylinder_tris(**kw):
    return cylinder(mesh="meshes/cylinder_medium.msh", **kw)


PRODUCTION = {"cylinder_butterfly": cylinder, "cylinder_tris": cylinder_tris}
