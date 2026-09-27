"""Interactive 3D visualization of the Re 3900 cylinder LES field (rotate/zoom/pan) with PyVista.

Extrudes the near-wake quad cells through the periodic Fourier span into hexahedra, computes the
vorticity magnitude from the saved instantaneous field (in-plane LSQ gradient + spectral d/dz, the
same operators the solver uses), and renders an isosurface of |omega| colored by streamwise velocity,
with the cylinder and a few streamlines for context.
    python visualize_cylinder3d_pyvista.py [--iso 6.0] [--box -1 4 1.8]
"""
import sys, os, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from src.uops import Gradient
import pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="meshes/cylinder_re3900.msh")
ap.add_argument("--field", default="results/ucyl3900_a100/ucyl3900_cylinder_re3900_nz64_wale/ucyl3900_cylinder_re3900_nz64_wale_final.npz")
ap.add_argument("--box", type=float, nargs=3, default=(-1.0, 4.0, 1.8), help="x0 x1 |y|max of the near-wake region to extrude")
ap.add_argument("--criterion", default="q", choices=["q", "omega"], help="vortex identifier to isosurface: Q-criterion (rotation-dominated regions, filters out shear) or |omega| (raw vorticity magnitude)")
ap.add_argument("--iso", type=float, default=None, help="isosurface level; default: a percentile of the field within the box")
ap.add_argument("--percentile", type=float, default=90.0)
ap.add_argument("--offscreen", default=None, help="save a screenshot to this path instead of opening an interactive window")
a = ap.parse_args()

print("loading mesh and field ...", flush=True)
nodes, cells, ctag, edges, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges, etag, names)
d = np.load(a.field); u, v, w = d["u"], d["v"], d["w"]; nz = int(d["nz"]); Lz = float(d["Lz"]); t = float(d["time"])
assert (m.nvert == 4).all(), "extrusion assumes an all-quad plane"

print("computing the velocity-gradient tensor (in-plane LSQ gradient + spectral d/dz) ...", flush=True)
g = Gradient(m)                                            # phi_b defaults to 0: exact for the cylinder no-slip wall, and the box below touches no other boundary
zb = np.zeros((g.Bx.shape[1], nz))
gu, gv, gw = g(u, zb), g(v, zb), g(w, zb)                     # (ncell, 2, nz): d/dx, d/dy
kz = 2 * np.pi / Lz * np.arange(nz // 2 + 1)
def ddz(f): return np.fft.irfft(np.fft.rfft(f, axis=1) * 1j * kz, n=nz, axis=1)
dudz, dvdz, dwdz = ddz(u), ddz(v), ddz(w)
dudx, dudy = gu[:, 0], gu[:, 1]; dvdx, dvdy = gv[:, 0], gv[:, 1]; dwdx, dwdy = gw[:, 0], gw[:, 1]

if a.criterion == "omega":
    field = np.sqrt((dwdy - dvdz) ** 2 + (dudz - dwdx) ** 2 + (dvdx - dudy) ** 2)   # |omega| (ncell, nz)
    field_name, field_label = "omega_mag", "|omega|"
else:
    # Q = -1/2 trace(J^2) = -1/2 (dudx^2+dvdy^2+dwdz^2) - (dudy dvdx + dudz dwdx + dvdz dwdy);
    # positive where rotation dominates strain -- the standard vortex-core identifier (filters out shear layers)
    field = -0.5 * (dudx ** 2 + dvdy ** 2 + dwdz ** 2) - (dudy * dvdx + dudz * dwdx + dvdz * dwdy)
    field_name, field_label = "Q", "Q"

x0, x1, ymax = a.box
C = m.centroid; sel = np.flatnonzero((C[:, 0] > x0) & (C[:, 0] < x1) & (np.abs(C[:, 1]) < ymax))
print(f"near-wake box x in [{x0},{x1}], |y|<{ymax}: {len(sel)} of {m.ncell} cells x {nz} planes = {len(sel)*nz:,} hexahedra", flush=True)

# ---- node re-indexing: only the nodes the selected cells use
cell_nodes = m.cells[sel, :4]                               # (nsel, 4)
uniq, inv = np.unique(cell_nodes.ravel(), return_inverse=True)
local = inv.reshape(-1, 4)                                  # local node index (0..len(uniq)-1) per selected cell corner
node_xy = m.nodes[uniq]                                     # (nuniq, 2)
nuniq = len(uniq)

# ---- 3D points: nuniq nodes x (nz+1) z-layers (layer nz repeats layer 0's data -- the span is periodic)
zlevels = np.linspace(0.0, Lz, nz + 1)
pts = np.empty((nuniq * (nz + 1), 3))
for k in range(nz + 1):
    pts[k * nuniq:(k + 1) * nuniq, 0] = node_xy[:, 0]
    pts[k * nuniq:(k + 1) * nuniq, 1] = node_xy[:, 1]
    pts[k * nuniq:(k + 1) * nuniq, 2] = zlevels[k]

# ---- hexahedra: one per (selected cell, z-slab k = 0..nz-1), slab k spans zlevels[k]..zlevels[k+1]
nsel = len(sel)
hexcells = np.empty((nsel * nz, 9), dtype=np.int64); hexcells[:, 0] = 8
for k in range(nz):
    rows = slice(k * nsel, (k + 1) * nsel)
    hexcells[rows, 1:5] = local + k * nuniq                 # bottom face
    hexcells[rows, 5:9] = local + (k + 1) * nuniq            # top face (k+1; k=nz-1 uses the closing layer nz)
cell_types = np.full(nsel * nz, pv.CellType.HEXAHEDRON, dtype=np.uint8)
grid = pv.UnstructuredGrid(hexcells.ravel(), cell_types, pts)

# ---- cell data: one value per (selected cell, z-slab k), ordered to match hexcells' (k, cell) layout above
field_sel = field[sel]                                      # (nsel, nz)
u_sel = u[sel]
grid.cell_data[field_name] = field_sel.T.ravel()             # (nz, nsel) -> ravel matches the k-major hex ordering
grid.cell_data["u"] = u_sel.T.ravel()
grid = grid.cell_data_to_point_data()

iso = a.iso if a.iso is not None else float(np.percentile(field_sel, a.percentile))
if a.criterion == "q" and iso <= 0: iso = float(np.percentile(field_sel[field_sel > 0], 50)) if (field_sel > 0).any() else 1e-3
print(f"isosurface {field_label} = {iso:.3f} (percentile {a.percentile} of the box; field range {field_sel.min():.2f}-{field_sel.max():.2f})", flush=True)
surf = grid.contour([iso], scalars=field_name)
print(f"isosurface: {surf.n_points:,} points, {surf.n_cells:,} triangles", flush=True)

# ---- cylinder geometry for reference
theta = np.linspace(0, 2 * np.pi, 64)
cyl = pv.CylinderStructured(radius=[0.5], theta_resolution=64, height=Lz, center=(0, 0, Lz / 2), direction=(0, 0, 1))

pl = pv.Plotter(window_size=(1400, 900), off_screen=bool(a.offscreen))
pl.add_mesh(surf, scalars="u", cmap="coolwarm", clim=(u_sel.min(), max(u_sel.max(), 1.3)), scalar_bar_args=dict(title="u / U_inf"), smooth_shading=True, specular=0.3)
pl.add_mesh(cyl, color=[0.25, 0.25, 0.25])
pl.add_axes(); pl.show_bounds(grid="back", location="outer", ticks="outside", xtitle="x/D", ytitle="y/D", ztitle="z/D")
print("surf bounds", surf.bounds, "grid bounds", grid.bounds)
pl.view_isometric(); pl.reset_camera()
pl.add_text(f"Re 3900 cylinder LES, t = {t:.0f}: {field_label} = {iso:.2f} isosurface, coloured by u/U_inf\n"
            f"box x in [{x0},{x1}], |y|<{ymax}, full span L_z = {Lz:.2f} D  ({surf.n_cells:,} triangles)", font_size=11)
if a.offscreen:
    pl.show(screenshot=a.offscreen); print("wrote", a.offscreen)
else:
    print("opening the interactive window (rotate: left-drag, pan: shift/right-drag, zoom: scroll) ...", flush=True)
    pl.show()
