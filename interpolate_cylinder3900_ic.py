"""Interpolate the finished V3 field onto a refined mesh, as a warm initial condition for the shear-layer
refinement run (reviewer item C, record section 65). Per-plane 2D linear interpolation (Delaunay on the
coarse mesh's cell centroids), same nz/Lz so no spanwise remapping is needed -- the two meshes share the
same Fourier span discretisation. Cells outside the coarse mesh's convex hull (there are none away from
the domain boundary, since both meshes cover the same box) fall back to nearest-neighbour.
    python interpolate_cylinder3900_ic.py --src results/ucyl3900_a100/.../_final.npz --dst-mesh meshes/cylinder_re3900_refined.msh --out results/ucyl3900_refined_ic.npz"""
import sys, os, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="results/ucyl3900_a100/ucyl3900_cylinder_re3900_nz64_wale/ucyl3900_cylinder_re3900_nz64_wale_final.npz")
ap.add_argument("--src-mesh", default="meshes/cylinder_re3900.msh"); ap.add_argument("--dst-mesh", default="meshes/cylinder_re3900_refined.msh")
ap.add_argument("--out", default="results/ucyl3900_refined_ic.npz")
a = ap.parse_args()

n0, c0, ct0, e0, et0, names0 = read_gmsh22(a.src_mesh); m0 = Mesh(n0, c0, e0, et0, names0)
n1, c1, ct1, e1, et1, names1 = read_gmsh22(a.dst_mesh); m1 = Mesh(n1, c1, e1, et1, names1)
d = np.load(a.src); u, v, w, p = d["u"], d["v"], d["w"], d["p"]; nz, Lz = int(d["nz"]), float(d["Lz"])
print(f"source: {a.src}  ({m0.ncell} cells, nz {nz})  ->  dest mesh: {a.dst_mesh}  ({m1.ncell} cells)")

tri = Delaunay(m0.centroid)
dst = m1.centroid
inside = tri.find_simplex(dst) >= 0
print(f"{inside.sum()} of {m1.ncell} dest cells inside the source's convex hull (linear interp); {(~inside).sum()} outside (nearest-neighbour, expected near the domain boundary)")

fields = {}
for name, f in (("u", u), ("v", v), ("w", w), ("p", p)):
    out = np.empty((m1.ncell, nz))
    for k in range(nz):
        li = LinearNDInterpolator(tri, f[:, k])
        vals = li(dst)
        bad = ~np.isfinite(vals)
        if bad.any():
            ni = NearestNDInterpolator(m0.centroid, f[:, k]); vals[bad] = ni(dst[bad])
        out[:, k] = vals
    fields[name] = out
    print(f"  {name}: range [{out.min():.4f}, {out.max():.4f}]  (source range [{f.min():.4f}, {f.max():.4f}])")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, u=fields["u"], v=fields["v"], w=fields["w"], p=fields["p"], nz=nz, Lz=Lz,
         src=a.src, src_mesh=a.src_mesh, dst_mesh=a.dst_mesh, src_time=float(d["time"]))
print("wrote", a.out)
