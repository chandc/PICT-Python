"""Interpolate a finished Gartling BFS field onto a different mesh, as a warm initial condition --
same pattern as interpolate_cylinder3900_ic.py / interpolate_cavity_ic.py. 2D linear interpolation
(Delaunay on the source mesh's cell centroids), nearest-neighbour fallback outside the source's convex
hull (expected only right at the domain boundary).
    python interpolate_gartling_bfs_ic.py --src results/ugartling_bfs_quad_xrefine/final.npz --src-mesh meshes/gartling_bfs_quad_xrefine.msh --dst-mesh meshes/gartling_bfs_tri_matched.msh --out results/ugartling_bfs_tri_matched_ic.npz"""
import sys, os, argparse, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.umesh import Mesh, read_gmsh22
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="results/ugartling_bfs_quad_xrefine/final.npz")
ap.add_argument("--src-mesh", default="meshes/gartling_bfs_quad_xrefine.msh")
ap.add_argument("--dst-mesh", default="meshes/gartling_bfs_tri_matched.msh")
ap.add_argument("--out", default="results/ugartling_bfs_tri_matched_ic.npz")
a = ap.parse_args()

n0, c0, ct0, e0, et0, names0 = read_gmsh22(a.src_mesh); m0 = Mesh(n0, c0, e0, et0, names0)
n1, c1, ct1, e1, et1, names1 = read_gmsh22(a.dst_mesh); m1 = Mesh(n1, c1, e1, et1, names1)
d = np.load(a.src)
print(f"source: {a.src}  ({m0.ncell} cells, t={float(d['time']):.2f})  ->  dest mesh: {a.dst_mesh}  ({m1.ncell} cells)")

tri = Delaunay(m0.centroid)
dst = m1.centroid
inside = tri.find_simplex(dst) >= 0
print(f"{inside.sum()} of {m1.ncell} dest cells inside the source's convex hull (linear interp); {(~inside).sum()} outside (nearest-neighbour, expected near the domain boundary)")

fields = {}
for name in ("u", "v", "p"):
    f = d[name]
    li = LinearNDInterpolator(tri, f)
    vals = li(dst)
    bad = ~np.isfinite(vals)
    if bad.any():
        ni = NearestNDInterpolator(m0.centroid, f); vals[bad] = ni(dst[bad])
    fields[name] = vals
    print(f"  {name}: range [{vals.min():.4f}, {vals.max():.4f}]  (source range [{f.min():.4f}, {f.max():.4f}])")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
# time=0.0: the new mesh gets its own fresh clock, same convention as the cavity/cylinder warm
# starts -- src_time records provenance (what time the source field was taken at) separately.
np.savez(a.out, u=fields["u"], v=fields["v"], p=fields["p"], time=0.0,
         src=a.src, src_mesh=a.src_mesh, dst_mesh=a.dst_mesh, src_time=float(d["time"]))
print("wrote", a.out)
