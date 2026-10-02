"""Surgical fix for the local cell-size-ratio outliers documented in skew_unstructured_literature.md
section 76 -- a handful of triangles on an otherwise well-graded unstructured mesh that are 3x+
smaller than their immediate neighbour, an artifact of Delaunay insertion-order path-dependence that
neither Gmsh's own Optimize pass nor its Netgen optimizer nor 1000 iterations of Laplacian smoothing
touches (all tried and confirmed ineffective, byte-identical output, before writing this). A FIRST
attempt here, centroid-splitting the larger cell of each bad pair, was tried and made things WORSE
(18 -> 127 bad pairs over 8 iterations): a 1-to-3 split creates three even-smaller cells, which then
clash with THEIR other neighbours, cascading the defect outward instead of fixing it.

Fix that actually works: EDGE COLLAPSE. Each bad pair's shared edge (p0, p1) is merged -- p1 is
retired, every triangle referencing it is relabelled to p0, and p0 is moved to the edge midpoint. The
two triangles that shared that edge become degenerate (two equal vertices) and are dropped; every
other triangle keeps its topology, just with one vertex renamed, so nothing is created and nothing
elsewhere is disturbed -- strictly coarsening, so it cannot cascade the way splitting did. Boundary
edges are relabelled the same way so tagging stays consistent.
    python meshes/fix_local_size_outliers.py meshes/gartling_bfs_tri_matched_fine_v2.msh [--ratio 2.5] [--max-iter 10]"""
import sys, argparse, numpy as np
sys.path.insert(0, ".")
from src.umesh import Mesh, read_gmsh22

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("--ratio", type=float, default=2.5)
ap.add_argument("--max-iter", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()
a.out = a.out or a.mesh.replace(".msh", "_fixed.msh")

nodes, tris, ctag, edges, etag, names = read_gmsh22(a.mesh)
nodes = [list(p) for p in nodes]; tris = [list(t) for t in tris]
edges = [list(e) for e in edges]; etag = list(etag)

def relabel(p_old, p_new, tris, edges):
    new_tris = []
    for t in tris:
        t2 = [p_new if v == p_old else v for v in t]
        if len(set(t2)) == 3:
            new_tris.append(t2)
    new_edges = [[p_new if v == p_old else v for v in e] for e in edges]
    return new_tris, new_edges

for it in range(a.max_iter):
    m = Mesh(np.array(nodes), np.array(tris), edges, etag, names)
    i = m.interior
    ratio = np.maximum(m.vol[m.owner[i]] / m.vol[m.neigh[i]], m.vol[m.neigh[i]] / m.vol[m.owner[i]])
    bad = np.flatnonzero(ratio > a.ratio)
    if len(bad) == 0:
        print(f"iter {it}: clean, {m.ncell} cells")
        break
    face_idx = np.flatnonzero(i)[bad]
    print(f"iter {it}: {len(bad)} bad pairs, worst {ratio.max():.2f}, collapsing each shared edge")
    # collapse at most one edge per affected node per pass (avoid chasing an already-moved node)
    touched = set()
    n_done = 0
    for k in face_idx[np.argsort(-ratio[bad])]:
        p0, p1 = int(m.face_nodes[k, 0]), int(m.face_nodes[k, 1])
        if p0 in touched or p1 in touched:
            continue
        mx, my = (nodes[p0][0] + nodes[p1][0]) / 2.0, (nodes[p0][1] + nodes[p1][1]) / 2.0
        nodes[p0] = [mx, my]
        tris, edges = relabel(p1, p0, tris, edges)
        touched.add(p0); touched.add(p1); n_done += 1
    print(f"  collapsed {n_done} edges")
else:
    print(f"[warn] still {len(bad)} bad pairs after {a.max_iter} iterations (worst {ratio.max():.2f})")

m = Mesh(np.array(nodes), np.array(tris), edges, etag, names)
print(f"final: {m.ncell} cells  audit {m.audit()}")
for line in m.quality(max_ratio_warn=a.ratio):
    print(" ", line)

with open(a.out, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    all_names = dict(names)
    f.write("$PhysicalNames\n"); f.write(f"{len(all_names)}\n")
    for tag, nm in all_names.items():
        dim = 2 if nm == "Fluid" else 1
        f.write(f'{dim} {tag} "{nm}"\n')
    f.write("$EndPhysicalNames\n")
    f.write("$Nodes\n"); f.write(f"{len(nodes)}\n")
    for idx, (x, y) in enumerate(nodes):
        f.write(f"{idx+1} {x:.10f} {y:.10f} 0\n")
    f.write("$EndNodes\n")
    fluid_tag = [t for t, nm in all_names.items() if nm == "Fluid"][0]
    f.write("$Elements\n"); f.write(f"{len(edges) + len(tris)}\n")
    eid = 1
    for k in range(len(edges)):
        p0, p1 = edges[k]
        f.write(f"{eid} 1 2 {etag[k]} {etag[k]} {p0+1} {p1+1}\n"); eid += 1
    for tri in tris:
        f.write(f"{eid} 2 2 {fluid_tag} {fluid_tag} {tri[0]+1} {tri[1]+1} {tri[2]+1}\n"); eid += 1
    f.write("$EndElements\n")
print("wrote", a.out)
