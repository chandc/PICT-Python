"""Unstructured 2D triangular mesh: Gmsh reader and cell/face connectivity.

WHY THIS EXISTS. The structured multiblock path cannot express a Cartesian tiling wrapped
around an O-ring: the four transition faces are forced onto matching node counts, the square
perimeter then receives 2p+2q nodes for 2p+2q-4 distinct positions, and four corners are
over-supplied by construction. Every attempt to shift that ownership turns duplicates into
gaps. On an unstructured mesh the question does not arise -- there are no blocks, no face
orientations, no ghost padding; a boundary face is simply a face with one owner.

It also lets us read HydroGym's OWN mesh. `medium.msh` is the mesh their published cylinder
results were computed on, so conformance stops being a thing we approximate with stretching
parameters and becomes a thing we either read or do not.

CONVENTIONS, all checked by `Mesh.audit()`:
  * cells are triangles, stored counter-clockwise, so the signed area is positive
  * every interior face has exactly two cells; a boundary face has `neigh == -1`
  * `normal` points OUT of `owner` (and into `neigh` where there is one) and is scaled by the
    face length, so it is the face AREA VECTOR in 2D
  * the face area vectors of a cell sum to zero -- the discrete divergence theorem, and the
    single cheapest check that the connectivity is right
"""
import numpy as np

GMSH_LINE, GMSH_TRI = 1, 2


def read_gmsh22(path):
    """Read a Gmsh 2.2 ASCII mesh. Returns (nodes, tris, tri_tag, edges, edge_tag, names).

    nodes     (N, 2) float    -- x, y (z discarded; this is a 2D solver)
    tris      (T, 3) int      -- node indices, 0-based
    tri_tag   (T,)   int      -- physical tag (the Fluid group)
    edges     (E, 2) int      -- boundary edges, node indices
    edge_tag  (E,)   int      -- physical tag (Inlet / Freestream / Outlet / Cylinder)
    names     {tag: name}
    """
    with open(path) as fh:
        lines = fh.read().split("\n")

    def block(tag):
        i = lines.index(tag)
        j = lines.index("$End" + tag[1:])
        return lines[i + 1:j]

    names = {}
    if "$PhysicalNames" in lines:
        pn = block("$PhysicalNames")
        for row in pn[1:]:
            p = row.split(maxsplit=2)
            if len(p) == 3:
                names[int(p[1])] = p[2].strip().strip('"')

    nb = block("$Nodes")
    nnode = int(nb[0])
    # Gmsh node ids are 1-based and need not be contiguous; map explicitly.
    ids = np.empty(nnode, dtype=np.int64)
    xy = np.empty((nnode, 2), dtype=float)
    for k in range(nnode):
        p = nb[1 + k].split()
        ids[k] = int(p[0])
        xy[k] = (float(p[1]), float(p[2]))
    lookup = np.full(ids.max() + 1, -1, dtype=np.int64)
    lookup[ids] = np.arange(nnode)

    eb = block("$Elements")
    nel = int(eb[0])
    tris, tri_tag, edges, edge_tag = [], [], [], []
    for k in range(nel):
        p = eb[1 + k].split()
        etype, ntag = int(p[1]), int(p[2])
        phys = int(p[3]) if ntag >= 1 else 0
        conn = [lookup[int(v)] for v in p[3 + ntag:]]
        if etype == GMSH_TRI:
            tris.append(conn); tri_tag.append(phys)
        elif etype == GMSH_LINE:
            edges.append(conn); edge_tag.append(phys)
    return (xy, np.array(tris, dtype=np.int64), np.array(tri_tag, dtype=np.int64),
            np.array(edges, dtype=np.int64), np.array(edge_tag, dtype=np.int64), names)


class Mesh:
    """Cell-centred finite-volume connectivity for a 2D triangular mesh."""

    def __init__(self, nodes, tris, edges=None, edge_tag=None, names=None, span=1.0):
        self.nodes = np.asarray(nodes, dtype=float)
        self.tris = np.asarray(tris, dtype=np.int64)
        self.names = names or {}
        self.span = float(span)                 # thickness of the 2D slab, for volumes/forces

        p = self.nodes[self.tris]               # (T, 3, 2)
        # SIGNED area, then orient. A clockwise triangle gives a negative area and would flip
        # every face normal built from it, so fix the winding once here rather than carry a
        # sign through every operator.
        cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                 - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        flip = cross < 0
        if flip.any():
            self.tris[flip] = self.tris[flip][:, ::-1]
            p = self.nodes[self.tris]
            cross = np.abs(cross)
        self.area = 0.5 * np.abs(cross)         # cell area (2D)
        self.vol = self.area * self.span        # cell volume
        self.centroid = p.mean(axis=1)          # triangle centroid
        self.ncell = len(self.tris)

        self._build_faces()
        self._tag_boundaries(edges, edge_tag)

    # -------------------------------------------------------------------------------------
    def _build_faces(self):
        """Edge list with owner/neighbour, area vectors, and cell-to-cell geometry."""
        T = self.tris
        # the three edges of every triangle, as sorted node pairs so the two sides of an
        # interior edge produce an identical key
        loc = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)   # (T, 3, 2)
        flat = loc.reshape(-1, 2)
        key = np.sort(flat, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        ks = key[order]
        same = np.empty(len(ks), dtype=bool)
        same[0] = False
        same[1:] = (ks[1:] == ks[:-1]).all(axis=1)
        start = np.flatnonzero(~same)
        count = np.diff(np.append(start, len(ks)))
        if count.max() > 2:
            raise ValueError("an edge is shared by more than two cells: mesh is not manifold")

        owner_cell = order[start] // 3
        first_local = order[start]
        self.face_nodes = flat[first_local]                       # oriented by the OWNER
        self.owner = owner_cell.astype(np.int64)
        self.neigh = np.full(len(start), -1, dtype=np.int64)
        second = start[count == 2] + 1
        self.neigh[count == 2] = order[second] // 3

        a = self.nodes[self.face_nodes[:, 0]]
        b = self.nodes[self.face_nodes[:, 1]]
        t = b - a
        self.length = np.hypot(t[:, 0], t[:, 1])
        # outward normal of the OWNER: rotate the owner-oriented edge tangent by -90 degrees.
        # With counter-clockwise cells that points out of the owner; asserted in audit().
        self.normal = np.stack([t[:, 1], -t[:, 0]], axis=1)       # length = |t|, so area vector
        self.sf = self.normal * self.span                         # face area vector (3D slab)
        self.fcentre = 0.5 * (a + b)
        self.nface = len(self.owner)
        self.interior = self.neigh >= 0
        self.boundary = ~self.interior

        # cell-to-cell vector, used by every gradient and by the non-orthogonal correction
        d = np.empty((self.nface, 2))
        d[self.interior] = (self.centroid[self.neigh[self.interior]]
                            - self.centroid[self.owner[self.interior]])
        d[self.boundary] = (self.fcentre[self.boundary]
                            - self.centroid[self.owner[self.boundary]])
        self.dcc = d
        self.dmag = np.hypot(d[:, 0], d[:, 1])
        # linear interpolation weight for the OWNER value at the face
        w = np.ones(self.nface)
        i = self.interior
        do = np.hypot(*(self.fcentre[i] - self.centroid[self.owner[i]]).T)
        dn = np.hypot(*(self.fcentre[i] - self.centroid[self.neigh[i]]).T)
        w[i] = dn / (do + dn)
        self.wf = w
        self._decompose()

    def _decompose(self):
        """OVER-RELAXED split of the face area vector, S_f = E_f + T_f.

            E_f = d (d.S)/(d.d)        T_f = S_f - E_f

        `E_f` is parallel to the line joining the two cell centres, so the part of the Laplacian
        built on it is a two-point stencil and stays diagonally dominant; `T_f` carries the
        non-orthogonality and is deferred to the right-hand side. Over-relaxed (rather than
        minimum-correction or orthogonal-correction) because |E_f| GROWS with skewness, which is
        what keeps the implicit operator dominant on the worst cells instead of the best.

        The guide's stated check for this step, `E_f + T_f == S_f`, is an algebraic identity --
        T_f is DEFINED as the remainder, so it cannot fail and tests nothing. The real check is
        that the Laplacian built from the split annihilates a linear field on a skewed mesh,
        which is what `test_uops` does.
        """
        d = self.dcc
        dd = (d * d).sum(axis=1)
        dS = (d * self.normal).sum(axis=1)
        self.Ef = d * (dS / np.maximum(dd, 1e-300))[:, None]
        self.Tf = self.normal - self.Ef
        # |E_f| / |d| is the coefficient the implicit Laplacian uses on each face
        self.ef_over_d = np.hypot(*self.Ef.T) / np.maximum(self.dmag, 1e-300)
        # cos of the angle between d and S: 1 = orthogonal, -> 0 = badly skewed
        self.orth = dS / np.maximum(np.hypot(*d.T) * np.hypot(*self.normal.T), 1e-300)

    # -------------------------------------------------------------------------------------
    def _tag_boundaries(self, edges, edge_tag):
        """Attach the Gmsh physical tag to each boundary face."""
        self.btag = np.zeros(self.nface, dtype=np.int64)
        if edges is None or len(edges) == 0:
            return
        key = {}
        for e, tg in zip(np.sort(np.asarray(edges), axis=1), np.asarray(edge_tag)):
            key[(int(e[0]), int(e[1]))] = int(tg)
        bf = np.flatnonzero(self.boundary)
        miss = 0
        for f in bf:
            k = tuple(sorted(int(v) for v in self.face_nodes[f]))
            if k in key:
                self.btag[f] = key[k]
            else:
                miss += 1
        self.unmatched_boundary = miss

    def faces_with_tag(self, tag):
        return np.flatnonzero(self.boundary & (self.btag == tag))

    def tag_of(self, name):
        for t, nm in self.names.items():
            if nm == name:
                return t
        raise KeyError(f"no physical group named {name!r}; have {sorted(self.names.values())}")

    # -------------------------------------------------------------------------------------
    def audit(self):
        """Invariants that produce silent, plausible-looking corruption if violated."""
        out = []
        if (self.area <= 0).any():
            out.append(f"[FAIL] {(self.area <= 0).sum()} cells with non-positive area")
        # closure: the face area vectors of a cell must sum to zero
        acc = np.zeros((self.ncell, 2))
        np.add.at(acc, self.owner, self.normal)
        np.add.at(acc, self.neigh[self.interior], -self.normal[self.interior])
        clo = np.abs(acc).max() / max(self.length.max(), 1e-30)
        if clo > 1e-10:
            out.append(f"[FAIL] cell face-vectors do not close: {clo:.3e}")
        # the owner normal must point from owner centroid towards the face
        s = ((self.fcentre - self.centroid[self.owner]) * self.normal).sum(axis=1)
        if (s <= 0).any():
            out.append(f"[FAIL] {(s <= 0).sum()} face normals do not point out of their owner")
        if getattr(self, "unmatched_boundary", 0):
            out.append(f"[warn] {self.unmatched_boundary} boundary faces carry no physical tag")
        untagged = int((self.boundary & (self.btag == 0)).sum())
        if untagged:
            out.append(f"[warn] {untagged} boundary faces untagged")
        return out

    def __repr__(self):
        return (f"Mesh({self.ncell} cells, {self.nface} faces, "
                f"{int(self.boundary.sum())} boundary, area {self.area.sum():.4f})")


def from_gmsh(path, span=1.0):
    nodes, tris, _tt, edges, etag, names = read_gmsh22(path)
    return Mesh(nodes, tris, edges, etag, names, span=span)


def rect_mesh(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0, span=1.0, perturb=0.0, seed=0,
              tags=("left", "right", "bottom", "top")):
    """Triangulated rectangle: nx by ny quads, each split into two triangles.

    `perturb` jitters the INTERIOR nodes by that fraction of the local spacing, which turns an
    otherwise perfectly orthogonal mesh into a skewed one. That matters: on a uniform right
    triangulation the non-orthogonal correction T_f is nearly zero, so a solver with a broken
    deferred-correction term still passes. Running every verification at perturb=0 AND at
    perturb>0 is what stops that.

    Boundary physical tags are 1..4 for left/right/bottom/top, named by `tags`.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if perturb:
        hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
        X[1:-1, 1:-1] += perturb * hx * (rng.random((nx - 1, ny - 1)) - 0.5)
        Y[1:-1, 1:-1] += perturb * hy * (rng.random((nx - 1, ny - 1)) - 0.5)
    nid = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]
            # alternate the diagonal so the mesh has no global bias direction
            if (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    tris = np.array(tris, dtype=np.int64)

    edges, etag = [], []
    for j in range(ny):
        edges.append([nid[0, j], nid[0, j + 1]]); etag.append(1)          # left
        edges.append([nid[nx, j], nid[nx, j + 1]]); etag.append(2)        # right
    for i in range(nx):
        edges.append([nid[i, 0], nid[i + 1, 0]]); etag.append(3)          # bottom
        edges.append([nid[i, ny], nid[i + 1, ny]]); etag.append(4)        # top
    names = {k + 1: tags[k] for k in range(4)}
    return Mesh(nodes, tris, np.array(edges), np.array(etag), names, span=span)
