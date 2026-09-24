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

GMSH_LINE, GMSH_TRI, GMSH_QUAD = 1, 2, 3


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
        if etype in (GMSH_TRI, GMSH_QUAD):
            tris.append(conn); tri_tag.append(phys)
        elif etype == GMSH_LINE:
            edges.append(conn); edge_tag.append(phys)
    # all-triangle -> (n,3) array as before; any quads -> ragged list, which Mesh pads itself
    cells = np.array(tris, dtype=np.int64) if all(len(c) == 3 for c in tris) else tris
    return (xy, cells, np.array(tri_tag, dtype=np.int64),
            np.array(edges, dtype=np.int64), np.array(edge_tag, dtype=np.int64), names)


class Mesh:
    """Cell-centred finite-volume connectivity for a 2D triangular mesh."""

    def __init__(self, nodes, cells, edges=None, edge_tag=None, names=None, span=1.0):
        """`cells`: (ncell, k) index array or a ragged list of index sequences -- triangles, quads
        or any mix. Stored padded to (ncell, maxk) with -1 as `self.cells`, vertex counts in
        `self.nvert`. `self.tris` remains as an alias ONLY when every cell is a triangle, for the
        few consumers that still assume it (plotting, the k-exact moments).

        Why polygons: on Cartesian quads the face weight is exactly 1/2, the pressure near-null
        mode is two-colourable, and the divergence-form gradient is exactly dual AND exactly blind
        to it -- the properties the uniform-triangle T5 pass rests on, and the ones every irregular
        triangulation lost (record, sections 22-24). A quad layer at the walls puts those
        properties where the boundary layer is; the triangle core keeps geometric flexibility.
        """
        self.nodes = np.asarray(nodes, dtype=float)
        C, nv = _pad_cells(cells)
        self.cells, self.nvert = C, nv
        self.names = names or {}
        self.span = float(span)                 # thickness of the 2D slab, for volumes/forces
        A, cen = self._poly_geometry()
        flip = A < 0
        if flip.any():                          # orient every cell counter-clockwise
            for r in np.flatnonzero(flip):
                k = nv[r]; C[r, :k] = C[r, :k][::-1].copy()
            A, cen = self._poly_geometry()
        self.area = np.abs(A)                   # cell area (2D)
        self.vol = self.area * self.span        # cell volume
        self.centroid = cen
        self.ncell = len(C)
        self.tris = C if (nv == 3).all() else None
        self._build_faces()
        self._tag_boundaries(edges, edge_tag)

    def _poly_geometry(self):
        """Signed shoelace area and area centroid of every (padded) polygon."""
        C, nv, X = self.cells, self.nvert, self.nodes
        A = np.zeros(len(C)); cx = np.zeros(len(C)); cy = np.zeros(len(C))
        for k in range(C.shape[1]):
            rows = np.flatnonzero(nv > k)
            a = C[rows, k]; b = C[rows, (k + 1) % nv[rows]]
            xa, ya, xb, yb = X[a, 0], X[a, 1], X[b, 0], X[b, 1]
            cr = xa * yb - xb * ya
            A[rows] += 0.5 * cr; cx[rows] += (xa + xb) * cr; cy[rows] += (ya + yb) * cr
        return A, np.stack([cx, cy], axis=1) / (6.0 * A)[:, None]

    def _build_faces(self):
        """Edge list with owner/neighbour, area vectors, and cell-to-cell geometry."""
        C, nv = self.cells, self.nvert
        # every edge of every polygon, cell-major (identical to the old triangle order when all
        # cells are triangles), as sorted node pairs so both sides of an interior edge share a key
        maxk = C.shape[1]
        kk = np.arange(maxk)[None, :]
        nxt = C[np.arange(self.ncell)[:, None], (kk + 1) % nv[:, None]]
        loc = np.stack([C, nxt], axis=2)                          # (ncell, maxk, 2); padded rows junk
        valid = kk < nv[:, None]
        flat = loc[valid]
        flat_cell = np.broadcast_to(np.arange(self.ncell)[:, None], (self.ncell, maxk))[valid]
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
        owner_cell = flat_cell[order[start]]
        first_local = order[start]
        self.face_nodes = flat[first_local]                       # oriented by the OWNER
        self.owner = owner_cell.astype(np.int64)
        self.neigh = np.full(len(start), -1, dtype=np.int64)
        second = start[count == 2] + 1
        self.neigh[count == 2] = flat_cell[order[second]]

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
        # Linear interpolation weight for the OWNER value at the face.
        #
        # NORMAL-PROJECTED, not distance-to-face-centre. The weight decides which POINT the
        # interpolation represents: w x_O + (1-w) x_N. For that point to be where the centroid
        # line actually crosses the face plane -- which is what the skewness correction then
        # corrects FROM -- the weight must use distances projected on the face normal. Using raw
        # distances to the face centre leaves a residual grad(phi).(x_int - x_w) that is O(h) and
        # does not vanish under refinement, and it is inconsistent with the skewness correction,
        # so the two together can be worse than neither.
        #
        # OpenFOAM (surfaceInterpolation::makeWeights) and code_saturne (_compute_face_distances)
        # both use this form; the raw-distance version appears in OpenFOAM ONLY as the fallback
        # for degenerate faces where the projected denominator underflows.
        #
        # On a uniform alternating-diagonal mesh both give exactly 0.5 by symmetry, so this
        # changes nothing there -- it matters on graded, clustered and perturbed meshes.
        w = np.ones(self.nface)
        i = self.interior
        sf = self.normal[i]
        so = np.abs((sf * (self.fcentre[i] - self.centroid[self.owner[i]])).sum(axis=1))
        sn = np.abs((sf * (self.centroid[self.neigh[i]] - self.fcentre[i])).sum(axis=1))
        den = so + sn
        bad = den < 1e-300 * np.maximum(np.hypot(*sf.T), 1e-300)
        w[i] = np.where(bad, 0.5, sn / np.maximum(den, 1e-300))
        self.wf = w
        self._decompose()

    def _decompose(self):
        """OVER-RELAXED split of the face area vector, S_f = E_f + T_f.

            E_f = d (S.S)/(d.S)        T_f = S_f - E_f

        `E_f` is parallel to the line joining the two cell centres, so the part of the Laplacian
        built on it is a two-point stencil and stays diagonally dominant; `T_f` carries the
        non-orthogonality and is deferred to the right-hand side. Over-relaxed (rather than
        minimum-correction or orthogonal-correction) because |E_f| GROWS with skewness, which is
        what keeps the implicit operator dominant on the worst cells instead of the best.

        THE FORMULA MATTERS, and the earlier one here did not match this docstring. `E_f =
        d (d.S)/(d.d)` is the MINIMUM-CORRECTION split, for which |E_f| = |S| cos(theta) and so
        |T_f|/|E_f| = tan(theta). The deferred correction is a lagged iteration with roughly that
        contraction factor, so it DIVERGES for any face beyond 45 degrees. The genuine
        over-relaxed split has |E_f| = |S|/cos(theta) and |T_f|/|E_f| = sin(theta) < 1 for every
        angle, which is the whole reason to prefer it.

        This was dormant while every test mesh was near-orthogonal and surfaced the moment the
        cavity mesh was clustered toward the walls: at cluster=1.5 the worst face is 69 degrees
        (tan = 2.6) and the run went to NaN before its first report. Both splits reduce to
        E_f = S_f when d is parallel to S, so uniform-mesh results are unchanged.

        The guide's stated check for this step, `E_f + T_f == S_f`, is an algebraic identity --
        T_f is DEFINED as the remainder, so it cannot fail and tests nothing. The real check is
        that the Laplacian built from the split annihilates a linear field on a skewed mesh,
        which is what `test_uops` does.
        """
        d = self.dcc
        dS = (d * self.normal).sum(axis=1)
        SS = (self.normal * self.normal).sum(axis=1)
        self.Ef = d * (SS / np.maximum(dS, 1e-300))[:, None]
        self.Tf = self.normal - self.Ef
        # |E_f| / |d| is the coefficient the implicit Laplacian uses on each face. For the
        # over-relaxed split this is just |S|^2/(d.S), with no square roots.
        self.ef_over_d = SS / np.maximum(dS, 1e-300)
        # cos of the angle between d and S: 1 = orthogonal, -> 0 = badly skewed
        self.orth = dS / np.maximum(np.hypot(*d.T) * np.hypot(*self.normal.T), 1e-300)

    # -------------------------------------------------------------------------------------
    def _tag_boundaries(self, edges, edge_tag):
        """Attach the Gmsh physical tag to each boundary face."""
        self.btag = np.zeros(self.nface, dtype=np.int64)
        # Boundary indexing belongs to the MESH. It used to be created as a side effect of
        # building the gradient, so any operator constructed first crashed on a mesh that
        # happened not to have one yet.
        self.bfaces = np.flatnonzero(self.boundary)
        self.bface_index = np.full(self.nface, -1, dtype=np.int64)
        self.bface_index[self.bfaces] = np.arange(len(self.bfaces))
        self.nbface = len(self.bfaces)
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

    def make_periodic(self, tag_a, tag_b, shift, rtol=1e-6):
        """Turn two boundary groups into one periodic seam.

        Every face tagged `tag_a` is paired with the face tagged `tag_b` whose centre is
        `fcentre_a + shift`; the pair becomes ONE interior face (the a-face survives with a's owner,
        its neighbour is b's owner, the b-face is deleted). The cell-to-cell vector of the merged
        face is `centroid[owner_b] - shift - centroid[owner_a]` (b's owner mapped back next to a), i.e. the period shift is carried
        in `dcc` so the vector points forward across the seam instead of jumping back across the
        domain -- the structured code's own scar ("ghost coordinates jump backwards across the
        seam, collapsing the Jacobian"). Everything downstream that reads geometry through `dcc`,
        `wf`, `Ef`/`Tf`, `normal` is then periodic for free; the two places that read
        `centroid[neigh]` directly were changed to `centroid[owner] + dcc`.
        Call once, after construction and before any operator is built."""
        shift = np.asarray(shift, dtype=float)
        fa = self.faces_with_tag(tag_a); fb = self.faces_with_tag(tag_b)
        if len(fa) == 0 or len(fa) != len(fb):
            raise ValueError(f"periodic pairing needs equal face counts: {len(fa)} vs {len(fb)}")
        from scipy.spatial import cKDTree
        dist, j = cKDTree(self.fcentre[fb]).query(self.fcentre[fa] + shift)
        scale = float(np.hypot(*shift)) if np.hypot(*shift) > 0 else 1.0
        if dist.max() > rtol * scale or len(np.unique(j)) != len(j):
            raise ValueError(f"periodic faces do not match: max offset {dist.max():.3e}, unique partners {len(np.unique(j))}/{len(j)}")
        partner = fb[j]
        self.neigh[fa] = self.owner[partner]
        # the neighbour across an a-face is the periodic IMAGE of b's owner, which sits at centroid_b - shift
        self.dcc[fa] = self.centroid[self.owner[partner]] - shift - self.centroid[self.owner[fa]]
        self.btag[fa] = 0
        keep = np.ones(self.nface, dtype=bool); keep[partner] = False
        for name in ("owner", "neigh", "face_nodes", "length", "normal", "sf", "fcentre", "dcc", "btag"):
            setattr(self, name, getattr(self, name)[keep])
        self.nface = len(self.owner)
        self.interior = self.neigh >= 0
        self.boundary = ~self.interior
        self.dmag = np.hypot(self.dcc[:, 0], self.dcc[:, 1])
        # normal-projected owner weights, with the neighbour centroid taken as owner + dcc
        w = np.ones(self.nface); i = self.interior; sf = self.normal[i]
        xo = self.centroid[self.owner[i]]; xn = xo + self.dcc[i]
        so = np.abs((sf * (self.fcentre[i] - xo)).sum(axis=1)); sn = np.abs((sf * (xn - self.fcentre[i])).sum(axis=1))
        den = so + sn; w[i] = np.where(den < 1e-300, 0.5, sn / np.maximum(den, 1e-300))
        self.wf = w
        self._decompose()
        self.bfaces = np.flatnonzero(self.boundary)
        self.bface_index = np.full(self.nface, -1, dtype=np.int64)
        self.bface_index[self.bfaces] = np.arange(len(self.bfaces))
        self.nbface = len(self.bfaces)
        self.periodic = getattr(self, "periodic", []) + [(int(tag_a), int(tag_b), shift.copy(), fa.copy())]
        return fa

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


def _pad_cells(cells):
    """(ncell, k) array or ragged list -> padded (ncell, maxk) int64 with -1, plus vertex counts."""
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        C = np.array(cells, dtype=np.int64)
        return C, np.full(len(C), C.shape[1], dtype=np.int64)
    nv = np.array([len(c) for c in cells], dtype=np.int64)
    C = np.full((len(cells), int(nv.max())), -1, dtype=np.int64)
    for r, c in enumerate(cells):
        C[r, :len(c)] = c
    return C, nv


def _cluster(n, a, b, beta):
    """n+1 points on [a,b], symmetrically clustered toward BOTH ends when beta > 0.

    Standard two-sided tanh stretch: uniform xi in [0,1] mapped through
    tanh(beta(2 xi - 1))/tanh(beta), renormalised to [0,1]. beta = 0 is uniform. The map is
    smooth and monotone, so the mesh stays valid for any beta, and symmetric, so both walls of
    a cavity get the same treatment.
    """
    xi = np.linspace(0.0, 1.0, n + 1)
    if not beta:
        return a + (b - a) * xi
    t = np.tanh(beta * (2.0 * xi - 1.0)) / np.tanh(beta)
    return a + (b - a) * 0.5 * (1.0 + t)


def rect_mesh(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0, span=1.0, perturb=0.0, seed=0,
              tags=("left", "right", "bottom", "top"), cluster=0.0, diag="alt",
              cells="tri", wall_layers=0, cluster_y=None):
    """Triangulated rectangle: nx by ny quads, each split into two triangles.

    `perturb` jitters the INTERIOR nodes by that fraction of the local spacing, which turns an
    otherwise perfectly orthogonal mesh into a skewed one. That matters: on a uniform right
    triangulation the non-orthogonal correction T_f is nearly zero, so a solver with a broken
    deferred-correction term still passes. Running every verification at perturb=0 AND at
    perturb>0 is what stops that.

    `cluster` (beta) stretches the node distribution toward all four walls, for cases whose
    error lives in a boundary layer rather than in the interior. At Re = 1000 the cavity wall
    layers are O(Re^-1/2) ~ 0.03, so a uniform 1/64 mesh puts only about two cells across them.

    CLUSTERING IS NOT FREE HERE. Non-uniform spacing destroys the exact interior orthogonality
    of the uniform right triangulation, which activates the non-orthogonal correction path --
    including the known `dp_compact` / `dp_wide` directional mismatch in Rhie-Chow, which is
    invisible at cluster=0. Measure `orth` before reading anything into a clustered result.

    Boundary physical tags are 1..4 for left/right/bottom/top, named by `tags`.
    """
    rng = np.random.default_rng(seed)
    xs = _cluster(nx, x0, x1, cluster)
    ys = _cluster(ny, y0, y1, cluster if cluster_y is None else cluster_y)   # cluster_y: wall-normal only (channel)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if perturb:
        # jitter by the LOCAL spacing, so a clustered mesh is not torn apart near the walls
        hx = np.minimum(np.diff(xs)[:-1], np.diff(xs)[1:])[:, None]
        hy = np.minimum(np.diff(ys)[:-1], np.diff(ys)[1:])[None, :]
        X[1:-1, 1:-1] += perturb * hx * (rng.random((nx - 1, ny - 1)) - 0.5)
        Y[1:-1, 1:-1] += perturb * hy * (rng.random((nx - 1, ny - 1)) - 0.5)
    nid = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]
            # `diag`: "alt" alternates the diagonal so the mesh has no global bias direction;
            # "same" splits every quad the same way. THIS IS NOT COSMETIC. Alternating diagonals
            # make the centroid-to-centroid line miss the face barycentre by exactly h/6, a fixed
            # fraction of the cell that does not shrink under refinement. A same-diagonal split
            # has EXACTLY ZERO skewness on every face family. Eymard, Herbin & Latche (M2AN 40
            # (2006) 501, Remarks 2.1 and 4.4) prove convergence for collocated FV Stokes only
            # when that segment crosses the face at its barycentre, and state that without it the
            # first-order rate is lost -- so this choice is a hypothesis of the theorem, not a
            # meshing preference.
            # `cells`: "tri" splits every quad; "quad" splits none; "hybrid" keeps the outermost
            # `wall_layers` rows/columns as quads and splits the core. A quad/triangle interface
            # shares one node pair, so the face builder sees an ordinary manifold edge.
            near_wall = (i < wall_layers or i >= nx - wall_layers
                         or j < wall_layers or j >= ny - wall_layers)
            if cells == "quad" or (cells == "hybrid" and near_wall):
                tris.append([a, b, c, d])
            elif diag == "same" or (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]

    edges, etag = [], []
    for j in range(ny):
        edges.append([nid[0, j], nid[0, j + 1]]); etag.append(1)          # left
        edges.append([nid[nx, j], nid[nx, j + 1]]); etag.append(2)        # right
    for i in range(nx):
        edges.append([nid[i, 0], nid[i + 1, 0]]); etag.append(3)          # bottom
        edges.append([nid[i, ny], nid[i + 1, ny]]); etag.append(4)        # top
    names = {k + 1: tags[k] for k in range(4)}
    return Mesh(nodes, tris, np.array(edges), np.array(etag), names, span=span)
