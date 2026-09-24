"""k-exact cell gradients on a vertex-neighbour stencil.

Why this exists: the face-neighbour least-squares gradient in `uops.lsq_gradient` is only FIRST
order pointwise, interior cells included (measured 0.97, 0.98, 0.99 under refinement). That is
not a stencil-width problem -- a LINEAR fit recovers the gradient of a curved field to O(h) no
matter how many points it is fitted through. Raising the order needs a higher polynomial degree,
and that in turn needs a wider stencil: a quadratic in 2D has 6 coefficients, 5 free once the
cell-average constraint is imposed, and on `rect_mesh` 100% of cells have fewer than 5 face
neighbours while 0% have fewer than 5 vertex neighbours (means 3.00 against 13.88).

`degree=1` on the same wide stencil is provided deliberately, to separate the two effects.

k-EXACT MEANS CELL AVERAGES, NOT CENTROID POINT VALUES. Fitting through centroid values would
reintroduce an O(h^2) error in the right-hand side and cap the gradient back at first order. So
the reconstruction is written about P as

    phi(x) = phi_P + g.(x - x_P) + 1/2 H : [ (x - x_P)(x - x_P) - M_P ]

whose average over P is exactly phi_P, and whose average over neighbour j is

    phi_P + g.d_j + 1/2 H : (M_j - M_P),

with M_c the second moment of cell c about x_P. M_c is evaluated by the three-edge-midpoint rule,
which is EXACT for quadratics on a triangle, so no quadrature error enters the fit.

Everything stays four explicit sparse matrices, as the adjoint decision requires, and the stencil
is geometry-determined -- no solution-dependent fallback, which would make the adjoint inexact.
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def _second_moment(mesh, cells, xP):
    """(1/A) int_cell (x-xP)(x-xP) dA, exact for quadratics via the edge-midpoint rule."""
    v = mesh.nodes[mesh.tris[cells]]                       # (n,3,2)
    mid = 0.5 * (v + np.roll(v, -1, axis=1))               # edge midpoints
    r = mid - xP[:, None, :]
    return np.einsum("nmi,nmj->nij", r, r) / 3.0


def kexact_gradient(mesh, degree=2, weight="inv_dist2"):
    nc = mesh.ncell
    v2c = defaultdict(list)
    for c, t in enumerate(mesh.tris):
        for v in t:
            v2c[v].append(c)
    bf_of = defaultdict(list)
    for f in mesh.bfaces:
        bf_of[mesh.owner[f]].append(f)

    ncoef = 2 if degree == 1 else 5
    rows_i, cols_i, vx_i, vy_i = [], [], [], []
    rows_b, cols_b, vx_b, vy_b = [], [], [], []
    cond = np.zeros(nc)

    for P in range(nc):
        xP = mesh.centroid[P]
        nbr = sorted({c for v in mesh.tris[P] for c in v2c[v]} - {P})
        bfs = bf_of.get(P, [])
        if len(nbr) + len(bfs) < ncoef:
            raise ValueError(f"cell {P}: stencil of {len(nbr)+len(bfs)} < {ncoef} coefficients")
        nbr = np.asarray(nbr, dtype=np.int64)
        d = mesh.centroid[nbr] - xP
        rowsA, dists = [], []
        if degree == 1:
            rowsA.append(d)
        else:
            MP = _second_moment(mesh, np.array([P]), xP[None, :])[0]
            Mj = _second_moment(mesh, nbr, np.repeat(xP[None, :], len(nbr), axis=0))
            dM = Mj - MP
            rowsA.append(np.column_stack([d[:, 0], d[:, 1],
                                          0.5 * dM[:, 0, 0], dM[:, 0, 1], 0.5 * dM[:, 1, 1]]))
        dists.append(d)
        if bfs:
            db = mesh.fcentre[bfs] - xP
            if degree == 1:
                rowsA.append(db)
            else:
                # a boundary face gives a POINT value, so its basis uses d(x)d(x) directly
                rowsA.append(np.column_stack([db[:, 0], db[:, 1],
                                              0.5 * (db[:, 0]**2 - MP[0, 0]),
                                              db[:, 0]*db[:, 1] - MP[0, 1],
                                              0.5 * (db[:, 1]**2 - MP[1, 1])]))
            dists.append(db)
        A = np.vstack(rowsA)
        dv = np.vstack(dists)
        w = 1.0 / np.maximum((dv*dv).sum(axis=1), 1e-300) if weight == "inv_dist2" \
            else np.ones(len(dv))
        sw = np.sqrt(w)
        Aw = sw[:, None] * A
        # COLUMN EQUILIBRATION. The quadratic basis mixes O(h) columns (dx, dy) with O(h^2)
        # ones (the moment terms), so the normal matrix spans h^2..h^4 and the condition number
        # grows like 1/h: measured median 43.5, 87.1, 174.1 at n = 16, 32, 64 -- a clean doubling
        # per refinement, i.e. the fit gets worse precisely as the mesh gets better. Scaling each
        # column to unit norm before the pseudo-inverse and unscaling the coefficients afterwards
        # makes the fit invariant to that, and is exact: with Aw = As diag(c),
        # argmin||Aw a - b|| = diag(1/c) argmin||As y - b||.
        cn = np.linalg.norm(Aw, axis=0)
        cn[cn == 0.0] = 1.0
        As = Aw / cn
        pinv = np.linalg.pinv(As) / cn[:, None]
        cond[P] = np.linalg.cond(As)
        coef = pinv[:2, :] * sw[None, :]            # (2, nrows): d(gx,gy)/d(phi_j - phi_P)
        k = len(nbr)
        rows_i.append(np.full(k, P)); cols_i.append(nbr)
        vx_i.append(coef[0, :k]); vy_i.append(coef[1, :k])
        rows_i.append(np.full(k, P)); cols_i.append(np.full(k, P))
        vx_i.append(-coef[0, :k]); vy_i.append(-coef[1, :k])
        if bfs:
            bi = mesh.bface_index[np.asarray(bfs)]
            rows_b.append(np.full(len(bfs), P)); cols_b.append(bi)
            vx_b.append(coef[0, k:]); vy_b.append(coef[1, k:])
            rows_i.append(np.full(len(bfs), P)); cols_i.append(np.full(len(bfs), P))
            vx_i.append(-coef[0, k:]); vy_i.append(-coef[1, k:])

    mk = lambda r, c, v, shape: sp.coo_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=shape).tocsr()
    Gx = mk(rows_i, cols_i, vx_i, (nc, nc)); Gy = mk(rows_i, cols_i, vy_i, (nc, nc))
    nb = mesh.nbface
    if rows_b:
        Bx = mk(rows_b, cols_b, vx_b, (nc, nb)); By = mk(rows_b, cols_b, vy_b, (nc, nb))
    else:
        Bx = sp.csr_matrix((nc, nb)); By = sp.csr_matrix((nc, nb))
    mesh.kexact_cond = cond
    return Gx, Gy, Bx, By


class KGradient:
    """Drop-in replacement for `uops.Gradient`."""

    def __init__(self, mesh, degree=2, weight="inv_dist2"):
        self.mesh = mesh
        self.Gx, self.Gy, self.Bx, self.By = kexact_gradient(mesh, degree, weight)

    def __call__(self, phi, phi_b=None):
        if phi_b is None:
            phi_b = np.zeros(self.Bx.shape[1])
        return np.stack([self.Gx @ phi + self.Bx @ phi_b,
                         self.Gy @ phi + self.By @ phi_b], axis=1)


class GGGradient:
    """Green-Gauss (face-averaged) cell gradient, as four sparse matrices.

        grad_P = (1/V_P) sum_f p_f S_f^(P),   p_f = w p_O + (1-w) p_N   (boundary: p_f = p_b)

    Deliberately NOT linear-exact on irregular triangles (Sozer et al. 2014; Syrakos et al.
    2017), and that is the point of offering it: a face-AVERAGED gradient is BLIND to a
    cell-to-cell checkerboard, because the face value of an alternating field is its mean, ~0.
    The face-neighbour LSQ gradient is exact for linear fields and, measured on T5, sees the
    checkerboard pressure error 15-20x more strongly than it sees the smooth part -- which is
    what transmits the checkerboard into the velocity and produces the mesh-independent floor.
    This is the gradient OpenFOAM's `fvc::grad(p)` (Gauss linear) actually is, and it is the one
    the collocated literature has in mind when it calls the wide gradient "blind" to checkerboard.
    Same interface as `Gradient` / `KGradient`.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        nc, nb = mesh.ncell, mesh.nbface
        i = mesh.interior; b = mesh.boundary
        o, n, w = mesh.owner, mesh.neigh, mesh.wf
        S = mesh.normal * mesh.span
        invV = 1.0 / mesh.vol
        rows = []; cols = []; vx = []; vy = []
        for cell, sgn in ((o[i], +1.0), (n[i], -1.0)):
            for other, coef in ((o[i], w[i]), (n[i], 1.0 - w[i])):
                rows.append(cell); cols.append(other)
                vx.append(sgn * coef * S[i, 0] * invV[cell]); vy.append(sgn * coef * S[i, 1] * invV[cell])
        self.Gx = sp.coo_matrix((np.concatenate(vx), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc)).tocsr()
        self.Gy = sp.coo_matrix((np.concatenate(vy), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc)).tocsr()
        bf = mesh.bfaces; bo = mesh.owner[bf]; bi = mesh.bface_index[bf]
        self.Bx = sp.coo_matrix((S[bf, 0] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr()
        self.By = sp.coo_matrix((S[bf, 1] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr()

    def __call__(self, phi, phi_b=None):
        if phi_b is None:
            phi_b = np.zeros(self.Bx.shape[1])
        return np.stack([self.Gx @ phi + self.Bx @ phi_b,
                         self.Gy @ phi + self.By @ phi_b], axis=1)


class GGDualGradient(GGGradient):
    """Green-Gauss gradient that is the EXACT adjoint of the interpolated-flux divergence.

    Same as `GGGradient` but with the face weights SWAPPED for the pressure:

        p_f = (1 - w) p_O + w p_N        (w = owner weight used for the velocity flux)

    With the velocity interpolated as u_f = w u_O + (1-w) u_N, the identity
    sum_P V_P u_P.(G p)_P + sum_P V_P p_P (D u)_P = 0 holds to machine precision for ANY w,
    because every face contributes (u_O p_O - u_N p_N).S_f and that telescopes by cell closure.
    With the ordinary weights it holds only when w = 1/2, i.e. only on a uniform mesh -- measured
    2.2e-15 uniform but 1.11 (relative) on a perturbed mesh.

    This is eq. (11) of Eymard, Herbin & Latche, ESAIM M2AN 40 (2006) 501, who build the gradient
    as the transpose of the divergence and note the swapped weighting "seems to be of crucial
    importance in the analysis of the stability of the scheme". It looks backwards -- the weight on
    p_N is the owner's distance fraction -- and that is correct.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        nc, nb = mesh.ncell, mesh.nbface
        i = mesh.interior
        o, n, w = mesh.owner, mesh.neigh, mesh.wf
        S = mesh.normal * mesh.span
        invV = 1.0 / mesh.vol
        rows = []; cols = []; vx = []; vy = []
        for cell, sgn in ((o[i], +1.0), (n[i], -1.0)):
            for other, coef in ((o[i], 1.0 - w[i]), (n[i], w[i])):      # SWAPPED
                rows.append(cell); cols.append(other)
                vx.append(sgn * coef * S[i, 0] * invV[cell]); vy.append(sgn * coef * S[i, 1] * invV[cell])
        self.Gx = sp.coo_matrix((np.concatenate(vx), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc)).tocsr()
        self.Gy = sp.coo_matrix((np.concatenate(vy), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc)).tocsr()
        bf = mesh.bfaces; bo = mesh.owner[bf]; bi = mesh.bface_index[bf]
        self.Bx = sp.coo_matrix((S[bf, 0] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr()
        self.By = sp.coo_matrix((S[bf, 1] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr()


class GGSkewGradient:
    """Green-Gauss with SKEWNESS-CORRECTED face values: divergence-form AND linear-exact.

        p_f = (1-w) p_O + w p_N  +  grad_p_f . (x_f - x_int),   grad_p_f = w' G_O + (1-w') G_N

    with `G` the face-neighbour LSQ cell gradient (exact for linear fields) and x_int the point
    where the centroid line crosses the face. The first term is the dual-weighted GG face value
    (exact for linear fields AT x_int); the correction moves it to the face centroid x_f, and is
    exact for linear fields because the LSQ gradient is. So the whole operator is exact for
    linear fields -- which plain GG is not (error 1.00 on `3x - 2y` on the uniform mesh) -- and
    it stays in divergence form, so it still telescopes and still does not transmit the
    checkerboard to the velocity.

    Why: sections 17, 20 and 22 of the record. Plain GG in the momentum term carries Sozer's
    `C != 1` bias on irregular cells. It showed as every clustered-cavity extremum undershooting
    by 1-5%, as a perturbed-mesh velocity order of only ~0.9, and -- decisively -- as a SMOOTH
    pressure error of ~6% that does not converge on the perturbed mesh (order -0.12), where the
    uniform mesh gives 1.75. A linear-exact face value removes the bias at its source.

    Duality: the interpolation term is exactly dual to the divergence (dual weights); the
    correction term is not, and is O(h) relative to it. So this is approximately dual, not
    exactly. Whether that matters is measured, not assumed.

    Still a fixed sparse matrix: [W + S . interp(G)] assembled once; adjoint is the transpose.
    """

    def __init__(self, mesh, weight="inv_dist2"):
        from src.uops import Gradient
        self.mesh = mesh
        nc, nb = mesh.ncell, mesh.nbface
        i = mesh.interior
        o, n, w = mesh.owner, mesh.neigh, mesh.wf
        S = mesh.normal * mesh.span
        invV = 1.0 / mesh.vol
        G = Gradient(mesh, weight)
        # skewness vector per interior face: face centroid minus centroid-line crossing
        d = mesh.dcc[i]                                                      # = centroid[n]-centroid[o], periodic-safe
        lam = ((mesh.fcentre[i] - mesh.centroid[o[i]]) * S[i]).sum(axis=1) / np.maximum((d * S[i]).sum(axis=1), 1e-300)
        # DUAL weights (1-w) p_O + w p_N are exact for a linear field at C_O + (1-lam) d -- the
        # MIRROR of the ordinary crossing point C_O + lam d about the centroid-line midpoint. The
        # correction must run from THAT point to the face centroid. Correcting from the ordinary
        # point was exact on the uniform mesh (lam = 1/2, the two coincide) and WRONG elsewhere:
        # linear-field error 3.13 on the perturbed mesh, worse than plain GG's 1.70.
        skew = mesh.fcentre[i] - (mesh.centroid[o[i]] + (1.0 - lam)[:, None] * d)   # (nfi, 2)
        # face value operator on cell values:  P_f = A_face @ p   (nfi x nc), plus boundary part
        nfi = int(i.sum()); fi = np.flatnonzero(i)
        rows = np.concatenate([np.arange(nfi), np.arange(nfi)])
        cols = np.concatenate([o[i], n[i]])
        vals = np.concatenate([1.0 - w[i], w[i]])                                   # DUAL weights
        Aface = sp.coo_matrix((vals, (rows, cols)), shape=(nfi, nc)).tocsr()
        # gradient interpolated to the face (ordinary weights), dotted with the skew vector
        Wo = sp.diags(w[i]); Wn = sp.diags(1.0 - w[i])
        Po = sp.coo_matrix((np.ones(nfi), (np.arange(nfi), o[i])), shape=(nfi, nc)).tocsr()
        Pn = sp.coo_matrix((np.ones(nfi), (np.arange(nfi), n[i])), shape=(nfi, nc)).tocsr()
        Gfx = Wo @ Po @ G.Gx + Wn @ Pn @ G.Gx; Gfy = Wo @ Po @ G.Gy + Wn @ Pn @ G.Gy
        Bfx = Wo @ Po @ G.Bx + Wn @ Pn @ G.Bx; Bfy = Wo @ Po @ G.By + Wn @ Pn @ G.By
        Aface = Aface + sp.diags(skew[:, 0]) @ Gfx + sp.diags(skew[:, 1]) @ Gfy
        Bface = sp.diags(skew[:, 0]) @ Bfx + sp.diags(skew[:, 1]) @ Bfy                 # (nfi x nb)
        # Green-Gauss assembly: cell P gets +p_f S_f for owner, -p_f S_f for neighbour
        Do = sp.coo_matrix((S[i, 0] * invV[o[i]], (o[i], np.arange(nfi))), shape=(nc, nfi)).tocsr() \
           - sp.coo_matrix((S[i, 0] * invV[n[i]], (n[i], np.arange(nfi))), shape=(nc, nfi)).tocsr()
        Dy = sp.coo_matrix((S[i, 1] * invV[o[i]], (o[i], np.arange(nfi))), shape=(nc, nfi)).tocsr() \
           - sp.coo_matrix((S[i, 1] * invV[n[i]], (n[i], np.arange(nfi))), shape=(nc, nfi)).tocsr()
        self.Gx = (Do @ Aface).tocsr(); self.Gy = (Dy @ Aface).tocsr()
        bf = mesh.bfaces; bo = mesh.owner[bf]; bi = mesh.bface_index[bf]
        self.Bx = (sp.coo_matrix((S[bf, 0] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr() + Do @ Bface).tocsr()
        self.By = (sp.coo_matrix((S[bf, 1] * invV[bo], (bo, bi)), shape=(nc, nb)).tocsr() + Dy @ Bface).tocsr()

    def __call__(self, phi, phi_b=None):
        if phi_b is None:
            phi_b = np.zeros(self.Bx.shape[1])
        return np.stack([self.Gx @ phi + self.Bx @ phi_b,
                         self.Gy @ phi + self.By @ phi_b], axis=1)
