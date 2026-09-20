"""Finite-volume operators on an unstructured triangular mesh.

EVERYTHING IS AN EXPLICIT SPARSE MATRIX. The discrete adjoint needs to transpose these
operators, so nothing here is matrix-free and nothing mutates its inputs. The cost is assembly
time once per mesh; the benefit is that `A.T` is the adjoint of `A` by construction rather than
by a second hand-written derivation that can drift out of step.

The gradient is LEAST SQUARES rather than Green-Gauss. On triangles a Green-Gauss gradient needs
face values, which need a gradient to reconstruct -- circular, and first-order on a skewed mesh.
Least squares reconstructs a linear field EXACTLY from any three non-collinear neighbours, which
is the property the verification leans on.
"""
import numpy as np
import scipy.sparse as sp


def lsq_gradient(mesh, weight="inv_dist2"):
    """Least-squares cell gradient as four sparse matrices.

        grad_x = Gx @ phi + Bx @ phi_b
        grad_y = Gy @ phi + By @ phi_b

    `phi` is the (ncell,) cell field and `phi_b` the (n_boundary_face,) boundary values, indexed
    by `mesh.bface_index`. Splitting the boundary out keeps both pieces linear, so the adjoint is
    just the transpose of each.

    For cell P the fit minimises  sum_j w_j ( g.d_j - (phi_j - phi_P) )^2  over the neighbours j
    reached through P's faces, with boundary faces contributing their FACE CENTRE as a neighbour.
    That last part matters: a corner cell can have only one interior neighbour, and a single
    difference vector leaves the 2x2 normal matrix singular.
    """
    nc, nf = mesh.ncell, mesh.nface
    bidx = np.full(nf, -1, dtype=np.int64)
    bf = np.flatnonzero(mesh.boundary)
    bidx[bf] = np.arange(len(bf))
    mesh.bface_index = bidx
    mesh.bfaces = bf

    d = mesh.dcc                                   # owner -> neighbour (or -> face centre)
    dd = (d * d).sum(axis=1)
    if weight == "inv_dist2":
        w = 1.0 / np.maximum(dd, 1e-300)
    elif weight == "unit":
        w = np.ones(nf)
    else:
        raise ValueError(f"unknown weight {weight!r}")

    # --- per-cell 2x2 normal matrix -----------------------------------------------------------
    M = np.zeros((nc, 2, 2))
    for a in range(2):
        for b in range(2):
            contrib = w * d[:, a] * d[:, b]
            np.add.at(M[:, a, b], mesh.owner, contrib)
            i = mesh.interior
            # the neighbour sees -d, and (-d_a)(-d_b) == d_a d_b, so the same contribution
            np.add.at(M[:, a, b], mesh.neigh[i], contrib[i])
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    if (np.abs(det) < 1e-300).any():
        bad = int((np.abs(det) < 1e-300).sum())
        raise ValueError(f"{bad} cells have a singular least-squares matrix "
                         f"(fewer than two independent neighbour directions)")
    inv = np.empty_like(M)
    inv[:, 0, 0] = M[:, 1, 1] / det
    inv[:, 1, 1] = M[:, 0, 0] / det
    inv[:, 0, 1] = -M[:, 0, 1] / det
    inv[:, 1, 0] = -M[:, 1, 0] / det
    mesh.lsq_cond = np.abs(det)

    # --- assemble ------------------------------------------------------------------------------
    # face f contributes  inv_P @ (w d)  to cell P, with +1 on the neighbour column and -1 on P.
    rows_i, cols_i, vx_i, vy_i = [], [], [], []
    rows_b, cols_b, vx_b, vy_b = [], [], [], []

    def emit(cells, dvec, wgt, other_col, boundary):
        """cells: the cell whose gradient this contributes to; other_col: the column that gets +."""
        cx = inv[cells, 0, 0] * (wgt * dvec[:, 0]) + inv[cells, 0, 1] * (wgt * dvec[:, 1])
        cy = inv[cells, 1, 0] * (wgt * dvec[:, 0]) + inv[cells, 1, 1] * (wgt * dvec[:, 1])
        if boundary:
            rows_b.append(cells); cols_b.append(other_col); vx_b.append(cx); vy_b.append(cy)
        else:
            rows_i.append(cells); cols_i.append(other_col); vx_i.append(cx); vy_i.append(cy)
        # and -1 on the cell's own column, always an interior (cell) column
        rows_i.append(cells); cols_i.append(cells); vx_i.append(-cx); vy_i.append(-cy)

    i = mesh.interior
    # owner side: d as stored, neighbour column gets +
    emit(mesh.owner[i], d[i], w[i], mesh.neigh[i], False)
    # neighbour side: sees -d, owner column gets +
    emit(mesh.neigh[i], -d[i], w[i], mesh.owner[i], False)
    # boundary: face centre acts as the neighbour, its value lives in phi_b
    b = mesh.boundary
    emit(mesh.owner[b], d[b], w[b], bidx[b], True)

    cat = np.concatenate
    Gx = sp.coo_matrix((cat(vx_i), (cat(rows_i), cat(cols_i))), shape=(nc, nc)).tocsr()
    Gy = sp.coo_matrix((cat(vy_i), (cat(rows_i), cat(cols_i))), shape=(nc, nc)).tocsr()
    Bx = sp.coo_matrix((cat(vx_b), (cat(rows_b), cat(cols_b))), shape=(nc, len(bf))).tocsr()
    By = sp.coo_matrix((cat(vy_b), (cat(rows_b), cat(cols_b))), shape=(nc, len(bf))).tocsr()
    return Gx, Gy, Bx, By


class Gradient:
    """Callable wrapper keeping the four matrices together."""

    def __init__(self, mesh, weight="inv_dist2"):
        self.mesh = mesh
        self.Gx, self.Gy, self.Bx, self.By = lsq_gradient(mesh, weight)

    def __call__(self, phi, phi_b=None):
        if phi_b is None:
            phi_b = np.zeros(self.Bx.shape[1])
        return np.stack([self.Gx @ phi + self.Bx @ phi_b,
                         self.Gy @ phi + self.By @ phi_b], axis=1)


def divergence(mesh, face_flux):
    """Cell divergence of a face flux (already the dotted, area-weighted value per face).

    Sum out of the owner, into the neighbour -- the discrete divergence theorem, and the exact
    inverse bookkeeping of the closure check in `Mesh.audit`.
    """
    out = np.zeros(mesh.ncell)
    np.add.at(out, mesh.owner, face_flux)
    i = mesh.interior
    np.add.at(out, mesh.neigh[i], -face_flux[i])
    return out / mesh.vol


def face_interp(mesh, phi, phi_b=None):
    """Linear (distance-weighted) interpolation of a cell field to faces."""
    f = np.empty(mesh.nface)
    i = mesh.interior
    f[i] = mesh.wf[i] * phi[mesh.owner[i]] + (1.0 - mesh.wf[i]) * phi[mesh.neigh[i]]
    b = mesh.boundary
    f[b] = phi[mesh.owner[b]] if phi_b is None else phi_b[mesh.bface_index[b]]
    return f


# ---------------------------------------------------------------------------------------------
# Boundary condition kinds. A boundary face is just a face with one owner, so a BC is a rule for
# what sits on the far side of it -- no ghost cells, no padding, no seam ownership.
DIRICHLET, NEUMANN = 0, 1


def laplacian(mesh, gamma_f, bkind, bval=None):
    """Volume-integrated Laplacian  sum_f gamma_f S_f . grad(phi)  as (A, rhs_fn).

    `A @ phi` is the IMPLICIT orthogonal part, built on E_f so the stencil is two-point and the
    matrix stays diagonally dominant. The remainder, gamma_f T_f . grad(phi)_f, is NOT in the
    matrix -- it depends on the cell gradients, which depend on phi -- and is returned by
    `rhs_fn(grad_phi)` for deferred correction.

    `bkind` is per boundary face: DIRICHLET (value pinned, contributes to the matrix diagonal and
    the right-hand side) or NEUMANN (zero normal gradient, contributes nothing to either -- the
    face flux is simply zero, which is what a symmetry plane and a zero-gradient outlet both
    want for the quantity they do not pin).
    """
    nc = mesh.ncell
    i = mesh.interior
    b = mesh.boundary
    coef = gamma_f * mesh.ef_over_d * mesh.span          # per face, the two-point coefficient

    o, n = mesh.owner, mesh.neigh
    rows = [o[i], n[i], o[i], n[i]]
    cols = [n[i], o[i], o[i], n[i]]
    vals = [coef[i], coef[i], -coef[i], -coef[i]]

    bi = mesh.bface_index[b]
    dir_mask = bkind[bi] == DIRICHLET
    bd = np.flatnonzero(b)[dir_mask]
    rows.append(o[bd]); cols.append(o[bd]); vals.append(-coef[bd])

    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc)).tocsr()

    def rhs_fn(grad_phi, bvalues=None):
        """Deferred non-orthogonal correction plus the Dirichlet boundary contribution."""
        out = np.zeros(nc)
        # non-orthogonal part, gamma T_f . grad_f, with grad_f a face interpolation of the
        # cell gradients. Exact for a linear field, which is what makes lap(linear) == 0.
        gf = np.empty((mesh.nface, 2))
        gf[i] = (mesh.wf[i, None] * grad_phi[o[i]]
                 + (1.0 - mesh.wf[i, None]) * grad_phi[n[i]])
        gf[b] = grad_phi[o[b]]
        cross = gamma_f * (mesh.Tf * gf).sum(axis=1) * mesh.span
        np.add.at(out, o, cross)
        np.add.at(out, n[i], -cross[i])
        if bvalues is not None and len(bd):
            # The flux out of the owner through a Dirichlet face is coef*(phi_b - phi_P). The
            # -coef*phi_P half sits in the matrix diagonal, so the RHS carries +coef*phi_b.
            np.add.at(out, o[bd], coef[bd] * bvalues[mesh.bface_index[bd]])
        return out

    return A, rhs_fn
