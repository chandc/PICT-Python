"""U1-U2 of the unstructured adjoint plan: the unstructured FV operators in torch.

Every function here is a line-for-line transcription of its numpy counterpart in `src/uops.py` /
`src/upiso.py`, written on face arrays with `index_add_`, so autograd differentiates the discrete
scheme the production solver actually runs. No operator is re-derived: the constant ones are
converted from the production scipy matrices (`SparseConst.from_scipy`), and the solution-
dependent ones are gated against production to round-off (`test_uadj_ops.py`).

NON-SMOOTH POINTS ARE MASKS, AND MASKS ARE DETACHED. The upwind direction of each face (sign of
F), the SIMPLEC floor max(rowsum, a_t V) and the boundary inflow/outflow split are decisions, not
functions: the gradient is the one of the branch production took. `Masks` records every decision
in call order on a reference forward and can replay it, so a finite-difference probe evaluates
the SAME branch on both sides of the difference (plan section 7.1). A central difference taken
across a switch would otherwise average two branches and disagree with any correct adjoint.
"""
import numpy as np
import scipy.sparse as sp
import torch

from src.uops import DIRICHLET, NEUMANN

DT = torch.float64


def _t(a, dtype=DT):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def _ti(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.long)


class Masks:
    """Detached branch decisions, recorded in call order and optionally replayed.

    mode "live":   compute every mask from the current values (production behaviour);
    mode "record": as live, and append each mask to `self.log`;
    mode "replay": return the recorded masks in the same order, ignoring the values.
    """

    def __init__(self, mode="live"):
        self.mode, self.log, self.k = mode, [], 0
        # replay: re-decide tie faces live, so a central FD straddles the kink (st_mask). Off for a
        # fully pinned operator, e.g. the linearity gate A4.
        self.straddle = True

    def __call__(self, name, compute):
        if self.mode == "replay":
            nm, m = self.log[self.k]
            if nm != name:
                raise RuntimeError(f"mask replay out of order: expected {nm!r}, got {name!r}")
            self.k += 1
            return m
        m = compute().detach()
        if self.mode == "record":
            self.log.append((name, m))
        return m

    def replay(self):
        """A replaying copy of a recorded log, rewound."""
        r = Masks("replay"); r.log = self.log; r.straddle = self.straddle; return r


TIE = 1e-12          # |F| <= TIE * max|F| (or rowsum within TIE of its floor) is a tie


def st_mask(x, value_mask, grad_weight):
    """x * value_mask in VALUE, with d/dx = grad_weight: the branch production took for the forward,
    and the average of the two one-sided derivatives (weight 1/2) where the decision is a tie.

    WHY: a symmetric state sits ON the kinks. On the faces crossing a symmetry axis v = 0, so F = 0
    exactly, and F * phi_upwind has one-sided derivatives phi_owner and phi_neighbour, which differ
    at O(1) even though F ~ 0. The branch round-off picks is asymmetric, and the adjoint then broke
    the mirror symmetry the forward keeps exactly (dC_D/domega 5.7e-4 of dC_L/domega where it must
    be 0; record finding 12). The midpoint of the two derivatives is what a central difference
    measures, and it is symmetric."""
    return x.detach() * value_mask + (x - x.detach()) * grad_weight


class SparseConst:
    """A constant sparse matrix as entry lists; y = A x is differentiable in x."""

    def __init__(self, rows, cols, vals, shape):
        self.rows, self.cols, self.vals, self.shape = _ti(rows), _ti(cols), _t(vals), tuple(shape)

    @classmethod
    def from_scipy(cls, A):
        A = sp.coo_matrix(A)
        return cls(A.row, A.col, A.data, A.shape)

    def __matmul__(self, x):
        return torch.zeros(self.shape[0], dtype=x.dtype).index_add_(0, self.rows, self.vals * x[self.cols])

    def rmatvec(self, y):
        """A^T y, for the adjoint-identity gate."""
        return torch.zeros(self.shape[1], dtype=y.dtype).index_add_(0, self.cols, self.vals * y[self.rows])


class TMesh:
    """Torch copies of the mesh arrays the step reads."""

    def __init__(self, m):
        self.m = m
        self.ncell, self.nface, self.nbface = m.ncell, m.nface, m.nbface
        self.owner = _ti(m.owner)
        i = np.flatnonzero(m.interior); b = np.flatnonzero(m.boundary)
        self.fi, self.fb = _ti(i), _ti(b)
        self.o_i, self.n_i = _ti(m.owner[i]), _ti(m.neigh[i])
        self.o_b = _ti(m.owner[b])
        self.bidx_b = _ti(m.bface_index[b])            # boundary face -> slot in a (nbface,) vector
        self.o_bf = _ti(m.owner[m.bfaces])             # owner of each boundary slot
        self.wf = _t(m.wf); self.w_i = self.wf[self.fi]
        self.normal = _t(m.normal); self.span = float(m.span)
        self.ef_over_d = _t(m.ef_over_d); self.Tf = _t(m.Tf)
        self.vol = _t(m.vol)
        self.fcentre = _t(m.fcentre); self.centroid = _t(m.centroid); self.dcc = _t(m.dcc)
        # boundary-outflow extrapolation vector and interior skewness offset: geometry only
        self.dx_b = self.fcentre[self.fb] - self.centroid[self.o_b]
        xin = self.centroid[self.o_i] + (1.0 - self.w_i)[:, None] * self.dcc[self.fi]
        self.skew_i = self.fcentre[self.fi] - xin

    # --- face <-> cell -------------------------------------------------------------------------
    def interp(self, phi, phi_b=None):
        """`uops.face_interp`: linear on interior faces, owner (or phi_b) on boundary faces."""
        f = torch.zeros(self.nface, dtype=phi.dtype)
        f = f.index_copy(0, self.fi, self.w_i * phi[self.o_i] + (1.0 - self.w_i) * phi[self.n_i])
        fb = phi[self.o_b] if phi_b is None else phi_b[self.bidx_b]
        return f.index_copy(0, self.fb, fb)

    def interp_vec(self, g):
        """Cell (n, 2) -> face (nface, 2), owner value on boundary faces."""
        return torch.stack([self.interp(g[:, 0]), self.interp(g[:, 1])], dim=1)

    def scatter(self, face_val):
        """Cell sum of a face quantity: +owner, -neighbour (interior only)."""
        out = torch.zeros(self.ncell, dtype=face_val.dtype).index_add_(0, self.owner, face_val)
        return out.index_add_(0, self.n_i, -face_val[self.fi])

    def divergence(self, F):
        """`uops.divergence`."""
        return self.scatter(F) / self.vol


class TBC:
    """`upiso.BC` with a differentiable value vector (the actuation input of plan U6)."""

    def __init__(self, tm, bc):
        self.tm = tm
        self.kind = np.asarray(bc.kind)
        self.neu = torch.as_tensor(self.kind == NEUMANN)
        self.value = _t(bc.value)

    def effective(self, phi, value=None):
        v = self.value if value is None else value
        return torch.where(self.neu, phi[self.tm.o_bf], v)


class TGradient:
    """A production gradient object (LSQ `Gradient` or `GGDualGradient`) as four constants."""

    def __init__(self, g):
        self.Gx, self.Gy = SparseConst.from_scipy(g.Gx), SparseConst.from_scipy(g.Gy)
        self.Bx, self.By = SparseConst.from_scipy(g.Bx), SparseConst.from_scipy(g.By)

    def __call__(self, phi, phi_b):
        return torch.stack([self.Gx @ phi + self.Bx @ phi_b, self.Gy @ phi + self.By @ phi_b], dim=1)


# ---------------------------------------------------------------------------------------------
# Matrices with a fixed SUPERSET pattern (plan U0): every interior face owns all four of
# (o,o), (o,n), (n,o), (n,n), and every cell its diagonal, so the pattern is geometric and the
# upwind switch changes values, never positions.

class Pattern:
    def __init__(self, tm):
        m = tm.m
        nc = m.ncell
        i = m.interior
        o, n = m.owner, m.neigh
        r = np.concatenate([o[i], o[i], n[i], n[i], np.arange(nc)])
        c = np.concatenate([o[i], n[i], o[i], n[i], np.arange(nc)])
        S = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(nc, nc)).tocsr()
        S.sum_duplicates(); S.sort_indices()
        coo = S.tocoo()
        order = np.lexsort((coo.col, coo.row))
        self.rows, self.cols = coo.row[order].astype(np.int64), coo.col[order].astype(np.int64)
        self.shape = (nc, nc)
        self.nnz = len(self.rows)
        self.keys = self.rows * nc + self.cols
        self.idx = (self.rows, self.cols)
        self.rows_t, self.cols_t = _ti(self.rows), _ti(self.cols)

        def pos(rr, cc):
            k = np.asarray(rr, dtype=np.int64) * nc + np.asarray(cc, dtype=np.int64)
            p = np.searchsorted(self.keys, k)
            assert np.all(self.keys[p] == k), "entry outside the superset pattern"
            return _ti(p)
        oi, ni = o[i], n[i]
        self.p_oo, self.p_on, self.p_no, self.p_nn = pos(oi, oi), pos(oi, ni), pos(ni, oi), pos(ni, ni)
        ob = o[m.boundary]
        self.p_bb = pos(ob, ob)                                  # boundary face -> owner diagonal
        self.p_diag = pos(np.arange(nc), np.arange(nc))

    def zeros(self):
        return torch.zeros(self.nnz, dtype=DT)

    def diag(self, vals):
        return vals[self.p_diag]

    def rowsum(self, vals):
        return torch.zeros(self.shape[0], dtype=DT).index_add_(0, self.rows_t, vals)

    def matvec(self, vals, x):
        return torch.zeros(self.shape[0], dtype=DT).index_add_(0, self.rows_t, vals * x[self.cols_t])

    def to_scipy(self, vals):
        return sp.csr_matrix((vals.detach().numpy(), self.idx), shape=self.shape)


def laplacian_vals(tm, pat, gam_f, dir_face_mask):
    """Implicit orthogonal part of `uops.laplacian` on the superset pattern. Linear in gam_f.

    `dir_face_mask` (nface,) bool: boundary faces whose kind is DIRICHLET."""
    coef = gam_f * tm.ef_over_d * tm.span
    ci = coef[tm.fi]
    v = pat.zeros()
    v = v.index_add(0, pat.p_on, ci).index_add(0, pat.p_no, ci)
    v = v.index_add(0, pat.p_oo, -ci).index_add(0, pat.p_nn, -ci)
    cb = torch.where(dir_face_mask[tm.fb], coef[tm.fb], torch.zeros_like(coef[tm.fb]))
    return v.index_add(0, pat.p_bb, -cb)


def laplacian_rhs(tm, gam_f, dir_face_mask, grad_phi, bvalues=None):
    """`rhs_fn` of `uops.laplacian`: deferred non-orthogonal flux plus the Dirichlet term."""
    gf = tm.interp_vec(grad_phi)
    cross = gam_f * tm.span * (tm.Tf * gf).sum(dim=1)
    noflux = torch.zeros(tm.nface, dtype=torch.bool)
    noflux[tm.fb] = ~dir_face_mask[tm.fb]                  # Neumann faces carry no flux at all
    cross = torch.where(noflux, torch.zeros_like(cross), cross)
    out = tm.scatter(cross)
    if bvalues is not None:
        coef = gam_f * tm.ef_over_d * tm.span
        cb = torch.where(dir_face_mask[tm.fb], coef[tm.fb] * bvalues[tm.bidx_b], torch.zeros_like(coef[tm.fb]))
        out = out.index_add(0, tm.o_b, cb)
    return out


def convection_vals(tm, pat, F, masks):
    """Implicit upwind part of `uops.convection` on the superset pattern."""
    pos = masks("upwind", lambda: F >= 0.0)
    tie = masks("upwind_tie", lambda: F.abs() <= TIE * F.abs().max())
    if masks.mode == "replay" and masks.straddle:
        # FD probes: pinned to the recorded branch EXCEPT at ties, where the live decision lets the
        # +-h pair straddle the kink, so the central difference measures the same midpoint slope
        # the adjoint returns (the value is continuous there: F * phi = 0 on both sides of F = 0)
        pos = torch.where(tie, F.detach() >= 0.0, pos)
    p = pos.to(DT)
    w = torch.where(tie, torch.full_like(p, 0.5), p)       # derivative weight of the owner branch
    Fi = F[tm.fi]
    pi, wi = p[tm.fi], w[tm.fi]
    Fo = st_mask(Fi, pi, wi)                               # owner-upwind part
    Fn = st_mask(Fi, 1.0 - pi, 1.0 - wi)                   # neighbour-upwind part
    v = pat.zeros()
    v = v.index_add(0, pat.p_oo, Fo).index_add(0, pat.p_no, -Fo)
    v = v.index_add(0, pat.p_on, Fn).index_add(0, pat.p_nn, -Fn)
    Fb = st_mask(F[tm.fb], p[tm.fb], w[tm.fb])             # boundary outflow: owner's own value
    return v.index_add(0, pat.p_bb, Fb), (pos, w)


def convection_rhs(tm, F, pos, phi, phi_b, grad_phi, scheme):
    """`rhs_fn` of `uops.convection`: boundary inflow/outflow terms and deferred central."""
    pos, wgt = pos
    p = pos.to(DT)
    out = torch.zeros(tm.ncell, dtype=DT)
    Fb, pb_, wb = F[tm.fb], p[tm.fb], wgt[tm.fb]
    inflow = st_mask(Fb, 1.0 - pb_, 1.0 - wb) * phi_b[tm.bidx_b]
    out = out.index_add(0, tm.o_b, inflow)
    ext = (grad_phi[tm.o_b] * tm.dx_b).sum(dim=1)
    outflow = st_mask(Fb, pb_, wb) * ext
    out = out.index_add(0, tm.o_b, outflow)
    if scheme == "upwind":
        return out
    Fi, pi, wi = F[tm.fi], p[tm.fi], wgt[tm.fi]
    po, pn = phi[tm.o_i], phi[tm.n_i]
    w = tm.w_i
    ce = w * po + (1.0 - w) * pn
    gf = w[:, None] * grad_phi[tm.o_i] + (1.0 - w)[:, None] * grad_phi[tm.n_i]
    ce = ce + (gf * tm.skew_i).sum(dim=1)
    corr = Fi * ce - (st_mask(Fi, pi, wi) * po + st_mask(Fi, 1.0 - pi, 1.0 - wi) * pn)
    return out.index_add(0, tm.o_i, corr).index_add(0, tm.n_i, -corr)
