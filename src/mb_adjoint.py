"""One DIFFERENTIABLE PISO step on a multi-block domain.

The multi-block twin of `adjoint_piso.MiniPISO`, and deliberately as small: it exists to make
gradients through a SEAM checkable, not to reproduce production physics. Rhie-Chow, BDF2, the
persistent flux and the corrector loop are Stage 7's subject; this is Stage 6's.

WHY THIS NEEDED ALMOST NO NEW ADJOINT MATHEMATICS. `MultiBlockPISO` assembles ONE global sparse
matrix over all blocks -- a connection contributes off-diagonal entries exactly as an interior
face does. `LinearSolve` does not care where a row came from, so the block coupling is already
inside $A$ and $A^{\\mathsf T}\\lambda = \\bar g$ carries sensitivity across a seam by the same
entries that carry flux across it forwards. No per-block gradient exchange, no halo adjoint.
That is the single largest risk in this kind of work, and the design retired it years before
anyone asked for a gradient.

WHAT IS FROZEN, AND WHY IT IS HONEST HERE. $A$ is assembled from the current velocity and passed
in DETACHED, so this measures $\\partial L/\\partial S$ with the matrix held fixed. That is the
same frozen-coefficient approximation Stage 4 measured on one block, and for Stage 6 it is not
an approximation at all: the gate compares a split domain against an unsplit one under
*identical* treatment, so anything frozen is frozen the same way on both sides. `exact_A` is
Stage 7's problem.
"""
import numpy as np
import torch
from scipy import sparse

from src.adjoint_piso import LinearSolve, csr_pattern


class MultiBlockMiniPISO:
    """One differentiable step over a `Domain`, with an additive source in the momentum RHS."""

    def __init__(self, domain, nu=0.05, dt=0.05):
        self.d = domain
        self.nu, self.dt = nu, dt
        nb = len(domain.blocks)
        self.Js = [domain.block_metrics_cached(b)[0] for b in range(nb)]
        self.ms = [domain.block_metrics_cached(b)[1] for b in range(nb)]
        self.N = domain.n_cells
        self.u0 = {b: self._field(blk) for b, blk in enumerate(domain.blocks)}

    @staticmethod
    def _field(blk):
        return (np.sin(2 * np.pi * blk.x) * np.cos(2 * np.pi * blk.y)
                * np.cos(2 * np.pi * blk.z))

    def flat(self, per_block):
        """Per-block arrays -> one global vector in the domain's own index order."""
        out = np.zeros(self.N)
        for b in range(len(self.d.blocks)):
            out[self.d.global_ids(b).ravel()] = np.asarray(per_block[b]).ravel()
        return out

    def block_slice(self, b):
        """Global indices belonging to block b, as a flat array."""
        return self.d.global_ids(b).ravel()

    def matrices(self):
        u = self.u0
        A = self.d.build_momentum_matrix(self.Js, self.ms, u, u, u, self.nu, self.dt, bdf2=False)
        M = self.d.build_diffusion_matrix(self.Js, self.ms)
        return sparse.csr_matrix(A), sparse.csr_matrix(M)

    def step(self, S, loss_idx=None):
        """S: torch tensor over all N cells (the source). Returns a scalar loss.

        `loss_idx` restricts the loss to a subset of global indices — used by 6.6 to put the
        loss support in one block and the parameter in the other, which is the only check here
        that cannot pass unless sensitivity genuinely crosses the seam.
        """
        A, M = self.matrices()
        Aidx, Ashape, Aval = csr_pattern(A)
        Midx, Mshape, Mval = csr_pattern(M)

        b_rhs = torch.as_tensor(self.flat({b: self.Js[b] * self.u0[b] / self.dt
                                           for b in range(len(self.d.blocks))})) + S
        # momentum: NON-symmetric, so the backward transposes
        u_star = LinearSolve.apply(Aval, b_rhs, (Aidx, Ashape), False, False)
        # pressure: symmetric and SINGULAR on a fully periodic box, so both passes project
        phi = LinearSolve.apply(Mval, u_star, (Midx, Mshape), True, True)

        if loss_idx is None:
            return (phi ** 2).sum() + (u_star ** 2).sum()
        sel = torch.as_tensor(np.asarray(loss_idx))
        return (phi[sel] ** 2).sum() + (u_star[sel] ** 2).sum()


def alignment(d_split, block_shape):
    """Permutation taking a split domain's global vector into the unsplit block's layout.

    The two domains hold the same nodes in different orders: unsplit is one flattened
    (nx, ny, nz), split is block 0's cells then block 1's. `x_split[alignment(...)]` lines up
    with `x_unsplit`, which is what makes 6.4 a comparison rather than a coincidence.
    """
    return np.concatenate([d_split.global_ids(b) for b in range(len(d_split.blocks))],
                          axis=0).ravel()


def periodic_box(ntot=12, n_split=1):
    """A fully periodic cube split into `n_split` blocks along x.

    Periodic in all three directions, so there are no walls and no boundary conditions to get
    wrong: the only thing that differs between n_split = 1 and n_split = 2 is the seam. Nodes
    PARTITION without duplicating the interface, as a periodic axis must.

    Lives here rather than in a test file because two test files need it, and importing one
    test from another runs it.
    """
    from src.multiblock import Block, Connection, Domain, face_id
    assert ntot % n_split == 0
    nxb = ntot // n_split
    ax = np.arange(ntot) / ntot
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    blocks = []
    for b in range(n_split):
        sl = slice(b * nxb, (b + 1) * nxb)
        blk = Block((nxb, ntot, ntot), X[sl], Y[sl], Z[sl], (1.0 / ntot,) * 3)
        for a in range(3):
            blk.faces[face_id(a, 0)] = blk.faces[face_id(a, 1)] = "periodic"
        blocks.append(blk)
    conns = []
    if n_split > 1:
        for b in range(n_split):
            nb = (b + 1) % n_split
            sh = (1.0, 0.0, 0.0) if nb == 0 else (0.0, 0.0, 0.0)
            conns.append(Connection(b, face_id(0, 1), nb, face_id(0, 0), shift=sh))
    return Domain(blocks, conns)


# --------------------------------------------------------------------------- face operators
# Stage 7 needs a genuine multi-step chain, which needs a gradient and a divergence that cross
# seams. Rather than write new stencils and hope they agree with the operator the solver
# actually inverts, these are built from the SAME face enumeration `build_diffusion_matrix`
# uses, and `check_consistency` below proves the identity
#
#     sum_a  F_a^T diag(w_a) F_a  ==  build_diffusion_matrix
#
# exactly. That turns "my gradient looks right" into "my gradient composes into the operator
# the pressure solve uses", which is the only property the projection actually needs.

def face_pairs(d, axis):
    """(lo global ids, hi global ids, h) for every face along `axis`.

    Enumerates exactly what the diffusion assembly enumerates: interior faces of every block,
    one wrap face per periodic axis, and one face per connection -- from the A side only, since
    adding it from both would double-count the coupling.
    """
    from src.multiblock import face_axis_side, face_id, face_slice
    lo, hi = [], []
    h = None
    for b, blk in enumerate(d.blocks):
        gid = d.global_ids(b)
        h = blk.h[axis]
        lo_s = [slice(None)] * 3; lo_s[axis] = slice(0, -1)
        hi_s = [slice(None)] * 3; hi_s[axis] = slice(1, None)
        lo.append(gid[tuple(lo_s)].ravel())
        hi.append(gid[tuple(hi_s)].ravel())
        if blk.faces[face_id(axis, 1)] == "periodic":
            f0 = [slice(None)] * 3; f0[axis] = 0
            fn = [slice(None)] * 3; fn[axis] = -1
            lo.append(gid[tuple(fn)].ravel())
            hi.append(gid[tuple(f0)].ravel())
    for c in d.connections:
        ax, _ = face_axis_side(c.fa)
        if ax != axis:
            continue
        ga, gb = d.pair_indices(c)
        lo.append(np.asarray(ga).ravel())
        hi.append(np.asarray(gb).ravel())
    return np.concatenate(lo), np.concatenate(hi), h


def face_difference_matrix(d, axis):
    """F: (n_faces, N) with -1/h on the low cell and +1/h on the high cell of each face."""
    lo, hi, h = face_pairs(d, axis)
    nf, N = len(lo), d.n_cells
    r = np.arange(nf)
    rows = np.concatenate([r, r])
    cols = np.concatenate([lo, hi])
    vals = np.concatenate([-np.ones(nf) / h, np.ones(nf) / h])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(nf, N))


def face_weights(d, Js, ms, axis, coefs=None):
    """The face coefficient the diffusion operator uses: 0.5*(Jg_lo + Jg_hi)."""
    key = ("xi", "eta", "zeta")[axis]
    per_block = []
    for b in range(len(d.blocks)):
        m = ms[b]
        g = m[f"{key}_x"] ** 2 + m[f"{key}_y"] ** 2 + m[f"{key}_z"] ** 2
        c = 1.0 if coefs is None else coefs[b]
        per_block.append(c * Js[b] * g)
    flat = np.zeros(d.n_cells)
    for b in range(len(d.blocks)):
        flat[d.global_ids(b).ravel()] = per_block[b].ravel()
    lo, hi, _ = face_pairs(d, axis)
    return 0.5 * (flat[lo] + flat[hi])


def check_consistency(d, Js, ms):
    """max |sum_a F_a^T W_a F_a - build_diffusion_matrix|, which must be ~0."""
    M = sparse.csr_matrix(d.build_diffusion_matrix(Js, ms))
    acc = sparse.csr_matrix(M.shape)
    for axis in range(3):
        F = face_difference_matrix(d, axis)
        W = sparse.diags(face_weights(d, Js, ms, axis))
        acc = acc + F.T @ W @ F
    diff = (acc - M).tocoo()
    return float(np.abs(diff.data).max()) if diff.nnz else 0.0


def cell_gradient_matrix(d, axis):
    """Central difference at cells: G = P F, with P averaging the two faces either side.

    Every cell of a fully periodic domain has exactly one face below and one above along each
    axis; `face_pairs` supplies both, seams included, so this crosses a connection with no
    special case.
    """
    lo, hi, _ = face_pairs(d, axis)
    nf, N = len(lo), d.n_cells
    r = np.arange(nf)
    P = sparse.csr_matrix((np.full(2 * nf, 0.5),
                           (np.concatenate([lo, hi]), np.concatenate([r, r]))), shape=(N, nf))
    return (P @ face_difference_matrix(d, axis)).tocsr()


def to_torch_sparse(mat):
    """scipy sparse -> a coalesced torch sparse COO tensor.

    `torch.sparse.mm(S, x)` is differentiable in the DENSE argument, which is all the chain
    needs: the operators are geometry and stay constant, only the fields carry gradients. The
    backward is S^T g, applied by torch, so nothing here hand-writes a transpose.
    """
    coo = mat.tocoo()
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64)
    val = torch.as_tensor(coo.data, dtype=torch.float64)
    return torch.sparse_coo_tensor(idx, val, coo.shape).coalesce()


def spmv(S, x):
    """S @ x for a sparse S and a 1-D dense x, differentiable in x."""
    return torch.sparse.mm(S, x.reshape(-1, 1)).reshape(-1)


class MultiBlockChain(MultiBlockMiniPISO):
    """A MULTI-STEP differentiable chain over the same domain, carrying state between steps.

    Stage 7's subject is the state PISO keeps across a step. This carries the BDF2 history and
    runs the real assembled operators -- the momentum matrix, the pressure operator, and the
    cell gradient proved consistent with it by `check_consistency` -- in a fixed sequence:

        rhs   = J (2 u^n - 0.5 u^{n-1}) / dt + S_k
        u*    = A^{-1} rhs
        phi   = M^{-1} (G_x u*)                 (symmetric, singular, projected both ways)
        u^{n+1} = u* - dt * G_x phi

    IT IS NOT THE PISO STEP, and the docstring says so rather than letting the class name
    imply it: there is one scalar field where the solver has three components, and the
    Rhie-Chow persistent flux is absent because `face_fluxes` and `pressure_face_fluxes` are
    not yet in torch. What IS genuine is every operator in the chain, the seam they cross, and
    the state carried from one step to the next -- which is exactly what 7.1, 7.4 and 7.5 test.
    F_prev and p_flux wait on the flux port; see the plan.
    """

    def __init__(self, domain, nu=0.05, dt=0.05):
        super().__init__(domain, nu, dt)
        self.G = cell_gradient_matrix(domain, 0)
        # BDF2 CONSISTENTLY, or the chain is an unstable recurrence. The RHS below carries
        # (2 u^n - 0.5 u^{n-1}), so the matrix must carry 1.5 J/dt on the diagonal -- that is
        # what `bdf2=True` does. Assembled with `bdf2=False` the diagonal is J/dt and the
        # recurrence becomes u^{n+1} = 2u^n - 0.5u^{n-1}, whose roots are 1.71 and 0.29: it
        # grows 1.71x per step forward, so the ADJOINT grows backwards and 7.5 measured a
        # blow-up that was mine, not the method's.
        A = sparse.csr_matrix(domain.build_momentum_matrix(
            self.Js, self.ms, self.u0, self.u0, self.u0, nu, dt, bdf2=True))
        M = self.matrices()[1]
        self.A_pat = csr_pattern(A)[:2], csr_pattern(A)[2]
        self.M_pat = csr_pattern(M)[:2], csr_pattern(M)[2]
        self.J_flat = self.flat({b: self.Js[b] for b in range(len(domain.blocks))})
        self.u_init = self.flat(self.u0)

    def rollout(self, sources, drop_history=False, final_only=False):
        """sources: list of per-step torch tensors. Returns the scalar loss.

        `drop_history` detaches u^{n-1} in the RHS -- the backward then ignores the BDF2 level,
        which is the mangle 7.4 requires to be DETECTED.

        `final_only` puts the loss on the LAST state instead of summing over the trajectory,
        which is what makes an adjoint-norm profile mean anything. With a summed loss, dL/dS_0
        collects a term from every subsequent step and dL/dS_last collects one, so the profile
        falls by a factor of order the horizon whatever the propagator does -- 100x over 20
        steps here, which reads as amplification and is not. With the loss on the final state
        only, dL/dS_k IS the adjoint propagator from step k, and its growth backwards in time is
        the quantity 7.5 is trying to bound.
        """
        (Aidx, Ashape), Aval = self.A_pat
        (Midx, Mshape), Mval = self.M_pat
        Gt = to_torch_sparse(self.G)
        u = torch.as_tensor(self.u_init)
        u_prev = torch.as_tensor(self.u_init)
        Jt = torch.as_tensor(self.J_flat)
        L = 0.0
        for S in sources:
            hist = u_prev.detach() if drop_history else u_prev
            rhs = Jt * (2.0 * u - 0.5 * hist) / self.dt + S
            u_star = LinearSolve.apply(Aval, rhs, (Aidx, Ashape), False, False)
            # THE PRESSURE RHS IS A DIVERGENCE, not the velocity. Solving M phi = u* applies a
            # Laplacian INVERSE to the velocity itself, which amplifies smooth modes by 1/k^2
            # and made the chain grow 1.15x per step -- an instability of the surrogate, not of
            # the adjoint. G_x u* is the divergence of a one-component velocity, and it puts
            # the RHS in the range of M by construction.
            phi = LinearSolve.apply(Mval, spmv(Gt, u_star), (Midx, Mshape), True, True)
            u_prev, u = u, u_star - self.dt * spmv(Gt, phi)
            if not final_only:
                L = L + (u ** 2).sum()
        return (u ** 2).sum() if final_only else L


def flux_divergence_matrix(d, Js, ms):
    """Sparse (N, 3N) taking (u, v, w) to div F, matching `divergence(face_fluxes(...))`.

    WHY THIS TARGET AND NOT `face_fluxes` ITSELF. The raw face array is indexed per block with
    one extra entry along the axis, so both domain-boundary faces of a block appear even when
    they are seam faces that another block also owns. Assembling that is fiddly and pointless:
    nothing consumes the face array directly, `divergence` does, and the composition is
    cell-to-cell with an index space that already exists.

    AND THE COMPOSITION COLLAPSES. On a domain whose faces are all interior or seam, every face
    flux is the average of the two cells it separates,

        F_face = 0.5 (JU_lo + JU_hi),

    so the divergence telescopes to a central difference of the contravariant component:

        div F |_c = sum_a  0.5 (JU_{c+} - JU_{c-}) / h / J_c

    which is exactly `cell_gradient_matrix` applied to JU_a. No new stencil, no new seam logic --
    the operator that was proved consistent in 7.0 is reused, and JU is a diagonal scaling of
    (u, v, w) by J times the metrics. The whole thing is three diagonal matrices and a gradient.

    This holds for periodic and connected faces. A wall or inflow face takes the boundary cell's
    own component instead of an average, which is a different row; that case is not assembled
    here and `verify_flux_divergence` will show it as a mismatch rather than a silent error.
    """
    N = d.n_cells
    flat = lambda per_block: _flatten(d, per_block)
    Jf = flat({b: Js[b] for b in range(len(d.blocks))})
    blocks = []
    for comp, letter in enumerate("xyz"):
        col = sparse.csr_matrix((N, N))
        for axis, key in enumerate(("xi", "eta", "zeta")):
            met = flat({b: ms[b][f"{key}_{letter}"] for b in range(len(d.blocks))})
            col = col + cell_gradient_matrix(d, axis) @ sparse.diags(Jf * met)
        blocks.append(sparse.diags(1.0 / Jf) @ col)
    return sparse.hstack(blocks).tocsr()


def _flatten(d, per_block):
    out = np.zeros(d.n_cells)
    for b in range(len(d.blocks)):
        out[d.global_ids(b).ravel()] = np.asarray(per_block[b]).ravel()
    return out


def verify_flux_divergence(d, Js, ms, rng=None):
    """max |D_flux (u,v,w) - divergence(face_fluxes(u,v,w))| over a random field."""
    rng = rng or np.random.default_rng(0)
    fields = []
    for _ in range(3):
        fields.append({b: rng.standard_normal(blk.shape) for b, blk in enumerate(d.blocks)})
    us, vs, ws = fields
    ref = _flatten(d, {b: d.divergence(b, d.face_fluxes(b, us, vs, ws), Js[b])
                       for b in range(len(d.blocks))})
    D = flux_divergence_matrix(d, Js, ms)
    got = D @ np.concatenate([_flatten(d, us), _flatten(d, vs), _flatten(d, ws)])
    return float(np.abs(got - ref).max()), float(np.abs(ref).max())


def face_select_matrices(d, axis):
    """(Sel_lo, Sel_hi): (n_faces, N) picking the low and high cell of each face."""
    lo, hi, _ = face_pairs(d, axis)
    nf, N = len(lo), d.n_cells
    r = np.arange(nf)
    one = np.ones(nf)
    return (sparse.csr_matrix((one, (r, lo)), shape=(nf, N)),
            sparse.csr_matrix((one, (r, hi)), shape=(nf, N)))


def rc_flux_divergence_matrix(d, Js, ms, coefs):
    """Sparse (N, N) for div(pressure_face_fluxes(p, rhie_chow=True)), coefficients frozen.

    THE WIDTH-2 STENCIL NEEDS NO WIDTH-2 MACHINERY. `implementation_plan.md` 5.1 flags the
    Rhie-Chow wide gradient as reaching two cells deep across a seam, so "the adjoint scatter at
    a connection is wider than the existing width-1 machinery assumes". True of the stencil, and
    it does not follow that new enumeration is needed: the term is

        0.5 (Jg_lo dpw_lo + Jg_hi dpw_hi),      dpw = central difference at a CELL

    which is a face AVERAGE (width 1) of a cell GRADIENT (width 1). Composing two width-1
    operators gives the width-2 stencil, and each factor already resolves its own seam, so the
    product does too. That is what `verify_rc_divergence` checks against the real function.

    The compact half is the same face difference the diffusion operator uses, weighted by the
    same face coefficient -- reused from 7.0 rather than rewritten.
    """
    N = d.n_cells
    Jf = _flatten(d, {b: Js[b] for b in range(len(d.blocks))})
    acc = sparse.csr_matrix((N, N))
    for axis, key in enumerate(("xi", "eta", "zeta")):
        F = face_difference_matrix(d, axis)
        G = cell_gradient_matrix(d, axis)
        Slo, Shi = face_select_matrices(d, axis)
        w = face_weights(d, Js, ms, axis, coefs=coefs)        # 0.5 (Jg_lo + Jg_hi)
        jg = _flatten(d, {b: coefs[b] * Js[b]
                          * sum(ms[b][f"{key}_{c}"] ** 2 for c in "xyz")
                          for b in range(len(d.blocks))})
        rc = sparse.diags(w) @ F - 0.5 * (Slo + Shi) @ sparse.diags(jg) @ G
        acc = acc - F.T @ rc                                  # div = -(1/J) F^T (flux)
    return (sparse.diags(1.0 / Jf) @ acc).tocsr()


def verify_rc_divergence(d, Js, ms, coefs, rng=None):
    """max |RC matrix @ p - divergence(pressure_face_fluxes(p, rhie_chow=True))|."""
    rng = rng or np.random.default_rng(0)
    ps = {b: rng.standard_normal(blk.shape) for b, blk in enumerate(d.blocks)}
    ref = _flatten(d, {b: d.divergence(
        b, d.pressure_face_fluxes(b, ps, coefs[b], coefs, include_cross=False, rhie_chow=True),
        Js[b]) for b in range(len(d.blocks))})
    got = rc_flux_divergence_matrix(d, Js, ms, coefs) @ _flatten(d, ps)
    return float(np.abs(got - ref).max()), float(np.abs(ref).max())


class MultiBlockChainRC(MultiBlockChain):
    """The chain with the RHIE-CHOW persistent pressure, `p_flux`, carried across steps.

    This is the structure `piso_multiblock.step` actually runs, in divergence form:

        u*        = A^{-1} [ J (2 u^n - 0.5 u^{n-1}) / dt + S ]
        div F     = D_flux(u*)  -  RC(p_flux)        <- p_flux enters HERE, from the last step
        phi       = M^{-1} div F
        u^{n+1}   = u* - dt G phi
        p_flux    = p_flux + phi                     <- and is carried out to the next

    `p_flux` is the genuine cross-step state of the production configuration: it is read at
    `piso_multiblock.py:338` whenever `persistent_flux=True`, which is every production case.
    `self.F_prev` is NOT -- it is read only under `if self.ddt_corr`, which is off everywhere,
    so it carries no gradient there at all. That is why 7.3 is the live mangle test and 7.2
    had to be re-scoped.

    Coefficients are frozen, so D_flux and RC are assembled once. Both were verified against
    the real `face_fluxes` and `pressure_face_fluxes` to 4e-16 (7.6, 7.7).
    """

    def __init__(self, domain, nu=0.05, dt=0.05):
        super().__init__(domain, nu, dt)
        A = sparse.csr_matrix(domain.build_momentum_matrix(
            self.Js, self.ms, self.u0, self.u0, self.u0, nu, dt, bdf2=True))
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        Jf = self.J_flat
        gamma_flat = Jf / rowsum
        gam = {}
        for b in range(len(domain.blocks)):
            gam[b] = gamma_flat[domain.global_ids(b).ravel()].reshape(domain.blocks[b].shape)
        self.gamma = gam
        self.D_flux = flux_divergence_matrix(domain, self.Js, self.ms)
        self.RC = rc_flux_divergence_matrix(domain, self.Js, self.ms, gam)
        self.N3 = 3 * self.N

    def rollout(self, sources, drop_history=False, final_only=False, drop_pflux=False):
        """`drop_pflux` detaches the carried pressure in the RC term -- the mangle for 7.3."""
        (Aidx, Ashape), Aval = self.A_pat
        (Midx, Mshape), Mval = self.M_pat
        Gt = to_torch_sparse(self.G)
        Du = to_torch_sparse(self.D_flux[:, :self.N])
        RCt = to_torch_sparse(self.RC)
        u = torch.as_tensor(self.u_init)
        u_prev = torch.as_tensor(self.u_init)
        p_flux = torch.zeros(self.N, dtype=torch.float64)
        Jt = torch.as_tensor(self.J_flat)
        L = 0.0
        for S in sources:
            hist = u_prev.detach() if drop_history else u_prev
            rhs = Jt * (2.0 * u - 0.5 * hist) / self.dt + S
            u_star = LinearSolve.apply(Aval, rhs, (Aidx, Ashape), False, False)
            pf = p_flux.detach() if drop_pflux else p_flux
            divF = spmv(Du, u_star) - spmv(RCt, pf)
            phi = LinearSolve.apply(Mval, divF, (Midx, Mshape), True, True)
            u_prev, u = u, u_star - self.dt * spmv(Gt, phi)
            p_flux = p_flux + phi
            if not final_only:
                L = L + (u ** 2).sum()
        return (u ** 2).sum() if final_only else L
