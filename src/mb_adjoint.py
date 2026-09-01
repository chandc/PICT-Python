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
