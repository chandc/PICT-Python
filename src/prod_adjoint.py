"""Stage 9.1: differentiable momentum assembly for the PRODUCTION solver.

A(u,v,w) from `build_momentum_matrix` (central convection, bdf2) is AFFINE
in the convecting velocity: vals(x) = vals0 + T x with x = (u|v|w) flat and
T a constant sparse (nnz x 3N) sensitivity, built ONCE per mesh by colored
probing of the production assembler. See reference/production_adjoint.md
for the coloring/attribution argument; test_prod_adjoint.py measures the
linearity rather than assuming it (gate 9.1a).
"""
import hashlib
import os

import numpy as np
import scipy.sparse as sparse
import torch


def _canonical(A):
    """COO in (row, col) lexicographic order: the fixed value layout."""
    A = A.tocoo()
    order = np.lexsort((A.col, A.row))
    return (A.row[order], A.col[order]), A.shape, A.data[order]


class MomentumAssembly:
    """Probe-built affine map x -> A-values for one (domain, nu, dt)."""

    def __init__(self, domain, nu, dt, cache_dir="results/cache", verbose=False):
        self.d, self.nu, self.dt = domain, nu, dt
        self.N = domain.n_cells
        self.Js = [domain.block_metrics_cached(b)[0] for b in range(len(domain.blocks))]
        self.ms = [domain.block_metrics_cached(b)[1] for b in range(len(domain.blocks))]
        self.gids = [domain.global_ids(b).ravel() for b in range(len(domain.blocks))]
        self.shapes = [domain.blocks[b].shape for b in range(len(domain.blocks))]

        (rows, cols), shape, vals0 = _canonical(self._build(np.zeros(3 * self.N)))
        self.idx, self.shape, self.vals0 = (rows, cols), shape, vals0
        self._key = {tuple(k): i for i, k in enumerate(zip(rows, cols))}

        path = os.path.join(cache_dir, f"momT_{self._fingerprint()}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.T = sparse.csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=(len(vals0), 3 * self.N))
            self.ncolors = int(z["ncolors"])
        else:
            self.T, self.ncolors = self._build_T(verbose)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, data=self.T.data, indices=self.T.indices,
                                indptr=self.T.indptr, ncolors=self.ncolors)
        self.cache_path = path

    # ------------------------------------------------------------ plumbing
    def _fingerprint(self):
        h = hashlib.sha256()
        for b in range(len(self.d.blocks)):
            blk = self.d.blocks[b]
            h.update(blk.x.ravel()[::13].tobytes())
            h.update(blk.y.ravel()[::13].tobytes())
        h.update(np.array([self.nu, self.dt, self.N]).tobytes())
        return h.hexdigest()[:16]

    def _fields(self, x):
        out = []
        for c in range(3):
            comp = x[c * self.N:(c + 1) * self.N]
            out.append({b: comp[self.gids[b]].reshape(self.shapes[b])
                        for b in range(len(self.d.blocks))})
        return out

    def _build(self, x):
        us, vs, ws = self._fields(x)
        return sparse.csr_matrix(self.d.build_momentum_matrix(
            self.Js, self.ms, us, vs, ws, self.nu, self.dt, bdf2=True))

    def values(self, x):
        """Production assembly at x, aligned to the canonical layout."""
        (rows, cols), _, vals = _canonical(self._build(x))
        if not (np.array_equal(rows, self.idx[0]) and np.array_equal(cols, self.idx[1])):
            raise AssertionError("assembly pattern changed with the velocity field")
        return vals

    # ------------------------------------------------------------- T build
    def _build_T(self, verbose):
        N = self.N
        rows, cols = self.idx
        S = sparse.csr_matrix(
            (np.ones(len(rows), bool), (rows, cols)), shape=self.shape)
        S = ((S + S.T + sparse.identity(N, dtype=bool)) > 0).astype(bool)
        S2 = ((S @ S) > 0).astype(bool)
        S4 = ((S2 @ S2) > 0).astype(bool)
        S5 = ((S4 @ S) > 0).astype(bool)          # conflict radius 5 (see doc)

        colors = -np.ones(N, dtype=int)
        indptr, indices = S5.indptr, S5.indices
        for c in range(N):
            nb = colors[indices[indptr[c]:indptr[c + 1]]]
            used = set(nb[nb >= 0].tolist())
            k = 0
            while k in used:
                k += 1
            colors[c] = k
        ncolors = colors.max() + 1
        if verbose:
            print(f"  T build: {ncolors} colors x 3 components "
                  f"= {3 * ncolors} assemblies", flush=True)

        S2c = S2.tocsc()
        Trows, Tcols, Tvals = [], [], []
        for comp in range(3):
            for k in range(ncolors):
                batch = np.where(colors == k)[0]
                x = np.zeros(3 * N)
                x[comp * N + batch] = 1.0
                dv = self.values(x) - self.vals0
                nz = np.nonzero(dv)[0]
                if not len(nz):
                    continue
                # unique in-batch cell within dist<=2 of each node
                sub = S2c[:, batch].tocoo()
                nearest = -np.ones(N, dtype=int)
                nearest[sub.row] = batch[sub.col]
                cell = nearest[rows[nz]]
                miss = cell < 0
                cell[miss] = nearest[cols[nz[miss]]]
                if np.any(cell < 0):
                    raise AssertionError("probe attribution failed: entry changed "
                                         "with no in-batch cell within dist 2")
                Trows.extend(nz.tolist())
                Tcols.extend((comp * N + cell).tolist())
                Tvals.extend(dv[nz].tolist())
            if verbose:
                print(f"  T build: component {comp} done", flush=True)
        T = sparse.csr_matrix((Tvals, (Trows, Tcols)),
                              shape=(len(self.vals0), 3 * N))
        return T, ncolors


class TorchMomentumAssembly:
    """The torch side: differentiable A-values from a flat (u|v|w) tensor."""

    def __init__(self, asm: MomentumAssembly):
        self.vals0 = torch.as_tensor(asm.vals0)
        T = asm.T.tocoo()
        self.T = torch.sparse_coo_tensor(
            np.vstack([T.row, T.col]), T.data, T.shape).coalesce()
        self.idx, self.shape = asm.idx, asm.shape

    def vals(self, x):
        return self.vals0 + torch.sparse.mm(self.T, x.reshape(-1, 1)).reshape(-1)


class PadMap:
    """Stage 9.2a: `pad_field` probed into an exact affine gather.

    pad(x) = c0 + x[src] * sel, with src/c0/sel found by probing the
    production recursion with a zero field and a global-id sentinel. The
    construction ASSERTS the affine-selection structure (every ghost is a
    verbatim copy of one cell, or a constant) and cross-checks two random
    fields at 1e-13 -- if pad_field ever averages or scales, this fails
    loudly rather than porting the wrong thing.
    """

    def __init__(self, d, b, width=1):
        nb, N = len(d.blocks), d.n_cells
        zero = {bb: np.zeros(d.blocks[bb].shape) for bb in range(nb)}
        c0, lo, hi = d.pad_field(b, zero, width)
        sent = {bb: d.global_ids(bb).astype(float) + 1.0 for bb in range(nb)}
        s, _, _ = d.pad_field(b, sent, width)
        rel = s - c0
        src = np.rint(rel).astype(np.int64) - 1
        sel = src >= 0
        if not np.allclose(rel[~sel], 0.0):
            raise AssertionError("pad_field ghost is neither a copy nor a constant")
        if not np.allclose(rel[sel], np.rint(rel[sel])):
            raise AssertionError("pad_field ghost is a combination, not a selection")
        rng = np.random.default_rng(1234 + b)
        for _ in range(2):
            x = rng.standard_normal(N)
            px, _, _ = d.pad_field(b, {bb: x[d.global_ids(bb)] for bb in range(nb)},
                                   width)
            pred = c0 + np.where(sel, x[np.clip(src, 0, None)], 0.0)
            if np.abs(px - pred).max() > 1e-13:
                raise AssertionError("probed pad map disagrees with pad_field")
        self.lo, self.hi = lo, hi
        self.c0 = torch.as_tensor(c0)
        self.src = torch.as_tensor(np.clip(src, 0, None).reshape(-1))
        self.sel = torch.as_tensor(sel.astype(float))
        self.pshape = c0.shape

    def apply(self, x):
        """x: global flat torch tensor (N,) -> padded block array."""
        return self.c0 + torch.take(x, self.src).reshape(self.pshape) * self.sel


def _sl(axis, s):
    out = [slice(None)] * 3
    out[axis] = s
    return tuple(out)


class TorchFluxKernels:
    """face_fluxes and divergence in torch, axis-aligned-seams branch only
    (the branch the butterfly takes); gated against the numpy originals in
    test_prod_adjoint.py 9.2. Fields travel as GLOBAL flat tensors."""

    def __init__(self, d):
        if not d._axis_aligned_seams():
            raise NotImplementedError("only the axis-aligned seam branch is ported")
        self.d = d
        nb, self.N = len(d.blocks), d.n_cells
        self.nb = nb
        self.pads = [PadMap(d, b, 1) for b in range(nb)]
        self.gid = [torch.as_tensor(d.global_ids(b).reshape(-1)) for b in range(nb)]
        self.shapes = [d.blocks[b].shape for b in range(nb)]
        self.h = [d.blocks[b].h for b in range(nb)]
        self.J = [torch.as_tensor(d.block_metrics_cached(b)[0]) for b in range(nb)]
        from src.phase5_fluxes import _KEYS
        self.mk = [{k: torch.as_tensor(d.block_metrics_cached(b)[1][k])
                    for keys in _KEYS for k in keys} for b in range(nb)]
        self._keys = _KEYS

    def _block(self, x, b):
        return torch.take(x, self.gid[b]).reshape(self.shapes[b])

    def contravariant_global(self, xu, xv, xw):
        """Per-axis GLOBAL vectors of the J-weighted contravariant components."""
        out = []
        for axis, keys in enumerate(self._keys):
            g = torch.zeros(self.N, dtype=torch.float64)
            for b in range(self.nb):
                u, v, w = (self._block(f, b) for f in (xu, xv, xw))
                comp = self.J[b] * (self.mk[b][keys[0]] * u + self.mk[b][keys[1]] * v
                                    + self.mk[b][keys[2]] * w)
                g = g.index_put((self.gid[b],), comp.reshape(-1))
            out.append(g)
        return out

    def face_fluxes_all(self, xu, xv, xw):
        """[per-block [per-axis face-flux tensor]] matching d.face_fluxes."""
        JUg = self.contravariant_global(xu, xv, xw)
        res = []
        for b in range(self.nb):
            pad = self.pads[b]
            padded = [pad.apply(JUg[a]) for a in range(3)]
            lo, hi = pad.lo, pad.hi
            core = [slice(lo[a], lo[a] + self.shapes[b][a]) for a in range(3)]
            F = []
            for axis in range(3):
                JU = padded[axis]
                n = self.shapes[b][axis]
                cs = list(core)
                k0 = 0 if lo[axis] > 0 else 1
                k1 = n if hi[axis] > 0 else n - 1
                cnt = k1 - k0 + 1
                cs[axis] = slice(lo[axis] + k0 - 1, lo[axis] + k0 - 1 + cnt)
                a_lo = JU[tuple(cs)]
                cs[axis] = slice(lo[axis] + k0, lo[axis] + k0 + cnt)
                a_hi = JU[tuple(cs)]
                pieces = [0.5 * (a_lo + a_hi)]
                if lo[axis] == 0:
                    cs[axis] = slice(0, 1)
                    pieces.insert(0, JU[tuple(cs)])
                if hi[axis] == 0:
                    cs[axis] = slice(lo[axis] + n - 1, lo[axis] + n)
                    pieces.append(JU[tuple(cs)])
                F.append(torch.cat(pieces, dim=axis))
            res.append(F)
        return res

    def divergence(self, b, F):
        out = torch.zeros(self.shapes[b], dtype=torch.float64)
        for axis in range(3):
            out = out + (F[axis][_sl(axis, slice(1, None))]
                         - F[axis][_sl(axis, slice(0, -1))]) / self.h[b][axis]
        return out / self.J[b]


def _tgrad(t, h, axis):
    """np.gradient(edge_order=2) in torch: central interior, one-sided edges."""
    n = t.shape[axis]
    mid = (t[_sl(axis, slice(2, None))] - t[_sl(axis, slice(0, -2))]) / (2 * h)
    lo = (-3.0 * t[_sl(axis, slice(0, 1))] + 4.0 * t[_sl(axis, slice(1, 2))]
          - t[_sl(axis, slice(2, 3))]) / (2 * h)
    hi = (3.0 * t[_sl(axis, slice(n - 1, n))] - 4.0 * t[_sl(axis, slice(n - 2, n - 1))]
          + t[_sl(axis, slice(n - 3, n - 2))]) / (2 * h)
    return torch.cat([lo, mid, hi], dim=axis)


class TorchPressureFlux:
    """Stage 9.2b: `pressure_face_fluxes` in torch -- the one BILINEAR kernel
    (p and the velocity-dependent coefficient both vary). Mirrors the
    production arithmetic including the width-2 Rhie-Chow wide gradient and
    its BC-aware boundary overrides (PICT_RC_BOUNDARY semantics, reading
    `d.pressure_pinned` exactly as the step does). Gated bit-level against
    the numpy original in test_prod_adjoint.py 9.2b."""

    _KEYS = (("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"),
             ("zeta_x", "zeta_y", "zeta_z"))

    def __init__(self, d, fk: "TorchFluxKernels", with_cross=True):
        self.d, self.fk = d, fk
        nb, self.N = fk.nb, fk.N
        self.pads2 = [PadMap(d, b, 2) for b in range(nb)]
        # static geometric factor g_ab = sum_c m[a_c]^2 per block/axis (own metrics)
        self.g = [[torch.as_tensor(sum(
            d.block_metrics_cached(b)[1][k] ** 2 for k in self._KEYS[a]))
            for a in range(3)] for b in range(nb)]
        if with_cross:
            self.pg = []
            for b in range(nb):
                Jp, mp, glo, ghi = d.padded_geometry(b, 1)
                self.pg.append((torch.as_tensor(Jp),
                                {k: torch.as_tensor(v) for k, v in mp.items()},
                                glo, ghi))

    def _jg_global(self, coef, axis):
        """coefs * J * g as ONE global field (association preserved)."""
        fk = self.fk
        g = torch.zeros(self.N, dtype=torch.float64)
        for b in range(fk.nb):
            cb = torch.take(coef, fk.gid[b]).reshape(fk.shapes[b])
            g = g.index_put((fk.gid[b],),
                            (cb * fk.J[b] * self.g[b][axis]).reshape(-1))
        return g

    def __call__(self, b, p, coef, include_orth=True, include_cross=False,
                 rhie_chow=False):
        d, fk = self.d, self.fk
        blk_shape, hb = fk.shapes[b], fk.h[b]
        pad1 = fk.pads[b]
        pp = pad1.apply(p)
        plo = pad1.lo
        if rhie_chow:
            pad2 = self.pads2[b]
            pp2 = pad2.apply(p)
            lo2, hi2 = pad2.lo, pad2.hi
            off = tuple(lo2[a] - plo[a] for a in range(3))
        if include_cross:
            Jp, mp, glo, ghi = self.pg[b]
            cc = pad1.apply(coef)
            dp = [_tgrad(pp, hb[a], a) for a in range(3)]

            def g_off(a1, a2):
                return sum(mp[self._KEYS[a1][c]] * mp[self._KEYS[a2][c]]
                           for c in range(3))

        mode = os.environ.get("PICT_RC_BOUNDARY", "auto")
        pinned = (frozenset() if mode == "ghost"
                  else getattr(d, "pressure_pinned", frozenset()))
        out = []
        for axis in range(3):
            Jg_pad = pad1.apply(self._jg_global(coef, axis))
            lo, hi = pad1.lo, pad1.hi
            n = blk_shape[axis]
            if include_cross:
                cross_cell = sum(cc * Jp * g_off(axis, o) * dp[o]
                                 for o in range(3) if o != axis)
            if rhie_chow:
                g2 = _tgrad(pp2, hb[axis], axis)
                for side, absent in ((0, lo2[axis] == 0), (1, hi2[axis] == 0)):
                    if not absent or mode == "legacy" or (b, axis, side) in pinned:
                        continue
                    nax = pp2.shape[axis]
                    sb = _sl(axis, slice(nax - 1, nax) if side else slice(0, 1))
                    sn = _sl(axis, slice(nax - 2, nax - 1) if side else slice(1, 2))
                    dd = (pp2[sb] - pp2[sn]) if side else (pp2[sn] - pp2[sb])
                    g2 = torch.cat(
                        [g2[_sl(axis, slice(0, nax - 1))], 0.5 * dd / hb[axis]]
                        if side else
                        [0.5 * dd / hb[axis], g2[_sl(axis, slice(1, None))]],
                        dim=axis)
                sl2 = tuple(slice(off[a], off[a] + pp.shape[a]) for a in range(3))
                dpw = g2[sl2]

            core = [slice(lo[a], lo[a] + blk_shape[a]) for a in range(3)]
            ccore = ([slice(glo[a], glo[a] + blk_shape[a]) for a in range(3)]
                     if include_cross else None)
            k0 = 0 if lo[axis] > 0 else 1
            k1 = n if hi[axis] > 0 else n - 1
            cnt = k1 - k0 + 1
            s1 = list(core); s1[axis] = slice(lo[axis] + k0 - 1, lo[axis] + k0 - 1 + cnt)
            s2 = list(core); s2[axis] = slice(lo[axis] + k0, lo[axis] + k0 + cnt)
            val = torch.zeros([cnt if a == axis else blk_shape[a] for a in range(3)],
                              dtype=torch.float64)
            if include_orth:
                cf = 0.5 * (Jg_pad[tuple(s1)] + Jg_pad[tuple(s2)])
                val = val + cf * (pp[tuple(s2)] - pp[tuple(s1)]) / hb[axis]
                if rhie_chow:
                    val = val - 0.5 * (Jg_pad[tuple(s1)] * dpw[tuple(s1)]
                                       + Jg_pad[tuple(s2)] * dpw[tuple(s2)])
            if include_cross:
                c1 = list(ccore); c1[axis] = slice(glo[axis] + k0 - 1,
                                                   glo[axis] + k0 - 1 + cnt)
                c2 = list(ccore); c2[axis] = slice(glo[axis] + k0,
                                                   glo[axis] + k0 + cnt)
                val = val + 0.5 * (cross_cell[tuple(c1)] + cross_cell[tuple(c2)])
            pieces = [val]
            zshape = [1 if a == axis else blk_shape[a] for a in range(3)]
            if lo[axis] == 0:
                pieces.insert(0, torch.zeros(zshape, dtype=torch.float64))
            if hi[axis] == 0:
                pieces.append(torch.zeros(zshape, dtype=torch.float64))
            out.append(torch.cat(pieces, dim=axis))
        return out
