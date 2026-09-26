"""Stage 9.2d: ONE production PISO step in torch, differentiable end-to-end.

Mirrors `MultiBlockPISO._step_impl` for the PRODUCTION configuration the
cylinder campaign runs (bdf2, rotational, Rhie-Chow persistent flux,
implicit_cross with the DC pressure solve, picard_iters x corrector_steps,
Dong outflow with copy factor; ddt_corr off, no velocity source, scalar nu,
dong_smooth off). Every ingredient is one of the Stage 9.1/9.2a-c certified
pieces:

    momentum A(u)      -> MomentumAssembly T           (9.1, affine, 2e-18)
    face/pressure flux -> TorchFluxKernels / TorchPressureFlux (bit-exact)
    cross diffusion    -> TorchFluxKernels.cross_diffusion    (bit-exact)
    M(coef)            -> DiffusionAssembly Tm          (linear, 5e-16)
    solves             -> LinearSolve (adjoint backward, dL/dA on pattern)

The constructor takes a CONFIGURED MultiBlockPISO so interior/bnd masks,
bc arrays, corner-pin enrollment and the outflow spec are identical by
construction. Torch state is global-flat (u, v, w, p, p_flux, u_prev).

Deliberate semantic notes, mirrored from production:
- Dong's U0 = max|u_n| enters as a detached scalar (production converts to
  a Python float, cutting any gradient through it).
- The DC sweep count follows PICT_CROSS_DC_TOL/ITERS with the convergence
  test on detached norms; each executed sweep is in the graph (the 7b
  lesson: the gradient is of the truncated iteration actually run).
- Solver tolerances: LinearSolve solves at its own TOL (tighter than
  production's); equivalence gates therefore tighten the production side
  (PICT_CROSS_DC_TOL) and compare at the linear-solve tolerance level.
"""
import os

import numpy as np
import scipy.sparse as sparse
import torch

from src.adjoint_piso import LinearSolve
from src.mb_adjoint import to_torch_sparse, spmv
from src.multiblock import face_axis_side, face_slice
from src.prod_adjoint import (DiffusionAssembly, MomentumAssembly,
                              TorchDiffusionAssembly, TorchFluxKernels,
                              TorchMomentumAssembly, TorchPressureFlux,
                              production_gradient_ops)


def _submap(idx, shape, rowsel, colsel):
    """Entry positions of A[rowsel][:, colsel] within the canonical value
    vector, plus the reduced pattern (rows, cols) and shape."""
    rows, cols = idx
    rpos = -np.ones(shape[0], dtype=np.int64)
    rpos[rowsel] = np.arange(len(rowsel))
    cpos = -np.ones(shape[1], dtype=np.int64)
    cpos[colsel] = np.arange(len(colsel))
    keep = (rpos[rows] >= 0) & (cpos[cols] >= 0)
    e = np.where(keep)[0]
    return e, (rpos[rows[e]], cpos[cols[e]]), (len(rowsel), len(colsel))


def _spmv_entries(rows_t, cols_t, vals, x, nrows):
    """y = A x for A given by entry lists; differentiable in vals and x."""
    return torch.zeros(nrows, dtype=torch.float64).index_add_(
        0, rows_t, vals * x[cols_t])


class TorchProductionStep:
    def __init__(self, m, verbose=False):
        d = m.d
        self.m, self.d = m, d
        nb, N = len(d.blocks), d.n_cells
        self.nb, self.N = nb, N
        self.dt, self.nu, self.tol = m.dt, float(m.nu), m.tol
        self.mom_tol = m.momentum_tol
        self.picard, self.correctors = m.picard_iters, m.corrector_steps
        self.mom_dc = m.momentum_dc_iters
        self.dong_copy = m.dong_copy

        self.asm = MomentumAssembly(d, self.nu, self.dt, verbose=verbose)
        self.tasm = TorchMomentumAssembly(self.asm)
        self.fk = TorchFluxKernels(d)
        self.pflux = TorchPressureFlux(d, self.fk, with_cross=True)
        self.da = DiffusionAssembly(d)
        self.tda = TorchDiffusionAssembly(self.da)
        # the PRODUCTION cell gradient, probed -- the chains'
        # cell_gradient_matrix differs O(1) from d.gradient on the butterfly
        self.G3 = [to_torch_sparse(g) for g in production_gradient_ops(d)]

        self.interior = m.interior.copy()
        self.bnd = m.bnd.copy()
        self.int_t = torch.as_tensor(self.interior)
        self.bnd_t = torch.as_tensor(self.bnd)
        self.Jg = torch.as_tensor(m._flat({b: self.asm.Js[b] for b in range(nb)}))

        # momentum submatrices as positions into the canonical A values
        e_ii, patt_ii, sh_ii = _submap(np.vstack(self.asm.idx), self.asm.shape,
                                       self.interior, self.interior)
        e_ib, patt_ib, sh_ib = _submap(np.vstack(self.asm.idx), self.asm.shape,
                                       self.interior, self.bnd)
        self.e_ii = torch.as_tensor(e_ii)
        self.patt_ii = ((patt_ii[0], patt_ii[1]), sh_ii)
        self.e_ib = torch.as_tensor(e_ib)
        self.ib_rows = torch.as_tensor(patt_ib[0])
        self.ib_cols = torch.as_tensor(patt_ib[1])
        # row aggregation for rowsum(A): entry -> row
        self.rows_all = torch.as_tensor(self.asm.idx[0].astype(np.int64))

        # pressure system: pattern of M, Dong node set (static), submaps
        pD, _ = m._dong_nodes()
        self.pD = pD
        mask = np.ones(N, dtype=bool)
        mask[pD] = False
        self.free = np.arange(N)[mask]
        self.free_t = torch.as_tensor(self.free)
        self.pD_t = torch.as_tensor(pD)
        e_ff, patt_ff, sh_ff = _submap(np.vstack(self.da.idx), self.da.shape,
                                       self.free, self.free)
        e_fD, patt_fD, sh_fD = _submap(np.vstack(self.da.idx), self.da.shape,
                                       self.free, pD)
        self.e_ff = torch.as_tensor(e_ff)
        self.patt_ff = ((patt_ff[0], patt_ff[1]), sh_ff)
        self.e_fD = torch.as_tensor(e_fD)
        self.fD_rows = torch.as_tensor(patt_fD[0])
        self.fD_cols = torch.as_tensor(patt_fD[1])

        # dong face geometry (static): boundary/interior gids, normal, dn
        self.dong = []
        for spec in m.outflow:
            if (spec[3] if len(spec) > 3 else "convective") != "dong":
                continue
            b, fid = spec[0], spec[1]
            axis, side = face_axis_side(fid)
            blk, bs = d.blocks[b], face_slice(fid)
            isl = [slice(None)] * 3
            isl[axis] = 1 if side == 0 else -2
            isl = tuple(isl)
            _, mb = d.block_metrics_cached(b)
            key = ("xi", "eta", "zeta")[axis]
            nx, ny, nz = mb[f"{key}_x"][bs], mb[f"{key}_y"][bs], mb[f"{key}_z"][bs]
            nrm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            sg = -1.0 if side == 0 else 1.0
            dn = np.sqrt((blk.x[bs] - blk.x[isl]) ** 2
                         + (blk.y[bs] - blk.y[isl]) ** 2
                         + (blk.z[bs] - blk.z[isl]) ** 2)
            self.dong.append(dict(
                gid=torch.as_tensor(d.global_ids(b)[bs].ravel()),
                gid_i=torch.as_tensor(d.global_ids(b)[isl].ravel()),
                n=[torch.as_tensor(sg * c / nrm).reshape(-1) for c in (nx, ny, nz)],
                dn=torch.as_tensor(dn.ravel())))
        self.dong_delta = m.dong_delta
        # warm-start seeds for the forward Krylov solves, keyed by call slot
        # (production's slot-seed trick; iterate path only, answer at TOL)
        self._seeds = {}
        # unique-node selection, mirroring _dong_nodes' np.unique(first)
        allg = np.concatenate([g["gid"].numpy() for g in self.dong]) if self.dong \
            else np.empty(0, dtype=int)
        _, self._dong_first = np.unique(allg, return_index=True)
        self._dong_first = torch.as_tensor(self._dong_first)

        # cross-active blocks (exact-math skip), mirroring _solve_cross_dc
        KEYS = (("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"),
                ("zeta_x", "zeta_y", "zeta_z"))
        self.xb = set()
        for b in range(nb):
            _, mb = d.block_metrics_cached(b)
            diag = max(float(np.abs(sum(mb[KEYS[a][c]] ** 2 for c in range(3))).max())
                       for a in range(3))
            off = max(float(np.abs(sum(mb[KEYS[a1][c]] * mb[KEYS[a2][c]]
                                       for c in range(3))).max())
                      for a1 in range(3) for a2 in range(3) if a1 != a2)
            if off > 1e-13 * diag:
                self.xb.add(b)

    # ------------------------------------------------------------- helpers
    def flat(self, per_block):
        return torch.as_tensor(self.m._flat(per_block))

    def state_from_solver(self):
        m = self.m
        st = dict(u=self.flat(m.u), v=self.flat(m.v), w=self.flat(m.w),
                  p=self.flat(m.p), p_flux=self.flat(m.p_flux),
                  ubc=self.flat(m.u_bc), vbc=self.flat(m.v_bc),
                  wbc=self.flat(m.w_bc))
        st["u_prev"] = (None if m.u_prev is None else
                        [self.flat(m.u_prev[k]) for k in range(3)])
        return st

    def dong_pressure(self, st):
        vals, gids = [], []
        for g in self.dong:
            ub = torch.take(st["u"], g["gid"])
            vb = torch.take(st["v"], g["gid"])
            wb = torch.take(st["w"], g["gid"])
            ui = torch.take(st["u"], g["gid_i"])
            vi = torch.take(st["v"], g["gid_i"])
            wi = torch.take(st["w"], g["gid_i"])
            un = ub * g["n"][0] + vb * g["n"][1] + wb * g["n"][2]
            un_i = ui * g["n"][0] + vi * g["n"][1] + wi * g["n"][2]
            U0 = float(torch.clamp(un.abs().max(), min=1e-12))   # detached, as production
            th = 0.5 * (1.0 - torch.tanh(un / (U0 * self.dong_delta)))
            pv = self.nu * (un - un_i) / g["dn"] \
                - 0.5 * (ub ** 2 + vb ** 2 + wb ** 2) * th
            vals.append(pv)
            gids.append(g["gid"])
        v = torch.cat(vals)
        return torch.index_select(v, 0, self._dong_first)

    def _update_outflow(self, st):
        """Dong advective update: bc <- bc - t (bc - interior), t = dong_copy."""
        t = self.dong_copy
        for g in self.dong:
            for f, fb in (("u", "ubc"), ("v", "vbc"), ("w", "wbc")):
                cur = torch.take(st[fb], g["gid"])
                new = cur - t * (cur - torch.take(st[f], g["gid_i"]))
                st[fb] = st[fb].index_put((g["gid"],), new)
                st[f] = st[f].index_put((g["gid"],), new)

    def cross_div(self, pfull, coef):
        """J-weighted cross-flux divergence, cross-active blocks only."""
        out = torch.zeros(self.N, dtype=torch.float64)
        for b in range(self.nb):
            if b not in self.xb:
                continue
            Phi = self.pflux(b, pfull, coef, include_orth=False,
                             include_cross=True)
            out = out.index_put((self.fk.gid[b],),
                                self.fk.divergence(b, Phi).reshape(-1))
        return self.Jg * out

    # ------------------------------------------------------------ the step
    def register_jets(self, jet_ids, jet_uv):
        """M2's actuator on the production step: profile-weighted Dirichlet
        velocities a * jet_uv at body wall nodes (jet_ids global, must lie
        in the Dirichlet set). apply_jets writes them into the state's bc
        arrays -- the correctors re-impose bc each sweep, so the actuation
        persists through the step and a stays in the graph."""
        ids = np.asarray(jet_ids, dtype=np.int64)
        missing = np.setdiff1d(ids, self.bnd)
        if missing.size:
            raise AssertionError(f"jet nodes not in the Dirichlet set: {missing[:5]}")
        self.jet_ids = torch.as_tensor(ids)
        self.jet_uv = torch.as_tensor(np.asarray(jet_uv, dtype=float))

    def apply_jets(self, st, a):
        st = dict(st)
        for c, (f, fb) in enumerate((("u", "ubc"), ("v", "vbc"), ("w", "wbc"))):
            val = a * self.jet_uv[:, c]
            st[fb] = st[fb].index_put((self.jet_ids,), val)
            st[f] = st[f].index_put((self.jet_ids,), val)
        return st

    def clear_seeds(self):
        """Drop warm-start seeds. Call when jumping to an unrelated state
        (a stale seed is at best useless, at worst a BiCGStab breakdown),
        and before FD probes (mutating seeds make probes nondeterministic
        at the solver-tolerance level)."""
        self._seeds = {}

    def _lsolve(self, vals, rhs, pattern, sym, sing, key):
        x0 = self._seeds.get(key)
        sol = LinearSolve.apply(vals, rhs, pattern, sym, sing, x0)
        self._seeds[key] = sol.detach().numpy().copy()
        return sol

    @staticmethod
    def _clone(st):
        return {k: (None if v is None else
                    [c.clone() for c in v] if isinstance(v, list) else v.clone())
                for k, v in st.items()}

    def step(self, st, src=None, detach_assembly=False, detach_dong=False):
        """One production step on torch state; returns the new state dict.
        Picard restore/early-exit mirror _step_impl (inf-norm test).
        `src`: optional per-component momentum sources (3 global tensors),
        production's velocity_source. `detach_assembly` freezes the
        gradient path THROUGH matrix assembly and the derived coefficient
        (the frozen-chain semantics -- the 9.3 mangle); `detach_dong` cuts
        the Dong-pressure path."""
        picard_state = self._clone(st)
        convect = None
        for pk in range(self.picard):
            if pk > 0:
                st = self._clone(picard_state)
            self._pk = pk
            out = self._step_once(st, convect, src=src,
                                  detach_assembly=detach_assembly,
                                  detach_dong=detach_dong)
            new_convect = (out["u"], out["v"], out["w"])
            if convect is not None:
                scale = max(max(float(c.abs().max()) for c in new_convect), 1e-30)
                rel = max(float((new_convect[i] - convect[i]).abs().max())
                          for i in range(3)) / scale
                if rel < self.tol:
                    st = out
                    break
            convect = new_convect
            st = out
        return st

    def _step_once(self, st, convect, src=None, detach_assembly=False,
                   detach_dong=False):
        N, nb, dt = self.N, self.nb, self.dt
        if self.dong:
            self._update_outflow(st)
        cvec = (torch.cat([st["u"], st["v"], st["w"]]) if convect is None
                else torch.cat(list(convect)))
        if detach_assembly:
            cvec = cvec.detach()
        Avals = self.tasm.vals(cvec)
        bdf2 = st["u_prev"] is not None

        # old-pressure gradient (rotational carries it into the predictor)
        gp = [spmv(self.G3[a], st["p"]) for a in range(3)]

        stars = []
        for k, (f, fb) in enumerate((("u", "ubc"), ("v", "vbc"), ("w", "wbc"))):
            phi_n = st[f]
            if bdf2:
                trans = (2.0 * phi_n - 0.5 * st["u_prev"][k]) / dt
            else:
                trans = phi_n / dt
            base = self.Jg * (trans - gp[k] + (src[k] if src is not None else 0.0))
            x = phi_n
            cur = phi_n
            for _dc in range(self.mom_dc):
                cd = torch.zeros(N, dtype=torch.float64)
                for b in range(nb):
                    cd = cd.index_put((self.fk.gid[b],),
                                      self.fk.cross_diffusion(b, cur).reshape(-1))
                rhs = base + self.Jg * (self.nu * cd)
                phi_b = torch.take(st[fb], self.bnd_t)
                elim = _spmv_entries(self.ib_rows, self.ib_cols,
                                     torch.take(Avals, self.e_ib), phi_b,
                                     len(self.interior))
                xi = self._lsolve(torch.take(Avals, self.e_ii),
                                  torch.take(rhs, self.int_t) - elim,
                                  self.patt_ii, False, False,
                                  ("mom", getattr(self, "_pk", 0), k, _dc))
                x = torch.zeros(N, dtype=torch.float64) \
                    .index_put((self.int_t,), xi) \
                    .index_put((self.bnd_t,), phi_b)
                cur = x
            stars.append(x)
        us, vs, ws = stars

        # Gamma from the GLOBAL row sums
        rowsum = torch.zeros(N, dtype=torch.float64).index_add_(
            0, self.rows_all, Avals)
        coef = self.Jg / rowsum

        Mvals = self.tda.vals(coef)
        if self.dong:
            dst = ({k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in st.items()} if detach_dong else st)
            pD_val = self.dong_pressure(dst)
        else:
            pD_val = None

        phi_tot = torch.zeros(N, dtype=torch.float64)
        div_star = None
        Fb = None
        for corr in range(self.correctors):
            if Fb is None:                      # persistent_flux: build ONCE
                pcur = st["p_flux"]
                Fb = self.fk.face_fluxes_all(us, vs, ws)
                for b in range(nb):
                    rc = self.pflux(b, pcur, coef, include_orth=True,
                                    include_cross=False, rhie_chow=True)
                    Fb[b] = [Fb[b][a] - rc[a] for a in range(3)]
            divF = torch.zeros(N, dtype=torch.float64)
            for b in range(nb):
                divF = divF.index_put((self.fk.gid[b],),
                                      self.fk.divergence(b, Fb[b]).reshape(-1))
            if div_star is None:
                div_star = divF
            rhs = -(self.Jg * divF)
            base_free = torch.take(rhs, self.free_t) - _spmv_entries(
                self.fD_rows, self.fD_cols, torch.take(Mvals, self.e_fD),
                pD_val, len(self.free))

            # DC cross pressure solve (implicit_cross), production env vars
            tol_dc = float(os.environ.get("PICT_CROSS_DC_TOL", "1e-3"))
            it_max = int(os.environ.get("PICT_CROSS_DC_ITERS", "6"))
            Mff_vals = torch.take(Mvals, self.e_ff)

            def full_p(sol_free):
                return torch.zeros(N, dtype=torch.float64) \
                    .index_put((self.free_t,), sol_free) \
                    .index_put((self.pD_t,), pD_val)

            b_free = base_free
            sol = None
            prev = None
            for _k in range(it_max):
                sol = self._lsolve(Mff_vals, b_free, self.patt_ff,
                                   False, False,
                                   ("dc", getattr(self, "_pk", 0), corr, _k))
                if prev is not None:
                    if float((sol - prev).norm()) <= \
                            tol_dc * max(float(sol.norm()), 1e-300):
                        prev = sol
                        break
                prev = sol
                b_free = base_free + torch.take(
                    self.cross_div(full_p(sol), coef), self.free_t)
            sol = self._lsolve(Mff_vals, b_free, self.patt_ff, False, False,
                               ("dcf", getattr(self, "_pk", 0), corr))
            pp = full_p(sol)

            gpp = [spmv(self.G3[a], pp) for a in range(3)]
            us = us - coef * gpp[0]
            vs = vs - coef * gpp[1]
            ws = ws - coef * gpp[2]
            # re-impose the Dirichlet values
            us = us.index_put((self.bnd_t,), torch.take(st["ubc"], self.bnd_t))
            vs = vs.index_put((self.bnd_t,), torch.take(st["vbc"], self.bnd_t))
            ws = ws.index_put((self.bnd_t,), torch.take(st["wbc"], self.bnd_t))
            for b in range(nb):
                Phis = self.pflux(b, pp, coef, include_orth=True,
                                  include_cross=True)
                Fb[b] = [Fb[b][a] - Phis[a] for a in range(3)]
            phi_tot = phi_tot + pp

        new = dict(st)
        new["p_flux"] = st["p_flux"] + phi_tot
        new["p"] = st["p"] + phi_tot - self.nu * div_star      # rotational
        new["u_prev"] = [st["u"], st["v"], st["w"]]
        new["u"], new["v"], new["w"] = us, vs, ws
        return new
