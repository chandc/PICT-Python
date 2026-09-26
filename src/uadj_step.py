"""U4 of the unstructured adjoint plan: one BDF2-PISO step of `src/upiso.PISO` in torch.

`TorchUPISO(piso)` is built from a CONFIGURED production solver, so boundary kinds, correction
counts, the convection scheme and the gradient choices are the production ones by construction,
and every constant operator is converted from the production matrices (no re-derived geometry).
`step(state)` mirrors `PISO.step` line for line; `test_uadj_step.py` holds it to the production
trajectory field for field (plan gate A8) before any gradient is trusted.

State (all torch, float64): u, v, p, Ff, Ff_prev, Ff_old, Fbar_old, u_old, v_old, plus the three
boundary-value vectors ub, vb, pb (the actuation inputs of U6). History entries may be None,
exactly as in production on the first step.

Loops. `n_inner`, `n_corr` and `n_outer` are fixed counts in production and here. The
non-orthogonal pressure loop in `solve_poisson` exits on a tolerance; here it does the same test
on detached values, so the executed count matches production, and the count of every call is
recorded in `self.poisson_counts`. `fixed_poisson=k` pins it (for FD gates), and the graph is
always of the iteration actually run.
"""
import numpy as np
import torch

from src.uops import DIRICHLET, NEUMANN
from src.uadj_ops import (DT, TIE, Masks, st_mask, Pattern, TBC, TGradient, TMesh, SparseConst,
                          convection_rhs, convection_vals, laplacian_rhs, laplacian_vals, _t)
from src.uadj_solve import LUFactor, lu_solve

STATE_KEYS = ("u", "v", "p", "Ff", "Ff_prev", "Ff_old", "Fbar_old", "u_old", "v_old", "ub", "vb", "pb")


class TorchUPISO:
    def __init__(self, piso):
        s = self.s = piso
        if s.time_scheme != "bdf2":
            raise NotImplementedError("TorchUPISO mirrors the BDF2 path; RK3 is plan stage U9")
        if s.p_neumann_extrap or s.grad_rc is not None:
            raise NotImplementedError("experimental Rhie-Chow / pressure-extrapolation options")
        m = s.m
        self.tm = tm = TMesh(m)
        self.pat = Pattern(tm)
        self.grad = TGradient(s.grad)
        self.grad_p = TGradient(s.grad_p) if s.grad_p is not None else self.grad
        self.bc_u, self.bc_v, self.bc_p = TBC(tm, s.bc_u), TBC(tm, s.bc_v), TBC(tm, s.bc_p)
        self.nu, self.dt = float(s.nu), float(s.dt)
        self.fx, self.fy = _t(s.fx), _t(s.fy)
        self.scheme, self.convect = s.scheme, s.convect

        def dirmask(bc):
            dm = torch.zeros(tm.nface, dtype=torch.bool)
            dm[tm.fb] = torch.as_tensor(bc.kind[m.bface_index[m.bfaces]] == DIRICHLET)
            return dm
        self.dir_u, self.dir_v, self.dir_p = dirmask(s.bc_u), dirmask(s.bc_v), dirmask(s.bc_p)
        # viscosity as a (differentiable) scalar tensor; the viscous operators are rebuilt from it
        # every step, which costs one face pass and puts nu in the graph (dL/dnu: plan P1, P2, P4)
        self.nu_t = torch.tensor(self.nu, dtype=DT)
        self.p_singular = bool(s.p_singular)

        # faces whose normal flux the velocity BCs prescribe: no Rhie-Chow damping there
        kd_u = s.bc_u.kind == DIRICHLET; kd_v = s.bc_v.kind == DIRICHLET
        Sb = m.normal[m.bfaces]; hyp = np.hypot(*Sb.T)
        ax = np.abs(Sb[:, 0]) > 1e-9 * hyp; ay = np.abs(Sb[:, 1]) > 1e-9 * hyp
        fixed = np.zeros(m.nface, dtype=bool)
        fixed[m.bfaces] = (kd_u & kd_v) | (kd_u & ~ay) | (kd_v & ~ax)
        self.fixed_u = torch.as_tensor(fixed)
        self.rc_scale = float(s.rc_scale)
        self.masks = Masks("live")
        self.fixed_poisson = None
        self.poisson_counts = []
        self._plan = None; self._pk = 0
        # plan gate A11: names of gradient paths to cut (forward values unchanged)
        self.detach = set()

    # ------------------------------------------------------------- branch record / replay
    def record(self):
        """Next forward(s) record every mask and every Poisson sweep count."""
        self.masks = Masks("record"); self.poisson_counts = []; self._plan = None

    def replay(self):
        """The next forward takes exactly the recorded branches and sweep counts (FD probes).
        Call before EVERY replayed forward: it rewinds to the start of the recording."""
        if self.masks.mode == "record":
            self._frozen = (self.masks.log, list(self.poisson_counts))
        self.replay_from(self._frozen)

    def replay_from(self, frozen):
        """Replay a recording kept elsewhere: (mask log, Poisson counts), as `replay` stores it."""
        self._frozen = frozen
        log, counts = frozen
        self.masks = Masks("replay"); self.masks.log = log
        self._plan = list(counts); self._pk = 0
        self.poisson_counts = []

    def live(self):
        self.masks = Masks("live"); self._plan = None; self.poisson_counts = []

    # -------------------------------------------------------------------------------- state
    def state_from_solver(self, requires_grad=False):
        s = self.s

        def c(a):
            if a is None:
                return None
            t = _t(np.array(a, dtype=float))
            return t.requires_grad_(True) if requires_grad else t
        return {"u": c(s.u), "v": c(s.v), "p": c(s.p), "Ff": c(s.Ff), "Ff_prev": c(s.Ff_prev),
                "Ff_old": c(s.Ff_old), "Fbar_old": c(s.Fbar_old), "u_old": c(s.u_old), "v_old": c(s.v_old),
                "ub": c(s.bc_u.value), "vb": c(s.bc_v.value), "pb": c(s.bc_p.value)}

    # ---------------------------------------------------------------------------- pieces
    def _d(self, name, x):
        return x.detach() if (x is not None and name in self.detach) else x

    def _momentum(self, comp, phi, phi_old, bc, bval, gp_comp, Fconv, phi_guess):
        tm, pat = self.tm, self.pat
        Fconv = self._d("conv_flux", Fconv)
        phi_old = self._d("bdf2_history", phi_old)
        if self.convect:
            Cv, pos = convection_vals(tm, pat, Fconv, self.masks)
        else:
            Cv, pos = pat.zeros(), None
        dirm = self.dir_u if comp == 0 else self.dir_v
        nuf = self.nu_t.expand(tm.nface)
        Lv = laplacian_vals(tm, pat, nuf, dirm)
        f = self.fx if comp == 0 else self.fy
        if phi_old is not None:
            a_t, rhs_t = 1.5 / self.dt, (2.0 * phi - 0.5 * phi_old) / self.dt
        else:
            a_t, rhs_t = 1.0 / self.dt, phi / self.dt
        A = pat.zeros().index_add(0, pat.p_diag, a_t * tm.vol) + Cv - Lv
        factor = LUFactor(pat, A)

        def rhs(g):
            pb = bc.effective(g, bval)
            gphi = self.grad(g, pb)
            b = (rhs_t + f - gp_comp) * tm.vol
            dm = torch.zeros_like(b)
            if self.convect:
                dm = dm - convection_rhs(tm, Fconv, pos, g, pb, gphi, self.scheme)
            dm = dm + laplacian_rhs(tm, nuf, dirm, gphi, pb)
            return b + self._d("deferred_momentum", dm)
        g0 = phi if phi_guess is None else phi_guess
        x = lu_solve(A, rhs(g0), pat, factor)
        for _ in range(self.s.n_inner - 1):
            x = lu_solve(A, rhs(x), pat, factor)
        aP = pat.diag(A)
        rs = pat.rowsum(A); floor = a_t * tm.vol
        keep = self.masks("simplec", lambda: rs >= floor)       # np.maximum(rowsum, a_t V)
        tie = self.masks("simplec_tie", lambda: (rs - floor).abs() <= TIE * floor)
        if self.masks.mode == "replay":
            keep = torch.where(tie, rs.detach() >= floor, keep)
        # value: production's branch; derivative: the midpoint where rowsum sits on its floor (the
        # viscous row sum is zero, so interior cells are within round-off of the floor: finding 12)
        wk = torch.where(tie, torch.full_like(floor, 0.5), keep.to(DT))
        aC = st_mask(rs, keep.to(DT), wk) + st_mask(floor, (~keep).to(DT), 1.0 - wk)
        return x, aP, aC

    def _gam(self, Dcell):
        tm = self.tm
        g = torch.zeros(tm.nface, dtype=DT)
        g = g.index_copy(0, tm.fi, tm.w_i * Dcell[tm.o_i] + (1.0 - tm.w_i) * Dcell[tm.n_i])
        return g.index_copy(0, tm.fb, Dcell[tm.o_b])

    def _rhie_chow(self, u, v, aP, p, gp, st):
        tm = self.tm
        ub = self.bc_u.effective(u, st["ub"]); vb = self.bc_v.effective(v, st["vb"])
        ubar = tm.interp(u, ub); vbar = tm.interp(v, vb)
        Fbar = (ubar * tm.normal[:, 0] + vbar * tm.normal[:, 1]) * tm.span
        Dcell = self.rc_scale * tm.vol / torch.clamp_min(self._d("aP", aP), 1e-300)
        D = self._gam(Dcell)
        pbv = self.bc_p.effective(p, st["pb"])
        gpf = tm.interp_vec(gp)
        dpc = torch.zeros(tm.nface, dtype=DT)
        dpc = dpc.index_copy(0, tm.fi, tm.ef_over_d[tm.fi] * (p[tm.n_i] - p[tm.o_i]))
        dpc = dpc.index_copy(0, tm.fb, tm.ef_over_d[tm.fb] * (pbv[tm.bidx_b] - p[tm.o_b]))
        dpc = dpc + (tm.Tf * gpf).sum(dim=1)
        dpw = (gpf * tm.normal).sum(dim=1)
        zero = torch.zeros(tm.nface, dtype=DT)
        damp = torch.where(self.fixed_u, zero, D * (dpc - dpw) * tm.span)
        F = Fbar - damp
        if st["Ff_old"] is not None and st["Fbar_old"] is not None:
            hist = self._d("rc_history", st["Ff_old"] - st["Fbar_old"])
            tr = torch.where(self.fixed_u, zero, (D / self.dt) * hist)
            F = F + tr
        return F, Fbar

    def _pressure_flux(self, gam, pp, gpp):
        tm = self.tm
        coef = gam * tm.ef_over_d * tm.span
        F = torch.zeros(tm.nface, dtype=DT).index_copy(0, tm.fi, coef[tm.fi] * (pp[tm.n_i] - pp[tm.o_i]))
        gf = tm.interp_vec(gpp)
        cross = gam * (tm.Tf * gf).sum(dim=1) * tm.span
        cross = cross.index_fill(0, tm.fb, 0.0)
        return F + cross

    def _solve_poisson(self, Ap_vals, gam, source, phi_b, tol=1e-9, max_outer=25):
        """`uops.solve_poisson` for the pressure correction (bkind = bc_p.kind, x0 = 0)."""
        tm, pat = self.tm, self.pat
        factor = LUFactor(pat, Ap_vals, singular=self.p_singular)
        rhs_target = source * tm.vol
        phi = torch.zeros(tm.ncell, dtype=DT)
        neu = self.bc_p.neu
        n = 0
        for it in range(max_outer):
            pb_eff = torch.where(neu, phi[tm.o_bf], phi_b)
            gr = self.grad(phi, pb_eff)
            b = rhs_target - self._d("deferred_poisson", laplacian_rhs(tm, gam, self.dir_p, gr, pb_eff))
            if self.p_singular:
                b = b - b.mean()
            new = lu_solve(Ap_vals, b, pat, factor)
            if self.p_singular:
                new = new - new.mean()
            nd, od = new.detach(), phi.detach()
            delta = float((nd - od).abs().max()) / max(float(nd.abs().max()), 1e-30)
            phi = new
            n = it + 1
            if self._plan is not None:
                if n >= self._plan[self._pk]:
                    break
            elif self.fixed_poisson is not None:
                if n >= self.fixed_poisson:
                    break
            elif delta < tol:
                break
        if self._plan is not None:
            self._pk += 1
        self.poisson_counts.append(n)
        return phi

    # ------------------------------------------------------------------------------ step
    def step(self, st):
        s, tm, pat = self.s, self.tm, self.pat
        if s.n_outer != 1:
            raise NotImplementedError("PIMPLE outer iterations: not needed for the U4 gates")
        Fn = st["Ff"]
        u_n, v_n = st["u"], st["v"]
        u, v, F, p = u_n, v_n, Fn, st["p"]
        if s.conv_flux_extrap and st["Ff_prev"] is not None:
            Fconv = 2.0 * Fn - st["Ff_prev"]
        else:
            Fconv = Fn
        pb = self.bc_p.effective(p, st["pb"])
        gp = self.grad_p(p, pb)
        u_star, aP, aC = self._momentum(0, u_n, st["u_old"], self.bc_u, st["ub"], gp[:, 0], Fconv, None)
        v_star, _, _ = self._momentum(1, v_n, st["v_old"], self.bc_v, st["vb"], gp[:, 1], Fconv, None)
        u, v = u_star, v_star
        pzero = torch.zeros(tm.nbface, dtype=DT)
        phi_b = pzero if self.p_singular else st["pb"]
        Fbar = None
        for _ in range(s.n_corr):
            F, Fbar = self._rhie_chow(u, v, aP, p, gp, st)
            Dcell = tm.vol / torch.clamp_min(self._d("aC", aC), 1e-300)
            gam = self._gam(Dcell)
            Ap = laplacian_vals(tm, pat, gam, self.dir_p)
            src = tm.divergence(F)
            pp = self._solve_poisson(Ap, gam, src, phi_b, tol=1e-9, max_outer=s.n_nonorth)
            gpp = self.grad(pp, self.bc_p.effective(pp, pzero))
            u = u - Dcell * gpp[:, 0]
            v = v - Dcell * gpp[:, 1]
            F = F - self._pressure_flux(gam, pp, gpp)
            p = p + pp
            gp = self.grad(p, self.bc_p.effective(p, st["pb"]))
        return {"u": u, "v": v, "p": p, "Ff": F, "Ff_prev": Fn, "Ff_old": F, "Fbar_old": Fbar,
                "u_old": u_n, "v_old": v_n, "ub": st["ub"], "vb": st["vb"], "pb": st["pb"]}
