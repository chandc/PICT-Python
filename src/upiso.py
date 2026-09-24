"""Collocated unstructured PISO: momentum assembly, Rhie-Chow, and the time loop.

RHIE-CHOW IS TIME-STEP INDEPENDENT here (Choi 1999; Yu et al. 2002), not the spatial-only form.
The naive damping,

    U_f = ubar_f . n  -  (V/a_P)_f [ (p_N - p_P)/|d|  -  gradp_bar_f . n ]

carries a coefficient (V/a_P) that contains 1/dt through a_P, so the amount of pressure-velocity
damping SCALES WITH THE TIME STEP. Shrinking dt then changes the steady state the solver
converges to, and in the limit the damping vanishes and the checkerboard returns. Carrying the
previous face flux,

    U_f^n = ubar_f . n - D_f [ dp_compact - dp_wide ] + (V/a_P/dt)_f ( U_f^{n-1} - ubar_f^{n-1} . n )

makes the added dissipation independent of dt, which is what the four-decade dt test checks.

The migration guide prints the spatial-only formula under a heading that says time-step
independent; its own verification (vary dt over four orders, expect no spurious modes) is the
test that formula fails.
"""
import numpy as np
import scipy.sparse as sp

from src.uops import (Gradient, laplacian, convection, divergence,
                      DIRICHLET, NEUMANN, solve_poisson)
from src.linsolve import SolveCache


class BC:
    """Per-boundary-face conditions for one scalar.

    `kind` is DIRICHLET or NEUMANN per boundary face; `value` is the Dirichlet value (ignored
    where the kind is NEUMANN). A boundary face is a face with one owner and a rule for the far
    side -- there is nothing else to it on an unstructured mesh.
    """

    def __init__(self, mesh, kind=None, value=None):
        self.mesh = mesh
        self.kind = np.full(mesh.nbface, NEUMANN) if kind is None else np.asarray(kind)
        self.value = np.zeros(mesh.nbface) if value is None else np.asarray(value, dtype=float)

    def effective(self, phi):
        """Boundary face values: the prescribed one where Dirichlet, the owner's where not."""
        out = self.value.copy()
        neu = self.kind == NEUMANN
        if neu.any():
            out[neu] = phi[self.mesh.owner[self.mesh.bfaces][neu]]
        return out


class PISO:
    """Cell-centred collocated PISO on an unstructured mesh.

    State is the cell velocity (u, v), the cell pressure p, and the FACE FLUX `Ff`. The face
    flux is a primary variable, not something re-derived from the cell velocities each step:
    Rhie-Chow makes the two differ by construction, and it is the face flux that the pressure
    projection makes solenoidal.
    """

    def __init__(self, mesh, nu, dt, bc_u, bc_v, bc_p, scheme="central",
                 n_corr=2, n_nonorth=2, body_force=None, backend="splu", convect=True,
                 n_inner=2):
        self.m = mesh
        self.nu, self.dt = float(nu), float(dt)
        self.bc_u, self.bc_v, self.bc_p = bc_u, bc_v, bc_p
        self.scheme = scheme
        self.convect = bool(convect)   # False -> Stokes, which isolates the pressure coupling
        self.n_corr, self.n_nonorth = n_corr, n_nonorth
        # Inner iterations of the momentum DEFERRED correction within one time step. With
        # n_inner = 1 the non-orthogonal cross term (and the deferred central convection) is
        # built from the PREVIOUS time level's gradient -- a one-step lag that is invisible at a
        # steady fixed point and O(dt) in an unsteady problem. Measured on T4 (unsteady Stokes,
        # decaying Taylor-Green, uniform mesh where the cross term lives only on the 26.6-deg
        # boundary faces): slope 0.91 instead of 2, and zeroing that term dropped the error at
        # dt = 0.1 by 4x straight onto the spatial floor. Iterating converges the correction at
        # t^{n+1} instead; the steady operator is unchanged, so nothing steady moves (measured
        # bit-identical on T5). Two iterations suffice: T4 orders 2.70 / 2.01 at n_inner = 2 against
        # 0.85 / 0.92 / 0.91 lagged. n_inner = 1 is kept only for comparison.
        self.n_inner = int(n_inner)
        self.fx = np.zeros(mesh.ncell) if body_force is None else np.asarray(body_force[0])
        self.fy = np.zeros(mesh.ncell) if body_force is None else np.asarray(body_force[1])

        self.grad = Gradient(mesh)
        self.u = np.zeros(mesh.ncell)
        self.v = np.zeros(mesh.ncell)
        self.p = np.zeros(mesh.ncell)
        self.Ff = np.zeros(mesh.nface)
        self.u_old = self.v_old = None            # BDF2 history
        self.Ff_old = None                        # previous face flux, for time-independent RC
        # Convecting mass flux for the momentum matrix: 2 F^n - F^{n-1}, the second-order
        # extrapolation to n+1 (F^n alone on the first step). Using F^n, the flux at the start of
        # the step, is a first-order-in-time linearisation: T8 (Orr-Sommerfeld, section 47 of the
        # skew record) measured the phase speed converging at order 1.00 in dt with it, against
        # 2.3-2.7 on the convection-free T4; with the extrapolation the phase-speed error is
        # dt-independent (-0.60/-0.58/-0.58% at dt 0.05/0.025/0.0125 on 48x200). Every T9/T10
        # result recorded before 2026-09-23 was run with the lagged flux.
        self.conv_flux_extrap = True
        # Outer (PIMPLE-style) iterations per time step: 1 = classic PISO. Each further pass
        # re-linearises convection about the corrected flux and re-solves momentum with the
        # corrected pressure, the time level held fixed; drives the splitting error to zero.
        self.n_outer = 1
        self.outer_change = []
        # Time scheme: "bdf2" (PISO/PIMPLE, the default) or "rk3" -- Le & Moin's three-stage
        # low-storage Runge-Kutta with explicit convection, Crank-Nicolson diffusion and a
        # pressure projection at every stage (LES plan L1a).
        self.time_scheme = "bdf2"
        self._rk_caches = {}
        self._rk_lap = {}
        self._N_prev = None
        self._rk_dt_prev = None
        self.Ff_prev = None                       # F^{n-1}
        self._flux_init = False                   # Ff built from (u, v) on the first step, see init_flux()
        self._Fconv = None
        self.Fbar_old = None                      # and its plain-interpolation counterpart
        self.ubar_old = None
        self.time, self.nstep = 0.0, 0

        self._mcache = SolveCache(backend=backend)
        self._pcache = SolveCache(backend=backend)
        self._gam_cache = None; self._Ap = None; self._Ap_rhs = None   # pressure Laplacian, rebuilt only when gam changes
        # the viscous operator never changes, so assemble it once
        self.Lu, self.Lu_rhs = laplacian(mesh, np.full(mesh.nface, self.nu), bc_u.kind)
        self.Lv, self.Lv_rhs = laplacian(mesh, np.full(mesh.nface, self.nu), bc_v.kind)
        # pressure operator: coefficient is 1/a_P interpolated to faces, set each step
        self.p_singular = bool((bc_p.kind == NEUMANN).all())
        self.rc_scale = 1.0        # diagnostic knob on the Rhie-Chow damping magnitude
        self.p_neumann_extrap = False  # linear extrapolation of p to Neumann faces in the momentum gradient (experiment)
        # Gradient used ONLY for the Rhie-Chow `dp_wide` term. It is deliberately separable from
        # self.grad: the damping (dp_compact - dp_wide) is dissipative only while the wide
        # gradient acts as a SMOOTHER. A higher-order gradient does not smooth -- it can overshoot
        # the compact difference and flip the sign of the dissipation. Substituting a quadratic
        # k-exact gradient here diverged geometrically at n=64 (max|p| growing ~2.7x per step)
        # while div F stayed at 1e-12, i.e. the projection was sound and the feedback loop was not.
        self.grad_rc = None        # None -> use self.grad
        # Gradient used ONLY for the pressure gradient in the MOMENTUM source. Separable so a
        # high-order gradient can be used where it buys accuracy without putting it inside the
        # pressure-correction feedback loop, which is where it destabilises.
        # Divergence-form pressure gradient for the momentum source (and hence Rhie-Chow's
        # dp_wide, which receives the same gp). DEFAULT: the exact adjoint of the interpolated-flux
        # divergence. The LSQ gradient in this slot gives order 0.02 on T5 (its gradient of the
        # pressure checkerboard carries a smooth mode the Stokes operator amplifies) and diverges
        # outright on a clustered mesh at n = 64; the dual Green-Gauss gives 1.98 / 1.99 and is
        # stable everywhere tested. Its zeroth-order inconsistency on irregular triangles is
        # harmless HERE (supraconvergence, measured) and must not be imported into the Laplacian
        # cross terms, the pressure-correction update or the Poisson deferred correction, which
        # keep self.grad. Record: reference/skew_unstructured_literature.md sections 16-24.
        from src.ugrad import GGDualGradient
        self.grad_p = GGDualGradient(mesh)   # set to None to fall back to self.grad

    # -----------------------------------------------------------------------------------------
    def _momentum(self, comp, phi, phi_old, bc, gp_comp, phi_guess=None):
        """Assemble and solve one momentum component. Returns (phi_star, a_P, a_C).

        `a_C` IS THE SIMPLEC COEFFICIENT, and it is not a refinement -- without it this solver
        diverges outright whenever diffusion dominates the momentum diagonal.

        The velocity correction u' = -(V/a) grad p' approximates A^{-1} by 1/a. SIMPLE takes
        a = a_P, the bare diagonal; that is only defensible when a_P is dominated by the
        transient term. A viscous operator has ZERO ROW SUM, so its contribution cancels out of
        the true response to a smooth pressure field while inflating a_P. Measured on the Stokes
        gate (nu=1, dt=0.05, h=1/16, diffusion number 12.8): a_P = 6.06 but the row sum is
        0.058594 = a_t*V exactly. D = V/a_P is then 92x too small, the pressure correction
        overshoots by that factor, and `p += pp` amplifies geometrically -- max|u| grew ~32x per
        step, reaching 7.8e+14 by step 12.

        SIMPLEC uses a_P - sum(a_N) instead, which IS the matrix row sum (with a_N = -A[P,N]).
        That is exact for a constant field, so the viscous part cancels as it should. Taking the
        row sum rather than a_P - sum|a_N| keeps the signs right for central convection, whose
        off-diagonals are not sign-definite.

        Corrector count is NOT the cure and measuring it is how this was pinned down: n_corr 1,
        2, 5, 20 and 50 all produced a bit-identical divergent trajectory, which rules the PISO
        splitting out and leaves the coefficient itself.
        """
        m = self.m
        if self.convect:
            C, C_rhs = convection(m, self._Fconv if self._Fconv is not None else self.Ff, bc.kind, scheme=self.scheme)
        else:
            C = sp.csr_matrix((m.ncell, m.ncell))
            C_rhs = lambda *a, **k: np.zeros(m.ncell)
        L = self.Lu if comp == 0 else self.Lv
        L_rhs = self.Lu_rhs if comp == 0 else self.Lv_rhs

        # BDF2 when a history exists, BDF1 on the first step (and after a dt change)
        if phi_old is not None:
            a_t, rhs_t = 1.5 / self.dt, (2.0 * phi - 0.5 * phi_old) / self.dt
        else:
            a_t, rhs_t = 1.0 / self.dt, phi / self.dt

        A = sp.diags(a_t * m.vol) + C - L
        # phi is the TIME LEVEL (transient rhs); phi_guess is the latest iterate for the lagged
        # deferred terms and the initial guess -- they differ only inside outer iterations
        g0 = phi if phi_guess is None else phi_guess
        pb = bc.effective(g0)
        gphi = self.grad(g0, pb)
        b = (rhs_t + (self.fx if comp == 0 else self.fy) - gp_comp) * m.vol
        b = b - C_rhs(g0, pb, gphi) + L_rhs(gphi, pb)
        Ac = A.tocsr()
        x = self._mcache.solve(Ac, b, x0=g0, symmetric=False, rtol=1e-10)
        f_comp = self.fx if comp == 0 else self.fy
        for _ in range(self.n_inner - 1):
            pb = bc.effective(x)
            gphi = self.grad(x, pb)
            b = (rhs_t + f_comp - gp_comp) * m.vol - C_rhs(x, pb, gphi) + L_rhs(gphi, pb)
            x = self._mcache.solve(Ac, b, x0=x, symmetric=False, rtol=1e-10)
        aP = np.asarray(Ac.diagonal())
        # SIMPLEC: a_P - sum(a_N) == the row sum. Floored at the transient term, which is the
        # response a row-sum-zero spatial operator can never push below.
        aC = np.maximum(np.asarray(Ac.sum(axis=1)).ravel(), a_t * m.vol)
        return x, aP, aC

    # -----------------------------------------------------------------------------------------
    RK3_GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
    RK3_ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)
    RK3_ALPHA = (4.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0)      # alpha = beta: Crank-Nicolson diffusion per stage

    def step_rk3(self):
        """Le & Moin (1991) RK3 / Crank-Nicolson fractional step, projection at every stage.

        Stage k advances u^k -> u^{k+1} over dt_k = (alpha_k + beta_k) dt:
            (V/dt) u* - beta_k L u*  =  V u^k/dt - gamma_k N(u^k) - zeta_k N(u^{k-1})
                                        + alpha_k L u^k + (alpha_k+beta_k) V (f - grad p^k)
        with N(u) = C(F^k) u the volume-integrated central convection on the stage's divergence-free
        flux F^k (explicit, so no linearisation lag -- the splitting error G0 measured in PISO is
        gone by construction), then Rhie-Chow face flux, a pressure correction pp on
        div((V/a_C) grad pp) = div F*, and u^{k+1} = u* - (V/a_C) grad pp, F^{k+1} = F* - F_pp,
        p^{k+1} = p^k + pp (Dcell carries dt_k, so pp is a pressure). Convection is third order in dt, diffusion and
        pressure second order. The three momentum matrices (one per beta_k) and the three
        pressure operators are factorised once and cached.
        """
        m = self.m; dt = self.dt
        u, v, F, p = self.u, self.v, self.Ff, self.p
        N_prev = None
        dt_save = self.dt
        for k in range(3):
            g, z, al = self.RK3_GAMMA[k], self.RK3_ZETA[k], self.RK3_ALPHA[k]; be = al
            dtk = (al + be) * dt
            # convective term on the stage flux (explicit, deferred central on top of upwind)
            if self.convect:
                C, C_rhs = convection(m, F, self.bc_u.kind, scheme=self.scheme)
                def N(phi, bc):
                    pb = bc.effective(phi); return C @ phi + C_rhs(phi, pb, self.grad(phi, pb))
            else:
                def N(phi, bc): return np.zeros(m.ncell)
            Nk = (N(u, self.bc_u), N(v, self.bc_v))
            if N_prev is None: N_prev = Nk
            pb = self.bc_p.effective(p); gp = (self.grad if self.grad_p is None else self.grad_p)(p, pb)
            a_t = 1.0 / dt
            if k not in self._rk_caches:
                self._rk_caches[k] = (SolveCache(backend=self._mcache.backend), SolveCache(backend=self._mcache.backend))
            out = []
            for comp, (phi, bc, L, L_rhs, f, Np, Npr) in enumerate(((u, self.bc_u, self.Lu, self.Lu_rhs, self.fx, Nk[0], N_prev[0]),
                                                                     (v, self.bc_v, self.Lv, self.Lv_rhs, self.fy, Nk[1], N_prev[1]))):
                pbc = bc.effective(phi); gphi = self.grad(phi, pbc)
                Ldiff_k = L @ phi + L_rhs(gphi, pbc)                       # full diffusion of u^k
                A = (sp.diags(a_t * m.vol) - be * L).tocsr()
                b = a_t * m.vol * phi - g * Np - z * Npr + al * Ldiff_k + (al + be) * m.vol * (f - gp[:, comp])
                x = phi
                for _ in range(max(1, self.n_inner)):                       # lagged cross-diffusion of u*
                    pbs = bc.effective(x); gs = self.grad(x, pbs)
                    bb = b + be * L_rhs(gs, pbs)
                    x = self._rk_caches[k][comp].solve(A, bb, x0=x, symmetric=False, rtol=1e-10)
                out.append((x, np.asarray(A.diagonal()), np.maximum(np.asarray(A.sum(axis=1)).ravel(), a_t * m.vol)))
            us, aP, aC = out[0][0], out[0][1], out[0][2]; vs = out[1][0]
            # projection: Rhie-Chow flux with the stage's time step, pressure correction
            # Plain per-stage Rhie-Chow (D_k = dt_k V/a_C), WITHOUT Choi's dt-independent transient
            # term: that term cancels the previous damping only when consecutive steps are equal, and
            # inside a three-stage RK it left a dt-independent 1.2-1.7%/turnover dissipation (with the
            # current or the previous stage step in its denominator alike), or blew up with the
            # predictor-based pair. Without it the stage damping is O(dt_k) and the measured loss on
            # the advected Taylor-Green is 0.13%/turnover at dt 0.005 (64^2), ~dt^1.8, checkerboard
            # indicator unchanged (record section 50).
            self.Ff_old = None
            self.dt = dtk
            Fs = self._rhie_chow(us, vs, aP / (al + be), aC / (al + be), p, gp)
            self.dt = dt_save
            Dcell = (al + be) * m.vol / np.maximum(aC, 1e-300)
            w = m.wf; i, bnd = m.interior, m.boundary
            gam = np.empty(m.nface); gam[i] = w[i] * Dcell[m.owner[i]] + (1 - w[i]) * Dcell[m.neigh[i]]; gam[bnd] = Dcell[m.owner[bnd]]
            if k not in self._rk_lap or not np.array_equal(gam, self._rk_lap[k][0]):
                Ap, Ap_rhs = laplacian(m, gam, self.bc_p.kind); self._rk_lap[k] = (gam.copy(), Ap, Ap_rhs, SolveCache(backend=self._mcache.backend))
            _, Ap, Ap_rhs, pc = self._rk_lap[k]
            src = divergence(m, Fs)
            pp, _h = solve_poisson(m, self.grad, Ap, Ap_rhs, src, np.zeros(m.nbface) if self.p_singular else self.bc_p.value,
                                   bkind=self.bc_p.kind, cache=pc, tol=1e-9, max_outer=self.n_nonorth, singular=self.p_singular)
            gpp = self.grad(pp, BC(m, self.bc_p.kind, np.zeros(m.nbface)).effective(pp))
            u = us - Dcell * gpp[:, 0]; v = vs - Dcell * gpp[:, 1]
            F = Fs - self._pressure_flux(gam, pp, gpp)
            p = p + pp                  # Dcell already carries dt_k, so pp is in pressure units
            N_prev = Nk
        self.u_old, self.v_old = self.u.copy(), self.v.copy()
        self.Ff_prev = self.Ff
        self.u, self.v, self.Ff, self.p = u, v, F, p
        self.Ff_old = None                          # the BDF2 path's Choi pair is not maintained under RK3
        self.time += dt; self.nstep += 1

    # -----------------------------------------------------------------------------------------
    def _rhie_chow(self, u, v, aP, aC, p, gp):
        """Face flux from cell velocities, with time-step-independent pressure damping."""
        m = self.m
        i, b = m.interior, m.boundary
        o, n = m.owner, m.neigh
        w = m.wf

        # plain interpolation of the cell velocity to faces, dotted with the area vector
        ubar = np.empty(m.nface); vbar = np.empty(m.nface)
        ubar[i] = w[i]*u[o[i]] + (1-w[i])*u[n[i]]
        vbar[i] = w[i]*v[o[i]] + (1-w[i])*v[n[i]]
        ub = self.bc_u.effective(u); vb = self.bc_v.effective(v)
        ubar[b] = ub[m.bface_index[b]]; vbar[b] = vb[m.bface_index[b]]
        Fbar = (ubar * m.normal[:, 0] + vbar * m.normal[:, 1]) * m.span

        # D_f = (V/a_P) interpolated to the face. THIS ONE STAYS a_P, unlike the pressure
        # equation's SIMPLEC coefficient: it is not a free stabilisation parameter but comes
        # straight out of the momentum equation's algebraic form,
        #     u_P = H_P/a_P - (V/a_P)(grad p)_P,
        # which is what makes the H/a_P terms cancel when the interpolated cell velocity is
        # subtracted from the face velocity. Substituting a_C here (tried, on the grounds that
        # V/a_P collapses to 3.17e-04 against the pressure equation's 2.93e-02 when diffusion
        # dominates) over-damps and blows up: L2(u) 4.0e+27 by n=32, then NaN.
        Dcell = self.rc_scale * m.vol / np.maximum(aP, 1e-300)
        D = np.empty(m.nface)
        D[i] = w[i]*Dcell[o[i]] + (1-w[i])*Dcell[n[i]]
        D[b] = Dcell[o[b]]

        # Compact minus wide pressure gradient -- the checkerboard is invisible to the wide one.
        #
        # BOTH MUST BE THE SAME DIRECTIONAL DERIVATIVE. The earlier form took the compact part as
        # (p_N - p_P)/|d|, which is grad p . dhat, and the wide part as gpf . nhat. Those agree
        # only where d is parallel to S, so the difference was contaminated by an O(1) directional
        # mismatch on any non-orthogonal face. It was invisible on the uniform mesh (interior
        # faces there are exactly orthogonal) and surfaces the moment the mesh is clustered:
        # `orth` min falls from 0.894 to 0.149 at cluster=2.0.
        #
        # The fix is to build the compact term with the same over-relaxed decomposition the
        # Laplacian and `_pressure_flux` already use, S_f = E_f + T_f, so the damping is a genuine
        # face-normal flux difference. On an orthogonal face E_f = S_f and T_f = 0, which reduces
        # this EXACTLY to the old expression -- verified bit-identical at cluster=0.
        pbv = self.bc_p.effective(p)
        gpw = gp if self.grad_rc is None else self.grad_rc(p, pbv)
        gpf = np.empty((m.nface, 2))
        gpf[i] = w[i, None]*gpw[o[i]] + (1-w[i])[:, None]*gpw[n[i]]
        gpf[b] = gpw[o[b]]
        dp_compact = np.zeros(m.nface)                  # |E_f| dp/d|d|  + T_f . grad_f
        dp_compact[i] = m.ef_over_d[i] * (p[n[i]] - p[o[i]])
        dp_compact[b] = m.ef_over_d[b] * (pbv[m.bface_index[b]] - p[o[b]])
        dp_compact = dp_compact + (m.Tf * gpf).sum(axis=1)
        dp_wide = (gpf * m.normal).sum(axis=1)          # both now are grad p . S_f
        area = m.span

        # NO RHIE-CHOW DAMPING ON A PRESCRIBED-VELOCITY BOUNDARY. There the flux IS the
        # boundary condition, u_wall . S, and nothing may be added to it. Damping those faces
        # injects a spurious NET mass flux, which makes the all-Neumann pressure problem
        # incompatible; the mean-subtraction that restores compatibility then smears that error
        # across every cell. It presented as a converged Poisson solve (outer delta 1e-15) whose
        # operator nevertheless missed its own source by 1.8e-03, and as max|div F| climbing to
        # 4e-03 over ten steps with every field still looking plausible.
        damp = D * (dp_compact - dp_wide) * area
        # A face's flux is prescribed when the velocity component ALONG ITS NORMAL is Dirichlet --
        # not only when both components are. A symmetry plane (u Neumann, v Dirichlet = 0 on a
        # y-normal face) prescribes its normal flux exactly as a wall does, and damping it would
        # inject the same spurious net mass as bug (18) in the record. Axis-aligned mixed faces are
        # handled here (the cylinder case's Freestream); an oblique face with mixed components is
        # not a well-posed prescription and falls back to requiring both.
        kd_u = self.bc_u.kind == DIRICHLET; kd_v = self.bc_v.kind == DIRICHLET
        Sb = m.normal[m.bfaces]; ax = np.abs(Sb[:, 0]) > 1e-9 * np.hypot(*Sb.T); ay = np.abs(Sb[:, 1]) > 1e-9 * np.hypot(*Sb.T)
        fixed_u = np.zeros(m.nface, dtype=bool)
        fixed_u[m.bfaces] = (kd_u & kd_v) | (kd_u & ~ay) | (kd_v & ~ax)
        damp[fixed_u] = 0.0
        F = Fbar - damp

        # TIME-STEP-INDEPENDENT TERM. Without it D carries 1/dt through a_P, so the damping
        # scales with the time step and the converged answer moves as dt is refined.
        #
        # BOTH HISTORY TERMS ARE FROM THE PREVIOUS TIME STEP, never from the previous CORRECTOR.
        # Updating them inside this function made the second corrector difference against the
        # first one instead, which is not a transient term at all -- it injected divergence the
        # projection had just removed, and max|div F| climbed from 1.4e-13 on step 1 to 4e-03 by
        # step 10 while every field still looked plausible.
        if self.Ff_old is not None and self.Fbar_old is not None:
            tr = (D / self.dt) * (self.Ff_old - self.Fbar_old)
            tr[fixed_u] = 0.0                      # same reason as the damping above
            F = F + tr
        self.Fbar_cur = Fbar
        return F

    # -----------------------------------------------------------------------------------------
    def init_flux(self):
        """Face flux from the current cell velocity (plain interpolation, boundary values from the BCs).

        Every run used to start with Ff = 0, so the FIRST step convected nothing: a one-off O(dt)
        error with an O(|u.grad u|) coefficient, invisible in windowed statistics (T8, T9) and in
        runs from rest, but it capped every temporal-order test on a non-trivial initial field at
        first order (record section 50: the advected Taylor-Green showed err ~ 1.36 dt for BDF2 and
        RK3 alike, mesh-independent). Called once, lazily, by step()/step_rk3() when nstep == 0 and
        Ff is identically zero while (u, v) is not; a run from rest is unchanged. Set Ff yourself
        (e.g. from a restart file) and it is left alone."""
        m = self.m; i, b = m.interior, m.boundary; w = m.wf
        ub = self.bc_u.effective(self.u); vb = self.bc_v.effective(self.v)
        uf = np.empty(m.nface); vf = np.empty(m.nface)
        uf[i] = w[i] * self.u[m.owner[i]] + (1 - w[i]) * self.u[m.neigh[i]]; vf[i] = w[i] * self.v[m.owner[i]] + (1 - w[i]) * self.v[m.neigh[i]]
        uf[b] = ub[m.bface_index[b]]; vf[b] = vb[m.bface_index[b]]
        self.Ff = (uf * m.normal[:, 0] + vf * m.normal[:, 1]) * m.span
        self._flux_init = True

    def _maybe_init_flux(self):
        if not self._flux_init and self.nstep == 0 and not np.any(self.Ff) and (np.any(self.u) or np.any(self.v)):
            self.init_flux()
        self._flux_init = True

    def step(self):
        self._maybe_init_flux()
        if self.time_scheme == "rk3":
            return self.step_rk3()
        m = self.m
        Fn = self.Ff
        u_n, v_n = self.u, self.v                   # time level n, fixed through the outer iterations
        u, v, F = u_n, v_n, Fn
        self.outer_change = []
        for outer in range(self.n_outer):
            if outer == 0:
                self._Fconv = (2.0 * Fn - self.Ff_prev) if (self.conv_flux_extrap and self.Ff_prev is not None) else None
                guess_u = guess_v = None
            else:
                # PIMPLE-style outer iteration: re-linearise convection about the latest corrected
                # flux and re-solve momentum with the latest pressure; the time level stays u^n.
                self._Fconv = F
                guess_u, guess_v = u, v
            pb = self.bc_p.effective(self.p)
            G = self.grad if self.grad_p is None else self.grad_p
            gp = G(self.p, pb)
            if self.p_neumann_extrap:
                # Neumann faces: p_b = p_P + grad p_P . (x_b - x_P) instead of the owner value (zeroth order).
                # One fixed-point pass with the gradient just computed. Experiment on the wall-excited
                # pressure checkerboard of non-bipartite meshes (record section 36/40); boundary term only,
                # the interior operator and its duality with the divergence are untouched.
                m_ = self.m; bf = m_.bfaces; neu = self.bc_p.kind == NEUMANN; bo = m_.owner[bf][neu]
                pb = pb.copy(); pb[neu] = self.p[bo] + ((m_.fcentre[bf][neu] - m_.centroid[bo]) * gp[bo]).sum(axis=1)
                gp = G(self.p, pb)

            u_star, aP, aC = self._momentum(0, u_n, self.u_old, self.bc_u, gp[:, 0], phi_guess=guess_u)
            v_star, _, _ = self._momentum(1, v_n, self.v_old, self.bc_v, gp[:, 1], phi_guess=guess_v)
            u_prev, v_prev = u, v
            u, v = u_star, v_star

            for _ in range(self.n_corr):
                F = self._rhie_chow(u, v, aP, aC, self.p, gp)
                # pressure equation: div( (V/a_C) grad p' ) = div F, a_C the SIMPLEC coefficient.
                # Rhie-Chow above keeps V/a_P -- that one comes from the momentum equation's own
                # algebraic form, not from the correction, so the two coefficients differ on purpose.
                Dcell = m.vol / np.maximum(aC, 1e-300)
                w = m.wf; i, b = m.interior, m.boundary
                gam = np.empty(m.nface)
                gam[i] = w[i]*Dcell[m.owner[i]] + (1-w[i])*Dcell[m.neigh[i]]
                gam[b] = Dcell[m.owner[b]]
                # gam depends only on a_C, which is fixed for a fixed dt and viscosity; rebuilding the
                # sparse operator every corrector was 8% of a step for nothing (profiled).
                if self._gam_cache is None or not np.array_equal(gam, self._gam_cache):
                    self._Ap, self._Ap_rhs = laplacian(m, gam, self.bc_p.kind); self._gam_cache = gam.copy()
                Ap, Ap_rhs = self._Ap, self._Ap_rhs
                src = divergence(m, F)
                pp, _hist = solve_poisson(m, self.grad, Ap, Ap_rhs, src,
                                          np.zeros(m.nbface) if self.p_singular else self.bc_p.value,
                                          bkind=self.bc_p.kind, cache=self._pcache,
                                          tol=1e-9, max_outer=self.n_nonorth,
                                          singular=self.p_singular)
                gpp = self.grad(pp, BC(m, self.bc_p.kind, np.zeros(m.nbface)).effective(pp))
                # correct the cell velocities and the face flux
                u = u - Dcell * gpp[:, 0]
                v = v - Dcell * gpp[:, 1]
                Fp = self._pressure_flux(gam, pp, gpp)
                F = F - Fp
                self.p = self.p + pp
                pb = self.bc_p.effective(self.p)
                gp = self.grad(self.p, pb)

            self.outer_change.append(float(max(np.abs(u - u_prev).max(), np.abs(v - v_prev).max())) if outer > 0 else np.nan)

        self.u_old, self.v_old = u_n.copy(), v_n.copy()
        self.Ff_prev = Fn                          # F^n becomes F^{n-1} for the next step's extrapolation
        self.u, self.v, self.Ff = u, v, F
        # history advances ONCE per step
        self.Ff_old = F.copy()
        self.Fbar_old = self.Fbar_cur.copy()
        self.time += self.dt
        self.nstep += 1
        return float(np.abs(divergence(m, F)).max())

    def _pressure_flux(self, gam, pp, gpp):
        """The face flux implied by the pressure correction.

        MUST INCLUDE THE NON-ORTHOGONAL PART. The Poisson operator is orthogonal-plus-deferred;
        subtracting only the orthogonal piece leaves exactly the deferred piece behind, so the
        corrected flux is not the one the solve made solenoidal. On an orthogonal mesh T_f = 0
        and the omission is invisible -- measured max|div F| 2.4e-15 there against 3.9e-04 on a
        skewed mesh, which is the kind of gap a nice test mesh hides entirely.
        """
        m = self.m
        i, b = m.interior, m.boundary
        w = m.wf
        F = np.zeros(m.nface)
        coef = gam * m.ef_over_d * m.span
        F[i] = coef[i] * (pp[m.neigh[i]] - pp[m.owner[i]])
        gf = np.empty((m.nface, 2))
        gf[i] = w[i, None] * gpp[m.owner[i]] + (1 - w[i])[:, None] * gpp[m.neigh[i]]
        gf[b] = gpp[m.owner[b]]
        cross = gam * (m.Tf * gf).sum(axis=1) * m.span
        cross[b] = 0.0                       # Neumann pressure faces carry no flux at all
        return F + cross
