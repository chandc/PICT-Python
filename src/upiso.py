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
        self.Ff_prev = None                       # F^{n-1}
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
    def _momentum(self, comp, phi, phi_old, bc, gp_comp):
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
        pb = bc.effective(phi)
        gphi = self.grad(phi, pb)
        b = (rhs_t + (self.fx if comp == 0 else self.fy) - gp_comp) * m.vol
        b = b - C_rhs(phi, pb, gphi) + L_rhs(gphi, pb)
        Ac = A.tocsr()
        x = self._mcache.solve(Ac, b, x0=phi, symmetric=False, rtol=1e-10)
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
    def step(self):
        m = self.m
        Fn = self.Ff
        self._Fconv = (2.0 * Fn - self.Ff_prev) if (self.conv_flux_extrap and self.Ff_prev is not None) else None
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

        u_star, aP, aC = self._momentum(0, self.u, self.u_old, self.bc_u, gp[:, 0])
        v_star, _, _ = self._momentum(1, self.v, self.v_old, self.bc_v, gp[:, 1])

        self.u_old, self.v_old = self.u.copy(), self.v.copy()
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
