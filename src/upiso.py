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
                 n_corr=2, n_nonorth=2, body_force=None, backend="scipy", convect=True):
        self.m = mesh
        self.nu, self.dt = float(nu), float(dt)
        self.bc_u, self.bc_v, self.bc_p = bc_u, bc_v, bc_p
        self.scheme = scheme
        self.convect = bool(convect)   # False -> Stokes, which isolates the pressure coupling
        self.n_corr, self.n_nonorth = n_corr, n_nonorth
        self.fx = np.zeros(mesh.ncell) if body_force is None else np.asarray(body_force[0])
        self.fy = np.zeros(mesh.ncell) if body_force is None else np.asarray(body_force[1])

        self.grad = Gradient(mesh)
        self.u = np.zeros(mesh.ncell)
        self.v = np.zeros(mesh.ncell)
        self.p = np.zeros(mesh.ncell)
        self.Ff = np.zeros(mesh.nface)
        self.u_old = self.v_old = None            # BDF2 history
        self.Ff_old = None                        # previous face flux, for time-independent RC
        self.Fbar_old = None                      # and its plain-interpolation counterpart
        self.ubar_old = None
        self.time, self.nstep = 0.0, 0

        self._mcache = SolveCache(backend=backend)
        self._pcache = SolveCache(backend=backend)
        # the viscous operator never changes, so assemble it once
        self.Lu, self.Lu_rhs = laplacian(mesh, np.full(mesh.nface, self.nu), bc_u.kind)
        self.Lv, self.Lv_rhs = laplacian(mesh, np.full(mesh.nface, self.nu), bc_v.kind)
        # pressure operator: coefficient is 1/a_P interpolated to faces, set each step
        self.p_singular = bool((bc_p.kind == NEUMANN).all())

    # -----------------------------------------------------------------------------------------
    def _momentum(self, comp, phi, phi_old, bc, gp_comp):
        """Assemble and solve one momentum component. Returns (phi_star, a_P)."""
        m = self.m
        if self.convect:
            C, C_rhs = convection(m, self.Ff, bc.kind, scheme=self.scheme)
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
        x = self._mcache.solve(A.tocsr(), b, x0=phi, symmetric=False, rtol=1e-10)
        aP = np.asarray(A.tocsr().diagonal())
        return x, aP

    # -----------------------------------------------------------------------------------------
    def _rhie_chow(self, u, v, aP, p, gp):
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

        # D_f = (V/a_P) interpolated to the face, times the face area factor
        Dcell = m.vol / np.maximum(aP, 1e-300)
        D = np.empty(m.nface)
        D[i] = w[i]*Dcell[o[i]] + (1-w[i])*Dcell[n[i]]
        D[b] = Dcell[o[b]]

        # compact minus wide pressure gradient -- the checkerboard is invisible to the wide one
        pbv = self.bc_p.effective(p)
        dp_compact = np.zeros(m.nface)
        dp_compact[i] = (p[n[i]] - p[o[i]]) / m.dmag[i]
        dp_compact[b] = (pbv[m.bface_index[b]] - p[o[b]]) / m.dmag[b]
        gpf = np.empty((m.nface, 2))
        gpf[i] = w[i, None]*gp[o[i]] + (1-w[i])[:, None]*gp[n[i]]
        gpf[b] = gp[o[b]]
        nhat = m.normal / np.maximum(np.hypot(*m.normal.T), 1e-300)[:, None]
        dp_wide = (gpf * nhat).sum(axis=1)
        area = np.hypot(*m.normal.T) * m.span

        # NO RHIE-CHOW DAMPING ON A PRESCRIBED-VELOCITY BOUNDARY. There the flux IS the
        # boundary condition, u_wall . S, and nothing may be added to it. Damping those faces
        # injects a spurious NET mass flux, which makes the all-Neumann pressure problem
        # incompatible; the mean-subtraction that restores compatibility then smears that error
        # across every cell. It presented as a converged Poisson solve (outer delta 1e-15) whose
        # operator nevertheless missed its own source by 1.8e-03, and as max|div F| climbing to
        # 4e-03 over ten steps with every field still looking plausible.
        damp = D * (dp_compact - dp_wide) * area
        fixed_u = np.zeros(m.nface, dtype=bool)
        fixed_u[m.bfaces] = (self.bc_u.kind == DIRICHLET) & (self.bc_v.kind == DIRICHLET)
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
        pb = self.bc_p.effective(self.p)
        gp = self.grad(self.p, pb)

        u_star, aP = self._momentum(0, self.u, self.u_old, self.bc_u, gp[:, 0])
        v_star, _ = self._momentum(1, self.v, self.v_old, self.bc_v, gp[:, 1])

        self.u_old, self.v_old = self.u.copy(), self.v.copy()
        u, v = u_star, v_star

        for _ in range(self.n_corr):
            F = self._rhie_chow(u, v, aP, self.p, gp)
            # pressure equation: div( (V/a_P) grad p' ) = div F
            Dcell = m.vol / np.maximum(aP, 1e-300)
            w = m.wf; i, b = m.interior, m.boundary
            gam = np.empty(m.nface)
            gam[i] = w[i]*Dcell[m.owner[i]] + (1-w[i])*Dcell[m.neigh[i]]
            gam[b] = Dcell[m.owner[b]]
            Ap, Ap_rhs = laplacian(m, gam, self.bc_p.kind)
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
