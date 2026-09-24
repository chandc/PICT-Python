"""2.5D incompressible solver: the unstructured collocated FV scheme in the x-y plane, Fourier in a
periodic span z (LES plan L2). One face-based 2D problem per spanwise mode; the nonlinear term in
physical space on 3/2-padded planes (dealiased); RK3 / Crank-Nicolson per stage with a pressure
projection per stage (the L1a integrator, `PISO.step_rk3`, lifted mode by mode).

Layout: every cell field is (ncell, nz) over the planes z_j = j Lz/nz, every face field (nface, nz).
Spectral arrays are (ncell, nk) complex with nk = nz//2 + 1 from `np.fft.rfft` along the last axis;
the Nyquist mode is kept at zero. Mode 0 is the 2D solver exactly (a z-independent field never
leaves it), which is the first thing the tests check.

Per stage k (dt_k = (alpha_k + beta_k) dt), for each of u, v, w and each mode:
    (V/dt) phi* - beta (L_2D - nu kz^2 V) phi*  =  V phi^k/dt - gamma N^k - zeta N^{k-1}
                                                   + alpha (L_2D phi^k - nu kz^2 V phi^k)
                                                   + (alpha+beta) V (f - grad_3D p^k)
    N = sum_f F_f phi_f  +  V d(w phi)/dz          (central + skewness in the plane, spectral in z,
                                                    products on 3nz/2 planes and truncated)
then Rhie-Chow per mode with D = dt_k V/a_P (no transient term, section 50 of the record),
    [Lap(gam) - kz^2 D V] pp  =  V div F* + i kz V w*,
    u = u* - D grad pp,  w = w* - D i kz pp,  F = F* - F_pp,  p += pp.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.uops import Gradient, laplacian, divergence, DIRICHLET, NEUMANN
from src.ugrad import GGDualGradient
from src.upiso import BC, PISO


class PISO25:
    RK3_GAMMA = PISO.RK3_GAMMA
    RK3_ZETA = PISO.RK3_ZETA
    RK3_ALPHA = PISO.RK3_ALPHA

    def __init__(self, mesh, nz, Lz, nu, dt, bc_u, bc_v, bc_w, bc_p, body_force=None,
                 n_nonorth=1, n_inner=2, dealias=True, convect=True):
        m = self.m = mesh
        self.nz, self.Lz = int(nz), float(Lz)
        self.nu, self.dt = float(nu), float(dt)
        self.bc_u, self.bc_v, self.bc_w, self.bc_p = bc_u, bc_v, bc_w, bc_p
        self.n_nonorth, self.n_inner = int(n_nonorth), int(n_inner)
        self.convect = bool(convect)
        self.nk = self.nz // 2 + 1
        self.kz = 2.0 * np.pi / self.Lz * np.arange(self.nk)          # rfft wavenumbers
        self.M = (3 * self.nz) // 2 if dealias else self.nz             # planes for the products
        self.kM = 2.0 * np.pi / self.Lz * np.arange(self.M // 2 + 1)
        self.z = np.arange(self.nz) * self.Lz / self.nz
        self.fx = np.zeros(m.ncell) if body_force is None else np.asarray(body_force[0], float)
        self.fy = np.zeros(m.ncell) if body_force is None else np.asarray(body_force[1], float)
        self.fz = np.zeros(m.ncell) if body_force is None or len(body_force) < 3 else np.asarray(body_force[2], float)

        self.u = np.zeros((m.ncell, self.nz)); self.v = np.zeros_like(self.u); self.w = np.zeros_like(self.u)
        self.p = np.zeros_like(self.u); self.Ff = np.zeros((m.nface, self.nz))
        self.grad = Gradient(m)
        self.grad_p = GGDualGradient(m)
        self.Lu, self.Lu_rhs = laplacian(m, np.full(m.nface, self.nu), bc_u.kind)
        self.Lv, self.Lv_rhs = laplacian(m, np.full(m.nface, self.nu), bc_v.kind)
        self.Lw, self.Lw_rhs = laplacian(m, np.full(m.nface, self.nu), bc_w.kind)
        self.p_singular = bool((bc_p.kind == NEUMANN).all())
        # owner(+)/neighbour(-) scatter of a face quantity into cells
        i = m.interior; o, n = m.owner, m.neigh
        self.Sc = (sp.coo_matrix((np.ones(m.nface), (o, np.arange(m.nface))), shape=(m.ncell, m.nface))
                   - sp.coo_matrix((np.ones(int(i.sum())), (n[i], np.flatnonzero(i))), shape=(m.ncell, m.nface))).tocsr()
        w = m.wf
        self.dskew = np.zeros((m.nface, 2))
        self.dskew[i] = m.fcentre[i] - (m.centroid[o[i]] + (1.0 - w[i])[:, None] * m.dcc[i])
        self.dxb = m.fcentre[m.bfaces] - m.centroid[o[m.bfaces]]
        kd_u = bc_u.kind == DIRICHLET; kd_v = bc_v.kind == DIRICHLET
        Sb = m.normal[m.bfaces]; ax = np.abs(Sb[:, 0]) > 1e-9 * np.hypot(*Sb.T); ay = np.abs(Sb[:, 1]) > 1e-9 * np.hypot(*Sb.T)
        self.fixed_u = np.zeros(m.nface, dtype=bool)
        self.fixed_u[m.bfaces] = (kd_u & kd_v) | (kd_u & ~ay) | (kd_v & ~ax)
        self._mom = {}          # (stage, comp-key, mode) -> (splu, aP, aC)
        self._pois = {}         # (stage, mode) -> (Ap_k, Ap_rhs, splu, gam)
        self.time, self.nstep = 0.0, 0
        self._N_prev = None
        self._flux_init = False
        # Subgrid model (LES plan L3): "none", "wale" or "smagorinsky". The eddy-viscosity term
        # div(nu_t (grad u + grad u^T)) is EXPLICIT, evaluated with the convection on the padded
        # planes (nu_t x grad u is a product, so it is dealiased with the rest) and carried by the
        # same RK3 gamma/zeta weights; the molecular viscosity stays Crank-Nicolson implicit. A
        # z-varying coefficient cannot sit inside the per-mode implicit solve, and the explicit
        # limit nu_t dt / h^2 is not binding for a wall-resolved LES. `nu_t` holds the last
        # evaluation on the nz planes for reporting; `sgs_nu_field` (ncell, nz), if set, is used
        # instead of a closure (manufactured tests).
        self.sgs_model = "none"
        self.sgs_kw = {}
        self.sgs_nu_field = None
        self.nu_t = np.zeros((m.ncell, self.nz))
        self._Tf_span = m.Tf * m.span
        self._bdir = {id(bc_u): bc_u.kind == DIRICHLET, id(bc_v): bc_v.kind == DIRICHLET, id(bc_w): bc_w.kind == DIRICHLET}

    # ---------------------------------------------------------------- spectral helpers
    def fft(self, f):
        fh = np.fft.rfft(f, axis=-1)
        if self.nz % 2 == 0: fh[..., -1] = 0.0
        return fh

    def ifft(self, fh):
        return np.fft.irfft(fh, n=self.nz, axis=-1)

    def pad(self, f):
        """(…, nz) -> (…, M) planes, same Fourier content (3/2 rule)."""
        if self.M == self.nz: return f
        fh = self.fft(f); out = np.zeros(f.shape[:-1] + (self.M // 2 + 1,), complex); out[..., :self.nk] = fh
        return np.fft.irfft(out, n=self.M, axis=-1) * (self.M / self.nz)

    def trunc(self, f):
        """(…, M) -> (…, nz) planes, modes above nk dropped."""
        if self.M == self.nz: return f
        fh = np.fft.rfft(f, axis=-1)[..., :self.nk]
        if self.nz % 2 == 0: fh[..., -1] = 0.0
        return np.fft.irfft(fh, n=self.nz, axis=-1) * (self.nz / self.M)

    def ddz(self, f, nplanes=None):
        n = f.shape[-1]; k = self.kM if n == self.M else self.kz
        fh = np.fft.rfft(f, axis=-1) * (1j * k)
        if n % 2 == 0: fh[..., -1] = 0.0
        return np.fft.irfft(fh, n=n, axis=-1)

    # ---------------------------------------------------------------- boundary values
    def beff(self, bc, phi):
        """Boundary face values for a (ncell, n) physical field: Dirichlet value, else owner's."""
        out = np.repeat(bc.value[:, None], phi.shape[1], axis=1)
        neu = bc.kind == NEUMANN
        if neu.any(): out[neu] = phi[self.m.owner[self.m.bfaces][neu]]
        return out

    def beff_hat(self, bc, phih):
        """Same for a (ncell, nk) spectral field: the Dirichlet value lives in mode 0 only."""
        out = np.zeros((self.m.nbface, phih.shape[1]), complex); out[:, 0] = bc.value * self.nz
        neu = bc.kind == NEUMANN
        if neu.any(): out[neu] = phih[self.m.owner[self.m.bfaces][neu]]
        return out

    # ---------------------------------------------------------------- operators on planes
    def init_flux(self):
        m = self.m; i, b = m.interior, m.boundary; w = m.wf
        ub = self.beff(self.bc_u, self.u); vb = self.beff(self.bc_v, self.v)
        uf = np.empty((m.nface, self.nz)); vf = np.empty_like(uf)
        uf[i] = w[i, None] * self.u[m.owner[i]] + (1 - w[i, None]) * self.u[m.neigh[i]]
        vf[i] = w[i, None] * self.v[m.owner[i]] + (1 - w[i, None]) * self.v[m.neigh[i]]
        uf[b] = ub[m.bface_index[b]]; vf[b] = vb[m.bface_index[b]]
        self.Ff = (uf * m.normal[:, 0, None] + vf * m.normal[:, 1, None]) * m.span
        self._flux_init = True

    def conv(self, phi, F, wz, bc):
        """Volume-integrated convection of phi on the padded planes: sum_f F_f phi_f + V d(wz phi)/dz."""
        m = self.m; i, b = m.interior, m.boundary; w = m.wf; o, n = m.owner, m.neigh
        pb = self.beff(bc, phi); g = self.grad(phi, pb)                        # (ncell, 2, M)
        phif = np.empty((m.nface, phi.shape[1]))
        gf = w[i, None, None] * g[o[i]] + (1 - w[i])[:, None, None] * g[n[i]]
        phif[i] = w[i, None] * phi[o[i]] + (1 - w[i, None]) * phi[n[i]] + (gf * self.dskew[i][:, :, None]).sum(axis=1)
        ob = o[m.bfaces]
        outflow = phi[ob] + (g[ob] * self.dxb[:, :, None]).sum(axis=1)
        phif[m.bfaces] = np.where(F[m.bfaces] >= 0.0, outflow, pb[m.bface_index[m.bfaces]])
        N = self.Sc @ (F * phif)
        if self.nz > 1: N = N + m.vol[:, None] * self.ddz(wz * phi)
        return N

    def sgs_term(self, u, v, w):
        """Volume-integrated  div( nu_t (grad u_i + (grad u)^T_i) )  for i = x, y, z on the given planes.

        In-plane faces: nu_f [ E_f/d (phi_N - phi_P) + T_f . grad_f phi ] (the Laplacian's own
        over-relaxed split) plus the transpose  nu_f (d u_j / d x_i)_f S_j ; spanwise:
        V d/dz [ nu_t (d u_i/dz + d w/d x_i) ] spectral. Dirichlet faces use the boundary value in
        the compact part; Neumann faces carry no flux. Returns (T_u, T_v, T_w) and sets self.nu_t
        when the planes are the solver's own."""
        import src.usgs as usgs
        m = self.m; i, b = m.interior, m.boundary; o, n = m.owner, m.neigh; wf = m.wf
        g = usgs.velocity_gradient(self, u, v, w)                       # g[i][j] (ncell, n)
        if self.sgs_nu_field is not None:
            nut = self.sgs_nu_field if self.sgs_nu_field.shape[1] == u.shape[1] else self.pad(self.sgs_nu_field)
        else:
            nut = usgs.eddy_viscosity(self, self.sgs_model, g=g, **self.sgs_kw)
        self.nu_t = nut if u.shape[1] == self.nz else self.trunc(nut)      # reported on the nz planes
        nuf = np.empty((m.nface, u.shape[1])); nuf[i] = wf[i, None] * nut[o[i]] + (1 - wf[i, None]) * nut[n[i]]; nuf[b] = nut[o[b]]
        out = []
        for comp, (phi, bc) in enumerate(((u, self.bc_u), (v, self.bc_v), (w, self.bc_w))):
            gphi = g[comp]                                               # [d phi/dx, d phi/dy, d phi/dz]
            gfx = np.empty((m.nface, u.shape[1])); gfy = np.empty_like(gfx)
            gfx[i] = wf[i, None] * gphi[0][o[i]] + (1 - wf[i, None]) * gphi[0][n[i]]; gfy[i] = wf[i, None] * gphi[1][o[i]] + (1 - wf[i, None]) * gphi[1][n[i]]
            gfx[b] = gphi[0][o[b]]; gfy[b] = gphi[1][o[b]]
            flux = np.zeros((m.nface, u.shape[1]))
            flux[i] = m.ef_over_d[i, None] * (phi[n[i]] - phi[o[i]]) + self._Tf_span[i, 0, None] / m.span * gfx[i] + self._Tf_span[i, 1, None] / m.span * gfy[i]
            pb = self.beff(bc, phi); dirb = self._bdir[id(bc)]
            bd = m.bfaces[dirb]
            flux[bd] = m.ef_over_d[bd, None] * (pb[m.bface_index[bd]] - phi[o[bd]])
            # transpose part: (d u_j / d x_comp)_f S_j  with j = x, y (in-plane faces), j = z below
            tx = np.empty_like(gfx); ty = np.empty_like(gfx)
            gu_c, gv_c = g[0][comp], g[1][comp]
            tx[i] = wf[i, None] * gu_c[o[i]] + (1 - wf[i, None]) * gu_c[n[i]]; ty[i] = wf[i, None] * gv_c[o[i]] + (1 - wf[i, None]) * gv_c[n[i]]
            tx[b] = gu_c[o[b]]; ty[b] = gv_c[o[b]]
            trans = tx * m.normal[:, 0, None] + ty * m.normal[:, 1, None]
            neu = m.bfaces[~dirb]; trans[neu] = 0.0
            T = self.Sc @ (nuf * (flux * m.span + trans * m.span))
            if self.nz > 1: T = T + m.vol[:, None] * self.ddz(nut * (gphi[2] + g[2][comp]))
            out.append(T)
        return out

    def nonlinear(self, u, v, w, F):
        """(N_u, N_v, N_w) on the nz planes, products formed on M planes and truncated."""
        if not self.convect:
            z = np.zeros_like(u); return z, z, z
        up, vp, wp, Fp = self.pad(u), self.pad(v), self.pad(w), self.pad(F)
        Nu, Nv, Nw = self.conv(up, Fp, wp, self.bc_u), self.conv(vp, Fp, wp, self.bc_v), self.conv(wp, Fp, wp, self.bc_w)
        if self.sgs_model != "none" or self.sgs_nu_field is not None:
            Tu, Tv, Tw = self.sgs_term(up, vp, wp); Nu = Nu - Tu; Nv = Nv - Tv; Nw = Nw - Tw   # N is what the rhs SUBTRACTS
        return self.trunc(Nu), self.trunc(Nv), self.trunc(Nw)

    def diffusion(self, phi, L, L_rhs, bc):
        """Full in-plane viscous term (orthogonal + cross) of a (ncell, n) field; the z part is spectral."""
        pb = self.beff(bc, phi); return L @ phi + L_rhs(self.grad(phi, pb), pb)

    # ---------------------------------------------------------------- per-mode pieces
    def _mom_solver(self, stage, comp, k, L, be):
        key = (stage, comp, k)
        if key not in self._mom:
            m = self.m
            A = (sp.diags(m.vol / self.dt) - be * L + sp.diags(be * self.nu * self.kz[k] ** 2 * m.vol)).tocsc()
            aP = np.asarray(A.diagonal()); aC = np.maximum(np.asarray(A.sum(axis=1)).ravel(), m.vol / self.dt)
            self._mom[key] = (spla.splu(A), aP, aC)
        return self._mom[key]

    def _pois_solver(self, stage, k, Dcell):
        key = (stage, k)
        if key not in self._pois:
            m = self.m; w = m.wf; i, b = m.interior, m.boundary
            gam = np.empty(m.nface); gam[i] = w[i] * Dcell[m.owner[i]] + (1 - w[i]) * Dcell[m.neigh[i]]; gam[b] = Dcell[m.owner[b]]
            Ap, Ap_rhs = laplacian(m, gam, self.bc_p.kind)
            Ak = (Ap - sp.diags(self.kz[k] ** 2 * Dcell * m.vol)).tocsc()
            if k == 0 and self.p_singular:
                # all-Neumann mode 0: pin cell 0. Exact for a compatible right-hand side (its mean
                # is removed), and the solution's mean is removed afterwards.
                Ak = Ak.tolil(); Ak[0, :] = 0.0; Ak[0, 0] = 1.0; Ak = Ak.tocsc()
            self._pois[key] = (Ak, Ap_rhs, spla.splu(Ak), gam)
        return self._pois[key]

    def _rhie_chow(self, uh, vh, ph, gph, Dcell):
        """Spectral Rhie-Chow: (ncell, nk) complex fields, Dcell (ncell, nk) real. Returns F* (nface, nk)."""
        m = self.m; i, b = m.interior, m.boundary; o, n = m.owner, m.neigh; w = m.wf
        ub = self.beff_hat(self.bc_u, uh); vb = self.beff_hat(self.bc_v, vh)
        uf = np.empty((m.nface, self.nk), complex); vf = np.empty_like(uf)
        uf[i] = w[i, None] * uh[o[i]] + (1 - w[i, None]) * uh[n[i]]; vf[i] = w[i, None] * vh[o[i]] + (1 - w[i, None]) * vh[n[i]]
        uf[b] = ub[m.bface_index[b]]; vf[b] = vb[m.bface_index[b]]
        Fbar = (uf * m.normal[:, 0, None] + vf * m.normal[:, 1, None]) * m.span
        D = np.empty((m.nface, self.nk)); D[i] = w[i, None] * Dcell[o[i]] + (1 - w[i, None]) * Dcell[n[i]]; D[b] = Dcell[o[b]]
        pbv = self.beff_hat(self.bc_p, ph)
        gpf = np.empty((m.nface, 2, self.nk), complex)
        gpf[i] = w[i, None, None] * gph[o[i]] + (1 - w[i])[:, None, None] * gph[n[i]]; gpf[b] = gph[o[b]]
        dpc = np.zeros((m.nface, self.nk), complex)
        dpc[i] = m.ef_over_d[i, None] * (ph[n[i]] - ph[o[i]])
        dpc[b] = m.ef_over_d[b, None] * (pbv[m.bface_index[b]] - ph[o[b]])
        dpc = dpc + (m.Tf[:, :, None] * gpf).sum(axis=1)
        dpw = (gpf * m.normal[:, :, None]).sum(axis=1)
        damp = D * (dpc - dpw) * m.span
        damp[self.fixed_u] = 0.0
        return Fbar - damp

    def _pressure_flux(self, gam, pph, gpph):
        m = self.m; i, b = m.interior, m.boundary; w = m.wf
        F = np.zeros((m.nface,) + pph.shape[1:], complex)
        coef = gam * m.ef_over_d * m.span
        F[i] = coef[i, None] * (pph[m.neigh[i]] - pph[m.owner[i]])
        gf = np.empty((m.nface, 2) + pph.shape[1:], complex)
        gf[i] = w[i, None, None] * gpph[m.owner[i]] + (1 - w[i])[:, None, None] * gpph[m.neigh[i]]; gf[b] = gpph[m.owner[b]]
        cross = gam[:, None] * (m.Tf[:, :, None] * gf).sum(axis=1) * m.span
        cross[b] = 0.0
        return F + cross

    def _solve_cols(self, lu, bh):
        """Complex rhs through a real factorisation: real and imaginary parts as two columns."""
        x = lu.solve(np.column_stack([bh.real, bh.imag]))
        return x[:, 0] + 1j * x[:, 1]

    # ---------------------------------------------------------------- the step
    def step(self):
        if not self._flux_init:
            if self.nstep == 0 and not np.any(self.Ff) and (np.any(self.u) or np.any(self.v)): self.init_flux()
            self._flux_init = True
        m = self.m; dt = self.dt; nk = self.nk; kz = self.kz
        u, v, w, F, p = self.u, self.v, self.w, self.Ff, self.p
        N_prev = None
        comps = ((self.bc_u, self.Lu, self.Lu_rhs, self.fx), (self.bc_v, self.Lv, self.Lv_rhs, self.fy), (self.bc_w, self.Lw, self.Lw_rhs, self.fz))
        for s in range(3):
            g, z, al = self.RK3_GAMMA[s], self.RK3_ZETA[s], self.RK3_ALPHA[s]; be = al
            Nk = self.nonlinear(u, v, w, F)
            if N_prev is None: N_prev = Nk
            pb = self.beff(self.bc_p, p); gp = self.grad_p(p, pb)                       # (ncell, 2, nz)
            gp3 = (gp[:, 0], gp[:, 1], self.ddz(p))
            stars = []; Dcells = []
            for comp, (bc, L, L_rhs, f) in enumerate(comps):
                phi = (u, v, w)[comp]
                Ld = self.diffusion(phi, L, L_rhs, bc)
                b = m.vol[:, None] * phi / dt - g * Nk[comp] - z * N_prev[comp] + al * Ld + (al + be) * m.vol[:, None] * (f[:, None] - gp3[comp])
                bh = self.fft(b)
                phih = self.fft(phi)
                bh -= al * self.nu * kz[None, :] ** 2 * m.vol[:, None] * phih           # explicit half of the z-diffusion
                xh = phih.copy()
                aP = np.empty((m.ncell, nk)); aC = np.empty((m.ncell, nk))
                for it in range(max(1, self.n_inner)):
                    x_phys = self.ifft(xh); pbs = self.beff(bc, x_phys)
                    corr = self.fft(L_rhs(self.grad(x_phys, pbs), pbs))                  # lagged cross-diffusion of phi*
                    for k in range(nk):
                        lu, aPk, aCk = self._mom_solver(s, comp, k, L, be)
                        if k == nk - 1 and self.nz % 2 == 0: xh[:, k] = 0.0; aP[:, k] = aPk; aC[:, k] = aCk; continue
                        xh[:, k] = self._solve_cols(lu, bh[:, k] + be * corr[:, k]); aP[:, k] = aPk; aC[:, k] = aCk
                stars.append(xh); Dcells.append((aP, aC))
            uh, vh, wh = stars
            aP, aC = Dcells[0]
            ph = self.fft(p); gph = self.fft(gp)                                          # (ncell, 2, nk)
            Drc = (al + be) * m.vol[:, None] / aP                                        # dt_k V / a_P
            Fs = self._rhie_chow(uh, vh, ph, gph, Drc)
            Dcell = (al + be) * m.vol[:, None] / aC
            pph = np.zeros((m.ncell, nk), complex); Fpp = np.zeros((m.nface, nk), complex); gpph = np.zeros((m.ncell, 2, nk), complex)
            for k in range(nk):
                if k == nk - 1 and self.nz % 2 == 0: continue
                Ak, Ap_rhs, lu, gam = self._pois_solver(s, k, Dcell[:, k])
                src = divergence(m, Fs[:, k].real) + 1j * divergence(m, Fs[:, k].imag) + 1j * kz[k] * wh[:, k]
                rhs = src * m.vol
                x = np.zeros(m.ncell, complex)
                zero_b = np.zeros(m.nbface, complex)
                for it in range(max(1, self.n_nonorth)):
                    pbk = zero_b.copy(); neu = self.bc_p.kind == NEUMANN; pbk[neu] = x[m.owner[m.bfaces][neu]]
                    gr = self.grad(np.column_stack([x.real, x.imag]), np.column_stack([pbk.real, pbk.imag]))
                    corr = Ap_rhs(gr, np.column_stack([pbk.real, pbk.imag])); corr = corr[:, 0] + 1j * corr[:, 1]
                    bk = rhs - corr
                    if k == 0 and self.p_singular: bk = bk - bk.mean(); bk[0] = 0.0
                    x = self._solve_cols(lu, bk)
                    if k == 0 and self.p_singular: x = x - x.mean()
                pbk = zero_b.copy(); neu = self.bc_p.kind == NEUMANN; pbk[neu] = x[m.owner[m.bfaces][neu]]
                gr = self.grad(np.column_stack([x.real, x.imag]), np.column_stack([pbk.real, pbk.imag]))
                gpph[:, :, k] = gr[:, :, 0] + 1j * gr[:, :, 1]
                pph[:, k] = x
                Fpp[:, k] = self._pressure_flux(gam, x[:, None], gpph[:, :, k][:, :, None])[:, 0]
            uh = uh - Dcell * gpph[:, 0]; vh = vh - Dcell * gpph[:, 1]; wh = wh - Dcell * (1j * kz[None, :]) * pph
            Fh = Fs - Fpp
            u, v, w = self.ifft(uh), self.ifft(vh), self.ifft(wh)
            F = self.ifft(Fh); p = p + self.ifft(pph)
            N_prev = Nk
        self.u, self.v, self.w, self.Ff, self.p = u, v, w, F, p
        self.time += dt; self.nstep += 1
        self._apply_mass_flow()

    # ---------------------------------------------------------------- forcing, checkpoint (LES plan L5)
    def set_mass_flow(self, U_bulk, direction=0):
        """Constant-mass-flow forcing: a uniform body force in `direction` adjusted every step so the
        bulk velocity returns to `U_bulk`. For a periodic channel d U_b / dt = f - tau_w / h exactly,
        so f^{n+1} = f^n + (U_target - U_b^{n+1}) / dt restores the bulk within a step and settles
        on the wall stress. `self.f_bulk` is the current force (the mean pressure gradient)."""
        self._mass_flow = (float(U_bulk), int(direction)); self.f_bulk = 0.0

    def bulk_velocity(self, direction=0):
        f = (self.u, self.v, self.w)[direction]
        return float((self.m.vol[:, None] * f).sum() / (self.m.vol.sum() * self.nz))

    def _apply_mass_flow(self):
        mf = getattr(self, "_mass_flow", None)
        if mf is None: return
        U_t, d = mf; Ub = self.bulk_velocity(d)
        self.f_bulk += (U_t - Ub) / self.dt
        arr = (self.fx, self.fy, self.fz)[d]; arr[:] = self.f_bulk

    def save(self, path):
        np.savez(path, u=self.u, v=self.v, w=self.w, p=self.p, Ff=self.Ff, time=self.time, nstep=self.nstep,
                 f_bulk=getattr(self, "f_bulk", 0.0), nz=self.nz, Lz=self.Lz)

    def load(self, path):
        d = np.load(path)
        self.u[:] = d["u"]; self.v[:] = d["v"]; self.w[:] = d["w"]; self.p[:] = d["p"]; self.Ff[:] = d["Ff"]
        self.time, self.nstep = float(d["time"]), int(d["nstep"]); self._flux_init = True
        if hasattr(self, "_mass_flow"):
            self.f_bulk = float(d["f_bulk"]); arr = (self.fx, self.fy, self.fz)[self._mass_flow[1]]; arr[:] = self.f_bulk

    # ---------------------------------------------------------------- diagnostics
    def energy(self):
        dz = self.Lz / self.nz
        return 0.5 * float((self.m.vol[:, None] * (self.u ** 2 + self.v ** 2 + self.w ** 2)).sum()) * dz

    def enstrophy(self):
        """int |omega|^2 dV with in-plane gradients from the cell gradient and spectral z."""
        m = self.m; dz = self.Lz / self.nz
        gu = self.grad(self.u, self.beff(self.bc_u, self.u)); gv = self.grad(self.v, self.beff(self.bc_v, self.v)); gw = self.grad(self.w, self.beff(self.bc_w, self.w))
        ox = gw[:, 1] - self.ddz(self.v); oy = self.ddz(self.u) - gw[:, 0]; oz = gv[:, 0] - gu[:, 1]
        return float((m.vol[:, None] * (ox ** 2 + oy ** 2 + oz ** 2)).sum()) * dz

    def dissipation(self):
        """The scheme's own viscous dissipation, -sum phi . (L phi) with the discrete operators actually
        used (in-plane orthogonal + cross, spectral z): -dE/dt equals this exactly when nothing else
        dissipates. Differs from nu * enstrophy() by the gradient's truncation error."""
        m = self.m; dz = self.Lz / self.nz; out = 0.0
        for phi, L, L_rhs, bc in ((self.u, self.Lu, self.Lu_rhs, self.bc_u), (self.v, self.Lv, self.Lv_rhs, self.bc_v), (self.w, self.Lw, self.Lw_rhs, self.bc_w)):
            out += -float((phi * self.diffusion(phi, L, L_rhs, bc)).sum()) * dz
            out += self.nu * float((m.vol[:, None] * self.ddz(phi) ** 2).sum()) * dz
        return out

    def divergence_max(self):
        out = np.empty_like(self.u)
        for j in range(self.nz): out[:, j] = divergence(self.m, self.Ff[:, j])
        return float(np.abs(out + self.ddz(self.w)).max())
