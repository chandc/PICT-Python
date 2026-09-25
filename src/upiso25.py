"""2.5D incompressible solver: the unstructured collocated FV scheme in the x-y plane, Fourier in a
periodic span z (LES plan L2). One face-based 2D problem per spanwise mode; the nonlinear term in
physical space on 3/2-padded planes (dealiased); RK3 / Crank-Nicolson per stage with a pressure
projection per stage (the L1a integrator, `PISO.step_rk3`, lifted mode by mode).

Layout: every cell field is (ncell, nz) over the planes z_j = j Lz/nz, every face field (nface, nz).
Spectral arrays are (ncell, nk) complex with nk = nz//2 + 1 from `rfft` along the last axis;
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

DEVICE (LES plan L4): `device="gpu"` runs the whole step on CuPy. Every array the step touches --
the mesh connectivity and metrics, the operator matrices (gradients, Laplacians and their deferred
corrections, the face scatter), the boundary-condition values and masks, the fields -- is copied to
the device once in `__init__` (`self.dm` for the mesh, `self.xp` for the array module); the code
below is written against `xp` and `dm` only, so the CPU path is the same code with numpy. The
per-mode implicit systems are then solved by the block AMG-PCG of `umodesolve.ModeFamily` on the
device (the hierarchy is built on the host by pyamg); the SuperLU path is CPU-only. Host callers
read fields through `self.host(arr)` and write them through `self.asdev(arr)`.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.uops import Gradient, laplacian, divergence, DIRICHLET, NEUMANN
from src.ugrad import GGDualGradient
from src.upiso import BC, PISO


class _NS:
    """attribute bag"""
    def __init__(self, **kw): self.__dict__.update(kw)


class _DevGradient:
    """The four gradient matrices on the device; same call signature as uops.Gradient."""
    def __init__(self, g, tomat, xp):
        self.Gx, self.Gy, self.Bx, self.By = tomat(g.Gx), tomat(g.Gy), tomat(g.Bx), tomat(g.By); self.xp = xp
    def __call__(self, phi, phi_b):
        return self.xp.stack([self.Gx @ phi + self.Bx @ phi_b, self.Gy @ phi + self.By @ phi_b], axis=1)


class PISO25:
    RK3_GAMMA = PISO.RK3_GAMMA
    RK3_ZETA = PISO.RK3_ZETA
    RK3_ALPHA = PISO.RK3_ALPHA

    def __init__(self, mesh, nz, Lz, nu, dt, bc_u, bc_v, bc_w, bc_p, body_force=None,
                 n_nonorth=1, n_inner=2, dealias=True, convect=True, solver="lu", amg_rtol=1e-9, mom_rtol=None, device="cpu"):
        m = self.m = mesh
        self.device = device
        if device == "gpu":
            import cupy as cp; from src.ucuda import DevCSR
            self.xp = cp; self._tomat = lambda A: DevCSR(A)                    # raw-kernel CSR x row-major block
            if solver == "lu": solver = "amg"                    # SuperLU is a host solver; the block AMG runs on the device
        else:
            self.xp = np; self._tomat = lambda A: sp.csr_matrix(A)
        xp = self.xp; ad = self.asdev
        self.nz, self.Lz = int(nz), float(Lz)
        self.nu, self.dt = float(nu), float(dt)
        self.bc_u, self.bc_v, self.bc_w, self.bc_p = bc_u, bc_v, bc_w, bc_p
        self.n_nonorth, self.n_inner = int(n_nonorth), int(n_inner)
        self.nonorth_tol = 1e-8                  # stop the deferred pressure passes once the iterate has settled
        self.convect = bool(convect)
        self.nk = self.nz // 2 + 1
        self.kz_h = 2.0 * np.pi / self.Lz * np.arange(self.nk)          # rfft wavenumbers (host copy for setup)
        self.kz = ad(self.kz_h)
        self.M = (3 * self.nz) // 2 if dealias else self.nz             # planes for the products
        self.kM = ad(2.0 * np.pi / self.Lz * np.arange(self.M // 2 + 1))
        self.z = np.arange(self.nz) * self.Lz / self.nz
        self.fx = ad(np.zeros(m.ncell) if body_force is None else np.asarray(body_force[0], float))
        self.fy = ad(np.zeros(m.ncell) if body_force is None else np.asarray(body_force[1], float))
        self.fz = ad(np.zeros(m.ncell) if body_force is None or len(body_force) < 3 else np.asarray(body_force[2], float))

        self.u = xp.zeros((m.ncell, self.nz)); self.v = xp.zeros_like(self.u); self.w = xp.zeros_like(self.u)
        self.p = xp.zeros_like(self.u); self.Ff = xp.zeros((m.nface, self.nz))
        # ---- device mesh: everything the step reads from the mesh
        i = m.interior; o, n = m.owner, m.neigh; w = m.wf
        self.dm = _NS(ncell=m.ncell, nface=m.nface, nbface=m.nbface, span=float(m.span),
                      owner=ad(o), neigh=ad(n), wf=ad(w), normal=ad(m.normal), vol=ad(m.vol), interior=ad(i), boundary=ad(m.boundary),
                      bfaces=ad(m.bfaces), bface_index=ad(m.bface_index), Tf=ad(m.Tf), ef_over_d=ad(m.ef_over_d),
                      owner_b=ad(o[m.bfaces]), bface_index_b=ad(m.bface_index[m.bfaces]), owner_i=ad(o[i]), neigh_i=ad(n[i]), wf_i=ad(w[i]),
                      iidx=ad(np.flatnonzero(i)), bidx=ad(np.flatnonzero(m.boundary)))
        dm = self.dm
        # ---- boundary conditions on the device
        self.bcd = {}
        for bc in (bc_u, bc_v, bc_w, bc_p):
            self.bcd[id(bc)] = _NS(value=ad(bc.value), neu=ad(bc.kind == NEUMANN), dir=ad(bc.kind == DIRICHLET), any_neu=bool((bc.kind == NEUMANN).any()),
                                   bfaces_dir=ad(m.bfaces[bc.kind == DIRICHLET]), bfaces_neu=ad(m.bfaces[bc.kind == NEUMANN]), has_dir=bool((bc.kind == DIRICHLET).any()))
        # ---- operators
        self.grad_h = Gradient(m); self.grad = _DevGradient(self.grad_h, self._tomat, xp)
        self.grad_p = _DevGradient(GGDualGradient(m), self._tomat, xp)
        self.Lu_h, Lu_rhs = laplacian(m, np.full(m.nface, self.nu), bc_u.kind)
        self.Lv_h, Lv_rhs = laplacian(m, np.full(m.nface, self.nu), bc_v.kind)
        self.Lw_h, Lw_rhs = laplacian(m, np.full(m.nface, self.nu), bc_w.kind)
        self.Lu, self.Lv, self.Lw = self._tomat(self.Lu_h), self._tomat(self.Lv_h), self._tomat(self.Lw_h)
        self.Lu_rhs, self.Lv_rhs, self.Lw_rhs = self._dev_rhs(Lu_rhs), self._dev_rhs(Lv_rhs), self._dev_rhs(Lw_rhs)
        self.p_singular = bool((bc_p.kind == NEUMANN).all())
        # owner(+)/neighbour(-) scatter of a face quantity into cells (also the divergence, divided by V)
        Sc = (sp.coo_matrix((np.ones(m.nface), (o, np.arange(m.nface))), shape=(m.ncell, m.nface))
              - sp.coo_matrix((np.ones(int(i.sum())), (n[i], np.flatnonzero(i))), shape=(m.ncell, m.nface))).tocsr()
        self.Sc = self._tomat(Sc)
        dskew = np.zeros((m.nface, 2)); dskew[i] = m.fcentre[i] - (m.centroid[o[i]] + (1.0 - w[i])[:, None] * m.dcc[i]); self.dskew_i = ad(dskew[i])
        self.dxb = ad(m.fcentre[m.bfaces] - m.centroid[o[m.bfaces]])
        kd_u = bc_u.kind == DIRICHLET; kd_v = bc_v.kind == DIRICHLET
        Sb = m.normal[m.bfaces]; ax = np.abs(Sb[:, 0]) > 1e-9 * np.hypot(*Sb.T); ay = np.abs(Sb[:, 1]) > 1e-9 * np.hypot(*Sb.T)
        fixed_u = np.zeros(m.nface, dtype=bool); fixed_u[m.bfaces] = (kd_u & kd_v) | (kd_u & ~ay) | (kd_v & ~ax); self.fixed_u = ad(fixed_u)
        self._mom = {}          # (stage, comp-key, mode) -> (splu, aP, aC)          [lu path]
        self._pois = {}         # (stage, mode) -> (Ap_k, Ap_rhs, splu, gam)         [lu path]
        # Linear solver for the per-mode families (LES plan L4): "lu" = one SuperLU per (stage, mode), cached;
        # "amg" = all modes at once by block PCG with a shared Ruge-Stuben hierarchy shifted per mode
        # (src/umodesolve.ModeFamily). Same answers to amg_rtol; the block form is what scales and what
        # runs on the device.
        self.solver = solver; self.amg_rtol = float(amg_rtol)
        # momentum tolerance: the projection makes the stage field solenoidal whatever the momentum residual, and
        # the integrator's own error is O(dt^3); 1e-7 measured indistinguishable from 1e-9 in the energy (section 57)
        self.mom_rtol = float(mom_rtol) if mom_rtol is not None else self.amg_rtol
        self._mom_fam = {}      # (stage, comp) -> (ModeFamily, aP (ncell, nk), aC)
        self._pois_fam = {}     # stage -> (ModeFamily, Ap_rhs (device), gam (device))
        self.solver_iters = []  # (stage, "mom"/"p", iterations) of the last step
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
        self.nu_t = xp.zeros((m.ncell, self.nz))
        self._Tf_span = dm.Tf * dm.span

    # ---------------------------------------------------------------- host/device
    def asdev(self, a):
        return self.xp.asarray(a)

    def host(self, a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def _dev_rhs(self, rhs_fn):
        Cx, Cy = self._tomat(rhs_fn.Cx), self._tomat(rhs_fn.Cy); Bd = self._tomat(rhs_fn.Bd) if rhs_fn.has_bd else None
        def f(grad_phi, bvalues=None):
            out = Cx @ grad_phi[:, 0] + Cy @ grad_phi[:, 1]
            if bvalues is not None and Bd is not None: out = out + Bd @ bvalues
            return out
        return f

    def _div(self, F):
        """cell divergence of a face flux block (nface, n) -> (ncell, n); real or complex."""
        if self.xp.iscomplexobj(F): return (self.Sc @ F.real + 1j * (self.Sc @ F.imag)) / self.dm.vol[:, None]
        return (self.Sc @ F) / self.dm.vol[:, None]

    # ---------------------------------------------------------------- spectral helpers
    def fft(self, f):
        fh = self.xp.fft.rfft(f, axis=-1)
        if self.nz % 2 == 0: fh[..., -1] = 0.0
        return fh

    def ifft(self, fh):
        return self.xp.fft.irfft(fh, n=self.nz, axis=-1)

    def pad(self, f):
        """(…, nz) -> (…, M) planes, same Fourier content (3/2 rule)."""
        if self.M == self.nz: return f
        xp = self.xp; fh = self.fft(f); out = xp.zeros(f.shape[:-1] + (self.M // 2 + 1,), complex); out[..., :self.nk] = fh
        return xp.fft.irfft(out, n=self.M, axis=-1) * (self.M / self.nz)

    def trunc(self, f):
        """(…, M) -> (…, nz) planes, modes above nk dropped."""
        if self.M == self.nz: return f
        xp = self.xp; fh = xp.fft.rfft(f, axis=-1)[..., :self.nk]
        if self.nz % 2 == 0: fh[..., -1] = 0.0
        return xp.fft.irfft(fh, n=self.nz, axis=-1) * (self.nz / self.M)

    def ddz(self, f, nplanes=None):
        xp = self.xp; n = f.shape[-1]; k = self.kM if n == self.M else self.kz
        fh = xp.fft.rfft(f, axis=-1) * (1j * k)
        if n % 2 == 0: fh[..., -1] = 0.0
        return xp.fft.irfft(fh, n=n, axis=-1)

    # ---------------------------------------------------------------- boundary values
    def beff(self, bc, phi):
        """Boundary face values for a (ncell, n) physical field: Dirichlet value, else owner's."""
        d = self.bcd[id(bc)]; out = self.xp.repeat(d.value[:, None], phi.shape[1], axis=1)
        if d.any_neu: out[d.neu] = phi[self.dm.owner_b[d.neu]]
        return out

    def beff_hat(self, bc, phih):
        """Same for a (ncell, nk) spectral field: the Dirichlet value lives in mode 0 only."""
        d = self.bcd[id(bc)]; out = self.xp.zeros((self.dm.nbface, phih.shape[1]), complex); out[:, 0] = d.value * self.nz
        if d.any_neu: out[d.neu] = phih[self.dm.owner_b[d.neu]]
        return out

    # ---------------------------------------------------------------- operators on planes
    def init_flux(self):
        xp = self.xp; dm = self.dm; i, b = dm.iidx, dm.bidx; w = dm.wf
        ub = self.beff(self.bc_u, self.u); vb = self.beff(self.bc_v, self.v)
        uf = xp.empty((dm.nface, self.nz)); vf = xp.empty_like(uf)
        uf[i] = w[i, None] * self.u[dm.owner[i]] + (1 - w[i, None]) * self.u[dm.neigh[i]]
        vf[i] = w[i, None] * self.v[dm.owner[i]] + (1 - w[i, None]) * self.v[dm.neigh[i]]
        uf[b] = ub[dm.bface_index[b]]; vf[b] = vb[dm.bface_index[b]]
        self.Ff = (uf * dm.normal[:, 0, None] + vf * dm.normal[:, 1, None]) * dm.span
        self._flux_init = True

    def conv(self, phi, F, wz, bc):
        """Volume-integrated convection of phi on the padded planes: sum_f F_f phi_f + V d(wz phi)/dz."""
        xp = self.xp; dm = self.dm; i = dm.iidx; w = dm.wf_i; o, n = dm.owner_i, dm.neigh_i
        pb = self.beff(bc, phi); g = self.grad(phi, pb)                        # (ncell, 2, M)
        phif = xp.empty((dm.nface, phi.shape[1]))
        gf = w[:, None, None] * g[o] + (1 - w)[:, None, None] * g[n]
        phif[i] = w[:, None] * phi[o] + (1 - w[:, None]) * phi[n] + (gf * self.dskew_i[:, :, None]).sum(axis=1)
        ob = dm.owner_b
        outflow = phi[ob] + (g[ob] * self.dxb[:, :, None]).sum(axis=1)
        phif[dm.bfaces] = xp.where(F[dm.bfaces] >= 0.0, outflow, pb[dm.bface_index_b])
        N = self.Sc @ (F * phif)
        if self.nz > 1: N = N + dm.vol[:, None] * self.ddz(wz * phi)
        return N

    def sgs_term(self, u, v, w):
        """Volume-integrated  div( nu_t (grad u_i + (grad u)^T_i) )  for i = x, y, z on the given planes.

        In-plane faces: nu_f [ E_f/d (phi_N - phi_P) + T_f . grad_f phi ] (the Laplacian's own
        over-relaxed split) plus the transpose  nu_f (d u_j / d x_i)_f S_j ; spanwise:
        V d/dz [ nu_t (d u_i/dz + d w/d x_i) ] spectral. Dirichlet faces use the boundary value in
        the compact part; Neumann faces carry no flux. Returns (T_u, T_v, T_w) and sets self.nu_t
        when the planes are the solver's own."""
        import src.usgs as usgs
        xp = self.xp; dm = self.dm; i, b = dm.iidx, dm.bidx; o, n = dm.owner, dm.neigh; wf = dm.wf
        g = usgs.velocity_gradient(self, u, v, w)                       # g[i][j] (ncell, n)
        if self.sgs_nu_field is not None:
            nut = self.sgs_nu_field if self.sgs_nu_field.shape[1] == u.shape[1] else self.pad(self.sgs_nu_field)
        else:
            nut = usgs.eddy_viscosity(self, self.sgs_model, g=g, **self.sgs_kw)
        self.nu_t = nut if u.shape[1] == self.nz else self.trunc(nut)      # reported on the nz planes
        nuf = xp.empty((dm.nface, u.shape[1])); nuf[i] = wf[i, None] * nut[o[i]] + (1 - wf[i, None]) * nut[n[i]]; nuf[b] = nut[o[b]]
        out = []
        for comp, (phi, bc) in enumerate(((u, self.bc_u), (v, self.bc_v), (w, self.bc_w))):
            gphi = g[comp]                                               # [d phi/dx, d phi/dy, d phi/dz]
            gfx = xp.empty((dm.nface, u.shape[1])); gfy = xp.empty_like(gfx)
            gfx[i] = wf[i, None] * gphi[0][o[i]] + (1 - wf[i, None]) * gphi[0][n[i]]; gfy[i] = wf[i, None] * gphi[1][o[i]] + (1 - wf[i, None]) * gphi[1][n[i]]
            gfx[b] = gphi[0][o[b]]; gfy[b] = gphi[1][o[b]]
            flux = xp.zeros((dm.nface, u.shape[1]))
            flux[i] = dm.ef_over_d[i, None] * (phi[n[i]] - phi[o[i]]) + self._Tf_span[i, 0, None] / dm.span * gfx[i] + self._Tf_span[i, 1, None] / dm.span * gfy[i]
            pb = self.beff(bc, phi); d = self.bcd[id(bc)]
            if d.has_dir:
                bd = d.bfaces_dir
                flux[bd] = dm.ef_over_d[bd, None] * (pb[dm.bface_index[bd]] - phi[o[bd]])
            # transpose part: (d u_j / d x_comp)_f S_j  with j = x, y (in-plane faces), j = z below
            tx = xp.empty_like(gfx); ty = xp.empty_like(gfx)
            gu_c, gv_c = g[0][comp], g[1][comp]
            tx[i] = wf[i, None] * gu_c[o[i]] + (1 - wf[i, None]) * gu_c[n[i]]; ty[i] = wf[i, None] * gv_c[o[i]] + (1 - wf[i, None]) * gv_c[n[i]]
            tx[b] = gu_c[o[b]]; ty[b] = gv_c[o[b]]
            trans = tx * dm.normal[:, 0, None] + ty * dm.normal[:, 1, None]
            if d.any_neu: trans[d.bfaces_neu] = 0.0
            T = self.Sc @ (nuf * (flux * dm.span + trans * dm.span))
            if self.nz > 1: T = T + dm.vol[:, None] * self.ddz(nut * (gphi[2] + g[2][comp]))
            out.append(T)
        return out

    def nonlinear(self, u, v, w, F):
        """(N_u, N_v, N_w) on the nz planes, products formed on M planes and truncated."""
        if not self.convect:
            z = self.xp.zeros_like(u); return z, z, z
        up, vp, wp, Fp = self.pad(u), self.pad(v), self.pad(w), self.pad(F)
        Nu, Nv, Nw = self.conv(up, Fp, wp, self.bc_u), self.conv(vp, Fp, wp, self.bc_v), self.conv(wp, Fp, wp, self.bc_w)
        if self.sgs_model != "none" or self.sgs_nu_field is not None:
            Tu, Tv, Tw = self.sgs_term(up, vp, wp); Nu = Nu - Tu; Nv = Nv - Tv; Nw = Nw - Tw   # N is what the rhs SUBTRACTS
        return self.trunc(Nu), self.trunc(Nv), self.trunc(Nw)

    def diffusion(self, phi, L, L_rhs, bc):
        """Full in-plane viscous term (orthogonal + cross) of a (ncell, n) field; the z part is spectral."""
        pb = self.beff(bc, phi); return L @ phi + L_rhs(self.grad(phi, pb), pb)

    # ---------------------------------------------------------------- per-mode pieces
    def _mom_solver(self, stage, comp, k, L_h, be):
        key = (stage, comp, k)
        if key not in self._mom:
            m = self.m
            A = (sp.diags(m.vol / self.dt) - be * L_h + sp.diags(be * self.nu * self.kz_h[k] ** 2 * m.vol)).tocsc()
            aP = np.asarray(A.diagonal()); aC = np.maximum(np.asarray(A.sum(axis=1)).ravel(), m.vol / self.dt)
            self._mom[key] = (spla.splu(A), aP, aC)
        return self._mom[key]

    def _mom_family(self, stage, comp, L_h, be):
        key = (stage, comp)
        if key not in self._mom_fam:
            from src.umodesolve import ModeFamily
            m = self.m; A0 = (sp.diags(m.vol / self.dt) - be * L_h).tocsr()                # SPD; the k-shift is be*nu*k^2*V
            fam = ModeFamily(-A0, be * self.nu * m.vol, self.kz_h ** 2, singular_k0=False, presmooth=1, postsmooth=1, device=self.device, precond="jacobi")
            aP = np.asarray(A0.diagonal())[:, None] + be * self.nu * m.vol[:, None] * (self.kz_h ** 2)[None, :]
            aC = np.maximum(np.asarray(A0.sum(axis=1)).ravel()[:, None] + be * self.nu * m.vol[:, None] * (self.kz_h ** 2)[None, :], (m.vol / self.dt)[:, None])
            self._mom_fam[key] = (fam, self.asdev(aP), self.asdev(aC))
        return self._mom_fam[key]

    def _gam_of(self, Dcell_h):
        m = self.m; w = m.wf; i, b = m.interior, m.boundary
        gam = np.empty(m.nface); gam[i] = w[i] * Dcell_h[m.owner[i]] + (1 - w[i]) * Dcell_h[m.neigh[i]]; gam[b] = Dcell_h[m.owner[b]]
        return gam

    def _pois_family(self, stage, Dcell_k0):
        if stage not in self._pois_fam:
            from src.umodesolve import ModeFamily
            m = self.m; Dh = self.host(Dcell_k0); gam = self._gam_of(Dh)
            Ap, Ap_rhs = laplacian(m, gam, self.bc_p.kind)
            fam = ModeFamily(Ap, Dh * m.vol, self.kz_h ** 2, singular_k0=self.p_singular, presmooth=1, postsmooth=1, device=self.device)
            self._pois_fam[stage] = (fam, self._dev_rhs(Ap_rhs), self.asdev(gam))
        return self._pois_fam[stage]

    def _pois_solver(self, stage, k, Dcell):
        key = (stage, k)
        if key not in self._pois:
            m = self.m; gam = self._gam_of(Dcell)
            Ap, Ap_rhs = laplacian(m, gam, self.bc_p.kind)
            Ak = (Ap - sp.diags(self.kz_h[k] ** 2 * Dcell * m.vol)).tocsc()
            if k == 0 and self.p_singular:
                # all-Neumann mode 0: pin cell 0. Exact for a compatible right-hand side (its mean
                # is removed), and the solution's mean is removed afterwards.
                Ak = Ak.tolil(); Ak[0, :] = 0.0; Ak[0, 0] = 1.0; Ak = Ak.tocsc()
            self._pois[key] = (Ak, Ap_rhs, spla.splu(Ak), gam)
        return self._pois[key]

    def _rhie_chow(self, uh, vh, ph, gph, Dcell):
        """Spectral Rhie-Chow: (ncell, nk) complex fields, Dcell (ncell, nk) real. Returns F* (nface, nk)."""
        xp = self.xp; dm = self.dm; i, b = dm.iidx, dm.bidx; o, n = dm.owner_i, dm.neigh_i; w = dm.wf_i
        ub = self.beff_hat(self.bc_u, uh); vb = self.beff_hat(self.bc_v, vh)
        uf = xp.empty((dm.nface, self.nk), complex); vf = xp.empty_like(uf)
        uf[i] = w[:, None] * uh[o] + (1 - w[:, None]) * uh[n]; vf[i] = w[:, None] * vh[o] + (1 - w[:, None]) * vh[n]
        uf[b] = ub[dm.bface_index_b]; vf[b] = vb[dm.bface_index_b]
        Fbar = (uf * dm.normal[:, 0, None] + vf * dm.normal[:, 1, None]) * dm.span
        D = xp.empty((dm.nface, self.nk)); D[i] = w[:, None] * Dcell[o] + (1 - w[:, None]) * Dcell[n]; D[b] = Dcell[dm.owner_b]
        pbv = self.beff_hat(self.bc_p, ph)
        gpf = xp.empty((dm.nface, 2, self.nk), complex)
        gpf[i] = w[:, None, None] * gph[o] + (1 - w)[:, None, None] * gph[n]; gpf[b] = gph[dm.owner_b]
        dpc = xp.zeros((dm.nface, self.nk), complex)
        dpc[i] = dm.ef_over_d[i, None] * (ph[n] - ph[o])
        dpc[b] = dm.ef_over_d[b, None] * (pbv[dm.bface_index_b] - ph[dm.owner_b])
        dpc = dpc + (dm.Tf[:, :, None] * gpf).sum(axis=1)
        dpw = (gpf * dm.normal[:, :, None]).sum(axis=1)
        damp = D * (dpc - dpw) * dm.span
        damp[self.fixed_u] = 0.0
        return Fbar - damp

    def _pressure_flux(self, gam, pph, gpph):
        xp = self.xp; dm = self.dm; i, b = dm.iidx, dm.bidx; w = dm.wf_i; o, n = dm.owner_i, dm.neigh_i
        F = xp.zeros((dm.nface,) + pph.shape[1:], complex)
        coef = gam * dm.ef_over_d * dm.span
        F[i] = coef[i, None] * (pph[n] - pph[o])
        gf = xp.empty((dm.nface, 2) + pph.shape[1:], complex)
        gf[i] = w[:, None, None] * gpph[o] + (1 - w)[:, None, None] * gpph[n]; gf[b] = gpph[dm.owner_b]
        cross = gam[:, None] * (dm.Tf[:, :, None] * gf).sum(axis=1) * dm.span
        cross[b] = 0.0
        return F + cross

    def _solve_cols(self, lu, bh):
        """Complex rhs through a real factorisation: real and imaginary parts as two columns."""
        x = lu.solve(np.column_stack([bh.real, bh.imag]))
        return x[:, 0] + 1j * x[:, 1]

    # ---------------------------------------------------------------- the step
    def step(self):
        xp = self.xp; dm = self.dm
        if not self._flux_init:
            if self.nstep == 0 and not bool(xp.any(self.Ff)) and (bool(xp.any(self.u)) or bool(xp.any(self.v))): self.init_flux()
            self._flux_init = True
        m = self.m; dt = self.dt; nk = self.nk; kz = self.kz
        u, v, w, F, p = self.u, self.v, self.w, self.Ff, self.p
        N_prev = None
        comps = ((self.bc_u, self.Lu, self.Lu_h, self.Lu_rhs, self.fx), (self.bc_v, self.Lv, self.Lv_h, self.Lv_rhs, self.fy), (self.bc_w, self.Lw, self.Lw_h, self.Lw_rhs, self.fz))
        vol = dm.vol[:, None]
        for s in range(3):
            g, z, al = self.RK3_GAMMA[s], self.RK3_ZETA[s], self.RK3_ALPHA[s]; be = al
            Nk = self.nonlinear(u, v, w, F)
            if N_prev is None: N_prev = Nk
            pb = self.beff(self.bc_p, p); gp = self.grad_p(p, pb)                       # (ncell, 2, nz)
            gp3 = (gp[:, 0], gp[:, 1], self.ddz(p))
            stars = []; Dcells = []
            for comp, (bc, L, L_h, L_rhs, f) in enumerate(comps):
                phi = (u, v, w)[comp]
                Ld = self.diffusion(phi, L, L_rhs, bc)
                b = vol * phi / dt - g * Nk[comp] - z * N_prev[comp] + al * Ld + (al + be) * vol * (f[:, None] - gp3[comp])
                bh = self.fft(b)
                phih = self.fft(phi)
                bh -= al * self.nu * kz[None, :] ** 2 * vol * phih                        # explicit half of the z-diffusion
                xh = phih.copy()
                aP = xp.empty((dm.ncell, nk)); aC = xp.empty((dm.ncell, nk))
                for it in range(max(1, self.n_inner)):
                    x_phys = self.ifft(xh); pbs = self.beff(bc, x_phys)
                    corr = self.fft(L_rhs(self.grad(x_phys, pbs), pbs))                  # lagged cross-diffusion of phi*
                    if self.solver == "amg":
                        fam, aP, aC = self._mom_family(s, comp, L_h, be)
                        # the family is the NEGATED (SPD) operator: (A0 + shift) x = b  with A0 = V/dt - be L
                        xh = fam.solve_complex(bh + be * corr, X0=xh, rtol=self.mom_rtol); self.solver_iters.append((s, "mom", fam.iterations))
                        if self.nz % 2 == 0: xh[:, nk - 1] = 0.0
                        continue
                    for k in range(nk):
                        lu, aPk, aCk = self._mom_solver(s, comp, k, L_h, be)
                        if k == nk - 1 and self.nz % 2 == 0: xh[:, k] = 0.0; aP[:, k] = aPk; aC[:, k] = aCk; continue
                        xh[:, k] = self._solve_cols(lu, bh[:, k] + be * corr[:, k]); aP[:, k] = aPk; aC[:, k] = aCk
                stars.append(xh); Dcells.append((aP, aC))
            uh, vh, wh = stars
            aP, aC = Dcells[0]
            ph = self.fft(p); gph = self.fft(gp)                                          # (ncell, 2, nk)
            Drc = (al + be) * vol / aP                                                    # dt_k V / a_P
            Fs = self._rhie_chow(uh, vh, ph, gph, Drc)
            Dcell = (al + be) * vol / aC
            pph = xp.zeros((dm.ncell, nk), complex); Fpp = xp.zeros((dm.nface, nk), complex); gpph = xp.zeros((dm.ncell, 2, nk), complex)
            neu = self.bcd[id(self.bc_p)].neu; any_neu = self.bcd[id(self.bc_p)].any_neu
            if self.solver == "amg":
                # all modes at once; Dcell differs across modes only through a_C(k) -- the family is built on
                # the k = 0 coefficient and the k-dependence of the coefficient itself is dropped from the
                # operator (it is O(nu k^2 dt), 1e-3 relative here) -- so the correction is solved with the
                # mode-0 diffusivity; the velocity/flux corrections below still use the exact Dcell(k)
                fam, Ap_rhs, gam = self._pois_family(s, Dcell[:, 0])
                rhs = (self._div(Fs) + 1j * kz[None, :] * wh) * vol; x = xp.zeros((dm.ncell, nk), complex)
                for it in range(max(1, self.n_nonorth)):
                    pbk = xp.zeros((dm.nbface, nk), complex)
                    if any_neu: pbk[neu] = x[dm.owner_b[neu]]
                    gr = self.grad(xp.concatenate([x.real, x.imag], axis=1), xp.concatenate([pbk.real, pbk.imag], axis=1))
                    corr = Ap_rhs(gr, xp.concatenate([pbk.real, pbk.imag], axis=1)); corr = corr[:, :nk] + 1j * corr[:, nk:]
                    xn = fam.solve_complex(-(rhs - corr), X0=x, rtol=self.amg_rtol)       # the family is the negated operator: (-Ap + k^2 D V) pp = -(rhs - corr)
                    delta = float(xp.abs(xn - x).max()) / max(float(xp.abs(xn).max()), 1e-300); x = xn
                    if delta < self.nonorth_tol: break
                self.solver_iters.append((s, "p", fam.iterations))
                if self.nz % 2 == 0: x[:, nk - 1] = 0.0
                pbk = xp.zeros((dm.nbface, nk), complex)
                if any_neu: pbk[neu] = x[dm.owner_b[neu]]
                gr = self.grad(xp.concatenate([x.real, x.imag], axis=1), xp.concatenate([pbk.real, pbk.imag], axis=1))
                gpph = gr[:, :, :nk] + 1j * gr[:, :, nk:]
                pph = x; Fpp = self._pressure_flux(gam, pph, gpph)
            for k in (range(nk) if self.solver != "amg" else ()):
                if k == nk - 1 and self.nz % 2 == 0: continue
                Ak, Ap_rhs, lu, gam = self._pois_solver(s, k, Dcell[:, k])
                src = divergence(m, Fs[:, k].real) + 1j * divergence(m, Fs[:, k].imag) + 1j * kz[k] * wh[:, k]
                rhs = src * m.vol
                x = np.zeros(m.ncell, complex)
                zero_b = np.zeros(m.nbface, complex)
                for it in range(max(1, self.n_nonorth)):
                    # deferred non-orthogonal passes: n_nonorth is the MAXIMUM, the loop stops when the
                    # iterate has settled (as solve_poisson does in 2D); a fixed count on a skewed mesh
                    # leaves the cross term one pass behind every stage
                    pbk = zero_b.copy()
                    if any_neu: pbk[neu] = x[m.owner[m.bfaces][neu]]
                    gr = self.grad(np.column_stack([x.real, x.imag]), np.column_stack([pbk.real, pbk.imag]))
                    corr = Ap_rhs(gr, np.column_stack([pbk.real, pbk.imag])); corr = corr[:, 0] + 1j * corr[:, 1]
                    bk = rhs - corr
                    if k == 0 and self.p_singular: bk = bk - bk.mean(); bk[0] = 0.0
                    xn = self._solve_cols(lu, bk)
                    if k == 0 and self.p_singular: xn = xn - xn.mean()
                    delta = float(np.abs(xn - x).max()) / max(float(np.abs(xn).max()), 1e-300); x = xn
                    if delta < self.nonorth_tol: break
                pbk = zero_b.copy()
                if any_neu: pbk[neu] = x[m.owner[m.bfaces][neu]]
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
        return float((self.dm.vol[:, None] * f).sum() / (self.dm.vol.sum() * self.nz))

    def _apply_mass_flow(self):
        mf = getattr(self, "_mass_flow", None)
        if mf is None: return
        U_t, d = mf; Ub = self.bulk_velocity(d)
        self.f_bulk += (U_t - Ub) / self.dt
        arr = (self.fx, self.fy, self.fz)[d]; arr[:] = self.f_bulk

    def save(self, path):
        h = self.host
        np.savez(path, u=h(self.u), v=h(self.v), w=h(self.w), p=h(self.p), Ff=h(self.Ff), time=self.time, nstep=self.nstep,
                 f_bulk=getattr(self, "f_bulk", 0.0), nz=self.nz, Lz=self.Lz)

    def load(self, path):
        d = np.load(path); ad = self.asdev
        self.u[:] = ad(d["u"]); self.v[:] = ad(d["v"]); self.w[:] = ad(d["w"]); self.p[:] = ad(d["p"]); self.Ff[:] = ad(d["Ff"])
        self.time, self.nstep = float(d["time"]), int(d["nstep"]); self._flux_init = True
        if hasattr(self, "_mass_flow"):
            self.f_bulk = float(d["f_bulk"]); arr = (self.fx, self.fy, self.fz)[self._mass_flow[1]]; arr[:] = self.f_bulk

    # ---------------------------------------------------------------- diagnostics
    def energy(self):
        dz = self.Lz / self.nz
        return 0.5 * float((self.dm.vol[:, None] * (self.u ** 2 + self.v ** 2 + self.w ** 2)).sum()) * dz

    def enstrophy(self):
        """int |omega|^2 dV with in-plane gradients from the cell gradient and spectral z."""
        dm = self.dm; dz = self.Lz / self.nz
        gu = self.grad(self.u, self.beff(self.bc_u, self.u)); gv = self.grad(self.v, self.beff(self.bc_v, self.v)); gw = self.grad(self.w, self.beff(self.bc_w, self.w))
        ox = gw[:, 1] - self.ddz(self.v); oy = self.ddz(self.u) - gw[:, 0]; oz = gv[:, 0] - gu[:, 1]
        return float((dm.vol[:, None] * (ox ** 2 + oy ** 2 + oz ** 2)).sum()) * dz

    def dissipation(self):
        """The scheme's own viscous dissipation, -sum phi . (L phi) with the discrete operators actually
        used (in-plane orthogonal + cross, spectral z): -dE/dt equals this exactly when nothing else
        dissipates. Differs from nu * enstrophy() by the gradient's truncation error."""
        dm = self.dm; dz = self.Lz / self.nz; out = 0.0
        for phi, L, L_rhs, bc in ((self.u, self.Lu, self.Lu_rhs, self.bc_u), (self.v, self.Lv, self.Lv_rhs, self.bc_v), (self.w, self.Lw, self.Lw_rhs, self.bc_w)):
            out += -float((phi * self.diffusion(phi, L, L_rhs, bc)).sum()) * dz
            out += self.nu * float((dm.vol[:, None] * self.ddz(phi) ** 2).sum()) * dz
        return out

    def divergence_max(self):
        return float(self.xp.abs(self._div(self.Ff) + self.ddz(self.w)).max())
