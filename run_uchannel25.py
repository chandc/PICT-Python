"""V2 of the LES plan: turbulent channel Re_tau = 180 on the 2.5D solver (unstructured x-y plane with
wall-clustered quads, Fourier span), the structured minimal-channel setup exactly:
    delta = u_tau = 1, nu = 1/180, y in [0, 2], Lx = pi (Lx+ 565), Lz = 0.34 pi (Lz+ 192)
    forcing: constant pressure gradient f_x = 1 (u_tau = 1 by construction; the measured wall stress is a
    CHECK) or --forcing mf (constant mass flow, Re_tau an outcome)
    initial condition: the in-house SEM DNS field (results/minchan_re180_field.npz, t = 18), interpolated
    model: WALE (default), Smagorinsky with van Driest, or none (implicit)
Reports the wall stress from the solver's own one-sided wall flux, the mean profile against the DNS
(stats_MERGED.npz) and the log law, the rms peaks, and the pressure two-colour indicator (the mode that
grew in the structured channel, channel_les_status.md). Restartable (--restart), checkpoints every
--checkpoint steps.
"""
import sys, os, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET
from src.upiso import BC
from src.upiso25 import PISO25
from src.ustats import Stats
DNS = "results/fosls_chan180_stats_t5.2_30.npz"        # SEM FOSLS DNS run02, window t = 5.2..30 (the K/E paths are fractional-step runs, not the reference)
MKM395 = "reference/mkm_chan395"                      # Moser, Kim & Mansour 1999, Re_tau 392.24, chan395.means / chan395.reystress
ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=24); ap.add_argument("--ny", type=int, default=80); ap.add_argument("--nz", type=int, default=32)
ap.add_argument("--dy0plus", type=float, default=1.0); ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--T", type=float, default=30.0); ap.add_argument("--t-stats", type=float, default=10.0)
ap.add_argument("--model", default="wale", choices=["wale", "smagorinsky", "none"])
ap.add_argument("--forcing", default="cpg", choices=["cpg", "mf"]); ap.add_argument("--Ub", type=float, default=15.63, help="bulk velocity for --forcing mf (DNS: 15.63)")
ap.add_argument("--tag", default=None); ap.add_argument("--checkpoint", type=int, default=2500); ap.add_argument("--restart", default=None)
ap.add_argument("--report", type=int, default=500); ap.add_argument("--nsteps", type=int, default=None)
ap.add_argument("--outdir", default="results", help="where <tag>_ckpt/_final/_stats.npz go (e.g. a mounted Google Drive folder, so a lost session loses nothing)")
ap.add_argument("--cfl-max", type=float, default=0.0, help="> 0: CFL-limited time step, dt halves (up to 3 times) whenever the face-flux Courant number would exceed this; 0 = fixed dt")
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="gpu: the whole step on CuPy with the block AMG solves (tools/a100/README.md)")
ap.add_argument("--re-tau", type=float, default=180.0, help="180 (FOSLS DNS reference, DNS initial field) or 395 (MKM 1999 reference; initial field = the 180 DNS field with its mean shifted to the MKM 395 mean)")
ap.add_argument("--Lx", type=float, default=np.pi, help="streamwise box (default pi, the minimal channel; MKM full box 2 pi)"); ap.add_argument("--Lz", type=float, default=0.34 * np.pi, help="span (default 0.34 pi; MKM full box pi)")
ap.add_argument("--mesh", default=None, help="gmsh msh2 file of the plane (physical tags 1 left, 2 right, 3 bottom, 4 top), e.g. meshes/channel_tri_graded.msh; overrides --nx/--ny/--cells")
ap.add_argument("--wall-layers-y", type=int, default=16, help="hybrid: quad layers at each wall, triangles in the core")
ap.add_argument("--cells", default="quad", choices=["quad", "tri", "hybrid"], help="tri: each quad split into two triangles (alternating diagonal), the non-bipartite mesh that carries a two-colour pressure mode in 2D")
ap.add_argument("--n-nonorth", type=int, default=None, help="deferred non-orthogonal passes in the pressure solve (default 1 for quads, 3 for triangles)")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
RE_TAU = a.re_tau; NU = 1.0 / RE_TAU; LX = a.Lx; LY = 2.0; LZ = a.Lz
if RE_TAU != 180 and abs(a.Ub - 15.63) < 1e-9: a.Ub = {395.0: 17.54}.get(RE_TAU, a.Ub)     # MKM 395: U_b/u_tau = 17.54
tag = a.tag or f"uchan{int(RE_TAU) if RE_TAU != 180 else ''}_{a.nx}x{a.ny}x{a.nz}_{a.model}_{a.forcing}" + ("_tri" if a.cells == "tri" else (f"_hyb{a.wall_layers_y}" if a.cells == "hybrid" else ""))
n_nonorth = a.n_nonorth if a.n_nonorth is not None else (1 if a.cells == "quad" else 6)
# ---- mesh: tanh clustering in y with the first cell dy0+ wall units high
def first_cell(beta): return LY * 0.5 * (1 + np.tanh(beta * (2.0 / a.ny - 1)) / np.tanh(beta))
lo, hi = 1e-3, 20.0; target = a.dy0plus / RE_TAU
for _ in range(200):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if first_cell(mid) > target else (lo, mid)
beta = 0.5 * (lo + hi)
if a.mesh:
    from src.umesh import Mesh, read_gmsh22
    nodes, cells, ctag, edges_, etag, names = read_gmsh22(a.mesh); m = Mesh(nodes, cells, edges_, etag, names)
    tag = a.tag or f"uchan_{os.path.splitext(os.path.basename(a.mesh))[0]}_{a.nz}_{a.model}_{a.forcing}"; a.cells = "mesh"
else:
    m = rect_mesh(a.nx, a.ny, 0, LX, 0, LY, cells=a.cells, cluster_y=beta, wall_layers_y=(a.wall_layers_y if a.cells == "hybrid" else None))
m.make_periodic(1, 2, (LX, 0.0))
# statistics bins: the tanh node distribution (80 bins) for every mesh, so an unstructured plane gets the same profile sampling
xi = np.linspace(0, 1, a.ny + 1); y_edges = LY * 0.5 * (1 + np.tanh(beta * (2 * xi - 1)) / np.tanh(beta)); y_edges[0], y_edges[-1] = -1e-9, LY + 1e-9
ys = np.unique(np.round(m.centroid[:, 1], 10)) if not a.mesh else 0.5 * (y_edges[1:] + y_edges[:-1]); dy = np.diff(np.concatenate([[0], 0.5 * (ys[1:] + ys[:-1]), [LY]])) if not a.mesh else np.diff(y_edges)
kd = np.full(m.nbface, DIRICHLET); z0 = np.zeros(m.nbface)
fx = np.ones(m.ncell) if a.forcing == "cpg" else np.zeros(m.ncell)
s = PISO25(m, a.nz, LZ, NU, a.dt, BC(m, kd, z0), BC(m, kd, z0), BC(m, kd, z0), BC(m), body_force=(fx, np.zeros(m.ncell), np.zeros(m.ncell)), n_nonorth=n_nonorth, device=a.device, solver=("amg" if a.device == "gpu" else "lu"), mom_rtol=1e-7)
s.sgs_model = a.model; s.cfl_max = a.cfl_max
if a.model == "smagorinsky":
    ywall = np.minimum(m.centroid[:, 1], LY - m.centroid[:, 1]); s.sgs_kw = dict(damping=ywall * RE_TAU)
if a.forcing == "mf": s.set_mass_flow(a.Ub)
print(f"channel Re_tau {RE_TAU:.0f} on {a.mesh or f'{a.nx}x{a.ny}'} {a.cells} ({m.ncell} cells, {int((m.nvert==3).sum())} triangles" + (f" from y+ {m.centroid[m.nvert==3,1].min()*RE_TAU:.0f}" if (m.nvert==3).any() else "") + f", orth min {m.orth.min():.3f}, n_nonorth {n_nonorth}) x {a.nz} modes: dx+ {LX/a.nx*RE_TAU:.1f}  dy+ wall {dy[0]*RE_TAU:.2f} centre {dy.max()*RE_TAU:.1f} (bins)  dz+ {LZ/a.nz*RE_TAU:.1f}  dt {a.dt}  model {a.model}  forcing {a.forcing}", flush=True)
# ---- reference profiles
def load_dns(path):
    d = np.load(path); yd, nu, n = d["y"], float(d["nu"]), int(d["nsamp"]); p = d["sums"] / n
    p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]]); ym = yd[-1] - yd; o = np.argsort(ym); q = np.empty_like(p)
    for kk in range(4): q[kk] = np.interp(yd, ym[o], p[kk][o])
    q[4] = -np.interp(yd, ym[o], p[4][o]); f = 0.5 * (p + q); h = yd <= 0.5 * yd[-1] + 1e-12
    ut = float(d["utau_series"][:, 1].mean()) if "utau_series" in d.files and d["utau_series"].ndim == 2 and d["utau_series"].shape[1] > 1 else float(np.sqrt(nu * abs((f[0][1] - f[0][0]) / (yd[1] - yd[0]))))
    return yd[h], f[0][h], np.sqrt(np.maximum(f[1][h], 0)), np.sqrt(np.maximum(f[2][h], 0)), np.sqrt(np.maximum(f[3][h], 0)), -f[4][h], ut
def load_mkm_mean():
    """MKM chan395.means: y, y+, Umean (in u_tau), ... on the half channel. Returns (y, U) with u_tau = 1."""
    A = np.loadtxt(os.path.join(MKM395, "chan395.means")); return A[:, 0], A[:, 2]
def load_mkm():
    """(y, U, u', v', w', -<u'v'>, u_tau) in the FOSLS loader's convention, u_tau = 1 (MKM normalises by u_tau)."""
    A = np.loadtxt(os.path.join(MKM395, "chan395.means")); B = np.loadtxt(os.path.join(MKM395, "chan395.reystress"))
    y, U = A[:, 0], A[:, 2]; uu, vv, ww, uv = B[:, 2], B[:, 3], B[:, 4], B[:, 5]                # R_uu R_vv R_ww R_uv
    return y, U, np.sqrt(np.maximum(uu, 0)), np.sqrt(np.maximum(vv, 0)), np.sqrt(np.maximum(ww, 0)), -uv, 1.0
# ---- initial condition: the SEM DNS field, Delaunay in x-y once, periodic linear in z
if a.restart:
    s.load(a.restart); print(f"  restart from {a.restart} at t = {s.time:.3f}, step {s.nstep}", flush=True)
else:
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    src = np.load("results/minchan_re180_field.npz"); X, Y, Z, U = src["x"], src["y"], src["z"], src["u"]; Lz_src = float(src["Lz"]); nzs = len(Z)
    pts = np.column_stack([X.ravel(), Y.ravel()]); tri = Delaunay(pts); C = m.centroid
    zf = (s.z / Lz_src * nzs) % nzs; k0 = np.floor(zf).astype(int) % nzs; k1 = (k0 + 1) % nzs; w1 = zf - np.floor(zf)
    plane = {}
    def src_plane(c, k):
        if (c, k) not in plane:
            vals = U[..., c, k].ravel(); got = LinearNDInterpolator(tri, vals)(C)
            if np.isnan(got).any(): got = np.where(np.isnan(got), NearestNDInterpolator(pts, vals)(C), got)
            plane[(c, k)] = got
        return plane[(c, k)]
    for c, f in enumerate((s.u, s.v, s.w)):
        F0 = np.empty((m.ncell, a.nz))
        for j in range(a.nz): F0[:, j] = (1 - w1[j]) * src_plane(c, k0[j]) + w1[j] * src_plane(c, k1[j])
        if c == 0 and RE_TAU != 180:
            # the 180 field's fluctuations (in wall units they change little between 180 and 395) on top of the
            # target mean: replace the plane-mean U_180(y) by the reference mean at the new Re_tau
            yd, Ud, *_ = load_dns(DNS); U180 = 0.5 * (np.interp(np.minimum(C[:, 1], LY - C[:, 1]), yd, Ud) + np.interp(np.minimum(C[:, 1], LY - C[:, 1]), yd, Ud))
            yr, Ur = load_mkm_mean(); Unew = np.interp(np.minimum(C[:, 1], LY - C[:, 1]), yr, Ur)
            F0 = F0 - U180[:, None] + Unew[:, None]
        f[:] = s.asdev(F0)
    uh0 = s.host(s.u); print(f"  DNS initial condition: U_bulk {s.bulk_velocity():.3f} (target {a.Ub}), |u|max {np.abs(uh0).max():.2f}, E3d fraction {1 - (uh0.mean(axis=1)**2).sum()/ (uh0**2).sum(axis=1).sum():.3f}", flush=True)
# ---- diagnostics
bot = m.bfaces[m.btag[m.bfaces] == 3]; top = m.bfaces[m.btag[m.bfaces] == 4]
def wall_stress():
    tw = []; uh = s.host(s.u)
    for faces in (bot, top):
        o = m.owner[faces]; dn = np.abs(m.fcentre[faces, 1] - m.centroid[o, 1])
        tw.append(float(NU * (np.abs(uh[o]) / dn[:, None]).mean()))
    return 0.5 * (tw[0] + tw[1])
ix = np.floor(m.centroid[:, 0] / (LX / a.nx)).astype(int); st_bins = Stats(s, edges=(y_edges if a.mesh else None))
quad_cells = np.flatnonzero(m.nvert == 4) if not a.mesh else np.array([], int); tri_pairs = np.flatnonzero(m.nvert == 3).reshape(-1, 2) if not a.mesh else np.zeros((0, 2), int)   # rect_mesh appends the two triangles of a split quad consecutively; a gmsh mesh has no pairs
jq = np.searchsorted(np.unique(np.round(m.centroid[quad_cells, 1], 8)), np.round(m.centroid[quad_cells, 1], 8)) if len(quad_cells) else None
sign_q = (-1.0) ** (ix[quad_cells] + jq) if len(quad_cells) else None
def checkerboard():
    """two-colour pressure content relative to the pressure fluctuation: on the quad cells the
    (-1)^(i+j) mode amplitude; on the triangle pairs the rms half-difference of the two halves of a
    split quad (the two-colour component of a non-bipartite mesh). Returns (quad, tri, high-pass share)."""
    ph = s.host(s.p); pp = ph - ph.mean(axis=0, keepdims=True); prms = max(np.sqrt((pp ** 2).mean()), 1e-300)
    cbq = np.abs((sign_q[:, None] * pp[quad_cells]).mean(axis=0)).mean() / prms if len(quad_cells) else 0.0
    cbt = np.sqrt(((0.5 * (pp[tri_pairs[:, 0]] - pp[tri_pairs[:, 1]])) ** 2).mean()) / prms if len(tri_pairs) else 0.0
    i = m.interior; nb = np.zeros_like(pp); cnt = np.zeros(m.ncell)
    np.add.at(nb, m.owner[i], pp[m.neigh[i]]); np.add.at(nb, m.neigh[i], pp[m.owner[i]]); np.add.at(cnt, m.owner[i], 1); np.add.at(cnt, m.neigh[i], 1)
    hp = pp - nb / np.maximum(cnt, 1)[:, None]
    return cbq, cbt, np.sqrt((hp ** 2).mean()) / prms
def cfl():
    return s.courant()                                                     # the face-flux Courant number the solver limits
nsteps = a.nsteps or int(round((a.T - s.time) / a.dt)); stats = Stats(s, edges=(y_edges if a.mesh else None)); t0 = time.time(); hist = []; diverged = False
k = -1
while True:
    k += 1
    if a.nsteps and k >= a.nsteps: break
    if not a.nsteps and s.time >= a.T - 1e-12: break
    s.step()
    if not bool(s.xp.isfinite(s.u).all()): print(f"  DIVERGED at step {s.nstep}, t = {s.time:.4f}, last CFL {s.cfl_last:.2f}", flush=True); diverged = True; break
    if s.time >= a.t_stats - 1e-12: stats.sample()
    tw = wall_stress(); hist.append((s.time, tw, s.bulk_velocity(), s.energy() / (LX * LY * LZ), float(s.nu_t.mean()) / NU, float(s.nu_t.max()) / NU))
    if (k + 1) % a.report == 0:
        cbq, cbt, hp = checkerboard()
        print(f"  t={s.time:7.3f}  u_tau {np.sqrt(tw):.4f} (Re_tau {np.sqrt(tw)/NU:6.1f})  U_b {hist[-1][2]:.3f}  E/V {hist[-1][3]:.3f}  <nu_t>/nu {hist[-1][4]:.3f} max {hist[-1][5]:.2f}  CFL {cfl():.2f} dt {s.dt:.5f}  p two-colour quad {cbq:.4f} tri {cbt:.4f} hp {hp:.3f}  stats {stats.n}  ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
    if (k + 1) % a.checkpoint == 0: s.save(f"{a.outdir}/{tag}_ckpt.npz")
if not diverged: s.save(f"{a.outdir}/{tag}_final.npz")
# ---- statistics against the DNS
pr = stats.profiles(); y = pr["y"]
if stats.n and not diverged:
    h = np.array(hist); win = h[:, 0] >= a.t_stats; ut = np.sqrt(h[win, 1].mean()); re_tau = ut / NU
    # fold onto the lower half
    half = y <= 1.0; yl = y[half]
    fold = lambda k, odd=False: 0.5 * (pr[k][half] + (-1 if odd else 1) * np.interp(LY - yl, y, pr[k]))
    U = fold("U"); urms = np.sqrt(np.maximum(fold("uu"), 0)); vrms = np.sqrt(np.maximum(fold("vv"), 0)); wrms = np.sqrt(np.maximum(fold("ww"), 0)); uv = fold("uv", odd=True)
    yp = yl * re_tau; Up = U / ut
    print(f"\nRESULT {tag}: window t={a.t_stats}..{s.time:.1f} ({stats.n} samples)  u_tau {ut:.4f}  Re_tau {re_tau:.1f} ({(re_tau/RE_TAU-1)*100:+.1f}%)  U_b {h[win,2].mean():.3f}  U_c+ {Up[-1]:.2f}  <nu_t>/nu {h[win,4].mean():.3f}", flush=True)
    log = np.log(yp) / 0.41 + 5.2; sel = (yp > 30) & (yp < 100)
    print(f"   log region (30 < y+ < 100): U+ - loglaw mean {(Up[sel]-log[sel]).mean():+.3f} u_tau units;  u_rms+ peak {(urms/ut).max():.3f} at y+ {yp[np.argmax(urms)]:.1f};  v_rms+ max {(vrms/ut).max():.3f};  w_rms+ max {(wrms/ut).max():.3f};  -<uv>+ max {(-uv/ut**2).max():.3f}", flush=True)
    if RE_TAU == 180 and os.path.exists(DNS) or RE_TAU != 180 and os.path.exists(MKM395):
        yd, Ud, ud, vd, wd, uvd, utd = load_dns(DNS) if RE_TAU == 180 else load_mkm(); ypd = yd * utd / NU if RE_TAU == 180 else yd * 392.24
        Ud_i = np.interp(yp, ypd, Ud / utd); dU = Up[sel] - Ud_i[sel]
        print(f"   vs {'FOSLS DNS' if RE_TAU == 180 else 'MKM 1999 Re_tau 392'} (u_tau {utd:.4f}): U+ difference in the log region mean {dU.mean():+.3f} max {np.abs(dU).max():.3f};  DNS u_rms+ peak {(ud/utd).max():.3f}, ours {(urms/ut).max():.3f} ({((urms/ut).max()/(ud/utd).max()-1)*100:+.1f}%);  DNS -<uv>+ max {(uvd/utd**2).max():.3f}, ours {(-uv/ut**2).max():.3f}", flush=True)
    cbq, cbt, hp = checkerboard(); print(f"   pressure two-colour mode: quad cells {cbq*100:.2f}% of p_rms, triangle pairs {cbt*100:.2f}% (criterion < 1%), face high-pass share {hp:.3f}", flush=True)
    np.savez(f"{a.outdir}/{tag}_stats.npz", y=yl, yp=yp, U=U, urms=urms, vrms=vrms, wrms=wrms, uv=uv, prms=np.sqrt(np.maximum(fold("pp"), 0)), ut=ut, re_tau=re_tau, hist=h, nsamp=stats.n, nu=NU, model=a.model, forcing=a.forcing, cells=a.cells, re_tau_target=RE_TAU, Lx=LX, Lz=LZ)
