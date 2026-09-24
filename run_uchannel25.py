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
RE_TAU = 180.0; NU = 1.0 / RE_TAU; LX = np.pi; LY = 2.0; LZ = 0.34 * np.pi
DNS = "results/fosls_chan180_stats_t5.2_30.npz"        # SEM FOSLS DNS run02, window t = 5.2..30 (the K/E paths are fractional-step runs, not the reference)
ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=24); ap.add_argument("--ny", type=int, default=80); ap.add_argument("--nz", type=int, default=32)
ap.add_argument("--dy0plus", type=float, default=1.0); ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--T", type=float, default=30.0); ap.add_argument("--t-stats", type=float, default=10.0)
ap.add_argument("--model", default="wale", choices=["wale", "smagorinsky", "none"])
ap.add_argument("--forcing", default="cpg", choices=["cpg", "mf"]); ap.add_argument("--Ub", type=float, default=15.63, help="bulk velocity for --forcing mf (DNS: 15.63)")
ap.add_argument("--tag", default=None); ap.add_argument("--checkpoint", type=int, default=2500); ap.add_argument("--restart", default=None)
ap.add_argument("--report", type=int, default=500); ap.add_argument("--nsteps", type=int, default=None)
a = ap.parse_args()
tag = a.tag or f"uchan_{a.nx}x{a.ny}x{a.nz}_{a.model}_{a.forcing}"
# ---- mesh: tanh clustering in y with the first cell dy0+ wall units high
def first_cell(beta): return LY * 0.5 * (1 + np.tanh(beta * (2.0 / a.ny - 1)) / np.tanh(beta))
lo, hi = 1e-3, 20.0; target = a.dy0plus / RE_TAU
for _ in range(200):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if first_cell(mid) > target else (lo, mid)
beta = 0.5 * (lo + hi)
m = rect_mesh(a.nx, a.ny, 0, LX, 0, LY, cells="quad", cluster_y=beta); m.make_periodic(1, 2, (LX, 0.0))
ys = np.unique(np.round(m.centroid[:, 1], 10)); dy = np.diff(np.concatenate([[0], 0.5 * (ys[1:] + ys[:-1]), [LY]]))
kd = np.full(m.nbface, DIRICHLET); z0 = np.zeros(m.nbface)
fx = np.ones(m.ncell) if a.forcing == "cpg" else np.zeros(m.ncell)
s = PISO25(m, a.nz, LZ, NU, a.dt, BC(m, kd, z0), BC(m, kd, z0), BC(m, kd, z0), BC(m), body_force=(fx, np.zeros(m.ncell), np.zeros(m.ncell)), n_nonorth=1)
s.sgs_model = a.model
if a.model == "smagorinsky":
    ywall = np.minimum(m.centroid[:, 1], LY - m.centroid[:, 1]); s.sgs_kw = dict(damping=ywall * RE_TAU)
if a.forcing == "mf": s.set_mass_flow(a.Ub)
print(f"channel Re_tau {RE_TAU:.0f} on {a.nx}x{a.ny} quads x {a.nz} modes: dx+ {LX/a.nx*RE_TAU:.1f}  dy+ wall {dy[0]*RE_TAU:.2f} centre {dy.max()*RE_TAU:.1f}  dz+ {LZ/a.nz*RE_TAU:.1f}  dt {a.dt}  model {a.model}  forcing {a.forcing}", flush=True)
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
        for j in range(a.nz): f[:, j] = (1 - w1[j]) * src_plane(c, k0[j]) + w1[j] * src_plane(c, k1[j])
    print(f"  DNS initial condition: U_bulk {s.bulk_velocity():.3f} (DNS 15.63), |u|max {np.abs(s.u).max():.2f}, E3d fraction {1 - (s.u.mean(axis=1)**2).sum()/ (s.u**2).sum(axis=1).sum():.3f}", flush=True)
# ---- diagnostics
bot = m.bfaces[m.btag[m.bfaces] == 3]; top = m.bfaces[m.btag[m.bfaces] == 4]
def wall_stress():
    tw = []
    for faces in (bot, top):
        o = m.owner[faces]; dn = np.abs(m.fcentre[faces, 1] - m.centroid[o, 1])
        tw.append(float(NU * (np.abs(s.u[o]) / dn[:, None]).mean()))
    return 0.5 * (tw[0] + tw[1])
ix = np.floor(m.centroid[:, 0] / (LX / a.nx)).astype(int); st_bins = Stats(s); sign = (-1.0) ** (ix + st_bins.inv)
def checkerboard():
    """amplitude of the (-1)^(i+j) pressure mode relative to the pressure fluctuation, and the
    face-neighbour high-pass share, both plane-averaged."""
    pp = s.p - s.p.mean(axis=0, keepdims=True); prms = np.sqrt((pp ** 2).mean())
    cb = np.abs((sign[:, None] * pp).mean(axis=0)).mean()
    i = m.interior; nb = np.zeros_like(pp); cnt = np.zeros(m.ncell)
    np.add.at(nb, m.owner[i], pp[m.neigh[i]]); np.add.at(nb, m.neigh[i], pp[m.owner[i]]); np.add.at(cnt, m.owner[i], 1); np.add.at(cnt, m.neigh[i], 1)
    hp = pp - nb / np.maximum(cnt, 1)[:, None]
    return cb / max(prms, 1e-300), np.sqrt((hp ** 2).mean()) / max(prms, 1e-300)
def cfl():
    hx = LX / a.nx; hy = np.repeat(dy, 1)[st_bins.inv]
    return float(max((np.abs(s.u) * a.dt / hx).max(), (np.abs(s.v) * a.dt / hy[:, None]).max(), (np.abs(s.w) * a.dt / (LZ / a.nz)).max()))
nsteps = a.nsteps or int(round((a.T - s.time) / a.dt)); stats = Stats(s); t0 = time.time(); hist = []
for k in range(nsteps):
    s.step()
    if not np.isfinite(s.u).all(): print(f"  DIVERGED at step {s.nstep}", flush=True); break
    if s.time >= a.t_stats - 1e-12: stats.sample()
    tw = wall_stress(); hist.append((s.time, tw, s.bulk_velocity(), s.energy() / (LX * LY * LZ), float(s.nu_t.mean()) / NU, float(s.nu_t.max()) / NU))
    if (k + 1) % a.report == 0:
        cb, hp = checkerboard()
        print(f"  t={s.time:7.3f}  u_tau {np.sqrt(tw):.4f} (Re_tau {np.sqrt(tw)/NU:6.1f})  U_b {hist[-1][2]:.3f}  E/V {hist[-1][3]:.3f}  <nu_t>/nu {hist[-1][4]:.3f} max {hist[-1][5]:.2f}  CFL {cfl():.2f}  p two-colour {cb:.4f} hp {hp:.3f}  stats {stats.n}  ({(time.time()-t0)/(k+1)*1e3:.0f} ms/step)", flush=True)
    if (k + 1) % a.checkpoint == 0: s.save(f"results/{tag}_ckpt.npz")
s.save(f"results/{tag}_final.npz")
# ---- statistics against the DNS
pr = stats.profiles(); y = pr["y"]
def load_dns(path):
    d = np.load(path); yd, nu, n = d["y"], float(d["nu"]), int(d["nsamp"]); p = d["sums"] / n
    p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]]); ym = yd[-1] - yd; o = np.argsort(ym); q = np.empty_like(p)
    for kk in range(4): q[kk] = np.interp(yd, ym[o], p[kk][o])
    q[4] = -np.interp(yd, ym[o], p[4][o]); f = 0.5 * (p + q); h = yd <= 0.5 * yd[-1] + 1e-12
    ut = float(d["utau_series"][:, 1].mean()) if "utau_series" in d.files and d["utau_series"].ndim == 2 and d["utau_series"].shape[1] > 1 else float(np.sqrt(nu * abs((f[0][1] - f[0][0]) / (yd[1] - yd[0]))))
    return yd[h], f[0][h], np.sqrt(np.maximum(f[1][h], 0)), np.sqrt(np.maximum(f[2][h], 0)), np.sqrt(np.maximum(f[3][h], 0)), -f[4][h], ut
if stats.n:
    h = np.array(hist); win = h[:, 0] >= a.t_stats; ut = np.sqrt(h[win, 1].mean()); re_tau = ut / NU
    # fold onto the lower half
    half = y <= 1.0; yl = y[half]; U = 0.5 * (pr["U"][half] + pr["U"][::-1][half]); urms = np.sqrt(np.maximum(0.5 * (pr["uu"][half] + pr["uu"][::-1][half]), 0))
    vrms = np.sqrt(np.maximum(0.5 * (pr["vv"][half] + pr["vv"][::-1][half]), 0)); wrms = np.sqrt(np.maximum(0.5 * (pr["ww"][half] + pr["ww"][::-1][half]), 0)); uv = 0.5 * (pr["uv"][half] - pr["uv"][::-1][half])
    yp = yl * re_tau; Up = U / ut
    print(f"\nRESULT {tag}: window t={a.t_stats}..{s.time:.1f} ({stats.n} samples)  u_tau {ut:.4f}  Re_tau {re_tau:.1f} ({(re_tau/RE_TAU-1)*100:+.1f}%)  U_b {h[win,2].mean():.3f}  U_c+ {Up[-1]:.2f}  <nu_t>/nu {h[win,4].mean():.3f}", flush=True)
    log = np.log(yp) / 0.41 + 5.2; sel = (yp > 30) & (yp < 100)
    print(f"   log region (30 < y+ < 100): U+ - loglaw mean {(Up[sel]-log[sel]).mean():+.3f} u_tau units;  u_rms+ peak {(urms/ut).max():.3f} at y+ {yp[np.argmax(urms)]:.1f};  v_rms+ max {(vrms/ut).max():.3f};  w_rms+ max {(wrms/ut).max():.3f};  -<uv>+ max {(-uv/ut**2).max():.3f}", flush=True)
    if os.path.exists(DNS):
        yd, Ud, ud, vd, wd, uvd, utd = load_dns(DNS); ypd = yd * utd / NU
        Ud_i = np.interp(yp, ypd, Ud / utd); dU = Up[sel] - Ud_i[sel]
        print(f"   vs DNS (u_tau {utd:.4f}): U+ difference in the log region mean {dU.mean():+.3f} max {np.abs(dU).max():.3f};  DNS u_rms+ peak {(ud/utd).max():.3f}, ours {(urms/ut).max():.3f} ({((urms/ut).max()/(ud/utd).max()-1)*100:+.1f}%);  DNS -<uv>+ max {(uvd/utd**2).max():.3f}, ours {(-uv/ut**2).max():.3f}", flush=True)
    cb, hp = checkerboard(); print(f"   pressure two-colour mode {cb*100:.2f}% of p_rms (criterion < 1%), face high-pass share {hp:.3f}", flush=True)
    np.savez(f"results/{tag}_stats.npz", y=yl, yp=yp, U=U, urms=urms, vrms=vrms, wrms=wrms, uv=uv, ut=ut, re_tau=re_tau, hist=h, nsamp=stats.n, nu=NU, model=a.model, forcing=a.forcing)
