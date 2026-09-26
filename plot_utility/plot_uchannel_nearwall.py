"""Usage: python plot_utility/plot_uchannel_nearwall.py [yplus] [run tag] [SEM state .npz]
Near-wall instantaneous planes, Re_tau 180 channel: pressure, streamwise velocity and streamwise
vorticity fluctuations on the wall-parallel plane y+ ~ 12, side by side for
   (1) the 2.5D unstructured LES (24x80 quads x 32 modes, WALE, t = 30),
   (2) the SEM field given as the third argument -- the FOSLS DNS state (Google Drive/My Drive/lssem_dns) is the
       intended reference; without it the K-path fractional-step SEM field used as the initial condition (t = 18) is shown.
Fluctuations about the plane mean, in wall units: p'/u_tau^2, u'/u_tau, omega_x nu/u_tau^2. Streaks at
lambda_z+ ~ 100 mean about two pairs across Lz+ = 192."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from src.umesh import rect_mesh
from src.uops import Gradient
RE_TAU, NU, LX, LY, LZ = 180.0, 1 / 180, np.pi, 2.0, 0.34 * np.pi
YPLUS = float(_sys.argv[1]) if len(_sys.argv) > 1 else 12.0
TAG = _sys.argv[2] if len(_sys.argv) > 2 else "uchan_24x80x32_wale_cpg"
cols = []
# ---- (1) ours
nx, ny, nz = 24, 80, 32
def first_cell(beta): return LY * 0.5 * (1 + np.tanh(beta * (2.0 / ny - 1)) / np.tanh(beta))
lo, hi = 1e-3, 20.0
for _ in range(200):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if first_cell(mid) > 1.0 / RE_TAU else (lo, mid)
m = rect_mesh(nx, ny, 0, LX, 0, LY, cells="quad", cluster_y=0.5 * (lo + hi)); m.make_periodic(1, 2, (LX, 0.0))
d = np.load(f"results/{TAG}_final.npz"); u, v, w, p = d["u"], d["v"], d["w"], d["p"]; t_ours = float(d["time"])
ys = np.unique(np.round(m.centroid[:, 1], 10)); jrow = np.searchsorted(ys, np.round(m.centroid[:, 1], 10)); ix = np.floor(m.centroid[:, 0] / (LX / nx)).astype(int)
g = Gradient(m); gw = g(w, np.zeros((m.nbface, nz)))                       # walls: w = 0 (Dirichlet), the only boundaries
kz = 2 * np.pi / LZ * np.arange(nz // 2 + 1); dvdz = np.fft.irfft(np.fft.rfft(v, axis=1) * 1j * kz, n=nz, axis=1)
omx = gw[:, 1] - dvdz
j = int(np.argmin(np.abs(ys * RE_TAU - YPLUS))); sel = jrow == j; order = np.argsort(ix[sel])
def zpad(f, mult=3):
    """Spectral refinement in z (exact for the solver's Fourier representation), for plotting only."""
    fh = np.fft.rfft(f, axis=1); out = np.zeros((f.shape[0], (mult * nz) // 2 + 1), complex); out[:, :fh.shape[1]] = fh; out[:, fh.shape[1] - 1] = 0
    return np.fft.irfft(out, n=mult * nz, axis=1) * mult
def plane(f): return zpad(f[sel][order])                                   # (nx, 3 nz): cell values in x, spectral in z
# u_tau from the wall flux of this field
bot = m.bfaces[m.btag[m.bfaces] == 3]; o = m.owner[bot]; ut = float(np.sqrt(NU * (np.abs(u[o]) / np.abs(m.fcentre[bot, 1] - m.centroid[o, 1])[:, None]).mean()))
x_o = (np.arange(nx) + 0.5) * LX / nx; z_o = np.arange(3 * nz) * LZ / (3 * nz)
cols.append(dict(name=f"2.5D LES, 24x80 quads x 32 modes, WALE, t = {t_ours:.0f}\n$u_\\tau$ {ut:.3f}, plane $y^+$ = {ys[j]*RE_TAU:.1f}; cells $\\Delta x^+$ 23.6, z spectral", x=x_o, z=z_o, ut=ut,
                 p=plane(p), u=plane(u), omx=plane(omx)))
# ---- (2) DNS field (initial condition)
s = np.load("results/minchan_re180_field.npz"); X, Y, Z, U, P = s["x"], s["y"], s["z"], s["u"], s["p"]; t_dns = 18.0; OMX_DNS = None
if len(_sys.argv) > 3:          # a different SEM state (same minchan mesh: 108 elements x 9 x 9, 17 rfft modes -> 32 planes), e.g. the t = 30 FOSLS result
    st = np.load(_sys.argv[3]); t_dns = float(st["t"]); OMX_DNS = None
    if st["U"].shape[-2] == 14:      # FOSLS checkpoint: 14 split-real fields (u v w ox oy oz p, then imaginary parts), z modes last
        Uc = st["U"][..., :7, :] + 1j * st["U"][..., 7:, :]; phys = np.fft.irfft(Uc, n=32, axis=-1)          # (108, 9, 9, 7, 32)
        U = phys[..., :3, :]; P = phys[..., 6, :]; OMX_DNS = phys[..., 3, :]                                    # omega_x is a FOSLS primary unknown
    else:                            # fractional-step state: U (.., 3, 17), p (.., 1, 17)
        U = np.fft.irfft(st["U"], n=32, axis=-1); P = np.fft.irfft(st["p"][..., 0, :], n=32, axis=-1)
    assert U.shape[:3] == X.shape, f"state mesh {U.shape[:3]} differs from the cached node coordinates {X.shape}"
pts = np.column_stack([X.ravel(), Y.ravel()]); tri = Delaunay(pts)
y_star = ys[j]; xd = (np.arange(96) + 0.5) * LX / 96
def dns_plane(c, yy, k): return LinearNDInterpolator(tri, (U[..., c, k] if c < 3 else P[..., k]).ravel())(np.column_stack([xd, np.full_like(xd, yy)]))
pd = np.array([dns_plane(3, ys[j], k) for k in range(32)]).T
ud = np.array([dns_plane(0, y_star, k) for k in range(32)]).T; vd = np.array([dns_plane(1, y_star, k) for k in range(32)]).T
dy = 0.004; wd_p = np.array([dns_plane(2, y_star + dy, k) for k in range(32)]).T; wd_m = np.array([dns_plane(2, y_star - dy, k) for k in range(32)]).T
kzd = 2 * np.pi / float(s["Lz"]) * np.arange(17); dvdz_d = np.fft.irfft(np.fft.rfft(vd, axis=1) * 1j * kzd, n=32, axis=1)
omx_d = (wd_p - wd_m) / (2 * dy) - dvdz_d
if OMX_DNS is not None:                                                    # the solver's own omega_x, interpolated like the rest
    omx_d = np.array([LinearNDInterpolator(tri, OMX_DNS[..., k].ravel())(np.column_stack([xd, np.full_like(xd, y_star)])) for k in range(32)]).T
# DNS u_tau from its own wall gradient: u at y = 0.5/180 and 1.0/180
u1 = np.array([dns_plane(0, 0.5 / RE_TAU, k) for k in range(32)]).mean(); utd = float(np.sqrt(NU * u1 / (0.5 / RE_TAU)))
dns_name = (f"SEM fractional-step field (K path), t = {t_dns:.0f}, the initial condition" if len(_sys.argv) <= 3 else f"SEM FOSLS DNS run02, t = {t_dns:.0f}" + (" ($\\omega_x$ = its own unknown)" if OMX_DNS is not None else "")) + f"\n$u_\\tau$ {utd:.3f}, same y; sampled at $\\Delta x^+$ 5.9"
cols.append(dict(name=dns_name, x=xd, z=Z, ut=utd, p=pd, u=ud, omx=omx_d))
# ---- figure
rows = [("p' / $u_\\tau^2$", "p", 1.0, 2), ("u' / $u_\\tau$", "u", 1.0, 1), ("$\\omega_x \\nu / u_\\tau^2$", "omx", NU, 2)]
fig, ax = plt.subplots(3, len(cols), figsize=(6.2 * len(cols), 9.5), squeeze=False)
for r, (lab, key, fac, pw) in enumerate(rows):
    vals = [c for c in cols if c[key] is not None]
    lim = max(np.abs((c[key] - c[key].mean()) * fac / c["ut"] ** pw).max() for c in vals) if vals else 1
    for cidx, c in enumerate(cols):
        a = ax[r, cidx]
        if c[key] is None: a.text(0.5, 0.5, "not stored in the DNS field file", ha="center", va="center", transform=a.transAxes); a.set_axis_off(); continue
        f = (c[key] - c[key].mean()) * fac / c["ut"] ** pw
        im = a.pcolormesh(c["x"] * RE_TAU, c["z"] * RE_TAU, f.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
        a.set_aspect("equal"); a.set_xlabel("$x^+$"); a.set_ylabel("$z^+$"); plt.colorbar(im, ax=a, shrink=0.8, pad=0.02)
        a.set_title((c["name"] + "\n" if r == 0 else "") + f"{lab}   rms {f.std():.3f}", fontsize=9)
plt.suptitle(f"Near-wall plane $y^+ \\approx$ {YPLUS:.0f}: instantaneous fluctuations about the plane mean, in wall units (bilinear shading in x, both columns). $L_z^+$ = 192.", fontsize=11)
plt.tight_layout(); out = f"figures/uchannel_re180_nearwall_yp{YPLUS:.0f}.png"; plt.savefig(out, dpi=140); print("wrote", out)
for c in cols:
    print(f"  {c['name'].splitlines()[0]:60s} u' rms/u_tau {((c['u']-c['u'].mean())/c['ut']).std():.3f}   omega_x rms nu/u_tau^2 {((c['omx']-c['omx'].mean())*NU/c['ut']**2).std():.4f}" + (f"   p' rms/u_tau^2 {((c['p']-c['p'].mean())/c['ut']**2).std():.3f}" if c['p'] is not None else ""))
