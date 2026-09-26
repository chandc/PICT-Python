"""Usage: python plot_utility/plot_uchannel_re395_nearwall_dns.py
Near-wall u, p and streamwise vorticity of the Re_tau 395 2.5D LES against DNS.
There is no DNS *field* at Re_tau 395 (MKM 1999 publish statistics), so the comparison is made twice:
  (1) figures/uchannel_re395_nearwall_vs_dns.png -- pictures: the LES plane at y+ ~ 10 beside the FOSLS SEM DNS
      plane at Re_tau 180 (t = 30), both in wall units at the same scale. Near-wall structure scales in wall
      units, so streak spacing, pressure-patch size and vortex size must agree even though Re_tau differs.
  (2) figures/uchannel_re395_nearwall_stats.png -- the same three quantities against the MKM 1999 Re_tau 395 data
      at the same Re: premultiplied spanwise spectra of u and p at y+ 9.5 and 20 (chan395.zspec), spanwise two-point
      correlations R_uu, R_pp at y+ 9.5 (chan395.zcorr), and the near-wall rms profiles of u', p' (20-turnover
      statistics; chan395.reystress, .velp) and omega_x (single field, both walls folded; chan395.vortvar)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from src.umesh import rect_mesh
from src.uops import Gradient
BASE = "results/uchan395/uchan395_96x160x128_wale_cpg/uchan395_96x160x128_wale_cpg"
FOSLS = "/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/scratch/_dns_drive/checkpoint_0037500.npz"   # Re_tau 180 SEM FOSLS run02, t = 30
MKM = "reference/mkm_chan395"
# ---------------- LES Re_tau 395
RE, NU, LX, LY, LZ = 395.0, 1 / 395, np.pi, 2.0, 0.34 * np.pi; nx, ny, nz = 96, 160, 128
def first_cell(beta): return LY * 0.5 * (1 + np.tanh(beta * (2.0 / ny - 1)) / np.tanh(beta))
lo, hi = 1e-3, 20.0
for _ in range(200):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if first_cell(mid) > 1.0 / RE else (lo, mid)
m = rect_mesh(nx, ny, 0, LX, 0, LY, cells="quad", cluster_y=0.5 * (lo + hi)); m.make_periodic(1, 2, (LX, 0.0))
d = np.load(BASE + "_final.npz"); u, v, w, p = d["u"], d["v"], d["w"], d["p"]; t = float(d["time"]); st = np.load(BASE + "_stats.npz"); ut_w = float(st["ut"])
ys = np.unique(np.round(m.centroid[:, 1], 8)); jrow = np.argmin(np.abs(m.centroid[:, 1][:, None] - ys[None]), axis=1); ix = np.floor(m.centroid[:, 0] / (LX / nx)).astype(int)
bot = m.bfaces[m.btag[m.bfaces] == 3]; o = m.owner[bot]; ut = float(np.sqrt(NU * (np.abs(u[o]) / np.abs(m.fcentre[bot, 1] - m.centroid[o, 1])[:, None]).mean()))
kz = 2 * np.pi / LZ * np.arange(nz // 2 + 1); gw = Gradient(m)(w, np.zeros((m.nbface, nz))); omx = gw[:, 1] - np.fft.irfft(np.fft.rfft(v, axis=1) * 1j * kz, n=nz, axis=1)
def plane(f, j): sel = jrow == j; return f[sel][np.argsort(ix[sel])]                       # (nx, nz)
def zpad(f, mult=2):
    fh = np.fft.rfft(f, axis=1); out = np.zeros((f.shape[0], (mult * f.shape[1]) // 2 + 1), complex); out[:, :fh.shape[1]] = fh; out[:, fh.shape[1] - 1] = 0
    return np.fft.irfft(out, n=mult * f.shape[1], axis=1) * mult
def fluct(F): return F - F.mean()
def power(F):
    """One-sided spanwise power per mode, averaged over the rows of F (.., nz): sum over modes = variance."""
    fh = np.fft.rfft(F - F.mean(axis=-1, keepdims=True), axis=-1) / F.shape[-1]; P = 2 * np.abs(fh) ** 2; P[..., 0] /= 2
    if F.shape[-1] % 2 == 0: P[..., -1] /= 2
    return P.reshape(-1, P.shape[-1]).mean(axis=0)
def corr(F):
    P = power(F); Pf = P.copy(); Pf[1:-1] /= 2; R = np.fft.irfft(Pf, n=F.shape[-1]) * F.shape[-1]; return R / R[0]   # circular autocorrelation, normalised
# ---------------- MKM Re_tau 395 reference
A = np.loadtxt(f"{MKM}/chan395.means"); B = np.loadtxt(f"{MKM}/chan395.reystress"); Vp = np.loadtxt(f"{MKM}/chan395.velp"); W = np.loadtxt(f"{MKM}/chan395.vortvar"); ypd = A[:, 1]
mkm = dict(u=np.sqrt(B[:, 2]), p=np.sqrt(Vp[:, 5]), omx=np.sqrt(W[:, 2]))
def mkm_at(key, yp): return float(np.interp(yp, ypd, mkm[key]))
# ---------------- FOSLS DNS Re_tau 180 field (t = 30), plane at y+ ~ 10
RE_D, NU_D = 180.0, 1 / 180
s = np.load("results/minchan_re180_field.npz"); X, Y, Z = s["x"], s["y"], s["z"]; LZ_D = float(s["Lz"]); LX_D = np.pi
stt = np.load(FOSLS); t_dns = float(stt["t"]); Uc = stt["U"][..., :7, :] + 1j * stt["U"][..., 7:, :]; phys = np.fft.irfft(Uc, n=32, axis=-1)
UD, PD, OMD = phys[..., :3, :], phys[..., 6, :], phys[..., 3, :]; assert UD.shape[:3] == X.shape
tri = Delaunay(np.column_stack([X.ravel(), Y.ravel()])); xd = (np.arange(96) + 0.5) * LX_D / 96
def dns_plane(arr, yy): return np.array([LinearNDInterpolator(tri, arr[..., k].ravel())(np.column_stack([xd, np.full_like(xd, yy)])) for k in range(32)]).T   # (96, 32)
u1 = dns_plane(UD[..., 0, :], 0.5 / RE_D).mean(); utd = float(np.sqrt(NU_D * u1 / (0.5 / RE_D)))
# ================= figure 1: pictures at y+ ~ 10, wall units, same scale
YP = 10.0; j = int(np.argmin(np.abs(ys * RE - YP))); yl = ys[j] * RE; yd = YP / RE_D
les = dict(p=fluct(zpad(plane(p, j))) / ut ** 2, u=fluct(zpad(plane(u, j))) / ut, omx=fluct(zpad(plane(omx, j))) * NU / ut ** 2)
dns = dict(p=fluct(zpad(dns_plane(PD, yd), 3)) / utd ** 2, u=fluct(zpad(dns_plane(UD[..., 0, :], yd), 3)) / utd, omx=fluct(zpad(dns_plane(OMD, yd), 3)) * NU_D / utd ** 2)
xl = (np.arange(nx) + 0.5) * LX / nx * RE; zl = np.arange(2 * nz) * LZ / (2 * nz) * RE; xdp = xd * RE_D; zdp = np.arange(96) * LZ_D / 96 * RE_D
labels = dict(p="$p'/u_\\tau^2$", u="$u'/u_\\tau$", omx="$\\omega_x\\,\\nu/u_\\tau^2$")
fig, ax = plt.subplots(3, 2, figsize=(17, 10.5), gridspec_kw=dict(width_ratios=[LX * RE, LX_D * RE_D + 60]))
for r, key in enumerate(("p", "u", "omx")):
    lim = max(np.abs(les[key]).max(), np.abs(dns[key]).max())
    for c, (F, xx, zz, nm) in enumerate(((les[key], xl, zl, "LES"), (dns[key], xdp, zdp, "DNS"))):
        a = ax[r, c]; im = a.pcolormesh(xx, zz, F.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud"); a.set_aspect("equal"); a.set_ylabel("$z^+$"); plt.colorbar(im, ax=a, pad=0.02, shrink=0.9)
        extra = f"   MKM 395 DNS rms at $y^+$ {yl:.1f}: {mkm_at(key, yl):.3f}" if c == 0 else ""
        a.set_title(f"{labels[key]}   plane rms {F.std():.3f}{extra}", fontsize=10)
        if r == 2: a.set_xlabel("$x^+$")
ax[0, 0].set_title(f"2.5D LES Re_tau 395, 96x160 quads x 128 modes, WALE, t = {t:.0f}; plane $y^+$ = {yl:.1f}; $u_\\tau$ {ut:.3f}\n" + ax[0, 0].get_title(), fontsize=10)
ax[0, 1].set_title(f"SEM FOSLS DNS Re_tau 180 (run02), t = {t_dns:.0f}; plane $y^+$ = {YP:.0f}; $u_\\tau$ {utd:.3f}\n" + ax[0, 1].get_title(), fontsize=10)
plt.suptitle(f"Near-wall plane $y^+ \\approx$ 10, fluctuations about the plane mean in wall units, both panels at the same wall-unit scale. LES box $L_x^+$ x $L_z^+$ = {LX*RE:.0f} x {LZ*RE:.0f}; DNS box {LX_D*RE_D:.0f} x {LZ_D*RE_D:.0f}.\n"
             "No DNS field exists at Re_tau 395 (MKM 1999 publish statistics only), so the picture reference is the Re_tau 180 DNS: near-wall structure scales in wall units.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/uchannel_re395_nearwall_vs_dns.png", dpi=140); print("wrote figures/uchannel_re395_nearwall_vs_dns.png")
for key in ("p", "u", "omx"): print(f"   {key:4s} plane rms: LES 395 {les[key].std():.3f}   MKM 395 {mkm_at(key, yl):.3f}   FOSLS 180 field {dns[key].std():.3f}")
# ================= figure 2: against MKM at the same Re
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
lam_les = LZ / np.arange(1, nz // 2 + 1) * RE; dk_les = 2 * np.pi / LZ
for col, (key, arr, fac, pw) in enumerate((("u", u, 1.0, 1), ("p", p, 1.0, 2))):
    a = ax[0, col]
    for yp_t, c in ((9.53, "C0"), (19.7, "C1")):
        E = np.loadtxt(f"{MKM}/chan395.zspec.{int(round(yp_t/10)*10)}"); kzd = E[:, 0]; Ed = E[:, 1 if key == "u" else 4]      # sum_k E = variance, dk = 2 (L_z = pi)
        a.semilogx(2 * np.pi / kzd[1:] * 392.24, kzd[1:] * Ed[1:] / 2, c, lw=2, label=f"MKM DNS $y^+$ {yp_t:.1f}")
        jt = int(np.argmin(np.abs(ys * RE - yp_t))); rows = np.stack([plane(arr, jt), plane(arr, ny - 1 - jt)]); P = power(rows) * fac ** 2 / ut ** (2 * pw)
        a.semilogx(lam_les, kz[1:] * P[1:] / dk_les, c, ls="none", marker="o", ms=5, mfc="none", label=f"LES $y^+$ {ys[jt]*RE:.1f} (one field, x and both walls averaged)")
    if key == "u":
        Pd = power(dns_plane(UD[..., 0, :], yd)) / utd ** 2; kzd_ = 2 * np.pi / LZ_D * np.arange(17); a.semilogx(LZ_D / np.arange(1, 17) * RE_D, kzd_[1:] * Pd[1:] / (2 * np.pi / LZ_D), "0.5", ls=":", marker="s", ms=3, label="FOSLS DNS Re_tau 180, $y^+$ 10 (its box mode $\\lambda_z^+$ 192 off scale)")
    a.axvline(100, color="k", lw=0.7, ls="--"); a.invert_xaxis(); a.set_ylim(0, 5 if key == "u" else 2.4); a.set(xlabel="$\\lambda_z^+$", ylabel=f"$k_z E_{{{key}{key}}}$ / wall units", title=f"premultiplied spanwise spectrum of {key}"); a.legend(fontsize=7.5)
a = ax[0, 2]; C = np.loadtxt(f"{MKM}/chan395.zcorr.10"); jt = int(np.argmin(np.abs(ys * RE - 9.53)))
for key, arr, ci, c in (("u", u, 2, "C0"), ("p", p, 5, "C3")):
    a.plot(C[:, 1], C[:, ci] / C[0, ci], c, lw=2, label=f"MKM DNS $R_{{{key}{key}}}$, $y^+$ 9.5")
    R = corr(np.stack([plane(arr, jt), plane(arr, ny - 1 - jt)])); dz = np.arange(nz) * LZ / nz * RE; hlf = dz <= LZ * RE / 2
    a.plot(dz[hlf], R[hlf], c, ls="none", marker="o", ms=4, mfc="none", label=f"LES $R_{{{key}{key}}}$, $y^+$ {ys[jt]*RE:.1f}")
    print(f"   R_{key}{key} at y+ 9.5: minimum at dz+ = LES {dz[hlf][np.argmin(R[hlf])]:.0f} (value {R[hlf].min():+.3f})   MKM {C[np.argmin(C[:, ci]), 1]:.0f} ({C[:, ci].min()/C[0, ci]:+.3f})")
a.axhline(0, color="k", lw=0.6); a.set(xlabel="$\\Delta z^+$", ylabel="R / R(0)", xlim=(0, LZ * RE / 2), title="spanwise two-point correlations at $y^+$ 9.5 (LES box half-width 211)"); a.legend(fontsize=8)
# rms profiles, near wall
h = ny // 2; ypl = ys[:h] * RE
def fold(arr): return 0.5 * (arr[:h] + arr[::-1][:h])
r_om = fold(np.array([plane(omx, jr).std() for jr in range(ny)])) * NU / ut ** 2; r_u1 = fold(np.array([plane(u, jr).std() for jr in range(ny)])) / ut; r_p1 = fold(np.array([plane(p, jr).std() for jr in range(ny)])) / ut ** 2
for a, key, stat, single, lab in ((ax[1, 0], "u", st["urms"] / ut_w, r_u1, "$u'_{rms}/u_\\tau$"), (ax[1, 1], "p", st["prms"] / ut_w ** 2, r_p1, "$p'_{rms}/u_\\tau^2$"), (ax[1, 2], "omx", None, r_om, "$\\omega_{x,rms}\\,\\nu/u_\\tau^2$")):
    a.plot(ypd, mkm[key], "k", lw=2.2, label="MKM 1999 DNS Re_tau 392")
    if stat is not None: a.plot(st["yp"], stat, "C0o-", ms=3, lw=1, label="LES, statistics t = 10-30")
    a.plot(ypl, single, "C3", lw=1, ls="--", label=f"LES, single field t = {t:.0f}, walls folded")
    a.set(xlabel="$y^+$", ylabel=lab, xlim=(0, 100), title=f"{lab} near the wall"); a.legend(fontsize=8)
    iy = np.argmin(np.abs(ypl - 10)); print(f"   rms profile {key:4s} at y+ 10: LES {'stats ' + format(np.interp(10, st['yp'], stat), '.3f') + ', ' if stat is not None else ''}single field {single[iy]:.3f}   MKM {mkm_at(key, 10):.3f};  peak LES single {single.max():.3f} at y+ {ypl[np.argmax(single)]:.0f}   MKM {mkm[key][ypd < 200].max():.3f} at y+ {ypd[np.argmax(mkm[key][ypd < 200])]:.0f}")
plt.suptitle(f"Re_tau 395 near-wall u, p and streamwise vorticity against the MKM 1999 DNS at the same Re: spectra, correlations and rms. LES 96x160 quads x 128 modes, WALE, $L_z^+$ {LZ*RE:.0f} (spanwise modes at $\\lambda_z^+$ = {LZ*RE:.0f}/n); DNS box $L_z^+$ 1232.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/uchannel_re395_nearwall_stats.png", dpi=140); print("wrote figures/uchannel_re395_nearwall_stats.png")
