"""Usage: python plot_utility/plot_uchannel_re395_nearwall.py [yplus] [run dir/tag]
Instantaneous structure of the Re_tau 395 2.5D LES field (96x160 quads x 128 modes, WALE, t = 30, A100 run).
MKM 1999 publishes statistics only, so the field is checked against the DNS the way the literature does:
  (1) wall-parallel plane y+ ~ 12: p'/u_tau^2, u'/u_tau, omega_x nu/u_tau^2 (streaks, lambda_z+ ~ 100 -> ~4 pairs across Lz+ = 420)
  (2) cross-stream plane (y-z) and streamwise plane (x-y) of u
  (3) spanwise premultiplied spectrum k_z E_uu at y+ 12 and 50: the peak must sit at lambda_z+ ~ 100 (MKM 1999 fig. 8; Kim-Moin-Moser 1987)
  (4) the plane rms against the MKM profile at the same y+."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.umesh import rect_mesh
from src.uops import Gradient
RE_TAU, NU, LX, LY, LZ = 395.0, 1 / 395, np.pi, 2.0, 0.34 * np.pi
YPLUS = float(_sys.argv[1]) if len(_sys.argv) > 1 else 12.0
BASE = _sys.argv[2] if len(_sys.argv) > 2 else "results/uchan395/uchan395_96x160x128_wale_cpg/uchan395_96x160x128_wale_cpg"
nx, ny, nz = 96, 160, 128
def first_cell(beta): return LY * 0.5 * (1 + np.tanh(beta * (2.0 / ny - 1)) / np.tanh(beta))
lo, hi = 1e-3, 20.0
for _ in range(200):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if first_cell(mid) > 1.0 / RE_TAU else (lo, mid)
m = rect_mesh(nx, ny, 0, LX, 0, LY, cells="quad", cluster_y=0.5 * (lo + hi)); m.make_periodic(1, 2, (LX, 0.0))
d = np.load(BASE + "_final.npz"); u, v, w, p = d["u"], d["v"], d["w"], d["p"]; t = float(d["time"]); assert u.shape == (m.ncell, nz)
st = np.load(BASE + "_stats.npz"); ut_stat = float(st["ut"])
ys = np.unique(np.round(m.centroid[:, 1], 8)); assert len(ys) == ny, len(ys); jrow = np.argmin(np.abs(m.centroid[:, 1][:, None] - ys[None]), axis=1); ix = np.floor(m.centroid[:, 0] / (LX / nx)).astype(int)
bot = m.bfaces[m.btag[m.bfaces] == 3]; o = m.owner[bot]; ut = float(np.sqrt(NU * (np.abs(u[o]) / np.abs(m.fcentre[bot, 1] - m.centroid[o, 1])[:, None]).mean()))
print(f"field t = {t:.2f}: u_tau from the bottom-wall gradient {ut:.4f} (window mean {ut_stat:.4f}); plane y+ target {YPLUS}")
g = Gradient(m); gw = g(w, np.zeros((m.nbface, nz))); kz = 2 * np.pi / LZ * np.arange(nz // 2 + 1)
omx = gw[:, 1] - np.fft.irfft(np.fft.rfft(v, axis=1) * 1j * kz, n=nz, axis=1)
def zpad(f, mult=2):
    fh = np.fft.rfft(f, axis=1); out = np.zeros((f.shape[0], (mult * nz) // 2 + 1), complex); out[:, :fh.shape[1]] = fh; out[:, fh.shape[1] - 1] = 0
    return np.fft.irfft(out, n=mult * nz, axis=1) * mult
def row(j): sel = jrow == j; return sel, np.argsort(ix[sel])
def plane(f, j): sel, order = row(j); return f[sel][order]                                  # (nx, nz)
j = int(np.argmin(np.abs(ys * RE_TAU - YPLUS))); x_o = (np.arange(nx) + 0.5) * LX / nx; z2 = np.arange(2 * nz) * LZ / (2 * nz)
# ---- MKM reference at the plane height
A = np.loadtxt("reference/mkm_chan395/chan395.means"); B = np.loadtxt("reference/mkm_chan395/chan395.reystress"); ypd = A[:, 1]
mkm_u = np.interp(ys[j] * RE_TAU, ypd, np.sqrt(B[:, 2])); mkm_v = np.interp(ys[j] * RE_TAU, ypd, np.sqrt(B[:, 3])); mkm_w = np.interp(ys[j] * RE_TAU, ypd, np.sqrt(B[:, 4]))
# ---- figure 1: wall-parallel planes
fig, ax = plt.subplots(3, 1, figsize=(11.5, 10.2))
for a, (lab, f, fac, pw) in zip(ax, (("$p'/u_\\tau^2$", p, 1.0, 2), ("$u'/u_\\tau$", u, 1.0, 1), ("$\\omega_x\\,\\nu/u_\\tau^2$", omx, NU, 2))):
    F = zpad(plane(f, j)); F = (F - F.mean()) * fac / ut ** pw; lim = np.abs(F).max()
    im = a.pcolormesh(x_o * RE_TAU, z2 * RE_TAU, F.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud"); a.set_aspect("equal"); a.set_ylabel("$z^+$"); plt.colorbar(im, ax=a, pad=0.01)
    a.set_title(f"{lab}   plane rms {F.std():.3f}" + (f"   (MKM 1999 $u'^+$ at this $y^+$: {mkm_u:.3f})" if f is u else ""), fontsize=10)
ax[-1].set_xlabel("$x^+$")
plt.suptitle(f"Re_tau 395 2.5D LES (96x160 quads x 128 modes, WALE), t = {t:.0f}, wall-parallel plane $y^+$ = {ys[j]*RE_TAU:.1f}\nfluctuations about the plane mean, wall units. "
             f"$L_x^+$ = {LX*RE_TAU:.0f}, $L_z^+$ = {LZ*RE_TAU:.0f}; $\\Delta x^+$ 12.9, z spectral (x2 refined)", fontsize=10)
plt.tight_layout(); plt.savefig(f"figures/uchannel_re395_nearwall_yp{YPLUS:.0f}.png", dpi=140); print("wrote", f"figures/uchannel_re395_nearwall_yp{YPLUS:.0f}.png")
# ---- figure 2: cross planes of u
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[LZ, LX]))
i0 = nx // 2; sel = ix == i0; jj = jrow[sel]; Uyz = np.zeros((ny, 2 * nz)); Uyz[jj] = zpad(u[sel]) / ut
im = ax[0].pcolormesh(z2 * RE_TAU, ys * RE_TAU, Uyz, cmap="turbo", shading="gouraud"); ax[0].set(xlabel="$z^+$", ylabel="$y^+$", title=f"$u/u_\\tau$, cross-stream plane x = {x_o[i0]:.2f}"); ax[0].set_aspect("equal"); plt.colorbar(im, ax=ax[0], pad=0.01)
k0 = 0; Uxy = np.zeros((ny, nx))
for jr in range(ny): Uxy[jr] = plane(u, jr)[:, k0] / ut
im = ax[1].pcolormesh(x_o * RE_TAU, ys * RE_TAU, Uxy, cmap="turbo", shading="gouraud"); ax[1].set(xlabel="$x^+$", ylabel="$y^+$", title="$u/u_\\tau$, streamwise plane z = 0"); ax[1].set_aspect("equal"); plt.colorbar(im, ax=ax[1], pad=0.01)
plt.suptitle(f"Re_tau 395 2.5D LES, t = {t:.0f}: instantaneous streamwise velocity (cell values in x and y, spectral in z)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uchannel_re395_planes.png", dpi=140); print("wrote figures/uchannel_re395_planes.png")
# ---- figure 3: spanwise premultiplied spectra of u at y+ 12 and 50 (x- and plane-averaged), plus the plane rms against MKM
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
lam = LZ / np.arange(1, nz // 2 + 1) * RE_TAU
for yp_t, c in ((12, "C0"), (50, "C1"), (150, "C2")):
    jt = int(np.argmin(np.abs(ys * RE_TAU - yp_t))); F = plane(u, jt); F = F - F.mean(axis=1, keepdims=True)
    E = (np.abs(np.fft.rfft(F, axis=1)) ** 2).mean(axis=0)[1:] * 2 / nz ** 2 / (2 * np.pi / LZ)          # one-sided, normalised so sum E dk = var
    kE = kz[1:] * E / ut ** 2; ax[0].semilogx(lam, kE, c, marker="o", ms=3, label=f"$y^+$ = {ys[jt]*RE_TAU:.0f}, two largest bins $\\lambda_z^+$ = {lam[np.argsort(kE)[-1]]:.0f}, {lam[np.argsort(kE)[-2]]:.0f}")
    print(f"   y+ {ys[jt]*RE_TAU:6.1f}: k_z E_uu peak at lambda_z+ = {lam[np.argmax(kE)]:.0f}  (bin edges {lam[max(np.argmax(kE)-1,0)]:.0f} .. {lam[min(np.argmax(kE)+1, len(lam)-1)]:.0f}); plane u' rms {F.std()/ut:.3f}  MKM {np.interp(ys[jt]*RE_TAU, ypd, np.sqrt(B[:, 2])):.3f}")
ax[0].axvline(100, color="k", lw=0.8, ls="--", label="$\\lambda_z^+$ = 100 (DNS streak spacing)"); ax[0].set(xlabel="$\\lambda_z^+$", ylabel="$k_z E_{uu}/u_\\tau^2$", title="spanwise premultiplied spectrum of u (one field, x-averaged)"); ax[0].legend(fontsize=8); ax[0].invert_xaxis()
# rms of each y-row from this single field vs the window statistics and MKM
r_u = np.array([plane(u, jr).std() for jr in range(ny)]) / ut; r_v = np.array([plane(v, jr).std() for jr in range(ny)]) / ut; r_w = np.array([plane(w, jr).std() for jr in range(ny)]) / ut
h = ny // 2; ypl = ys[:h] * RE_TAU
for arr, ls, nm in ((np.sqrt(B[:, 2]), "-", "u'"), (np.sqrt(B[:, 3]), "--", "v'"), (np.sqrt(B[:, 4]), ":", "w'")): ax[1].plot(ypd, arr, "k", ls=ls, lw=2, label=f"MKM DNS {nm}")
for arr, ls in ((r_u, "-"), (r_v, "--"), (r_w, ":")): ax[1].plot(ypl, 0.5 * (arr[:h] + arr[::-1][:h]), "C3", ls=ls, lw=1.2)
for k, ls in (("urms", "-"), ("vrms", "--"), ("wrms", ":")): ax[1].plot(st["yp"], st[k] / ut_stat, "C0", ls=ls, lw=1.0, alpha=0.8)
ax[1].set(xlabel="$y^+$", ylabel="rms / $u_\\tau$", xlim=(0, 395), title="red: this single field (t = 30, walls folded)\nblue: 20-turnover statistics; black: MKM DNS"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/uchannel_re395_spectra.png", dpi=140); print("wrote figures/uchannel_re395_spectra.png")
