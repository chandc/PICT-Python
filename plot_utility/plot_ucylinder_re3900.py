"""Usage: python plot_utility/plot_ucylinder_re3900.py [<tag>_stats.npz]
Re 3900 cylinder LES (run_ucylinder3900.py) against the published data:
  figures/ucylinder_re3900_forces.png  -- C_D, C_L history, C_L spectrum (St), C_p around the cylinder, dt history
  figures/ucylinder_re3900_wake.png    -- mean U on the wake centreline (recirculation length), cross-wake profiles of
                                          U, V, u'u', v'v', u'v' at x = 1.06, 1.54, 2.02 D (Parnaudeau's PIV stations)
  figures/ucylinder_re3900_fields.png  -- mean U, mean V, u'u', v'v', <nu_t>/nu in the near wake
References (Re 3900, span pi D): Parnaudeau et al. 2008 (PIV) St 0.208, L_r 1.51; Kravchenko & Moin 2000 C_D 1.04, C_pb -0.94,
L_r 1.35, St 0.21; Lehmkuhl et al. 2013 DNS L_r 1.26 (H) / 1.55 (L), C_D 1.05 / 0.98, C_pb -0.94 / -0.88; Norberg exp. C_D 0.98, C_pb -0.88."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
f = _sys.argv[1] if len(_sys.argv) > 1 else "results/ucyl3900/ucyl3900_cylinder_re3900_nz64_wale_stats.npz"
d = np.load(f, allow_pickle=True); h = d["hist"]; t_stats = float(d["t_stats"]); n = int(d["nsamp"]); Re = float(d["Re"]); nu = 1 / Re
print(f"{f}: {n} samples, t = {t_stats}..{float(d['t_end']):.1f}, Re {Re:.0f}, {d['sgs']}, nz {int(d['nz'])}")
win = h[:, 0] >= t_stats; hw = h[win] if win.sum() > 10 else h[len(h) // 2:]
t, cd, cl, cdp, cpb, dt = hw.T
REF = dict(St=(0.208, "Parnaudeau PIV"), Cd=((0.98, 1.05), "Norberg .. Lehmkuhl H"), Cpb=((-0.94, -0.88), "K&M .. Norberg"), Lr=((1.26, 1.55), "Lehmkuhl H .. L (PIV 1.51)"))
# ---- Strouhal from zero crossings and from the spectrum
clf = cl - cl.mean(); z = np.flatnonzero(np.diff(np.sign(clf)) > 0); St_zc = float("nan")
if len(z) >= 4: tz = t[z] - clf[z] * (t[z + 1] - t[z]) / (clf[z + 1] - clf[z]); St_zc = 1 / np.diff(tz).mean()
tu = np.linspace(t[0], t[-1], 4096); F = np.abs(np.fft.rfft((np.interp(tu, t, clf) * np.hanning(4096)))) ** 2; fr = np.fft.rfftfreq(4096, tu[1] - tu[0]); St_sp = fr[1 + np.argmax(F[1:])]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
ax[0].plot(h[:, 0], h[:, 1], "C0", lw=0.7, label="$C_D$"); ax[0].plot(h[:, 0], h[:, 2], "C1", lw=0.7, label="$C_L$"); ax[0].axvspan(t_stats, h[-1, 0], color="0.92", label="statistics window")
ax[0].set(xlabel="t U/D", ylim=(-1.5, 2.0), title=f"forces: $C_D$ {cd.mean():.3f}, $C_L$ rms {cl.std():.3f}, $C_{{pb}}$ {cpb.mean():+.3f}"); ax[0].legend(fontsize=8)
ax[1].semilogy(fr[1:], F[1:] / F[1:].max(), "C1", lw=0.8); ax[1].axvline(0.208, color="k", ls="--", lw=0.8, label="St 0.208 (PIV)"); ax[1].set(xlim=(0, 1), xlabel="f D/U", title=f"$C_L$ spectrum: St {St_sp:.4f} (zero crossings {St_zc:.4f})"); ax[1].legend(fontsize=8)
th = d["theta_wall"]; cp = d["cp_wall"]; o = np.argsort(np.abs(th)); thf = 180 - np.abs(th[o])                # angle from the FRONT stagnation point
ax[2].plot(thf, cp[o], "C0.", ms=3, label="LES, time and span mean"); ax[2].axhline(-0.88, color="0.4", ls=":", lw=0.8, label="$C_{pb}$ -0.88 (Norberg)"); ax[2].axhline(-0.94, color="0.4", ls="--", lw=0.8, label="$C_{pb}$ -0.94 (K&M)")
ax[2].set(xlabel="angle from the front stagnation point (deg)", ylabel="$C_p$", xlim=(0, 180), title="pressure coefficient around the cylinder"); ax[2].legend(fontsize=8)
ax[3].plot(h[:, 0], h[:, 5], "C2", lw=0.7); ax[3].set(xlabel="t U/D", ylabel="dt", title="time step (CFL-limited)")
plt.suptitle(f"Re 3900 cylinder, 2.5D unstructured LES ({d['mesh']}, {int(d['nz'])} Fourier planes, L_z = {float(d['Lz'])/np.pi:.0f} pi D, {d['sgs']}): statistics t = {t_stats}..{float(d['t_end']):.0f} ({n} samples)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_forces.png", dpi=130); print("wrote figures/ucylinder_re3900_forces.png")
# ---- mean field interpolants
C = d["centroid"]; tri = Delaunay(C); interp = {k: LinearNDInterpolator(tri, d[k]) for k in ("u", "v", "uu", "vv", "uv", "ww", "nut", "p")}
xc = np.linspace(0.5, 10, 600); Uc = interp["u"](np.column_stack([xc, np.zeros_like(xc)]))
neg = Uc < 0; Lr = float("nan")
if neg.any():
    i1 = np.flatnonzero(neg)[-1]
    if i1 + 1 < len(xc): Lr = xc[i1] + (0 - Uc[i1]) * (xc[i1 + 1] - xc[i1]) / (Uc[i1 + 1] - Uc[i1]) - 0.5      # measured from the rear stagnation point
fig, ax = plt.subplots(2, 4, figsize=(21, 8.4))
a = ax[0, 0]; a.plot(xc - 0.5, Uc, "C0", lw=1.5); a.axhline(0, color="k", lw=0.6); a.axvspan(1.26, 1.55, color="0.9", label="Lehmkuhl DNS states 1.26 .. 1.55"); a.axvline(1.51, color="k", ls="--", lw=0.8, label="Parnaudeau PIV 1.51"); a.axvline(1.35, color="0.4", ls=":", lw=0.8, label="K&M 1.35")
a.set(xlabel="(x - 0.5)/D, from the rear stagnation point", ylabel="$U/U_\\infty$", xlim=(0, 6), title=f"wake centreline: recirculation length {Lr:.2f} D, $U_{{min}}$ {np.nanmin(Uc):.3f}"); a.legend(fontsize=8)
yy = np.linspace(-2.5, 2.5, 301)
for j, xs in enumerate((1.06, 1.54, 2.02)):
    P = np.column_stack([np.full_like(yy, xs), yy]); a = ax[0, j + 1]
    a.plot(yy, interp["u"](P), "C0", label="$U$"); a.plot(yy, interp["v"](P), "C1", label="$V$"); a.axhline(0, color="k", lw=0.5); a.set(xlabel="y/D", title=f"x = {xs} D: mean U, V"); a.legend(fontsize=8)
    a = ax[1, j + 1]; a.plot(yy, interp["uu"](P), "C0", label="$\\langle u'u'\\rangle$"); a.plot(yy, interp["vv"](P), "C1", label="$\\langle v'v'\\rangle$"); a.plot(yy, interp["uv"](P), "C2", label="$\\langle u'v'\\rangle$"); a.plot(yy, interp["ww"](P), "C3", ls="--", lw=0.8, label="$\\langle w'w'\\rangle$")
    a.set(xlabel="y/D", title=f"x = {xs} D: Reynolds stresses"); a.legend(fontsize=8)
a = ax[1, 0]; a.plot(xc - 0.5, interp["uu"](np.column_stack([xc, np.zeros_like(xc)])), "C0", label="$\\langle u'u'\\rangle$"); a.plot(xc - 0.5, interp["vv"](np.column_stack([xc, np.zeros_like(xc)])), "C1", label="$\\langle v'v'\\rangle$")
a.set(xlabel="(x - 0.5)/D", xlim=(0, 6), title="centreline Reynolds stresses"); a.legend(fontsize=8)
plt.suptitle("Re 3900 near wake: mean and second moments (span and time averaged). Parnaudeau et al. 2008 PIV: L_r 1.51 D; U-shaped mean profile at x = 1.06 turning V-shaped by 2.02; u'u' twin peaks at |y| ~ 0.5 at x = 1.06", fontsize=10)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_wake.png", dpi=130); print("wrote figures/ucylinder_re3900_wake.png")
# ---- near-wake fields
from matplotlib.collections import PolyCollection
polys = [d["nodes"][c[:k], :2] for c, k in zip(d["cells"], d["nvert"])]; sel = (C[:, 0] > -1.5) & (C[:, 0] < 6) & (np.abs(C[:, 1]) < 2)
fig, ax = plt.subplots(1, 5, figsize=(26, 4.6))
for a, (key, lab, cm, arr) in zip(ax, (("u", "mean U", "turbo", d["u"]), ("v", "mean V", "RdBu_r", d["v"]), ("uu", "<u'u'>", "magma", d["uu"]), ("vv", "<v'v'>", "magma", d["vv"]), ("nut", "<nu_t>/nu", "viridis", d["nut"] / nu))):
    pc = PolyCollection([polys[i] for i in np.flatnonzero(sel)], array=arr[sel], cmap=cm, edgecolors="face", linewidths=0.1); a.add_collection(pc); plt.colorbar(pc, ax=a, pad=0.02, shrink=0.85)
    a.add_patch(plt.Circle((0, 0), 0.5, fc="0.85", ec="k", lw=0.8)); a.set(xlim=(-1.5, 6), ylim=(-2, 2), title=lab); a.set_aspect("equal")
plt.suptitle(f"Re 3900 near wake, span and time averaged over t = {t_stats}..{float(d['t_end']):.0f}", fontsize=11); plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_fields.png", dpi=120); print("wrote figures/ucylinder_re3900_fields.png")
print(f"RESULT: St {St_sp:.4f} (spectrum) / {St_zc:.4f} (zero crossings) [ref 0.208, criterion 3%: {abs(St_sp/0.208-1)*100:.1f}%]   Cd {cd.mean():.3f} [0.98..1.05]   Cpb {cpb.mean():+.3f} [-0.94..-0.88]   Cl_rms {cl.std():.3f}   L_r {Lr:.2f} D [1.26..1.55; PIV 1.51, criterion 10%: {abs(Lr/1.51-1)*100:.1f}%]   <dt> {dt.mean():.4f}")
