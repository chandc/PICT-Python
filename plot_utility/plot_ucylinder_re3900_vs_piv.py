"""Usage: python plot_utility/plot_ucylinder_re3900_vs_piv.py
Wake-centreline mean U and variance <u'u'> against the field data of Parnaudeau et al. 2008 (Phys.
Fluids 20, 085101; open access at https://www.irisa.fr/fluminance/team/Carlier/publications/
ParnaudeauCarlierHeitzLamballaisPOF.pdf), redrawn from digitized values -- publication figures cannot
reproduce a graph from someone else's paper, so this plots only our own curve plus (a) points digitized
from their Figs. 9-10 by calibrated pixel analysis of the open-access PDF (median per x-column across
every series shown there; provenance and a sanity check against their Table II in
reference/parnaudeau2008_digitized/README.md) and (b) their Table II numbers verbatim (text, not an
image). No machine-readable PIV dataset is public -- their own paper says so (ref. 31).
x/D is measured from the cylinder centre, Parnaudeau's own convention (their Fig. 1 caption)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

d = np.load("results/ucyl3900/ucyl3900_cylinder_re3900_nz64_wale_stats.npz")
C = d["centroid"]; tri = Delaunay(C)
interp = {k: LinearNDInterpolator(tri, d[k]) for k in ("u", "uu")}
xc = np.linspace(0.5, 10, 950); pts = np.column_stack([xc, np.zeros_like(xc)])
Uc, uuc = interp["u"](pts), interp["uu"](pts)
imin = np.nanargmin(Uc); Umin, xUmin = float(Uc[imin]), float(xc[imin])
ipk = np.nanargmax(uuc); uumax, Lr_uu = float(uuc[ipk]), float(xc[ipk] - 0.5)   # base-referenced, matching Table II's own definition
neg = Uc < 0; Lr = float("nan")
if neg.any():
    i1 = np.flatnonzero(neg)[-1]
    if i1 + 1 < len(xc): Lr = xc[i1] + (0 - Uc[i1]) * (xc[i1 + 1] - xc[i1]) / (Uc[i1 + 1] - Uc[i1]) - 0.5
PIV = dict(Lr=1.51, Umin=-0.34, Luu=0.87)                   # Table II, Parnaudeau et al. 2008 (text, verbatim)
print("this LES (span & time mean, lengths measured from the cylinder base):")
print(f"  L_r         {Lr:.3f} D   PIV {PIV['Lr']:.2f} D   ({(Lr/PIV['Lr']-1)*100:+.1f}%)")
print(f"  U_min       {Umin:.3f}    PIV {PIV['Umin']:.2f}    ({(Umin/PIV['Umin']-1)*100:+.1f}%)  at x = {xUmin:.2f} D from the centre")
print(f"  L_<u'u'>    {Lr_uu:.3f} D   PIV {PIV['Luu']:.2f} D   ({(Lr_uu/PIV['Luu']-1)*100:+.1f}%)  peak <u'u'>/Uinf^2 = {uumax:.3f}")

# ---- digitized points (median per x-column, IQR band across the eight series Parnaudeau plot)
def load_digitized(path, smooth=9):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    x, med, lo, hi = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    k = np.ones(smooth) / smooth
    pad = smooth // 2
    med_s = np.convolve(np.pad(med, pad, mode="edge"), k, mode="valid")
    lo_s = np.convolve(np.pad(lo, pad, mode="edge"), k, mode="valid")
    hi_s = np.convolve(np.pad(hi, pad, mode="edge"), k, mode="valid")
    return x, med_s, lo_s, hi_s
xd9, u9, u9lo, u9hi = load_digitized("reference/parnaudeau2008_digitized/fig9_centerline_u.csv")
xd10, uu10, uu10lo, uu10hi = load_digitized("reference/parnaudeau2008_digitized/fig10_centerline_uu.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
ax[0].fill_between(xd9, u9lo, u9hi, color="0.75", label="Parnaudeau et al. 2008, digitized\n(median $\\pm$ spread across their 8 series)")
ax[0].plot(xd9, u9, color="0.35", lw=1.2)
ax[0].plot(xc, Uc, "C0", lw=2, label="this LES")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].plot([], [], " ", label=f"PIV (Table II): $U_{{min}}$ = {PIV['Umin']:.2f}, $L_r$ = {PIV['Lr']:.2f} D")
ax[0].plot(xUmin, Umin, "C0o", ms=8, zorder=5); ax[0].annotate(f"  LES $U_{{min}}$ = {Umin:.3f}\n  at x/D = {xUmin:.2f}", (xUmin, Umin), fontsize=9)
ax[0].axhline(PIV["Umin"], color="k", ls="--", lw=1)
ax[0].set(xlim=(0, 10), ylim=(-0.5, 1.0), xlabel="x/D (from the cylinder centre)", ylabel="$\\langle u\\rangle/U_\\infty$", title="wake-centreline mean streamwise velocity"); ax[0].legend(fontsize=8.5, loc="lower right")

ax[1].fill_between(xd10, uu10lo, uu10hi, color="0.75", label="Parnaudeau et al. 2008, digitized\n(median $\\pm$ spread across their 8 series)")
ax[1].plot(xd10, uu10, color="0.35", lw=1.2)
ax[1].plot(xc, uuc, "C1", lw=2, label="this LES")
ax[1].plot([], [], " ", label=f"PIV (Table II): $L_{{u'u'}}$ = {PIV['Luu']:.2f} D")
ax[1].plot(xc[ipk], uumax, "C1o", ms=8, zorder=5); ax[1].annotate(f"  LES peak {uumax:.3f}\n  at x/D = {xc[ipk]:.2f}", (xc[ipk], uumax), fontsize=9)
ax[1].axvline(PIV["Luu"] + 0.5, color="k", ls="--", lw=1)
ax[1].set(xlim=(0, 10), ylim=(0, 0.16), xlabel="x/D (from the cylinder centre)", ylabel="$\\langle u'u'\\rangle/U_\\infty^2$", title="wake-centreline streamwise variance"); ax[1].legend(fontsize=8.5)

plt.suptitle("Re 3900 cylinder, wake centreline: this LES against the field data of Parnaudeau et al. 2008 (Phys. Fluids 20, 085101)\n"
             "Grey band: digitized from their Figs. 9-10 (pixel-calibrated, not a reproduced image); dashed lines: their Table II values, verbatim", fontsize=10.5)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_vs_piv.png", dpi=150); print("wrote figures/ucylinder_re3900_vs_piv.png")
