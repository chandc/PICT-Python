"""Usage: python plot_utility/plot_ucylinder_re3900_vs_piv.py
Wake-centreline mean U and variance <u'u'> against the actual PIV figures of Parnaudeau et al. 2008
(Phys. Fluids 20, 085101; open access at https://www.irisa.fr/fluminance/team/Carlier/publications/
ParnaudeauCarlierHeitzLamballais_2008_POF.pdf), not just the summary numbers already used for the V3
gate. Their Table II gives U_min = -0.34 (minimum of <u> on the centreline) and L_<u'u'> = 0.87 D
(distance from the cylinder base to the x-location of the highest <u'u'> peak) for the PIV data,
neither of which the L_r-only V3 gate checked. Their own Figs. 9-10 (cropped from the freely available
PDF, reference/parnaudeau2008_fig9/10.png) are reproduced alongside our curves for a direct visual
check against the actual field data -- no machine-readable PIV dataset is public (their own reference
31 says so explicitly); this is the closest available comparison without contacting the authors.
x/D here is measured from the CYLINDER CENTRE, Parnaudeau's own convention (their Fig. 1 caption)."""
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
PIV = dict(Lr=1.51, Umin=-0.34, Luu=0.87)                   # Table II, Parnaudeau et al. 2008
print(f"this LES (span & time mean, x measured from the cylinder base):")
print(f"  L_r         {Lr:.3f} D   PIV {PIV['Lr']:.2f} D   ({(Lr/PIV['Lr']-1)*100:+.1f}%)")
print(f"  U_min       {Umin:.3f}    PIV {PIV['Umin']:.2f}    ({(Umin/PIV['Umin']-1)*100:+.1f}%)  at x = {xUmin:.2f} D from the centre")
print(f"  L_<u'u'>    {Lr_uu:.3f} D   PIV {PIV['Luu']:.2f} D   ({(Lr_uu/PIV['Luu']-1)*100:+.1f}%)  peak <u'u'>/Uinf^2 = {uumax:.3f}")

fig = plt.figure(figsize=(15, 10.5))
ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)
ax1.plot(xc, Uc, "C0", lw=1.8, label="this LES"); ax1.axhline(0, color="k", lw=0.5)
ax1.axhline(PIV["Umin"], color="k", ls="--", lw=1, label=f"PIV $U_{{min}}$ = {PIV['Umin']:.2f} (Table II)")
ax1.plot(xUmin, Umin, "C0o", ms=8); ax1.annotate(f"  LES $U_{{min}}$ = {Umin:.3f}\n  at x/D = {xUmin:.2f}", (xUmin, Umin), fontsize=9)
ax1.set(xlim=(0, 10), ylim=(-0.5, 1.0), xlabel="x/D (from the cylinder centre)", ylabel="$\\langle u\\rangle/U_\\infty$", title="wake-centreline mean streamwise velocity"); ax1.legend(fontsize=9)
ax2.imshow(plt.imread("reference/parnaudeau2008_fig9.png")); ax2.set_axis_off(); ax2.set_title("Parnaudeau et al. 2008, Fig. 9 (their PIV/HWA/LES + 5 other refs.)", fontsize=10)
ax3.plot(xc, uuc, "C1", lw=1.8, label="this LES"); ax3.axvline(PIV["Luu"] + 0.5, color="k", ls="--", lw=1, label=f"PIV peak at x/D = {PIV['Luu']+0.5:.2f} ($L_{{u'u'}}$ = {PIV['Luu']:.2f} D, Table II)")
ax3.plot(xc[ipk], uumax, "C1o", ms=8); ax3.annotate(f"  LES peak {uumax:.3f}\n  at x/D = {xc[ipk]:.2f}", (xc[ipk], uumax), fontsize=9)
ax3.set(xlim=(0, 10), ylim=(0, 0.16), xlabel="x/D (from the cylinder centre)", ylabel="$\\langle u'u'\\rangle/U_\\infty^2$", title="wake-centreline streamwise variance"); ax3.legend(fontsize=9)
ax4.imshow(plt.imread("reference/parnaudeau2008_fig10.png")); ax4.set_axis_off(); ax4.set_title("Parnaudeau et al. 2008, Fig. 10", fontsize=10)
plt.suptitle("Re 3900 cylinder, wake centreline: this LES against the actual PIV field data of Parnaudeau et al. 2008 (Phys. Fluids 20, 085101), not just the summary L_r used for the V3 gate\n"
             "Their Table II PIV values used as reference points; their own Figs. 9-10 (open access) shown alongside for a direct look at the field data. x/D from the cylinder centre throughout.", fontsize=10.5)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_vs_piv.png", dpi=140); print("wrote figures/ucylinder_re3900_vs_piv.png")
