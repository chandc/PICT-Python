"""Usage: python plot_utility/plot_ucylinder_re3900_refinement.py
The shear-layer-refined cylinder mesh (record section 65, reviewer item C) against the original V3
mesh: did refining the ring and near-wake (1.7x the cells, wall cell and arc roughly halved) raise the
under-resolved centreline turbulence-intensity peak toward the PIV/LES band, as the diagnosis predicted?"""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

def centerline(path):
    d = np.load(path); C = d["centroid"]; tri = Delaunay(C)
    interp = {k: LinearNDInterpolator(tri, d[k]) for k in ("u", "uu")}
    xc = np.linspace(0.5, 10, 950); pts = np.column_stack([xc, np.zeros_like(xc)])
    Uc, uuc = interp["u"](pts), interp["uu"](pts)
    return xc, Uc, uuc, d

xc0, Uc0, uu0, d0 = centerline("results/ucyl3900/ucyl3900_cylinder_re3900_nz64_wale_stats.npz")
xc1, Uc1, uu1, d1 = centerline("results/ucyl3900_refined/ucyl3900_cylinder_re3900_refined_nz64_wale_stats.npz")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].plot(xc0, Uc0, "C0", lw=1.8, label=f"V3, 60k cells")
ax[0].plot(xc1, Uc1, "C3", lw=1.8, label=f"refined, 104k cells")
ax[0].axhline(0, color="k", lw=0.5); ax[0].set(xlim=(0, 6), ylim=(-0.5, 1.0), xlabel="x/D (from the cylinder centre)", ylabel="$\\langle u\\rangle/U_\\infty$", title="wake-centreline mean velocity"); ax[0].legend(fontsize=9)
ax[1].plot(xc0, uu0, "C0", lw=1.8, label="V3, 60k cells"); ax[1].plot(xc1, uu1, "C3", lw=1.8, label="refined, 104k cells")
ax[1].axhspan(0.10, 0.12, color="0.85", label="PIV/LES band 0.10-0.12")
ax[1].set(xlim=(0, 6), ylim=(0, 0.16), xlabel="x/D", ylabel="$\\langle u'u'\\rangle/U_\\infty^2$", title="wake-centreline streamwise variance"); ax[1].legend(fontsize=9)
plt.suptitle("Re 3900 cylinder: does mesh refinement raise the under-resolved u'u' peak? Answer: no -- essentially unchanged (0.084 -> 0.085)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/ucylinder_re3900_refinement_compare.png", dpi=140); print("wrote figures/ucylinder_re3900_refinement_compare.png")

for name, xc, Uc, uuc, d in (("V3 (60k)", xc0, Uc0, uu0, d0), ("refined (104k)", xc1, Uc1, uu1, d1)):
    imin = np.nanargmin(Uc); ipk = np.nanargmax(uuc)
    neg = Uc < 0; i1 = np.flatnonzero(neg)[-1]
    Lr = xc[i1] + (0 - Uc[i1]) * (xc[i1+1]-xc[i1]) / (Uc[i1+1]-Uc[i1]) - 0.5
    print(f"{name}: L_r {Lr:.3f} D, U_min {Uc[imin]:.3f} at x/D {xc[imin]:.2f}, peak <u'u'> {uuc[ipk]:.4f} at x/D {xc[ipk]:.2f}, "
          f"St/Cd/Cpb from hist: Cd_mean {float(d['cd_mean']):.4f}, Cpb {float(d['cpb_mean']):+.4f}, Cl_rms {float(d['cl_rms']):.4f}")
