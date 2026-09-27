"""Usage: python plot_utility/plot_uadj_dpc_streamlines.py
Streamlines and velocity magnitude for the coarse Re 100 cylinder: the uncontrolled limit cycle (a shedding
instant) against the lift-penalised ZNMF DPC controller's steady state (from uadj_dpc_cylinder_fields.py)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
d = np.load("results/uadj_dpc/cylinder_fields.npz")
C = d["centroid"]; tri_pts = np.column_stack([C[:, 0], C[:, 1]])
x = np.linspace(-2, 8, 400); y = np.linspace(-2.5, 2.5, 250); X, Y = np.meshgrid(x, y); R = np.hypot(X, Y)
def grid(f):
    G = LinearNDInterpolator(tri_pts, f)(X, Y); G[R < 0.5] = np.nan; return G
Uu, Vu = grid(d["u_unc"]), grid(d["v_unc"]); Uc, Vc = grid(d["u_con"]), grid(d["v_con"])
Mu, Mc = np.hypot(Uu, Vu), np.hypot(Uc, Vc); vmax = np.nanmax([Mu, Mc])
fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
for a_, (U, V, M, lab, cd, cl) in zip(ax, ((Uu, Vu, Mu, "uncontrolled (shedding limit cycle)", float(d["cd_unc"]), float(d["cl_unc"])),
                                             (Uc, Vc, Mc, "ZNMF DPC controller, lift-penalised (steady state)", float(d["cd_con"]), float(d["cl_con"])))):
    im = a_.pcolormesh(X, Y, M, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    a_.streamplot(x, y, U, V, color="white", density=1.6, linewidth=0.7, arrowsize=0.8)
    a_.add_patch(plt.Circle((0, 0), 0.5, fc="0.3", ec="k", lw=1, zorder=5))
    a_.set(xlim=(-2, 8), ylim=(-2.5, 2.5), aspect="equal", ylabel="y / D", title=f"{lab}:  $C_D$ = {cd:.3f},  $C_L$ = {cl:+.3f}")
    plt.colorbar(im, ax=a_, pad=0.01, shrink=0.9, label="$|u|/U_\\infty$")
ax[1].set(xlabel="x / D")
plt.suptitle(f"Re 100 coarse-butterfly cylinder: uncontrolled shedding vs. the DPC-trained $\\pm$90$^\\circ$ ZNMF jet controller (unstructured replay adjoint)\n"
             f"$C_D$ {float(d['cd_con'])/float(d['cd_unc'])-1:+.1%}, jets settle to {d['action_con'][0]:+.3f} / {d['action_con'][1]:+.3f} $U_\\infty$ (essentially idle once the wake is centred)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/uadj_dpc_cylinder_streamlines.png", dpi=140); print("wrote figures/uadj_dpc_cylinder_streamlines.png")
