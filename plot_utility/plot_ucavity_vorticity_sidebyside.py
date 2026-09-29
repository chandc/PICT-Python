"""Side-by-side vorticity: UniFlow's own cavity run (medium mesh, run_ucavity_hg.py) vs HydroGym's
Firedrake reference (fine mesh, tools/hydrogym_cmp/hg_cavity.py), same colour scale. Different mesh
resolution AND method at once (see reference/skew_unstructured_literature.md sections 66-67); this is
a snapshot-in-time comparison, not phase-aligned (the two runs are on unrelated clocks -- there is no
shared t=0 or shared perturbation seed to lock the shedding phase to)."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from src.umesh import Mesh
from src.uops import Gradient

LEV = np.linspace(-15, 15, 31)

# --- UniFlow (medium mesh) ---
du = np.load("results/ucavity_hg_medium/final.npz")
cells_u = [du["cells"][k, :du["nvert"][k]] for k in range(len(du["nvert"]))]
mu = Mesh(du["nodes"], cells_u, span=1.0)
names = {2: "Inlet", 3: "Freestream", 4: "Outlet", 5: "Slip", 6: "Wall", 7: "Control", 8: "Sensor"}
inv = {v: k for k, v in names.items()}
bt = du["btag"][mu.bfaces]
uo, vo = du["u"][mu.owner[mu.bfaces]], du["v"][mu.owner[mu.bfaces]]
ubv = np.where(bt == inv["Inlet"], 1.0, np.where(np.isin(bt, [inv["Wall"], inv["Control"], inv["Sensor"]]), 0.0, uo))
vbv = np.where(bt == inv["Outlet"], vo, 0.0)
g = Gradient(mu); wc = g(du["v"], vbv)[:, 0] - g(du["u"], ubv)[:, 1]
nv = len(du["nodes"]); ws = np.zeros(nv); Wu = np.zeros(nv)
for c, vol, ww in zip(cells_u, mu.vol, wc): ws[c] += vol; Wu[c] += vol * ww
Wu /= ws
Xu = du["nodes"]; Tu = mtri.Triangulation(Xu[:, 0], Xu[:, 1], triangles=np.array(cells_u))
polys_u = [Xu[c] for c in cells_u]

# --- Firedrake (fine mesh), CG1-projected vorticity straight from their own checkpoint ---
df = np.load("results/hydrogym_cmp/cavity_re7500_fine/final_fields.npz")
Xf = df["vert"]; Tf = mtri.Triangulation(Xf[:, 0], Xf[:, 1], triangles=df["triangles"])
Wf = df["vort_cg1"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharex=True, sharey=True)
panels = [("UniFlow, medium mesh (65536 cells), t=50", Tu, Wu, polys_u, 0.05),
          ("HydroGym Firedrake, fine mesh (224849 cells), t=50", Tf, Wf, None, 0.0)]
for ax, (ttl, T, W, polys, lw) in zip(axes, panels):
    cf = ax.tricontourf(T, np.clip(W, LEV[0], LEV[-1]), LEV, cmap="RdBu_r", extend="both")
    if polys is not None:
        ax.add_collection(PolyCollection(polys, facecolor="none", edgecolor="k", linewidth=lw, alpha=0.3, zorder=2))
    ax.set_xlim(-1.2, 2.5); ax.set_ylim(-1.0, 0.5); ax.set_aspect("equal"); ax.set_title(ttl, fontsize=11)
    plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.01, label="vorticity")

fig.suptitle("Open cavity, Re 7500: vorticity, UniFlow vs HydroGym Firedrake (same geometry/BCs, different mesh + method)", fontsize=12)
fig.text(0.5, 0.01, "Not phase-aligned -- unrelated clocks/perturbation seeds, so shedding phase is not expected to match.\n"
         "Firedrake's CG1 field also carries the expected sharp reentrant-corner spikes at the two cavity edges (clipped here to +-15 for comparability).",
         ha="center", fontsize=8.5, style="italic")
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.savefig("figures/ucavity_vorticity_sidebyside.png", dpi=150)
print("wrote figures/ucavity_vorticity_sidebyside.png")
