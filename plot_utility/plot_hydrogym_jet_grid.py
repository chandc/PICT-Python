"""The jet-resolved butterfly grid for HydroGym's `Cylinder`, and whether it resolves the jet.

The question this figure exists to answer is the last panel's: HydroGym's actuator is a 10-degree
COSINE, and a grid that cannot represent that shape gives a policy control authority that is a
discretisation artefact. The production grid puts THREE points across it, and its tangential
clustering is at the east corner -- so theta = +-90, where the jets live, is the COARSEST part
of the ring. That is backwards for this case.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from cylinder_ring_grid import ring_rect_domain, D
from cylinder_rect_bc import classify
from src.multiblock import face_slice

R, SDT, NZ, NE = D / 2, 0.008, 2, 97
OMEGA = np.pi / 18                      # HydroGym jet width, radians (10 deg)


def body_angles(d):
    roles, th = classify(d), []
    for (b, f), r in roles.items():
        if r != "body":
            continue
        fs = face_slice(f)
        x, y, z = (d.blocks[b].x[fs].ravel(), d.blocks[b].y[fs].ravel(), d.blocks[b].z[fs].ravel())
        m = np.abs(z - z[0]) < 1e-9
        th.extend(np.degrees(np.arctan2(y[m], x[m])).tolist())
    return np.unique(np.round(np.array(th), 6))


d, idx = ring_rect_domain(n_east=NE, side_dt=SDT, nz=NZ)
d0, _ = ring_rect_domain(n_east=NE, side_dt=0.025, nz=NZ)
th, th0 = body_angles(d), body_angles(d0)
ncell = sum(int(np.prod(b.shape)) for b in d.blocks)
ncell0 = sum(int(np.prod(b.shape)) for b in d0.blocks)

fig = plt.figure(figsize=(16, 9))
fig.suptitle(f"Jet-resolved butterfly grid for HydroGym `Cylinder`  --  side_dt={SDT}, nz={NZ}, "
             f"{ncell:,} cells (production: {ncell0:,})", fontsize=13)

def draw(ax, zoom, title):
    for b in d.blocks:
        x, y = b.x[:, :, 0], b.y[:, :, 0]
        ax.plot(x, y, lw=0.25, color="#4a6fa5", alpha=.8)
        ax.plot(x.T, y.T, lw=0.25, color="#4a6fa5", alpha=.8)
    ax.add_artist(plt.Circle((0, 0), R, facecolor="0.75", edgecolor="k", zorder=5))
    for s in (+1, -1):
        a = np.linspace(s * np.pi / 2 - OMEGA / 2, s * np.pi / 2 + OMEGA / 2, 40)
        ax.plot(R * np.cos(a) * 1.02, R * np.sin(a) * 1.02, lw=3.5, color="#e4572e", zorder=6)
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)

draw(fig.add_axes([0.04, 0.08, 0.30, 0.80]), 10.5, "full domain (jets in orange)")
draw(fig.add_axes([0.36, 0.08, 0.30, 0.80]), 1.6, "near body")

# the decisive panel: the discrete jet against the analytic cosine
ax = fig.add_axes([0.71, 0.14, 0.26, 0.68])
tt = np.linspace(-9, 9, 400)
prof = np.where(np.abs(np.radians(tt)) < OMEGA / 2,
                np.cos((np.pi / OMEGA) * np.radians(tt)), 0.0)
ax.plot(tt, prof, color="0.35", lw=1.6, label="HydroGym cos profile")
ax.axvspan(-5, 5, color="#e4572e", alpha=.10)
for arr, col, lab, mk in ((th, "#e4572e", f"side_dt={SDT}", "o"),
                          (th0, "#7a7a7a", "production (0.025)", "s")):
    dd = arr[np.abs(arr - 90.0) <= 9.0] - 90.0
    p = np.where(np.abs(np.radians(dd)) < OMEGA / 2,
                 np.cos((np.pi / OMEGA) * np.radians(dd)), 0.0)
    n_in = int((np.abs(dd) <= 5).sum())
    ax.plot(dd, p, mk, color=col, ms=7, mfc="none", mew=1.8,
            label=f"{lab}: {n_in} pts in jet")
ax.set_xlabel("theta - 90  [deg]"); ax.set_ylabel("jet amplitude")
ax.set_title("does the grid resolve the actuator?", fontsize=10)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3); ax.set_xlim(-9, 9)

fig.savefig("figures/hydrogym_jet_grid.png", dpi=110)
print(f"  wrote figures/hydrogym_jet_grid.png")
print(f"  cells {ncell:,} (nz={NZ}); azimuthal {len(th)}; "
      f"in 10-deg jet {(np.abs(th-90)<=5).sum()} (production {(np.abs(th0-90)<=5).sum()})")
