"""The HydroGym-conforming cylinder grid. Every number here is quoted from source.

medium.geo (DEFAULT_MESH):  x_ninf=-5, x_pinf=15, y_inf=5, n1=35
flow.py:  INLET u=(1,0) | FREESTREAM v=0 (slip) | OUTLET p=0 | CYLINDER no-slip + jets
          jets at +-90 deg, omega=pi/18 (10 deg), cos profile, BOTH outward, MAX_CONTROL 0.1

The previous grid was built at x[-10,30], y[-10,10] -- double the domain, half the blockage --
because flow.py was read and medium.geo never opened.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, warnings
warnings.filterwarnings("ignore")
from cylinder_ring_grid import ring_rect_domain, D
from cylinder_rect_bc import classify
from src.multiblock import face_slice

R, OMEGA = D/2, np.pi/18
d, idx = ring_rect_domain(nz=2)   # module defaults ARE the HydroGym-conforming set
roles = classify(d)
th, ringmax = [], 0.0
for (b, f), r in roles.items():
    if r != "body": continue
    fs = face_slice(f)
    x, y, z = (d.blocks[b].x[fs].ravel(), d.blocks[b].y[fs].ravel(), d.blocks[b].z[fs].ravel())
    m = np.abs(z - z[0]) < 1e-9
    th.extend(np.degrees(np.arctan2(y[m], x[m])).tolist())
    rr = np.sqrt(d.blocks[b].x**2 + d.blocks[b].y**2)
    ringmax = max(ringmax, float(rr.max()) - R)
th = np.unique(np.round(np.array(th), 6))
xs = np.concatenate([b.x.ravel() for b in d.blocks]); ys = np.concatenate([b.y.ravel() for b in d.blocks])
ncell = sum(int(np.prod(b.shape)) for b in d.blocks)
beta = D / (ys.max() - ys.min())

fig = plt.figure(figsize=(17, 9))
fig.suptitle(f"HydroGym-conforming cylinder grid  --  x[{xs.min():.0f},{xs.max():.0f}]  "
             f"y[{ys.min():.0f},{ys.max():.0f}]  beta={beta:.3f}  {ncell:,} cells (nz=2)",
             fontsize=13)

def mesh(ax, zoom):
    for b in d.blocks:
        X, Y = b.x[:, :, 0], b.y[:, :, 0]
        ax.plot(X, Y, lw=0.25, color="#4a6fa5", alpha=.85)
        ax.plot(X.T, Y.T, lw=0.25, color="#4a6fa5", alpha=.85)
    ax.add_artist(plt.Circle((0, 0), R, fc="0.72", ec="k", zorder=6))
    for s in (+1, -1):
        a = np.linspace(s*np.pi/2 - OMEGA/2, s*np.pi/2 + OMEGA/2, 40)
        ax.plot(R*np.cos(a)*1.03, R*np.sin(a)*1.03, lw=4, color="#e4572e", zorder=7)
    ax.set_aspect("equal")

ax = fig.add_axes([0.04, 0.55, 0.44, 0.36]); mesh(ax, None)
ax.set_xlim(-5, 15); ax.set_ylim(-5, 5)
ax.set_title("full domain: inlet x=-5, outlet x=+15, slip laterals y=+-5", fontsize=10)

ax = fig.add_axes([0.04, 0.07, 0.44, 0.40]); mesh(ax, None)
ax.add_artist(plt.Circle((0, 0), R + 0.5*D, fc="none", ec="#c0392b", ls="--", lw=1.8, zorder=8))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_title(f"near body: O-ring reaches {ringmax:.3f} D from the surface "
             f"(dashed = your 0.5 D limit)", fontsize=10)

ax = fig.add_axes([0.55, 0.55, 0.40, 0.36])
tt = np.linspace(-9, 9, 400)
ax.plot(tt, np.where(np.abs(np.radians(tt)) < OMEGA/2,
                     np.cos((np.pi/OMEGA)*np.radians(tt)), 0.0),
        color="0.35", lw=1.6, label="HydroGym cos profile")
ax.axvspan(-5, 5, color="#e4572e", alpha=.10)
dd = th[np.abs(th - 90.0) <= 9.0] - 90.0
ax.plot(dd, np.where(np.abs(np.radians(dd)) < OMEGA/2,
                     np.cos((np.pi/OMEGA)*np.radians(dd)), 0.0),
        "o", color="#e4572e", ms=8, mfc="none", mew=2,
        label=f"ours: {(np.abs(dd)<=5).sum()} pts in jet")
ax.set_xlabel("theta - 90  [deg]"); ax.set_ylabel("jet amplitude"); ax.set_xlim(-9, 9)
ax.set_title("jet resolution vs HydroGym's own (~3 pts at n1=35)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = fig.add_axes([0.55, 0.07, 0.40, 0.40]); ax.axis("off")
near = np.sort(th[np.abs(th - 90.0) <= 12.0]); gap = np.diff(near)
rows = [("domain x", "[-5, 15]", f"[{xs.min():.0f}, {xs.max():.0f}]", xs.min() == -5 and xs.max() == 15),
        ("domain y", "[-5, 5]", f"[{ys.min():.0f}, {ys.max():.0f}]", abs(ys.min()+5) < 1e-9),
        ("blockage beta", "0.100", f"{beta:.3f}", abs(beta-0.1) < 1e-3),
        ("spacing at +-90 deg", "3.27 deg", f"{gap.min():.2f}-{gap.max():.2f}", 2.5 < gap.mean() < 4.0),
        ("pts in 10-deg jet", "~3", f"{(np.abs(dd)<=5).sum()}", 2 <= (np.abs(dd)<=5).sum() <= 5),
        ("O-ring from surface", "<= 0.5 D", f"{ringmax:.3f} D", ringmax <= 0.5+1e-9)]
ax.text(0.0, 1.0, f"{'quantity':<22}{'target':>12}{'ours':>16}   ", fontsize=10,
        family="monospace", va="top", weight="bold")
for i, (k, t, v, ok) in enumerate(rows):
    ax.text(0.0, 0.90 - i*0.11, f"{k:<22}{t:>12}{v:>16}   {'OK' if ok else 'FAIL'}",
            fontsize=10, family="monospace", va="top",
            color="#1e7a46" if ok else "#c0392b")
ax.text(0.0, 0.90 - len(rows)*0.11 - 0.06,
        "targets quoted from medium.geo and flow.py", fontsize=8, color="0.4", va="top")
fig.savefig("figures/hydrogym_conforming_grid.png", dpi=105)
print(f"  wrote figures/hydrogym_conforming_grid.png")
print(f"  {ncell:,} cells | azim {len(th)} | jet pts {(np.abs(dd)<=5).sum()} | "
      f"ring {ringmax:.3f} D | beta {beta:.3f}")
