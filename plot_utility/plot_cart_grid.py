"""Render the 16-block Cartesian-background cylinder grid."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cylinder_cart_grid import cart_ring_domain, D, R_CYL

d, idx = cart_ring_domain()
inv = {v: k for k, v in idx.items()}
RING = [idx["ring" + s] for s in "ENWS"]
TRAN = [idx["tran" + s] for s in "ENWS"]

fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
for ax, (xl, yl) in zip(axes, [((-5, 15), (-5, 5)), ((-2.0, 2.0), (-2.0, 2.0))]):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        col = "#c0392b" if b in RING else ("#e67e22" if b in TRAN else "#3465a4")
        lw = 0.35 if b in RING + TRAN else 0.28
        st = max(1, X.shape[0] // 90); st2 = max(1, X.shape[1] // 90)
        for i in range(0, X.shape[0], st): ax.plot(X[i, :], Y[i, :], color=col, lw=lw)
        for j in range(0, X.shape[1], st2): ax.plot(X[:, j], Y[:, j], color=col, lw=lw)
        ax.plot(X[-1, :], Y[-1, :], color=col, lw=lw); ax.plot(X[:, -1], Y[:, -1], color=col, lw=lw)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_CYL * np.cos(th), R_CYL * np.sin(th), color="0.75", zorder=5)
    ax.plot((R_CYL + 0.5 * D) * np.cos(th), (R_CYL + 0.5 * D) * np.sin(th),
            "r--", lw=1.6, zorder=6)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
n = sum(int(np.prod(b.shape)) for b in d.blocks) // 2
axes[0].set_title(f"16 blocks: 4 O-ring (red) + 4 transition (orange) + 8 RECTANGULAR (blue)"
                  f"  --  {n:,} cells 2D")
axes[1].set_title("near body: dashed = the 0.5 D O-ring limit, met exactly at the corners")
fig.suptitle("HydroGym-conforming cylinder grid, Cartesian background  --  "
             f"x[-5,15] y[-5,5] beta=0.100", fontsize=13)
fig.tight_layout()
fig.savefig("figures/cylinder_cart_grid.png", dpi=115)
print(f"  wrote figures/cylinder_cart_grid.png  |  {n:,} cells | "
      f"{len(d.blocks)} blocks | {len(d.connections)} connections")
