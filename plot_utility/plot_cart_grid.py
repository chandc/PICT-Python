"""The 16-block Cartesian-background cylinder grid, with the skew map that justifies it."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cylinder_cart_grid import cart_ring_domain, D, R_CYL

d, idx = cart_ring_domain(nz=2)
inv = {v: k for k, v in idx.items()}
RING = [idx["ring" + s] for s in "ENWS"]
TRAN = [idx["tran" + s] for s in "ENWS"]


def skew(b):
    B = d.blocks[b]
    t0 = np.stack([np.diff(B.x[:, :, 0], axis=0), np.diff(B.y[:, :, 0], axis=0)], -1)[:, :-1]
    t1 = np.stack([np.diff(B.x[:, :, 0], axis=1), np.diff(B.y[:, :, 0], axis=1)], -1)[:-1, :]
    n0, n1 = np.linalg.norm(t0, axis=-1), np.linalg.norm(t1, axis=-1)
    return np.abs((t0 * t1).sum(-1)) / np.maximum(n0 * n1, 1e-30)


fig = plt.figure(figsize=(17.5, 9.5))
ax0 = fig.add_subplot(2, 2, 1)
ax1 = fig.add_subplot(2, 2, 3)
ax2 = fig.add_subplot(1, 2, 2)

for ax, xl, yl in ((ax0, (-5, 15), (-5, 5)), (ax1, (-2.2, 2.2), (-2.2, 2.2))):
    for b, blk in enumerate(d.blocks):
        X, Y = blk.x[:, :, 0], blk.y[:, :, 0]
        col = "#c0392b" if b in RING else ("#e67e22" if b in TRAN else "#3465a4")
        lw = 0.4 if b in RING + TRAN else 0.3
        s0 = max(1, X.shape[0] // 80); s1 = max(1, X.shape[1] // 80)
        for i in range(0, X.shape[0], s0): ax.plot(X[i, :], Y[i, :], color=col, lw=lw)
        for j in range(0, X.shape[1], s1): ax.plot(X[:, j], Y[:, j], color=col, lw=lw)
        ax.plot(X[-1, :], Y[-1, :], color=col, lw=lw); ax.plot(X[:, -1], Y[:, -1], color=col, lw=lw)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_CYL * np.cos(th), R_CYL * np.sin(th), color="0.75", zorder=5)
    ax.plot((R_CYL + 0.5 * D) * np.cos(th), (R_CYL + 0.5 * D) * np.sin(th), "r--", lw=1.7, zorder=6)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")

n = sum(int(np.prod(b.shape)) for b in d.blocks) // 2
ax0.set_title(f"4 O-ring (red) + 4 transition (orange) + 8 RECTANGULAR (blue) -- {n:,} cells 2D")
ax1.set_title("O-ring is a TRUE ANNULUS: r in [0.50, 1.00], exactly 0.5 D thick (dashed)")

vmax = 0.75
for b, blk in enumerate(d.blocks):
    c = skew(b)
    im = ax2.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0],
                        np.pad(c, ((0, 1), (0, 1)), mode="edge"),
                        cmap="inferno_r", vmin=0, vmax=vmax, shading="auto")
ax2.fill(R_CYL * np.cos(th), R_CYL * np.sin(th), color="0.75", zorder=5)
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2); ax2.set_aspect("equal")
allc = np.concatenate([skew(b).ravel() for b in range(len(d.blocks))])
ring = np.concatenate([skew(b).ravel() for b in RING]).mean()
tran = np.concatenate([skew(b).ravel() for b in TRAN]).mean()
ax2.set_title(f"non-orthogonality |cos| between grid lines\n"
              f"O-ring mean {ring:.4f} (was 0.251)   transition mean {tran:.4f} (was 0.414)")
fig.colorbar(im, ax=ax2, fraction=0.046, label="|cos|   (0 = orthogonal)")
fig.suptitle("HydroGym-conforming cylinder grid, Cartesian background  --  "
             "x[-5,15] y[-5,5] beta=0.100", fontsize=13)
fig.tight_layout()
fig.savefig("figures/cylinder_cart_grid.png", dpi=112)
print(f"  wrote figures/cylinder_cart_grid.png | {n:,} cells | ring skew {ring:.4f} | "
      f"tran skew {tran:.4f} | domain mean {allc.mean():.4f}")
