"""Cavity Re=1000 centreline profiles against Ghia, Ghia & Shin (1982)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src.ghia as G

d = np.load("results/ucavity_re1000.npz")
y, u, x, v, nc = d["y"], d["u"], d["x"], d["v"], int(d["ncell"])
eu = np.interp(G.Y_RE1000[::-1], y, u)[::-1] - G.U_RE1000
ev = np.interp(G.X_RE1000[::-1], x, v)[::-1] - G.V_RE1000

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax[0].plot(u, y, "-", color="#3465a4", lw=2, label=f"this solver ({nc:,} cells)")
ax[0].plot(G.U_RE1000, G.Y_RE1000, "o", color="#c0392b", ms=7, label="Ghia et al. (1982)")
ax[0].set_xlabel("u"); ax[0].set_ylabel("y"); ax[0].set_title("u along the vertical centreline x = 0.5")
ax[0].axvline(0, color="0.8", lw=0.8); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(x, v, "-", color="#1a7a3a", lw=2, label="this solver")
ax[1].plot(G.X_RE1000, G.V_RE1000, "s", color="#c0392b", ms=7, label="Ghia et al. (1982)")
ax[1].set_xlabel("x"); ax[1].set_ylabel("v"); ax[1].set_title("v along the horizontal centreline y = 0.5")
ax[1].axhline(0, color="0.8", lw=0.8); ax[1].legend(); ax[1].grid(alpha=0.3)

ax[2].plot(G.U_RE1000, G.Y_RE1000, "o-", color="#c0392b", ms=5, lw=0.8, alpha=0.4, label="Ghia u")
ax[2].plot(eu, G.Y_RE1000, "o-", color="#3465a4", ms=6, label="error in u (vs y)")
ax[2].plot(ev, G.X_RE1000, "s-", color="#1a7a3a", ms=6, label="error in v (vs x)")
ax[2].axvline(0, color="k", lw=1)
ax[2].set_xlabel("error"); ax[2].set_ylabel("y  /  x")
ax[2].set_title(f"pointwise error\nu: max {np.abs(eu).max():.4f}   v: max {np.abs(ev).max():.4f}")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
fig.suptitle("Lid-driven cavity, Re = 1000 — unstructured collocated PISO vs Ghia, Ghia & Shin (1982)",
             fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig("figures/ucavity_re1000_ghia.png", dpi=115)
print(f"  wrote figures/ucavity_re1000_ghia.png")
print(f"  u: max|err| {np.abs(eu).max():.4f}  rms {np.sqrt((eu**2).mean()):.4f}")
print(f"  v: max|err| {np.abs(ev).max():.4f}  rms {np.sqrt((ev**2).mean()):.4f}")
