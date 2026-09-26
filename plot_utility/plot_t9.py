"""T9 post-processing: force histories, St/Cd/Cl for every results/t9/*.npz, and a wake vorticity
snapshot per mesh. Run after run_ucylinder.py has written its .npz files."""
import sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.tri as mtri
sys.path.insert(0, ".")
from src.umesh import Mesh
from src.uops import Gradient

files = sorted(glob.glob("results/t9/*_re100.npz"))
if not files: sys.exit("no results/t9/*_re100.npz yet")
fig, ax = plt.subplots(2, 2, figsize=(16, 9.5)); rows = []
for f in files:
    d = np.load(f); h = d["hist"]; t, cd, cl, cdp = h.T; nm = f.split("/")[-1].replace("_re100.npz", "")
    n2 = int(0.6 * len(t)); tw, clw = t[n2:], cl[n2:] - cl[n2:].mean()
    fr = np.fft.rfftfreq(len(tw), d=t[1] - t[0]); A = np.abs(np.fft.rfft(clw * np.hanning(len(tw))))
    # St from linearly interpolated upward zero-crossings of C_L (the FFT bin is ~0.017 wide with
    # ten periods in the window; parabolic peak refinement gave 0.1783 vs 0.1775 by zero-crossings)
    z = np.flatnonzero(np.diff(np.sign(clw)) > 0); tz = tw[z] - clw[z] * (tw[z + 1] - tw[z]) / (clw[z + 1] - clw[z])
    per = np.diff(tz); St = 1.0 / per.mean(); print(f"  {nm}: {len(per)} periods, mean {per.mean():.4f} +- {per.std():.4f}")
    rows.append((nm, len(d["u"]), St, cd[n2:].mean(), 0.5 * (cl[n2:].max() - cl[n2:].min()), np.sqrt((clw ** 2).mean()), cdp[n2:].mean(), t[-1]))
    ax[0, 0].plot(t, cd, lw=1.0, label=f"{nm} ({len(d['u'])} cells)"); ax[0, 1].plot(t, cl, lw=0.8, label=nm)
    ax[1, 0].plot(fr[1:], A[1:] / A[1:].max(), lw=1.2, label=f"{nm}: St={St:.4f}")
ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("C_D"); ax[0, 0].set_title("drag coefficient"); ax[0, 0].set_ylim(1.0, 2.0)
ax[0, 1].set_xlabel("t"); ax[0, 1].set_ylabel("C_L"); ax[0, 1].set_title("lift coefficient")
ax[1, 0].set_xlim(0, 0.6); ax[1, 0].set_xlabel("frequency"); ax[1, 0].set_ylabel("|FFT(C_L)| (normalised)"); ax[1, 0].set_title("lift spectrum, last 40% of the run")
ax[1, 0].axvline(0.165, color="k", ls=":", lw=1, label="St=0.165 (unconfined literature)")
for a in ax.ravel()[:3]: a.grid(alpha=0.3); a.legend(fontsize=8)
# wake vorticity snapshot from the LAST file (final fields)
d = np.load(files[-1]); m = Mesh(d["nodes"], [d["cells"][k, :d["nvert"][k]] for k in range(len(d["nvert"]))], span=1.0)
g = Gradient(m); C = m.centroid
bt = d["btag"][m.bfaces]; fb = m.fcentre[m.bfaces]; ub = np.where(bt == 2, 1.0, 0.0); ub = np.where(bt == 5, 0.0, np.where(bt == 2, 1.0, np.nan))
# boundary values for the gradient: inlet 1, cylinder 0, elsewhere owner value
uo = d["u"][m.owner[m.bfaces]]; ubv = np.where(np.isnan(ub), uo, ub); vbv = np.where(np.isin(bt, [2, 3, 5]), 0.0, d["v"][m.owner[m.bfaces]])
w = g(d["v"], vbv)[:, 0] - g(d["u"], ubv)[:, 1]
a = ax[1, 1]; tc = mtri.Triangulation(C[:, 0], C[:, 1]); c = a.tricontourf(tc, np.clip(w, -3, 3), np.linspace(-3, 3, 31), cmap="RdBu_r", extend="both")
from matplotlib.collections import PolyCollection
a.add_collection(PolyCollection([d["nodes"][d["cells"][k, :d["nvert"][k]]] for k in range(len(d["nvert"]))], facecolor="none", edgecolor="k", linewidth=0.12))
th = np.linspace(0, 2 * np.pi, 100); a.fill(0.5 * np.cos(th), 0.5 * np.sin(th), "0.3"); plt.colorbar(c, ax=a, shrink=0.8)
a.set_xlim(-2, 12); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_title(f"vorticity at t={rows[-1][-1]:.0f}, {files[-1].split('/')[-1]}", fontsize=10)
plt.suptitle("T9: circular cylinder, Re=100, HydroGym Firedrake specification (domain [-5,15]x[-5,5], beta=0.10)", fontsize=12)
plt.tight_layout(); plt.savefig("figures/t9_forces.png", dpi=130)
print(f"{'mesh':22s} {'cells':>6} {'St':>7} {'Cd':>7} {'Cl_amp':>7} {'Cl_rms':>7} {'Cd_p':>7} {'T':>5}")
for r in rows: print(f"{r[0]:22s} {r[1]:6d} {r[2]:7.4f} {r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f} {r[6]:7.4f} {r[7]:5.0f}")
print("literature, UNCONFINED Re=100: St 0.164-0.167, Cd 1.33-1.35, Cl_amp ~0.33  (beta=0.10 blockage raises Cd and St slightly)")
print("wrote figures/t9_forces.png")
