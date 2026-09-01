"""
Axial velocity profiles at the stations used by the Python_SEM outflow study.

That study reports x/h = 0.5, 1, 2, 4, but only as RELATIVE DIFFERENCES between its own short
(x<=2.5) and long (x<=8.5) domains -- 0.02% / 0.1% / 0.5% / 1.8% -- not as absolute profiles.
So there is nothing to overlay; what can be compared is the reattachment length (their 8.20
against our 8.224) and the qualitative shape of the profiles through the bubble.

Our outlet is at 30 S, far downstream of reattachment at ~8.2, so by that study's own criterion
we sit in the regime where the outflow condition is benign. Their SHORT domain, where the outlet
sits inside the recirculation and free outflow blew up on step 1, is the discriminating case and
we have never run its equivalent.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import checkpoint as ck
from armaly_bfs5_grid import S, H_IN, INLET, RECIRC_U, RECIRC_L, RECOV_U, RECOV_L

RUN = "results/fields/bfs5_Re389_dong_rc_amgx_n26720.npz"
SEM = [0.5, 1.0, 2.0, 4.0]          # the stations the SEM study tabulates
EXTRA = [6.0, 8.2, 12.0]            # through and past reattachment (x_r/S = 8.224)
XR_OURS, XR_SEM = 8.224, 8.20


def load(path):
    fl, meta = ck.load_fields(path)
    g = {b: np.load(path.replace(".npz", f"_geom{b}.npz")) for b in range(5)}
    k = g[0]["x"].shape[2] // 2
    xu = np.concatenate([g[b]["x"][:, 0, k] for b in (INLET, RECIRC_U, RECOV_U)])
    xl = np.concatenate([g[b]["x"][:, 0, k] for b in (RECIRC_L, RECOV_L)])
    return dict(
        xu=xu, xl=xl, yu=g[RECIRC_U]["y"][0, :, k], yl=g[RECIRC_L]["y"][0, :, k],
        up={q: np.concatenate([fl[q][b][:, :, k] for b in (INLET, RECIRC_U, RECOV_U)], axis=0)
            for q in ("u", "v")},
        lo={q: np.concatenate([fl[q][b][:, :, k] for b in (RECIRC_L, RECOV_L)], axis=0)
            for q in ("u", "v")},
        xr=float(meta["extra"]["xr"]), Re=float(meta["extra"]["Re"]))


def prof(d, q, xs):
    iu = int(np.argmin(np.abs(d["xu"] - xs)))
    if xs < 0:
        return d["up"][q][iu, :], d["yu"]
    il = int(np.argmin(np.abs(d["xl"] - xs)))
    return (np.concatenate([d["lo"][q][il, :], d["up"][q][iu, :]]),
            np.concatenate([d["yl"], d["yu"]]))


d = load(RUN)
fig = plt.figure(figsize=(16, 8.6))

ax = fig.add_axes([0.06, 0.60, 0.90, 0.30])
XR = np.linspace(-5, 14, 420); YR = np.linspace(-S, H_IN, 170)
GX, GY = np.meshgrid(XR, YR, indexing="ij")
from scipy.interpolate import RegularGridInterpolator
iu = RegularGridInterpolator((d["xu"], d["yu"]), d["up"]["u"], bounds_error=False, fill_value=None)
il = RegularGridInterpolator((d["xl"], d["yl"]), d["lo"]["u"], bounds_error=False, fill_value=None)
U = np.where(GY >= 0, iu(np.stack([GX, GY], -1)),
             il(np.stack([np.maximum(GX, 0), GY], -1)))
U = np.where((GX < 0) & (GY < 0), np.nan, U)
ax.contourf(XR, YR, np.nan_to_num(U).T, 24, cmap="viridis")
ax.contour(XR, YR, np.nan_to_num(U).T, levels=[0.0], colors="crimson", linewidths=1.5)
ax.add_patch(plt.Rectangle((-5, -S), 5, S, color="0.9", zorder=5))
ax.plot([0, 0], [-S, 0], "k", lw=4, zorder=6); ax.plot([-5, 0], [0, 0], "k", lw=3, zorder=6)
cols = plt.cm.plasma(np.linspace(.1, .85, len(SEM + EXTRA)))
for c, xs in zip(cols, SEM + EXTRA):
    ax.axvline(xs, color=c, lw=1.8, alpha=.9)
    ax.text(xs, H_IN * 1.08, f"{xs:g}", color=c, ha="center", fontsize=8, fontweight="bold")
ax.plot([d["xr"]], [-S], "r^", ms=12, zorder=8)
ax.set_xlim(-5, 14); ax.set_ylim(-S, H_IN * 1.02); ax.set_aspect("equal")
ax.set_xlabel("x / S"); ax.set_ylabel("y / S")
ax.set_title(f"Re = {d['Re']:.0f}, AmgX + Rhie-Chow + Dong.  red line: u = 0;  "
             f"triangle: reattachment $x_r/S$ = {d['xr']:.3f}  "
             f"(Python_SEM: {XR_SEM:.2f})", fontsize=11)

for n, (grp, ttl) in enumerate(((SEM, "stations tabulated by the Python_SEM study"),
                                (EXTRA, "through and beyond reattachment"))):
    ax = fig.add_axes([0.06 + n * 0.50, 0.08, 0.40, 0.42])
    for c, xs in zip(cols[len(SEM) * n:], grp):
        u, y = prof(d, "u", xs)
        ax.plot(u, y, color=c, lw=2.0, label=f"x/S = {xs:g}")
    ax.axvline(0, color="0.6", lw=.8)
    ax.axhline(0, color="tab:blue", ls=":", lw=1.1)
    ax.axhline(-S, color="k", lw=2.5); ax.axhline(H_IN, color="k", lw=2.5)
    ax.set_ylim(-S * 1.03, H_IN * 1.03)
    ax.set_xlabel("$u\\,/\\,U_{bulk}$"); ax.set_ylabel("y / S")
    ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(ttl, fontsize=10)

fig.suptitle("Armaly BFS Re = 389 — axial velocity profiles, five domains, Dong outflow",
             fontsize=13)
fig.savefig("figures/bfs5_axial_Re389.png", dpi=145, bbox_inches="tight")

print(f"  x_r/S ours {d['xr']:.3f}   Python_SEM {XR_SEM:.2f}   "
      f"diff {100*abs(d['xr']-XR_SEM)/XR_SEM:.2f}%")
print(f"  {'x/S':>6}{'min u':>10}{'max u':>10}{'reversed?':>11}")
for xs in SEM + EXTRA:
    u, y = prof(d, "u", xs)
    print(f"  {xs:6g}{u.min():10.4f}{u.max():10.4f}{'yes' if u.min() < -1e-6 else 'no':>11}")
print("wrote figures/bfs5_axial_Re389.png")
