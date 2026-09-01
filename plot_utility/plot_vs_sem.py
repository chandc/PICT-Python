"""
Our Re=389 BFS against the Python_SEM "LONG / P+Z" reference, digitised from its figure.

GEOMETRIES DIFFER AND THE COMPARISON IS ONLY FAIR AFTER RESCALING. Their channel is y in [0,1]
with the step at y = 0.5, so their step height is h = 0.5 and their expansion ratio is
1.0/0.5 = 2.00. Ours uses Armaly's actual ratio, h_in = 1.0612 with S = 1, ER = 1.9423. Mapping
their wall-normal coordinate onto ours by step height, y/S = (y_them - 0.5)/0.5, puts both
bottom walls at -1 and both step levels at 0 -- but their top wall lands at +1 and ours at
+1.0612. That 6% difference in channel height is real and unremovable, so exact agreement near
the top wall is not expected and any disagreement there is geometry, not solver error.

Stations map as x/h = 2 * x_them, which is our x/S directly. Both cases have inlet peak
velocity 1.5, so u needs no rescaling.

The reference curves come from digitising the green lines out of figs/bfs_cmp_profiles.png --
pixel extraction, not published data -- so they carry their own error, of order the line width.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import checkpoint as ck
from armaly_bfs5_grid import S, H_IN, INLET, RECIRC_U, RECIRC_L, RECOV_U, RECOV_L

RUN = "results/fields/bfs5_Re389_dong_rc_amgx_n26720.npz"
SEM = json.load(open("results/sem_green_digitized.json"))
STATIONS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]   # (their x, our x/S = x/h)


def load(path):
    fl, meta = ck.load_fields(path)
    g = {b: np.load(path.replace(".npz", f"_geom{b}.npz")) for b in range(5)}
    k = g[0]["x"].shape[2] // 2
    return dict(
        xu=np.concatenate([g[b]["x"][:, 0, k] for b in (INLET, RECIRC_U, RECOV_U)]),
        xl=np.concatenate([g[b]["x"][:, 0, k] for b in (RECIRC_L, RECOV_L)]),
        yu=g[RECIRC_U]["y"][0, :, k], yl=g[RECIRC_L]["y"][0, :, k],
        up={q: np.concatenate([fl[q][b][:, :, k] for b in (INLET, RECIRC_U, RECOV_U)], axis=0)
            for q in ("u", "v", "p")},
        lo={q: np.concatenate([fl[q][b][:, :, k] for b in (RECIRC_L, RECOV_L)], axis=0)
            for q in ("u", "v", "p")},
        xr=float(meta["extra"]["xr"]))


def prof(d, q, xs):
    iu = int(np.argmin(np.abs(d["xu"] - xs)))
    il = int(np.argmin(np.abs(d["xl"] - xs)))
    return (np.concatenate([d["lo"][q][il, :], d["up"][q][iu, :]]),
            np.concatenate([d["yl"], d["yu"]]))


d = load(RUN)
# reference the pressure to the inlet-plane mean, as the SEM figure does
i_in = int(np.argmin(np.abs(d["xu"] - 0.0)))
p_ref = d["up"]["p"][i_in, :].mean()

fig, axes = plt.subplots(3, len(STATIONS), figsize=(16, 10.5))
rows = [("u", "$u$"), ("v", "$v$"), ("p", "$p$ (rel. inlet mean)")]
stats = []
for ci, (xt, xs) in enumerate(STATIONS):
    for ri, (q, lab) in enumerate(rows):
        ax = axes[ri, ci]
        v, y = prof(d, q, xs)
        if q == "p":
            v = v - p_ref
        ax.plot(v, y, color="#1f77b4", lw=2.2, label="ours (PISO, ER 1.94)", zorder=3)
        key = f"{q}_x{xt:g}"
        if key in SEM:
            sy = (np.array(SEM[key]["y"]) - 0.5) / 0.5      # their y -> our y/S
            sv = np.array(SEM[key]["v"])
            ax.plot(sv, sy, color="#2ca02c", lw=1.6, ls="--",
                    label="Python_SEM LONG/P+Z (ER 2.00)", zorder=4)
            # compare only where both channels exist: y/S in [-1, +1]
            m = (sy >= -1) & (sy <= 1)
            ours = np.interp(sy[m], y, v)
            rel = np.abs(ours - sv[m]).max() / max(np.abs(sv[m]).max(), 1e-12)
            stats.append((q, xs, rel))
        ax.axhline(0, color="tab:blue", ls=":", lw=1)
        ax.axhline(-S, color="k", lw=2); ax.axhline(H_IN, color="k", lw=2)
        ax.axhline(1.0, color="0.5", lw=1.2, ls="-.")     # THEIR top wall
        ax.set_ylim(-1.05, H_IN * 1.05); ax.grid(alpha=.25)
        if ri == 0:
            ax.set_title(f"x/S = {xs:g}   (their x = {xt:g})", fontsize=10)
        if ci == 0:
            ax.set_ylabel(f"{lab}\ny / S", fontsize=9)
        if ri == 2:
            ax.set_xlabel("value")
axes[0, 0].legend(fontsize=7.5, loc="lower right")
axes[0, 0].plot([], [], color="0.5", lw=1.2, ls="-.", label="their top wall")
fig.suptitle(f"Armaly BFS Re = 389 — ours vs Python_SEM LONG/P+Z (digitised).  "
             f"$x_r/S$: ours {d['xr']:.3f}, theirs 8.20", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/bfs5_vs_sem_Re389.png", dpi=145, bbox_inches="tight")

print(f"  x_r/S  ours {d['xr']:.3f}   SEM 8.20   diff {100*abs(d['xr']-8.20)/8.20:.2f}%\n")
print(f"  {'field':>6}{'x/S':>7}{'max rel diff':>15}")
for q, xs, rel in stats:
    print(f"  {q:>6}{xs:7g}{rel:14.1%}")
print("\nwrote figures/bfs5_vs_sem_Re389.png")
