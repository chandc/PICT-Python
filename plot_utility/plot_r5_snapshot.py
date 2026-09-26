"""R5 snapshot views: mean profile, then u', omega_x and p' contours.

Mean profile from the run's accumulated statistics (not the snapshot); the
contours from the rolling checkpoint. omega_x = dW/dy - dV/dz on the cell
field with wide gradients -- fine for structure, not for budgets.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RE_TAU, NU = 180.0, 1.0 / 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chan_re180_v3_mac")
    ap.add_argument("--yplus", type=float, default=12.0)
    a = ap.parse_args()

    from src import checkpoint
    from src.domains import channel_box, clustered_y

    st = np.load(f"results/{a.tag}_stats.npz")
    S = st["sums"] / int(st["nsamp"])
    y = st["y"]
    Um = S[0]
    yh = np.unique(np.round(np.minimum(y, 2.0 - y), 12))
    Uh = 0.5 * (np.interp(yh, y, Um) + np.interp(2.0 - yh, y, Um))

    y80 = clustered_y(80, re_tau=RE_TAU)
    d = channel_box(24, 80, 24, 2, Lx=np.pi, Lz=0.34 * np.pi, y_nodes=y80)
    nb = len(d.blocks)
    f, meta = checkpoint.load_fields(f"results/fields/{a.tag}.npz")
    t = float(meta.get("time", 0.0))
    U = np.concatenate([f["u"][b] for b in range(nb)], axis=0)
    V = np.concatenate([f["v"][b] for b in range(nb)], axis=0)
    W = np.concatenate([f["w"][b] for b in range(nb)], axis=0)
    P = np.concatenate([f["p"][b] for b in range(nb)], axis=0)
    x = np.concatenate([d.blocks[b].x[:, 0, 0] for b in range(nb)])
    z = d.blocks[0].z[0, 0, :]

    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])

    # (a) mean profile
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx(yh[1:] * RE_TAU, Uh[1:], "k-", lw=2,
                label=f"R5 mean, t={st['t0']:.1f}..{st['t1']:.1f} "
                      f"({int(st['nsamp'])} samples)")
    yy = np.logspace(0, np.log10(200), 120)
    ax.semilogx(yy[yy < 13], yy[yy < 13], ":", color="gray", lw=1)
    ax.semilogx(yy[yy > 8], 2.5 * np.log(yy[yy > 8]) + 5.5, ":", color="gray",
                lw=1, label="$u^+=y^+$;  $2.5\\ln y^+ + 5.5$")
    akm = np.loadtxt("../Apple_MLX_CFD/sem_demo/reference/akm_chan180/ch180.dat",
                     skiprows=81, max_rows=65) if False else None
    ax.set(xlabel="$y^+$", ylabel="$U^+$", xlim=(1, 200), ylim=(0, 21),
           title="(a) mean velocity")
    ax.legend(fontsize=8, loc="upper left")

    # (b) instantaneous mean-profile check: plane-averaged snapshot u
    ax = fig.add_subplot(gs[0, 1])
    Usnap = U.mean(axis=(0, 2))
    ax.semilogx(np.maximum(y80, 1e-3)[1:40] * RE_TAU, Usnap[1:40], "-",
                color="tab:blue", lw=1.5, label=f"snapshot t={t:.2f} (lower half)")
    ax.semilogx(yh[1:] * RE_TAU, Uh[1:], "k--", lw=1, label="time mean")
    ax.set(xlabel="$y^+$", ylabel="$\\langle u\\rangle_{xz}^+$", xlim=(1, 200),
           ylim=(0, 21), title="(b) snapshot vs time mean")
    ax.legend(fontsize=8, loc="upper left")

    j = int(np.argmin(np.abs(y80 * RE_TAU - a.yplus)))
    xp, zp = x * RE_TAU, z * RE_TAU
    Xg, Zg = np.meshgrid(xp, zp, indexing="ij")

    def plane(axp, q, name, cmap):
        lim = float(np.abs(q).max())
        c = axp.pcolormesh(Xg, Zg, q, cmap=cmap, vmin=-lim, vmax=lim,
                           shading="gouraud")
        plt.colorbar(c, ax=axp, shrink=0.9)
        axp.set(xlabel="$x^+$", ylabel="$z^+$", title=name, aspect="equal")

    up = U[:, j, :] - U[:, j, :].mean()
    pp = P[:, j, :] - P[:, j, :].mean()
    plane(fig.add_subplot(gs[1, 0]), up,
          f"(c) $u'^+$ at $y^+\\!={y80[j]*RE_TAU:.0f}$, t={t:.2f}", "RdBu_r")
    plane(fig.add_subplot(gs[1, 1]), pp,
          f"(e) $p'$ at $y^+\\!={y80[j]*RE_TAU:.0f}$", "coolwarm")

    # (d) omega_x in a cross-stream (z, y) plane at mid-x
    i0 = U.shape[0] // 2
    dWdy = np.gradient(W[i0], y80, axis=0, edge_order=2)
    dVdz = np.gradient(V[i0], z, axis=1, edge_order=2)
    ox = dWdy - dVdz
    ax = fig.add_subplot(gs[0, 2])
    Zc, Yc = np.meshgrid(zp, y80 * RE_TAU, indexing="xy")
    lim = np.percentile(np.abs(ox), 98)
    c = ax.pcolormesh(Zc, Yc, ox, cmap="RdBu_r", vmin=-lim, vmax=lim,
                      shading="gouraud")
    plt.colorbar(c, ax=ax, shrink=0.9)
    ax.set(xlabel="$z^+$", ylabel="$y^+$", ylim=(0, 180),
           title=f"(d) $\\omega_x$, cross-section at $x^+\\!={xp[i0]:.0f}$")

    fig.suptitle(f"R5 {a.tag} snapshot, t = {t:.2f}", y=0.995)
    fig.tight_layout()
    out = f"figures/{a.tag}_snapshot.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
