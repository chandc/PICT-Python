"""Near-wall structure of the Re_tau = 180 channel LES: streaks and pressure.

WHAT THIS IS FOR. The profile comparison says the streamwise fluctuation is 9% too strong and
its peak sits too far from the wall, while both cross-stream components are ~14% too weak. That
is the signature of an under-resolved near-wall cycle, but a profile cannot show WHY. The
wall-parallel plane can: the near-wall cycle is carried by low- and high-speed STREAKS, and
their spacing is the one length in wall turbulence that is genuinely universal --
lambda_z+ = 100, near enough independent of Reynolds number.

So this plot is a falsifiable check, not an illustration. The box is L_z+ = 192, so a correctly
resolved field must show ABOUT TWO streak pairs across the span. Too few means the box is
squeezing them; too many, or none, means the near-wall cycle is not being sustained.

THE FLUCTUATION IS PLOTTED, NOT THE VELOCITY. u itself is dominated by the mean profile, which
varies by a factor of twenty across the plane's own y-station and would swamp the structure.
Subtracting the plane mean leaves exactly the streaks.

PRESSURE IS SHOWN ON THE SAME PLANE because it is the other half of the cycle: the streamwise
vortices that generate the streaks leave a pressure signature, and a field with plausible
streaks but structureless pressure would be a warning that the streaks are a numerical artefact
rather than a dynamical structure.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RE_TAU, NU = 180.0, 1.0 / 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="results/fields/chan_re180_v2.npz")
    ap.add_argument("--yplus", type=float, default=12.0)
    ap.add_argument("--out", default="figures/channel_re180_nearwall.png")
    a = ap.parse_args()

    from src import checkpoint
    from src.domains import channel_box, clustered_y

    y80 = clustered_y(80, re_tau=RE_TAU)
    Lx, Lz = np.pi, 0.34 * np.pi
    d = channel_box(24, 80, 24, 2, Lx=Lx, Lz=Lz, y_nodes=y80)
    nb = len(d.blocks)
    f, meta = checkpoint.load_fields(a.field)
    t = float(meta.get("time", 0.0))

    # stitch the two x-blocks back together
    U = np.concatenate([f["u"][b] for b in range(nb)], axis=0)
    P = np.concatenate([f["p"][b] for b in range(nb)], axis=0)
    V = np.concatenate([f["v"][b] for b in range(nb)], axis=0)
    x = np.concatenate([d.blocks[b].x[:, 0, 0] for b in range(nb)])
    z = d.blocks[0].z[0, 0, :]

    j = int(np.argmin(np.abs(y80 * RE_TAU - a.yplus)))
    yp = y80[j] * RE_TAU
    up = U[:, j, :] - U[:, j, :].mean()
    pp = P[:, j, :] - P[:, j, :].mean()
    vp = V[:, j, :] - V[:, j, :].mean()

    xp, zp = x * RE_TAU, z * RE_TAU
    Xg, Zg = np.meshgrid(xp, zp, indexing="ij")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for k, (q, name, cmap) in enumerate((
            (up, r"streamwise fluctuation $u'^+$", "RdBu_r"),
            (vp, r"wall-normal fluctuation $v'^+$", "PuOr_r"),
            (pp, r"pressure fluctuation $p'$", "coolwarm"))):
        lim = float(np.abs(q).max())
        c = ax[k].pcolormesh(Xg, Zg, q, cmap=cmap, vmin=-lim, vmax=lim, shading="gouraud")
        ax[k].contour(Xg, Zg, q, levels=[-0.5 * lim, 0.5 * lim], colors="k",
                      linewidths=0.4, alpha=0.5)
        fig.colorbar(c, ax=ax[k], fraction=0.046, pad=0.03)
        ax[k].set_xlabel(r"$x^+$")
        ax[k].set_ylabel(r"$z^+$")
        ax[k].set_title(name, fontsize=10)
        ax[k].set_aspect("equal")

    # the falsifiable bit: count streaks from the spanwise spectrum of u'
    su = np.abs(np.fft.rfft(up - up.mean(), axis=1)) ** 2
    su = su.mean(axis=0)
    kz = np.fft.rfftfreq(up.shape[1], d=(z[1] - z[0]))
    kpk = int(np.argmax(su[1:]) + 1)
    lam = 1.0 / kz[kpk] * RE_TAU
    fig.suptitle(rf"Channel $Re_\tau=180$ LES, wall-parallel plane at $y^+={yp:.1f}$, $t={t:.1f}$"
                 rf" — $L_z^+={Lz*RE_TAU:.0f}$, peak spanwise wavelength $\lambda_z^+={lam:.0f}$"
                 rf" (universal near-wall value $\approx100$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs("figures", exist_ok=True)
    fig.savefig(a.out, dpi=150)

    print(f"  plane y+ = {yp:.2f} (index {j}), t = {t:.2f}")
    print(f"  u' range {up.min():+.3f} to {up.max():+.3f}   rms {up.std():.3f}")
    print(f"  v' range {vp.min():+.3f} to {vp.max():+.3f}   rms {vp.std():.3f}")
    print(f"  p' range {pp.min():+.3f} to {pp.max():+.3f}   rms {pp.std():.3f}")
    print(f"  spanwise: L_z+ = {Lz*RE_TAU:.0f}, peak lambda_z+ = {lam:.0f} "
          f"-> {Lz*RE_TAU/lam:.1f} streak pairs across the span")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
