"""Channel profiles from SNAPSHOTS, against the reference DNS.

WHY SNAPSHOTS AND NOT STATISTICS. The production run stalled at t ~ 5, before its sampling
window opened at t = 12, so no converged LES statistics exist yet. What does exist is the clean
t = 4 field and the interpolated initial condition. Plane-averaging a single snapshot over the
24 x 24 = 576 points at each wall-normal station gives a mean profile that is quite smooth and
second moments that are NOT: a single snapshot measures the SPATIAL variance within a plane,
where the DNS reference is a time average over 8571 samples. The stresses here are therefore
indicative of shape, not of magnitude, and are labelled as such.

WHAT THE COMPARISON IS ACTUALLY FOR. Plotting t = 0 beside t = 4 asks a sharper question than
either alone: the initial condition IS the DNS field, interpolated, so any gap that has opened
by t = 4 is what the LES did to it. Drift towards the laminar parabola would mean the model or
the mesh is killing the turbulence; drift in the stresses with the mean holding would mean
something subtler. Neither is visible from a single curve.

Both profiles are scaled by their OWN u_tau, and for the LES that number is trustworthy only
because van Driest damping drives nu_t to zero at the wall -- undamped Smagorinsky leaves
nu_t/nu = 0.92 there, which makes sqrt(nu du/dy) meaningless as a friction velocity.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DNS = os.path.expanduser("~/Dropbox/Apple_MLX_CFD/sem_demo/results/minchan_re180_K/"
                         "stats_MERGED.npz")
RE_TAU, NU = 180.0, 1.0 / 180.0


def fold(y, p):
    """(y, U, u', v', w', -<u'v'>) on the lower half; U,uu,vv,ww EVEN, <u'v'> ODD."""
    Ly = y[-1]
    ym = Ly - y
    o = np.argsort(ym)
    q = np.empty_like(p)
    for k in range(4):
        q[k] = np.interp(y, ym[o], p[k][o])
    q[4] = -np.interp(y, ym[o], p[4][o])
    f = 0.5 * (p + q)
    h = y <= 0.5 * Ly + 1e-12
    return (y[h], f[0][h], np.sqrt(np.maximum(f[1][h], 0)), np.sqrt(np.maximum(f[2][h], 0)),
            np.sqrt(np.maximum(f[3][h], 0)), -f[4][h])


def from_dns():
    d = np.load(DNS)
    p = d["sums"] / int(d["nsamp"])
    p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]])
    y = d["y"]
    out = fold(y, p)
    ut = float(np.sqrt(float(d["nu"]) * abs((out[1][1] - out[1][0]) / (y[1] - y[0]))))
    return out, ut


def from_snapshot(uvw, d):
    """Raw moments of one snapshot, plane-averaged, then folded."""
    ys = np.concatenate([b.y.ravel() for b in d.blocks])
    y = np.unique(np.round(ys, 9))
    idx = [np.searchsorted(y, np.round(b.y, 9)) for b in d.blocks]
    cnt = np.zeros(len(y))
    for b in range(len(d.blocks)):
        np.add.at(cnt, idx[b].ravel(), 1.0)
    acc = np.zeros((5, len(y)))
    for b in range(len(d.blocks)):
        i = idx[b].ravel()
        u, v, w = (a.ravel() for a in uvw[b])
        np.add.at(acc[0], i, u)
        np.add.at(acc[1], i, u * u)
        np.add.at(acc[2], i, v * v)
        np.add.at(acc[3], i, w * w)
        np.add.at(acc[4], i, u * v)
    acc /= cnt
    p = np.array([acc[0], acc[1] - acc[0] ** 2, acc[2], acc[3], acc[4]])
    out = fold(y, p)
    ut = float(np.sqrt(NU * abs((out[1][1] - out[1][0]) / (y[1] - y[0]))))
    return out, ut


def main():
    from src import checkpoint
    from src.domains import channel_box, clustered_y
    from channel_ic import interpolate_to

    y80 = clustered_y(80, re_tau=RE_TAU)
    d = channel_box(24, 80, 24, 2, y_nodes=y80)
    nb = len(d.blocks)

    ic = interpolate_to(d)
    snap0, ut0 = from_snapshot({b: ic[b] for b in range(nb)}, d)

    f, meta = checkpoint.load_fields("results/fields/chan_re180_t4.npz")
    uvw4 = {b: (f["u"][b], f["v"][b], f["w"][b]) for b in range(nb)}
    snap4, ut4 = from_snapshot(uvw4, d)

    dns, utd = from_dns()

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.1))
    series = [("DNS (SEM, 8571 samples)", dns, utd, "k", "-", 2.4),
              (r"LES $t=0$ (interpolated IC)", snap0, ut0, "C0", "--", 1.8),
              (r"LES $t=4$ (1 snapshot)", snap4, ut4, "C3", "-.", 1.8)]

    for lab, s, ut, c, ls, lw in series:
        yp = s[0] * ut / NU
        ax[0].semilogx(yp[1:], s[1][1:] / ut, ls, c=c, lw=lw,
                       label=f"{lab}, $u_\\tau$={ut:.3f}")
    yv = np.logspace(-1, 0.7, 20)
    ax[0].semilogx(yv, yv, ":", c="0.6", lw=1.2)
    yl = np.logspace(1.2, 2.3, 20)
    ax[0].semilogx(yl, np.log(yl) / 0.41 + 5.2, ":", c="0.6", lw=1.2)
    ax[0].text(1.5, 1.5, r"$U^+=y^+$", color="0.45", fontsize=8.5, rotation=40)
    ax[0].text(42, 13.4, r"$\frac{1}{\kappa}\ln y^++B$", color="0.45", fontsize=8.5)
    ax[0].set_xlabel(r"$y^+$"); ax[0].set_ylabel(r"$U^+$")
    ax[0].set_title("mean velocity")
    ax[0].set_xlim(0.5, 200); ax[0].set_ylim(0, 22)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[0].grid(alpha=0.22, which="both")

    for j, (name, k) in enumerate((("$u'^+$", 2), ("$v'^+$", 3), ("$w'^+$", 4))):
        for lab, s, ut, c, ls, lw in series:
            ax[1].plot(s[0] * ut / NU, s[k] / ut, ls, c=f"C{j}" if c != "k" else "k",
                       lw=lw if c == "k" else 1.5,
                       alpha=1.0 if c == "k" else (0.85 if c == "C0" else 0.9))
    for j, name in enumerate(("$u'^+$", "$v'^+$", "$w'^+$")):
        ax[1].plot([], [], "-", c=f"C{j}", lw=2, label=name)
    ax[1].plot([], [], "k-", lw=2.4, label="DNS")
    ax[1].plot([], [], "k--", lw=1.5, label=r"LES $t=0$")
    ax[1].plot([], [], "k-.", lw=1.5, label=r"LES $t=4$")
    ax[1].set_xlabel(r"$y^+$"); ax[1].set_ylabel("r.m.s., wall units")
    ax[1].set_title("Reynolds stresses  (LES: single snapshot, spatial variance only)")
    ax[1].set_xlim(0, 180)
    ax[1].legend(frameon=False, fontsize=8.5, ncol=2)
    ax[1].grid(alpha=0.22)

    fig.suptitle(r"Channel $Re_\tau=180$ minimal box — LES snapshots vs in-house SEM DNS. "
                 "LES statistics are NOT converged: the production run stalled at $t\\approx5$, "
                 "before sampling began.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs("figures", exist_ok=True)
    out = "figures/channel_re180_snapshots.png"
    fig.savefig(out, dpi=150)

    print(f"  {'quantity':<26}{'DNS':>10}{'LES t=0':>10}{'LES t=4':>10}")
    rows = [("u_tau", utd, ut0, ut4),
            ("U+ centreline", dns[1][-1] / utd, snap0[1][-1] / ut0, snap4[1][-1] / ut4),
            ("u'+ peak", dns[2].max() / utd, snap0[2].max() / ut0, snap4[2].max() / ut4),
            ("v'+ max", dns[3].max() / utd, snap0[3].max() / ut0, snap4[3].max() / ut4),
            ("w'+ max", dns[4].max() / utd, snap0[4].max() / ut0, snap4[4].max() / ut4),
            ("-<u'v'>+ max", dns[5].max() / utd**2, snap0[5].max() / ut0**2,
             snap4[5].max() / ut4**2)]
    for n, a, b, c in rows:
        print(f"  {n:<26}{a:>10.3f}{b:>10.3f}{c:>10.3f}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
