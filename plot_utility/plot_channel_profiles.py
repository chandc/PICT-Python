"""Channel LES against the in-house Re_tau = 180 DNS: mean profile and Reynolds stresses.

TWO CONVENTIONS HAD TO BE RECONCILED and both are handled here rather than at the point of
measurement, so the raw archives stay untouched:

  1  the DNS stores RAW second moments <u^2>; the fluctuation is <u^2> - U^2. At the centreline
     that is 348.51 - 348.16, i.e. 0.1% of the number being subtracted, so the difference is
     taken in float64 and the result clipped at zero rather than trusted blindly.
  2  the DNS measures its friction velocity (1.0218); the LES FIXES it at 1 by construction,
     because it is driven by a constant pressure gradient. Each profile is therefore scaled by
     its OWN u_tau, which is what makes wall units comparable at all.

The mean profile is shown against the two laws it must lie between -- U+ = y+ below y+ ~ 5 and
the log law above y+ ~ 30 -- with the caveat that at Re_tau = 180 there is barely a log layer,
so agreement with the DNS matters and agreement with the log law does not.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DNS = os.path.expanduser("~/Dropbox/Apple_MLX_CFD/sem_demo/results/minchan_re180_K/"
                         "stats_MERGED.npz")
KAPPA, B = 0.41, 5.2


def _load(path):
    """-> (y, U, u', v', w', -<u'v'>, u_tau, nu), folded onto the lower half."""
    d = np.load(path)
    y, nu, n = d["y"], float(d["nu"]), int(d["nsamp"])
    p = d["sums"] / n
    p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]])       # raw -> central
    Ly = y[-1]
    ym = Ly - y
    o = np.argsort(ym)
    q = np.empty_like(p)
    for k in range(4):
        q[k] = np.interp(y, ym[o], p[k][o])
    q[4] = -np.interp(y, ym[o], p[4][o])                            # <u'v'> is ODD
    f = 0.5 * (p + q)
    h = y <= 0.5 * Ly + 1e-12
    ut = float(np.sqrt(nu * abs((f[0][1] - f[0][0]) / (y[1] - y[0]))))
    return (y[h], f[0][h], np.sqrt(np.maximum(f[1][h], 0)), np.sqrt(np.maximum(f[2][h], 0)),
            np.sqrt(np.maximum(f[3][h], 0)), -f[4][h], ut, nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("les", help="results/<tag>_stats.npz")
    ap.add_argument("--out", default="figures/channel_re180_profiles.png")
    ap.add_argument("--label", default="LES")
    a = ap.parse_args()

    yd, Ud, ud, vd, wd, uvd, utd, nud = _load(DNS)
    yl, Ul, ul, vl, wl, uvl, utl, nul = _load(a.les)
    ypd, ypl = yd * utd / nud, yl * utl / nul

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax[0].semilogx(ypd[1:], Ud[1:] / utd, "k-", lw=2.2, label=f"DNS (SEM), $u_\\tau$={utd:.3f}")
    ax[0].semilogx(ypl[1:], Ul[1:] / utl, "C3--", lw=2.0, marker="o", ms=4, mfc="none",
                   label=f"{a.label}, $u_\\tau$={utl:.3f}")
    yv = np.logspace(-1, 0.7, 30)
    ax[0].semilogx(yv, yv, ":", c="0.55", lw=1.3)
    yl2 = np.logspace(1.2, 2.3, 30)
    ax[0].semilogx(yl2, np.log(yl2) / KAPPA + B, ":", c="0.55", lw=1.3)
    ax[0].text(1.6, 1.4, r"$U^+=y^+$", color="0.4", fontsize=9, rotation=42)
    ax[0].text(45, 13.2, r"$\frac{1}{\kappa}\ln y^++B$", color="0.4", fontsize=9)
    ax[0].set_xlabel(r"$y^+$"); ax[0].set_ylabel(r"$U^+$")
    ax[0].set_title("mean velocity")
    ax[0].set_xlim(0.5, 200); ax[0].set_ylim(0, 22)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[0].grid(alpha=0.25, which="both")

    for q_d, q_l, c, lab in ((ud, ul, "C0", "$u'^+$"), (vd, vl, "C2", "$v'^+$"),
                             (wd, wl, "C1", "$w'^+$"),
                             (np.sqrt(np.abs(uvd)), np.sqrt(np.abs(uvl)), "C4",
                              r"$\sqrt{-\langle u'v'\rangle^+}$")):
        s_d = utd if lab.startswith("$") and "sqrt" not in lab else utd
        ax[1].plot(ypd, q_d / s_d, "-", c=c, lw=2.0)
        ax[1].plot(ypl, q_l / utl, "--", c=c, lw=1.8, marker="o", ms=3.5, mfc="none",
                   markevery=3, label=lab)
    ax[1].plot([], [], "k-", lw=2.0, label="DNS")
    ax[1].plot([], [], "k--", lw=1.8, label=a.label)
    ax[1].set_xlabel(r"$y^+$"); ax[1].set_ylabel("r.m.s. / stress, wall units")
    ax[1].set_title("Reynolds stresses")
    ax[1].set_xlim(0, 180)
    ax[1].legend(frameon=False, fontsize=9, ncol=2)
    ax[1].grid(alpha=0.25)

    fig.suptitle(r"Turbulent channel, $Re_\tau=180$ minimal box "
                 rf"($L_x^+=565$, $L_z^+=192$) — {a.label} vs in-house SEM DNS", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)

    # the numbers that decide it, printed so they can be quoted rather than read off a plot
    def at(yp, q, t):
        return float(np.interp(t, yp, q))
    print(f"  {'quantity':<28}{'DNS':>10}{a.label:>10}{'diff':>10}")
    rows = [("u_tau", utd, utl),
            ("U+ centreline", Ud[-1] / utd, Ul[-1] / utl),
            ("u'+ peak", ud.max() / utd, ul.max() / utl),
            ("y+ of u'+ peak", ypd[np.argmax(ud)], ypl[np.argmax(ul)]),
            ("v'+ max", vd.max() / utd, vl.max() / utl),
            ("w'+ max", wd.max() / utd, wl.max() / utl),
            ("-<u'v'>+ max", uvd.max() / utd ** 2, uvl.max() / utl ** 2)]
    for name, a_, b_ in rows:
        print(f"  {name:<28}{a_:>10.3f}{b_:>10.3f}{100*(b_-a_)/abs(a_):>9.1f}%")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
