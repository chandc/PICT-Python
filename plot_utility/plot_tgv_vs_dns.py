"""Taylor-Green at Re = 400: our LES against a resolved spectral-element DNS.

THE REFERENCE. `results/tgv_diag_re400.npz`, 6x6 spectral elements at order 8 with Nz = 48,
carrying `-dE/dt / 2 nu Om = 1.0000` throughout -- it removes no energy of its own, which is
what makes it usable as a reference at all.

THE COMPARISON IS NOT DOF-FOR-DOF, and the figure says so. An order-8 spectral element resolves
roughly 2.5-3x more per direction than second-order central differences, so our 48^3 sits well
BELOW the reference's 48^3 in resolving power. That is the correct setup for an LES test -- a
coarse run judged against a resolved one -- but quoting "48^3 vs 48^3" would imply a parity that
does not exist.

WHAT THE FIGURE HAS TO SHOW HONESTLY. Four of the five 48^3 configurations FAILED, and a curve
that simply stops is easy to miss. Failures are marked where they end, and the run that reached
the end is drawn heavier, because "Smagorinsky tracks to 1.6% mean error" and "Smagorinsky was
the only one that finished" are different claims and both belong on the plot.
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

NU, V = 0.0025, (2 * np.pi) ** 3
SRC = _sys.argv[1] if len(_sys.argv) > 1 else "/tmp/tgv400_all.txt"
# LINE STYLE, NOT COLOUR, distinguishes the models. Five coloured curves lying on top of one
# another was unreadable precisely where it mattered -- at 64^3 the four working closures agree
# to about a percent, so colour was encoding a difference the eye could not resolve anyway.
# Dashes separate them without implying the differences are large.
STYLE = {"smagorinsky": (0, ()), "wale": (0, (6, 2)), "vreman": (0, (2, 2)),
         "sigma": (0, (7, 2, 1, 2)), "none": (0, (1, 3))}


def load_runs(path):
    runs, key = {}, None
    for line in open(path):
        if line.startswith("###"):
            _, n, m = line.split()
            key = (int(n), m)
            runs[key] = []
        elif key and line.strip():
            p = line.split()
            if len(p) >= 7:
                runs[key].append([float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])])
    return {k: np.array(v) for k, v in runs.items() if v}


def fail_time(a, tr, Dr):
    """When it went, which is EARLIER than when it was noticed.

    A run reported as fine at t = 9 may have passed the enstrophy bar at t = 7. Reporting the
    last sample would credit it with two time units it did not earn.
    """
    d = np.interp(a[:, 0], tr, Dr)
    bad = np.where((a[:, 2] < 0) | ~np.isfinite(a[:, 2]) | (a[:, 3] > ENSTROPHY_BLOWUP * d))[0]
    return a[bad[0], 0] if len(bad) else a[-1, 0]


def note(ax, failed):
    """Name the configurations that blew up, rather than drawing them.

    A diverging curve shoots off the axes and drags the y-limits with it, which compresses every
    surviving run into a band and hides the differences the figure exists to show. But omitting
    the failures silently would be worse: "four models agree to a percent" and "four of nine
    configurations did not survive" are both results, and the second is arguably the more
    important one. So they are listed, with the time they failed.
    """
    if not failed:
        return
    txt = "did not survive:\n" + "\n".join(f"   {m} (t = {t:.0f})" for m, t in failed)
    ax.text(0.985, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f6f0ee", ec="#c0392b", lw=0.9))


ENSTROPHY_BLOWUP = 2.0


def status(a, tr=None, Dr=None):
    """done | failed | running.

    A first version called anything short of t = 11.5 a failure, which would have labelled five
    HEALTHY 64^3 runs, still mid-flight, as blown up. So the test is on the SOLUTION, not on how
    far it got. Two signatures, and both are needed:

      NEGATIVE DISSIPATION. Energy cannot increase in a decaying flow, so eps < 0 is
      unambiguous. This catches a run after it has already gone.

      RESOLVED ENSTROPHY FAR ABOVE THE DNS. It catches one on the way. 48^3 Vreman still had
      eps > 0 at t = 9 and was reported as "running" for hours, while its 2 nu Z stood at
      0.02572 against the DNS's 0.01098 -- 2.3x. There is no physical way for a coarse grid to
      hold MORE enstrophy than a resolved one; that is grid-scale vorticity accumulating, and
      every failure in this study passed through it. The bar is 2x, which no surviving run
      approaches: the worst is WALE at the 64^3 peak, 1.16x.
    """
    eps = a[:, 2]
    if not np.all(np.isfinite(eps)) or eps[-1] < 0:
        return "failed"
    if tr is not None:
        d = np.interp(a[:, 0], tr, Dr)
        if np.any(a[:, 3] > ENSTROPHY_BLOWUP * d):
            return "failed"
    return "done" if a[-1, 0] >= 11.5 else "running"


def main():
    z = np.load("results/tgv_diag_re400.npz")
    tr, Er, Zr = z["t"], z["E"] / V, z["Om"] / V
    runs = load_runs(SRC)

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.6))
    for col, n in enumerate((48, 64)):
        ax = axes[0, col]
        ax.plot(tr, Er, color="#b0b0b0", lw=6, solid_capstyle="round",
                label="SEM DNS (order 8, Nz=48)", zorder=1)
        failed = []
        for m in ("smagorinsky", "wale", "vreman", "sigma", "none"):
            a = runs.get((n, m))
            if a is None:
                continue
            st = status(a, tr, 2 * NU * Zr)
            if st == "failed":
                failed.append((m, fail_time(a, tr, 2 * NU * Zr)))
                continue
            ax.plot(a[:, 0], a[:, 1], color="k", lw=1.5, ls=STYLE[m], zorder=3,
                    label=m + ("" if st == "done" else "  (running)"))
        note(ax, failed)
        ax.set_xlim(0, 15); ax.set_ylim(0, 0.14)
        ax.set_ylabel("E"); ax.set_title(f"{n}$^3$ — kinetic energy", fontsize=11)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
        ax.grid(alpha=0.25)

        ax = axes[1, col]
        ax.plot(tr, 2 * NU * Zr, color="#b0b0b0", lw=6, solid_capstyle="round",
                label="DNS, $2\\nu Z$", zorder=1)
        failed = []
        for m in ("smagorinsky", "wale", "vreman", "sigma", "none"):
            a = runs.get((n, m))
            if a is None:
                continue
            st = status(a, tr, 2 * NU * Zr)
            if st == "failed":
                failed.append((m, fail_time(a, tr, 2 * NU * Zr)))
                continue
            ax.plot(a[:, 0], a[:, 2], color="k", lw=1.5, ls=STYLE[m], zorder=3,
                    label=m + ("" if st == "done" else "  (running)"))
        note(ax, failed)
        ax.axvline(6.0, color="k", ls=":", lw=0.9)
        ax.text(6.15, 0.0148, "DNS peak $t=6.00$", fontsize=8)
        ax.set_xlim(0, 15); ax.set_ylim(0, 0.016)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("t"); ax.set_ylabel(r"$\epsilon = -dE/dt$")
        ax.set_title(f"{n}$^3$ — dissipation rate  (negative = the run has gone unstable)",
                     fontsize=11)
        ax.legend(fontsize=8, loc="center right", framealpha=0.95)
        ax.grid(alpha=0.25)

    fig.suptitle("Taylor–Green, Re = 400, against a spectral-element DNS.  Our 48$^3$ has "
                 "roughly 1/3 the resolving power per direction of the order-8 reference —\n"
                 "a coarse LES judged against a resolved run, which is the point.\n"
                 "Only surviving runs are drawn; configurations that diverged are named in the "
                 "box.", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "figures/tgv400_vs_dns.png"
    fig.savefig(out, dpi=115)
    print(f"  wrote {out}")
    for (n, m), a in sorted(runs.items()):
        e = np.interp(a[:, 0], tr, Er)
        err = np.abs(a[:, 1] - e) / e
        st = status(a, tr, 2 * NU * Zr)
        end = {"done": "reached t=%.0f" % a[-1, 0],
               "failed": "DIVERGED at t=%.0f" % fail_time(a, tr, 2 * NU * Zr),
               "running": "running, at t=%.0f" % a[-1, 0]}[st]
        print(f"  {n}^3 {m:<12} {end:<18} mean |err| {100*err.mean():5.2f}%  "
              f"max {100*err.max():5.2f}%")


if __name__ == "__main__":
    main()
