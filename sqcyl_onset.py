"""Critical Reynolds number for the square cylinder, from growth rates rather than from runs
that shed.

WHY THIS IS THE CHEAP EXPERIMENT. Re_c is a sharp published number -- about 45-47 at ~5%
blockage -- and finding it by asking "did this case shed?" needs runs long enough to SATURATE at
every Re, which is 380 time units each. The growth rate does not: after the kick the amplitude
grows as exp(sigma t) for as long as the perturbation stays small, so ~70 time units at each Re
measures sigma, and near onset

    sigma = k (Re - Re_c)

so a straight line through a handful of sigma values crosses zero at Re_c. Every run is a
quarter of the length and the answer is a fit rather than a bisection.

IT ALSO TESTS THE TOLERANCE FINDING. At tol = 1e-4 the same sweep must return sigma <= 0 at
every Re, because the under-resolved pressure correction damps the growing mode. That is a
sharper statement than "it converged to steady" and it is falsifiable.

WHICH PART OF THE SIGNAL IS THE GROWTH RATE. Not all of it. The first few time units after the
kick are the perturbation rearranging itself onto the unstable eigenmode, and the top of the
range is nonlinear saturation. Both bias sigma low. `growth_rate` fits only the window between
them and reports how many e-folds it actually used, because a fit over half a decade is not
worth the same as one over two.
"""
import glob
import os
import sys

import numpy as np

RE_C_REF = (45.0, 47.0)          # published onset band, ~5% blockage
WIN = None                       # kept for the API; the envelope needs no window


def envelope(t, v, win=None):
    """Oscillation amplitude from CONSECUTIVE LOCAL EXTREMA: (t_mid, |v_hi - v_lo| / 2).

    Two estimators were tried first and both failed on this signal:

    Peak-to-peak in a fixed window needs the window short to resolve the growth and long to
    average the phase, and cannot be both: at a third of a period the fit residual was 0.40 in
    log amplitude, which swamps a growth rate measured over two e-folds.

    A Hilbert envelope is local in principle and global in practice -- it is computed by FFT, so
    the saturated tail of a 300-time-unit record leaks into the low-amplitude growth phase at
    the start and put the first envelope sample at 0.085 where the true amplitude was 0.01.
    Analysing only the growth segment would fix it, but the segment is what we are trying to
    find.

    Successive extrema are local by construction, need no window, and give two samples per
    period -- about a dozen through a growth phase, which is enough to fit a line through.
    """
    dv = np.diff(v)
    turn = np.where(np.sign(dv[1:]) != np.sign(dv[:-1]))[0] + 1     # local extrema of v
    if len(turn) < 3:
        return np.empty((0, 2))
    tv, vv = t[turn], v[turn]
    return np.column_stack([0.5 * (tv[1:] + tv[:-1]), 0.5 * np.abs(np.diff(vv))])


def growth_rate(t, v, t_kick=None, win_pts=7, win=None):
    """The linear growth rate: the PEAK of the local d(ln a)/dt, not a fit over a long window.

    THE THIRD ESTIMATOR, and the previous two were both biased by the same thing from opposite
    sides. The local slope is not constant across a run:

        Re = 55   -0.018  -0.001  +0.009  +0.028  +0.033  +0.032  +0.030  +0.028  +0.025
        Re = 65   +0.004  +0.027  +0.044  +0.055  +0.048  +0.038  +0.028  +0.018  +0.012

    It RISES while the kick's stable components decay and the unstable eigenmode takes over,
    PLATEAUS -- that plateau is sigma -- and then FALLS as the amplitude becomes nonlinear. A
    least-squares fit over any long window averages all three regimes together, and how much of
    each it catches depends on Re, so the bias is not even consistent across a sweep: fitting
    the whole record gave 0.0300 and 0.0395 at Re = 55 and 65, whose ratio implies Re_c = 23,
    while the plateaus give 0.033 and 0.055, whose ratio implies Re_c = 40 against a published
    45-47.

    So: slide a `win_pts`-point least-squares fit along the envelope and take the largest slope.
    `plateau` reports how flat the neighbourhood of that maximum is -- a sharp peak means the
    linear regime was never resolved and the number should not be trusted.
    """
    if t_kick is not None:
        m = t >= t_kick
        t, v = t[m], v[m]
    env = envelope(t, v, win)
    if len(env) < win_pts + 2:
        return None
    te, a = env[:, 0], np.log(env[:, 1])
    slopes, centres = [], []
    for i in range(len(te) - win_pts + 1):
        sl = np.polyfit(te[i:i + win_pts], a[i:i + win_pts], 1)[0]
        slopes.append(sl)
        centres.append(te[i:i + win_pts].mean())
    slopes, centres = np.array(slopes), np.array(centres)
    k = int(slopes.argmax())
    near = slopes[max(k - 2, 0):k + 3]
    return {"sigma": float(slopes[k]), "t_peak": float(centres[k]),
            "n_windows": len(slopes),
            "plateau": float(near.min() / slopes[k]) if slopes[k] > 0 else float("nan"),
            "amp_at_peak": float(np.exp(np.interp(centres[k], te, a))),
            "e_folds": float(a.max() - a.min())}


def critical_reynolds(re, sigma, re_max=None):
    """Least-squares sigma = k (Re - Re_c); returns (Re_c, k, r2).

    `re_max` DROPS points above a cutoff, and it matters more than it looks. The relation is a
    NEAR-ONSET expansion: sigma is linear in Re only while Re - Re_c is small. Measured here,
    sigma = 0.0342, 0.0561, 0.0902 at Re = 55, 65, 100 -- the 55/65 pair alone gives Re_c = 39,
    and adding Re = 100 flattens the slope and drags it to 22 with an R^2 of 0.97 that looks
    perfectly healthy. A good fit to the wrong model is the failure mode to watch for; use the
    lowest Reynolds numbers available and check the residual of the ones left out.
    """
    re, sigma = np.asarray(re, float), np.asarray(sigma, float)
    if re_max is not None:
        m = re <= re_max
        re, sigma = re[m], sigma[m]
    k, b = np.polyfit(re, sigma, 1)
    pred = k * re + b
    ss = 1.0 - ((sigma - pred)**2).sum() / max(((sigma - sigma.mean())**2).sum(), 1e-300)
    return float(-b / k), float(k), float(ss)


def main(pattern="results/sqcyl_onset_Re*_history.npy", t_kick=None):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"  no histories match {pattern}")
        return
    print(f"  {'Re':>6}{'sigma':>10}{'e-folds':>9}{'windows':>9}{'fit rms':>10}"
          f"{'window t':>16}{'saturated':>11}")
    re_list, sig_list = [], []
    for p in paths:
        tag = os.path.basename(p).replace("_history.npy", "")
        try:
            Re = float(tag.split("_Re")[1].split("_")[0])
        except (IndexError, ValueError):
            print(f"  {tag}: cannot read Re from the name, skipped")
            continue
        h = np.load(p)
        g = growth_rate(h[:, 0], h[:, 1], t_kick)
        if g is None:
            print(f"  {Re:>6.0f}   too few windows to fit")
            continue
        print(f"  {Re:>6.0f}{g['sigma']:>10.5f}{g['e_folds']:>9.2f}{g['n_windows']:>9d}"
              f"{g['plateau']:>10.3f}{g['t_peak']:>16.0f}{g['amp_at_peak']:>11.4f}")
        re_list.append(Re); sig_list.append(g["sigma"])
    if len(re_list) >= 2:
        Re_c, k, r2 = critical_reynolds(re_list, sig_list)
        print(f"\n  sigma = {k:.6f} (Re - {Re_c:.2f}),  R^2 = {r2:.5f}")
        print(f"  Re_c = {Re_c:.2f}   against the published {RE_C_REF[0]:.0f}-{RE_C_REF[1]:.0f} "
              f"at ~5% blockage")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sqcyl_onset_Re*_history.npy",
         float(sys.argv[2]) if len(sys.argv) > 2 else None)
