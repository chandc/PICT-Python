"""Critical Reynolds number for the square cylinder, from growth rates rather than from runs
that shed.

WHY THIS IS THE CHEAP EXPERIMENT. Re_c is a sharp published number -- 51.2 +/- 1.0 at 5%
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

FLATNESS_MAX = 0.1
# THE PREVIOUS VALUE HERE WAS THE WRONG COMPARISON. (45, 47) is the ZERO-blockage experimental
# estimate (Norberg, quoted as 47 +/- 2 in Sohankar, Norberg & Davidson, IJNMF 26:39-56, 1998).
# Our grid is 5% blockage, and at 5% the same paper computes Re_cr = 51.2 +/- 1.0, stating that
# the critical Reynolds number increases with blockage. Comparing our 5%-blockage result against
# a zero-blockage reference made a correct answer look 10% wrong -- section 1 of
# measurement_traps.md, "a bar taken from the wrong regime", committed again in this file.
RE_C_REF = (50.2, 52.2)          # Sohankar et al. 1998, Re_cr = 51.2 +/- 1.0 at 5% blockage
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


def growth_rate(t, v, t_kick=None, win_pts=7, win=None, span=5):
    """The linear growth rate: the PLATEAU of the local d(ln a)/dt, found by flatness.

    THE FOURTH ESTIMATOR. The third one is documented below because its reasoning was right and
    its implementation had the failure built in. The local slope is not constant across a run:

        Re = 55   -0.018  -0.001  +0.009  +0.028  +0.033  +0.032  +0.030  +0.028  +0.025
        Re = 65   +0.004  +0.027  +0.044  +0.055  +0.048  +0.038  +0.028  +0.018  +0.012

    It RISES while the kick's stable components decay and the unstable eigenmode takes over,
    PLATEAUS -- that plateau is sigma -- and then FALLS as the amplitude becomes nonlinear. A
    least-squares fit over any long window averages all three regimes together.

    THE THIRD ESTIMATOR PICKED THE PLATEAU BY POSITION AND THAT IS WHAT BROKE IT. For a growing
    case it took the maximum slope, which is fine. For a decaying case it argued that the
    least-stable eigenvalue is approached from below and so took the LAST window -- which is the
    end of the record, where the signal is weakest and noisiest. On the near-onset sweep that
    returned sigma = +0.79 at Re = 44, a violently GROWING mode read off a case that had decayed
    by four orders of magnitude, and Re_c = -132 from the sweep as a whole. The noise floor was
    set relative to the FIRST envelope sample, so on a signal that falls 4 decades it sat 4
    decades above the noise and excluded nothing.

    Both faults have the same cause: choosing the fitting window by WHERE it is in the record
    rather than by whether the slope is actually stationary there. So:

      * the floor is relative to the LARGEST envelope sample, not the first, which is what makes
        it track the decay;
      * the plateau is the run of `span` consecutive slopes with the smallest spread, chosen the
        same way whether the mode grows or decays. `flatness` is that spread relative to the
        slope itself, and a large value means no linear regime was resolved.

    Checked against a fit over a fixed amplitude band -- above round-off, below saturation --
    which is an independent way to isolate the same regime: this returns -0.0069 and +0.031 at
    Re = 52 and 55 against the band fit's -0.0069 and +0.031.
    """
    if t_kick is not None:
        m = t >= t_kick
        t, v = t[m], v[m]
    env = envelope(t, v, win)
    if len(env) < win_pts + span + 2:
        return None
    # A NOISE FLOOR, because a strongly stable case decays into round-off. Re = 44 fell from
    # 2.1e-03 to 6e-12 with 308 exact zeros in its envelope: past that point the "extrema" are
    # floating-point noise, not oscillations. RELATIVE TO THE MAXIMUM, so that it still bites
    # after the signal has fallen several decades -- relative to the first sample it does not.
    floor = max(1e-12, 1e-5 * float(env[:, 1].max()))
    keep = env[:, 1] > floor
    if keep.sum() < win_pts + span + 1:
        return None
    te, a = env[keep, 0], np.log(env[keep, 1])
    slopes, centres = [], []
    for i in range(len(te) - win_pts + 1):
        slopes.append(np.polyfit(te[i:i + win_pts], a[i:i + win_pts], 1)[0])
        centres.append(te[i:i + win_pts].mean())
    slopes, centres = np.array(slopes), np.array(centres)
    if len(slopes) < span:
        return None
    # THE PLATEAU IS THE FLATTEST RUN, not the largest and not the last. Same rule for growth
    # and decay, so a near-onset sweep that produces both is measured one way throughout.
    spread = np.array([slopes[i:i + span].std() for i in range(len(slopes) - span + 1)])
    k = int(spread.argmin())
    seg = slopes[k:k + span]
    sig = float(seg.mean())
    return {"sigma": sig, "t_peak": float(centres[k:k + span].mean()),
            "growing": bool(sig > 0), "n_windows": len(slopes),
            "flatness": float(seg.std() / abs(sig)) if sig != 0 else float("nan"),
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
    print(f"  {'Re':>6}{'sigma':>10}{'e-folds':>9}{'windows':>9}{'flatness':>10}"
          f"{'window t':>16}{'saturated':>11}")
    re_list, sig_list, flat = [], [], []
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
              f"{g['flatness']:>10.3f}{g['t_peak']:>16.0f}{g['amp_at_peak']:>11.4f}")
        re_list.append(Re); sig_list.append(g["sigma"])
        flat.append(g["flatness"])
    if len(re_list) >= 2:
        re_a = np.array(re_list, float); sg = np.array(sig_list); fl = np.array(flat)
        # A CASE THAT SATURATES BEFORE IT RESOLVES A LINEAR PHASE HAS NO GROWTH RATE TO REPORT.
        # At Re = 80 and 95 the flattest run of slopes lies in the saturated tail, which is flat
        # for the wrong reason, and `flatness` shows it -- 0.39 and 0.41 against 0.00-0.04 for
        # the cases that do resolve one. Those points are excluded rather than fitted.
        good = fl <= FLATNESS_MAX
        if (~good).any():
            print(f"\n  excluded from the fit, no linear phase resolved (flatness > "
                  f"{FLATNESS_MAX}): Re {', '.join(f'{r:.0f}' for r in re_a[~good])}")
        # THE BRACKET FIRST, because it needs no model: the lowest Re with sigma > 0 and the
        # highest with sigma < 0 straddle Re_c whatever the shape of sigma(Re).
        neg, pos = re_a[good & (sg < 0)], re_a[good & (sg > 0)]
        if len(neg) and len(pos):
            print(f"  sign bracket, no model assumed:  {neg.max():.0f} < Re_c < {pos.min():.0f}")
        re_a, sg = re_a[good], sg[good]
        if len(re_a) >= 2:
            o = np.argsort(np.abs(sg))[:4]           # the four points nearest onset
            Re_c, k, r2 = critical_reynolds(re_a[o], sg[o])
            print(f"  near-onset fit over Re {sorted(re_a[o].astype(int).tolist())}:  "
                  f"sigma = {k:.6f} (Re - {Re_c:.2f}),  R^2 = {r2:.5f}")
            Re_all, k_all, r2_all = critical_reynolds(re_a, sg)
            print(f"  all {len(re_a)} points:  Re_c = {Re_all:.2f}, R^2 = {r2_all:.5f}  "
                  f"-- shown to be discarded: sigma(Re) is curved, so a fit reaching away from\n"
                  f"     onset crosses zero too early however good its R^2 looks")
            print(f"  published Re_cr = 51.2 +/- 1.0 at 5% blockage (Sohankar et al. 1998); "
                  f"the zero-blockage\n  experimental estimate is 47 +/- 2, which is NOT the "
                  f"right comparison for this grid")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sqcyl_onset_Re*_history.npy",
         float(sys.argv[2]) if len(sys.argv) > 2 else None)
