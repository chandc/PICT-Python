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


def growth_rate(t, v, t_kick=None, frac_hi=0.3, skip_t=2.0, win=WIN):
    """Fit sigma over the exponential range: after the kick transient, before saturation.

    THE RANGE SELECTION IS THE WHOLE PROBLEM. A fit over the full post-kick record returns
    almost nothing -- on sqcyl_v3 it gives 0.0019 against a true 0.07 -- because 250 of the 300
    time units are a saturated limit cycle at constant amplitude, and least squares happily
    averages a flat line into the slope. The cut is:

      * drop the first `skip_t` time units: the kick is not the eigenmode, and its projection
        onto the stable modes decays first;
      * stop where the envelope first reaches `frac_hi` of its maximum, which is where the
        nonlinearity starts bending the curve down.

    `frac_hi` = 0.3 is a compromise and it biases sigma LOW: the Landau correction to the growth
    rate goes as the square of the amplitude ratio, so including data up to 30% of saturation
    costs of order 10%. Tightening it on a run that saturates quickly leaves too few extrema to
    fit. The sweep runs are deliberately stopped BEFORE saturation, where this cut does nothing
    and the bias does not arise; it only matters when re-reading a long saturated record such as
    sqcyl_v3.

    If the run never saturated the second cut does nothing and the whole record after the
    transient is used, which is the intended behaviour for a case run only long enough to
    measure sigma.
    """
    if t_kick is not None:
        m = t >= t_kick
        t, v = t[m], v[m]
    env = envelope(t, v, win)
    if len(env) < 8:
        return None
    te, a = env[:, 0], env[:, 1]
    hit = np.where(a >= frac_hi * a.max())[0]
    i_sat = int(hit[0]) if len(hit) else len(a)
    sel = np.where((te >= te[0] + skip_t) & (np.arange(len(a)) < i_sat))[0]
    if len(sel) < 4:
        return None
    s, c = np.polyfit(te[sel], np.log(a[sel]), 1)
    resid = np.log(a[sel]) - (s * te[sel] + c)
    return {"sigma": float(s), "n_samples": int(len(sel)),
            "t_lo": float(te[sel].min()), "t_hi": float(te[sel].max()),
            "e_folds": float(np.log(a[sel].max() / a[sel].min())),
            "rms_resid": float(np.sqrt((resid**2).mean())),
            "saturated": bool(i_sat < len(a) - 2)}


def critical_reynolds(re, sigma):
    """Least-squares sigma = k (Re - Re_c); returns (Re_c, k, r2)."""
    re, sigma = np.asarray(re, float), np.asarray(sigma, float)
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
        window = f"{g['t_lo']:.0f}-{g['t_hi']:.0f}"
        print(f"  {Re:>6.0f}{g['sigma']:>10.5f}{g['e_folds']:>9.2f}{g['n_samples']:>9d}"
              f"{g['rms_resid']:>10.4f}{window:>16}{str(g['saturated']):>11}")
        re_list.append(Re); sig_list.append(g["sigma"])
    if len(re_list) >= 2:
        Re_c, k, r2 = critical_reynolds(re_list, sig_list)
        print(f"\n  sigma = {k:.6f} (Re - {Re_c:.2f}),  R^2 = {r2:.5f}")
        print(f"  Re_c = {Re_c:.2f}   against the published {RE_C_REF[0]:.0f}-{RE_C_REF[1]:.0f} "
              f"at ~5% blockage")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sqcyl_onset_Re*_history.npy",
         float(sys.argv[2]) if len(sys.argv) > 2 else None)
