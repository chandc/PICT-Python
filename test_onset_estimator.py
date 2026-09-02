"""The growth-rate estimator against signals whose answer is known.

This estimator has now been wrong twice on real data and both times it was found by a
downstream number looking absurd -- Re_c = 23, then Re_c = -132 -- rather than by anything in
the estimator itself. A synthetic signal removes that: sigma is an input, so the check cannot
be argued with.

The cases are chosen to cover what the sweep actually produces, INCLUDING the two that broke it:

  strong decay      Re = 44 fell four decades into round-off and the old estimator, whose noise
                    floor was set relative to the FIRST envelope sample, returned sigma = +0.79
  marginal decay    Re = 52, the point that sets the lower end of the bracket
  saturating growth the plateau must be found before nonlinearity bends the curve down
"""
import numpy as np

from sqcyl_onset import growth_rate

ST, DT, T_END = 0.117, 0.01, 200.0
TOL = 2e-3


def signal(sigma, a0=0.01, noise=1e-13, seed=0, sat=None):
    t = np.arange(0.0, T_END, DT)
    a = a0 * np.exp(sigma * t)
    if sat is not None:
        a = sat * a / np.sqrt(1.0 + (a / sat) ** 2)      # smooth saturation
    v = a * np.sin(2 * np.pi * ST * t)
    v += noise * np.random.default_rng(seed).standard_normal(len(t))
    return t, v


def check(name, sigma, **kw):
    t, v = signal(sigma, **kw)
    g = growth_rate(t, v)
    if g is None:
        print(f"  [FAIL] {name}: estimator returned None")
        return False
    err = abs(g["sigma"] - sigma)
    ok = err <= TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} true {sigma:+.4f}  "
          f"got {g['sigma']:+.5f}  err {err:.2e}  flatness {g['flatness']:.3f}")
    return ok


def main():
    print("=" * 78)
    print("  growth rate recovered from signals with a known exponent")
    print("=" * 78)
    r = []
    for s in (-0.05, -0.02, -0.007, 0.01, 0.033, 0.06):
        r.append(check(f"pure exponential sigma={s:+.3f}", s))
    # The Re = 44 shape: four decades of decay ending in round-off. The old floor was 1e-7 of
    # the FIRST sample, so on a signal that falls this far it excluded nothing and the estimator
    # fitted the noise tail.
    r.append(check("deep decay into round-off", -0.05, a0=0.3, noise=1e-9))
    # Growth that saturates: the plateau must be found before the curve bends over.
    r.append(check("growth into saturation", 0.033, a0=1e-4, sat=0.05))
    # A pure limit cycle has no growth rate; flatness must not claim one.
    t, v = signal(0.0, a0=0.05)
    g = growth_rate(t, v)
    ok = abs(g["sigma"]) < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] {'steady limit cycle':<34} true +0.0000  "
          f"got {g['sigma']:+.5f}")
    r.append(ok)
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
