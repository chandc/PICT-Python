"""Shedding validation THROUGH the HydroGym env: restart from the R11 final
state (fully developed vortex street, t = 380) via make_env(restart=...),
run uncontrolled control steps, and require the campaign physics back:

  St within 1% of R11's 0.1673, mean C_D within 1% of 1.321, and the C_L
  amplitude sustained (last-period rms within 10% of the first -- a restart
  that loses the Rhie-Chow state or BDF2 history damps visibly).

St is measured from C_L upward zero-crossings (linear interpolation),
C_D averaged over whole periods. Saves the trace to
results/hydrogym_shedding_validation.npz.

Run:  .venv/bin/python validate_hydrogym_shedding.py [--steps 1800]
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from hydrogym_pict import make_env

ST_REF, CD_REF = 0.1673, 1.321          # R11 (Y10 freestream, this mesh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1800, help="~3 shedding periods")
    p.add_argument("--restart", default="results/fields/cylrect_r11_final.npz")
    a = p.parse_args()

    env = make_env(mesh="production", restart=a.restart)
    obs, _ = env.reset()
    print(f"  restarted through FlowEnv: t={env.flow.t:.1f}, "
          f"initial (C_L, C_D) = ({obs[0]:+.4f}, {obs[1]:.4f})", flush=True)

    cl, cd = [], []
    t0 = time.time()
    for i in range(a.steps):
        obs, r, term, trunc, _ = env.step(np.array([0.0]))
        cl.append(float(obs[0]))
        cd.append(float(obs[1]))
        if (i + 1) % 200 == 0:
            print(f"  step {i+1:5d}/{a.steps}  C_L {cl[-1]:+.4f}  C_D {cd[-1]:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    cl, cd = np.array(cl), np.array(cd)
    dt = env.solver.dt
    t = dt * np.arange(len(cl))

    # St from upward zero-crossings of C_L
    s = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    tz = t[s] - cl[s] * dt / (cl[s + 1] - cl[s])
    ok = True
    if len(tz) >= 3:
        period = np.diff(tz).mean()
        St = 1.0 / period
        n_whole = int((t[-1] - tz[0]) // period)
        mask = (t >= tz[0]) & (t <= tz[0] + n_whole * period)
        cd_mean = cd[mask].mean()
        rms_first = np.sqrt((cl[(t >= tz[0]) & (t < tz[0] + period)] ** 2).mean())
        rms_last = np.sqrt((cl[(t >= tz[-1] - period) & (t < tz[-1])] ** 2).mean())
        dSt = abs(St - ST_REF) / ST_REF
        dCd = abs(cd_mean - CD_REF) / CD_REF
        dAmp = abs(rms_last - rms_first) / rms_first
        for cond, msg in (
            (dSt < 0.01, f"St = {St:.4f} vs R11 {ST_REF} ({dSt:.2%})"),
            (dCd < 0.01, f"mean C_D = {cd_mean:.4f} vs R11 {CD_REF} ({dCd:.2%})"),
            (dAmp < 0.10, f"C_L rms sustained: first {rms_first:.4f} "
                          f"last {rms_last:.4f} ({dAmp:.2%} drift)"),
        ):
            print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)
            ok = ok and cond
    else:
        print(f"  [FAIL] only {len(tz)} zero-crossings -- no shedding?", flush=True)
        ok = False

    np.savez("results/hydrogym_shedding_validation.npz", t=t, cl=cl, cd=cd)
    print(f"\n  {'VALIDATED' if ok else 'FAILED'} -- trace saved to "
          f"results/hydrogym_shedding_validation.npz", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
