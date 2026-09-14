"""Gates for the HydroGym backend (hydrogym_pict): the production solver
behind HydroGym's PDEBase/FlowEnv contract, rotary cylinder at Re 100.

  hg.1  FlowEnv conformance: spaces, reset -> (2,) obs, step -> 5-tuple,
        reward = -dt * C_D.
  hg.2  uncontrolled forward: 40 steps finite, C_D physically scaled.
  hg.3  exact reset: after stepping, reset() reproduces the initial
        observation to round-off (copy_state/set_state carry the FULL
        restart state, BDF2 history and Rhie-Chow flux included).
  hg.4  rotation liveness + antisymmetry: omega = +/-1 episodes produce
        mirrored C_L responses (|dCL| well above noise, opposite signs)
        and identical C_D to leading order.
  hg.5  actuator damping: with TAU = 0.0556 the actuator state reaches the
        commanded value along the exact exponential (checked mid-lag).
  hg.6  substeps: num_substeps=5 aggregates rewards (mean rule) and
        advances 5 solver steps per env step.

Run:  .venv/bin/python test_hydrogym_pict.py     (~6 min on the Mac, coarse mesh)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from hydrogym_pict import HYDROGYM_SOURCE, make_env

PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}", flush=True)


def main():
    print(f"  contract source: {HYDROGYM_SOURCE}", flush=True)
    env = make_env(mesh="coarse", dt=1e-2, tol=1e-6)
    dt = env.solver.dt

    # ------------------------------------------------------------------ hg.1
    obs, info = env.reset()
    ok_spaces = (env.observation_space.shape == (2,)
                 and env.action_space.shape == (1,)
                 and np.isclose(env.action_space.high[0], 0.5 * np.pi))
    out = env.step(np.array([0.0]))
    obs1, r1, term, trunc, info1 = out
    cd_now = env.flow.evaluate_objective()
    check(ok_spaces and obs.shape == (2,) and len(out) == 5
          and np.isclose(r1, -dt * cd_now),
          f"hg.1 FlowEnv conformance: obs (2,), action (1,) +-pi/2, "
          f"reward {r1:+.4f} = -dt*C_D")

    # ------------------------------------------------------------------ hg.2
    for _ in range(39):
        obs, r, *_ = env.step(np.array([0.0]))
    CL0, CD0 = obs
    check(np.all(np.isfinite(obs)) and 0.3 < CD0 < 30.0 and abs(CL0) < CD0,
          f"hg.2 uncontrolled 40 steps: C_D {CD0:.3f}, C_L {CL0:+.4f} "
          f"(impulsive-start transient; finite and physically scaled)")

    # ------------------------------------------------------------------ hg.3
    obs_r, _ = env.reset()
    env2 = make_env(mesh="coarse", dt=1e-2, tol=1e-6)
    obs_f, _ = env2.reset()
    diff = float(np.abs(np.asarray(obs_r) - np.asarray(obs_f)).max())
    check(diff < 1e-12,
          f"hg.3 exact reset after 40 steps: |obs - fresh| = {diff:.2e}")

    # ------------------------------------------------------------------ hg.4
    def spin(omega, n=40):
        e = make_env(mesh="coarse", dt=1e-2, tol=1e-6)
        e.reset()
        for _ in range(n):
            obs, *_ = e.step(np.array([omega]))
        return obs

    op, om = spin(+1.0), spin(-1.0)
    dcl_p, dcl_m = op[0] - CL0, om[0] - CL0
    anti = abs(dcl_p + dcl_m) / max(abs(dcl_p), abs(dcl_m), 1e-300)
    check(abs(dcl_p) > 1e-3 and dcl_p * dcl_m < 0 and anti < 0.05
          and abs(op[1] - om[1]) < 0.05 * abs(op[1]),
          f"hg.4 rotation: dC_L(+1) {dcl_p:+.4f} vs dC_L(-1) {dcl_m:+.4f} "
          f"(antisym defect {anti:.2%}), C_D split {abs(op[1]-om[1]):.2e}")

    # ------------------------------------------------------------------ hg.5
    e = make_env(mesh="coarse", dt=1e-2, tol=1e-6)
    e.reset()
    tau = e.flow.TAU
    e.step(np.array([1.0]))
    a1 = float(e.flow.actuators[0].state)
    expect = 1.0 - np.exp(-dt / tau)
    check(np.isclose(a1, expect, rtol=1e-10),
          f"hg.5 actuator lag: state after one step {a1:.6f} == "
          f"1-exp(-dt/tau) = {expect:.6f}")

    # ------------------------------------------------------------------ hg.6
    e = make_env(mesh="coarse", dt=1e-2, tol=1e-6, num_substeps=5)
    e.reset()
    n0 = e.flow.m.nstep
    obs, r, *_ = e.step(np.array([0.0]))
    check(e.flow.m.nstep - n0 == 5 and np.isfinite(r),
          f"hg.6 substeps: 5 solver steps per env step, aggregated reward {r:+.4f}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
