"""
Steady-state detection, tied to what the linear solver can actually resolve.

THE OLD TEST WAS INOPERATIVE. The drivers broke when `max|u^{n+1} - u^n| < 1e-9`, on the
streamwise component alone. Measured on the five-domain BFS restarted from a converged field,
the step-to-step residual floors at ~3e-05 -- four orders ABOVE that threshold. It could never
fire, which is why every sweep reported exactly its 3000-step cap and why those reattachment
numbers are "after 3000 steps", not "at steady state".

WHY THE THRESHOLD MUST FOLLOW THE SOLVER TOLERANCE. Each step's velocity update comes from a
pressure solve converged to relative residual `tol`, so the update itself carries relative error
of that order. A step-to-step change smaller than `tol` is indistinguishable from solver noise:
asking whether the flow moved less than that is asking a question the solve cannot answer.
Hence `rel_tol = max(solver_tol, floor)`.

AND WHY A SMALL RESIDUAL IS NOT ALWAYS GOOD NEWS. Measured floors on the same case:

    solver tol   max|du|     rel du
      1e-11      2.97e-05    1.98e-05
      1e-06      2.98e-05    1.98e-05
      1e-04      2.04e-08    1.36e-08

The 1e-11 and 1e-06 rows agree, so ~2e-05 is the PHYSICAL residual -- the flow genuinely still
evolving. At 1e-04 the residual is three orders SMALLER, which looks like better convergence and
is the opposite: with only ~48 Krylov iterations the pressure correction is under-resolved, so
each velocity update is damped and the iteration moves less because it is solving less. Step-to-
step change therefore measures ITERATION MOVEMENT, not distance from steady state, and the two
decouple as the tolerance loosens.

So this reports honestly: convergence here means "no longer changing by an amount this solver
can resolve", NOT "converged in an absolute sense". At a loose tolerance those are very
different claims, and `SteadyState.reason` records which tolerance the verdict rests on.
"""
import numpy as np

# Below this the test is meaningless regardless of `tol`: float64 round-off accumulated over a
# multi-block step lands around here, so a tighter demand can never be satisfied.
ABSOLUTE_FLOOR = 1e-12


class SteadyState:
    """Relative, multi-component steady-state test whose threshold follows the solver tolerance.

    Checks u, v AND w -- the old test looked only at u, so a flow evolving in the wall-normal
    direction with a static streamwise component would have read as converged.

    Requires the test to hold for `hold` CONSECUTIVE steps. A single quiet step is not
    convergence; near the floor the residual fluctuates, and one lucky step below threshold
    would stop a run that is still moving.
    """

    def __init__(self, solver_tol, safety=1.0, hold=5):
        self.rel_tol = max(float(solver_tol) * safety, ABSOLUTE_FLOOR)
        self.hold = hold
        self.streak = 0
        self.last = None
        self.residual = np.inf
        self.reason = ""

    def update(self, fields):
        """`fields` is a sequence of the current (u, v, w) per-block dicts or arrays.

        Returns True when the relative step-to-step change has stayed below the threshold for
        `hold` consecutive steps.
        """
        cur = [np.concatenate([np.asarray(v).ravel() for v in f.values()])
               if isinstance(f, dict) else np.asarray(f).ravel() for f in fields]
        if self.last is None:
            self.last = cur
            return False
        # ONE common velocity scale for all components, not each normalised by its own max.
        # The spanwise component is ~zero in a 2D flow, so |dw| / |w|max explodes even when dw
        # is negligible -- measured 1.29e+00 "relative change" on a near-steady field, which is
        # meaningless. The physically right denominator is the velocity magnitude.
        scale = max(max(np.abs(c).max() for c in cur), 1e-30)
        rel = max(np.abs(c - p).max() for c, p in zip(cur, self.last)) / scale
        self.last = cur
        self.residual = rel
        if rel < self.rel_tol:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.hold:
            self.reason = (f"relative change {rel:.2e} < {self.rel_tol:.2e} "
                           f"(solver tol) for {self.hold} steps")
            return True
        return False

    def report(self, nsteps, converged):
        if converged:
            return f"converged: {self.reason}"
        return (f"NOT converged after {nsteps} steps: relative change {self.residual:.2e} "
                f"still above {self.rel_tol:.2e}")
