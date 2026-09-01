# The linear-solve tolerance, and how much of the cost it was buying

The pressure Poisson solve is where this solver spends its time. Profiling ten steps of the
five-domain BFS:

| component | share of step time |
|---|---|
| SciPy sparse, total | **93.0%** |
| `scipy.sparse._sparsetools.csr_matvec` | **48.8%** |
| our own assembly (`multiblock.py`) | ~4% |

Half the runtime is one kernel — the sparse matrix-vector product inside the Krylov iterations.
So **iteration count is runtime**, and the tolerance is what sets iteration count.

## What the tolerance was set to, and what it cost

`MultiBlockPISO` used to default to `tol=1e-13`, and the BFS drivers passed `1e-11`. Nothing
justified those numbers; they were tight because tight felt safe.

Measured at Re = 300 on the five-domain grid, each case **restarted from the same converged
field** and run 400 steps, so the only variable is the tolerance:

| rtol | Krylov it/step | s/step | divF (interior) | x_r/S | max\|u − u_ref\| | speedup |
|---|---|---|---|---|---|---|
| 1e-12 | 1,757 | 0.505 | 6.1e-15 | **6.714** | — | 1.00x |
| 1e-10 | 1,662 | 0.482 | 6.2e-15 | **6.714** | 1.0e-08 | 1.05x |
| 1e-08 | 1,568 | 0.457 | 1.8e-12 | **6.714** | 2.5e-06 | 1.10x |
| 1e-06 | 861 | 0.269 | 2.0e-10 | **6.714** | 1.0e-04 | 1.88x |
| **1e-04** | **48** | **0.059** | 5.0e-08 | 6.720 | 9.0e-03 | **8.58x** |

**Reattachment is identical to four significant figures from 1e-12 down to 1e-6**, and moves
0.09% at 1e-4. The current default is **1e-4**.

## THE DEFAULT IS UNSAFE FOR UNSTEADY FLOWS

Everything above was measured on a STEADY flow, where a damped iteration still converges to the
same fixed point. On a marginally unstable one it does not converge to the same answer -- it
converges to the wrong kind of answer.

Square cylinder, Re = 100, where a vortex street is the physical solution:

| tol | outcome |
|---|---|
| 1e-4 | converges **bitwise steady**, no shedding at all |
| 1e-6 | **sheds**, St = 0.1467, saturated amplitude 0.62 |

Nothing else differed -- same grid, same Reynolds number, same boundary conditions. The
mechanism is the one described two sections down in this very file: at 1e-4 the pressure
correction is under-resolved, so **each velocity update is damped**. Damping every update is
precisely what stops a growing mode from growing, and the shedding instability never gets off
the ground.

The failure is silent and total. It does not degrade the Strouhal number; it removes the
physics, and the result looks like a perfectly converged steady solution. Worse, it survives
every check you would think to run -- the flow stayed steady at Re 200 AND Re 300, where a
square cylinder must shed, and at every grid resolution tried, so both the obvious explanations
(too coarse, effectively too viscous) were ruled out before the tolerance was suspected.

**So: 1e-4 for steady problems, 1e-6 or tighter for anything that oscillates.** If you do not
know which you have, use 1e-6 -- the cost is real but bounded, and a steady run merely gets
slower, whereas an unsteady run at 1e-4 gets a plausible wrong answer.

Tight tolerances are affordable with AMG, whose iteration count is O(1); Jacobi's goes from 48 to
~1568 between 1e-4 and 1e-8, which is why this trade looked worse than it is.

## Why this is not recklessness

The discretisation error sets the floor. Spatial accuracy is O(h^2) ~ 1e-3 on these grids and
the temporal error at dt = 0.02 is comparable, so a linear residual of 1e-6 sits two orders
BELOW the error already in the answer. Solving to 1e-11 was computing eleven digits of the
solution to a discrete system that approximates the continuous one to about three.

The iteration collapse makes the point sharply: 1,757 -> 861 -> 48 across the last two decades.
That steepness is the signature of a badly conditioned operator, where most iterations buy very
little residual. It is also why the tolerance matters more here than it would on a well
conditioned problem.

## What to watch, and it is not accuracy

The risk is not that answers get slightly worse. It is that **verification stops verifying**.

Several of this project's most valuable gates assert at machine precision:

* `J*div(Phi(p)) == -M*p` to 2-4e-16, which ruled out an operator mismatch and redirected a
  whole investigation
* seam fluxes agreeing bitwise (0.00e+00), which localised the reentrant-corner bug to a single
  cell
* multi-block matching single-block to ~1e-15
* bit-exact checkpoint restarts (0.00e+00)

At `tol=1e-4` two runs agree to ~1e-3, not ~1e-15. A machine-precision assertion under a loose
tolerance still PASSES — it is just measuring the solver's noise floor instead of the property
it was written to gate. Passing for the wrong reason is worse than failing.

**So the split is deliberate: production runs at 1e-4, verification suites at 1e-12 or tighter.**
The suites already pass `tol` explicitly (1e-12, 1e-13, 1e-14), so they do not inherit the loose
default. Any new test that compares fields at machine precision must do the same.

Regressions after the change: multiblock 55/55, rhie_chow 28/28, checkpoint 23/23, energy 6/6.

## This invalidates the GPU speedup figures

Every AmgX number measured before this change compared SciPy grinding to 1e-11 against AmgX at
1e-10 — 56x at Re = 389, 12.6x at Re = 100, 25x at the matrix level. Two separate problems:

1. **The tolerances were never matched.** SciPy was doing ten times more work by construction.
2. **The binding ignores the requested tolerance.** `SolveCache.solve()` takes an `rtol` and
   honours it on the SciPy path; `_amgx_solve()` discards it and uses whatever the AmgX JSON
   config says. A caller tightening `tol` sees no effect on the GPU path. That is a defect, not
   just a benchmarking slip.

Against SciPy at 1e-4 — 48 iterations per step rather than 1,757 — the GPU advantage will be
much smaller, possibly to the point where AmgX does not pay at these problem sizes. **The
numbers in `src/amgx/README.md` and the CuPy plan are stale in a direction that flatters the
GPU** and must be re-measured at matched tolerance before anyone acts on them.

The wider lesson: a large part of what looked like a hardware problem was a tolerance nobody had
questioned. 1.88x at 1e-6, or 8.58x at 1e-4, needs no GPU, no port, and no new dependency.

## Precision

Everything is float64 — velocity, pressure, coordinates, Jacobian, metrics, matrix values
(indices int32); AmgX runs in matching `dDDI` mode. Float32 would roughly halve the bytes per
nonzero and so buy ~1.5x on the dominant bandwidth-bound kernel, but it was **deliberately
rejected**: float32 carries ~7 decimal digits, so the machine-precision gates listed above could
no longer resolve what they check. That is a poor trade against a change that costs nothing.
