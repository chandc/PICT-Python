# LES checkerboard -- PARKED

Parked deliberately, not abandoned. Everything needed to resume is here.

## Where it stands

The Rhie-Chow boundary defect was real, is fixed, and is verified (T1-T6, see
`rhie_chow_boundary.md`). It explains the CYLINDERS' far-field odd-even oscillation. It does
**not** explain the channel, and that is established from three independent directions:

1. the zero-damping faces were entirely WALL-NORMAL (576 per wall = nx*nz), while the channel's
   mode is STREAMWISE (99.3% of energy at lambda_x+ = 2 dx+), where x is periodic and damping was
   never missing;
2. the cell-Reynolds table explains all five boundary observations without the channel needing
   the gap at all -- a wall is refined until Re_cell is O(1) and viscosity covers for the missing
   damping, which is why both cylinders' body surfaces were clean and only their far fields
   (Re_cell 28.5 and 89-99) were not;
3. T7 ran 4750 steps with both arms **identical to three digits** in the streamwise mode.

## The live hypothesis, and the experiment that decides it

This repo is a FRACTIONAL-STEP PROJECTION method that solves for an increment and reconstructs
the pressure. Upstream PICT and OpenFOAM are segregated PISO solvers that solve for p DIRECTLY
and replace it -- PICT has zero occurrences of "rotational"/"incremental" and **no `p += ...`
anywhere in its C++, CUDA or Python**; `CopyPressureResultToBlocks` copies rather than
accumulates. Neither reference can exhibit this failure mode.

    chorin:       p = phi                      # rebuilt each step, no memory
    incremental:  p = p + phi                  # accumulates
    rotational:   p = p + phi - nu*div(u*)     # accumulates

`p = p + phi` is an INTEGRATOR: an undamped mode in phi accumulates step over step. That is a
better explanation of the 15,000x scheme ratio (rotational 15.292x against chorin 0.001x at
t = 12) than anything about the schemes' accuracy.

**Two candidates, one discriminating arm -- and it has never been run.** The old probe only ever
ran `base`, `noforce` and `oneblock`.

| | mechanism | prediction for `incremental` |
|---|---|---|
| A: accumulation | `p = p + phi` integrates any undamped mode | amplifies, like rotational |
| B: the rotational term | `-nu*div(u*)` injects a mode the flux never saw | clean, like chorin |

A -> the fix is STRUCTURAL: adopt the upstream PISO formulation, solve for p directly.
B -> the fix is LOCAL: make `-nu*div(u*)` consistent with the flux, as the RC term now is.

### How to run it when this comes off the shelf

1. **Minutes, do this first.** `self._diag = {"phi": phi_tot, "div_star": div_star}` is already
   stashed every step. Measure the Nyquist content of `div_star` against `phi`. If `div_star`
   carries no checkerboard, B has no mechanism and dies before an hour is spent.
2. **~1 hour.** Three arms -- chorin, incremental, rotational -- same seed, same dt, in parallel.
   Stop at t ~ 4.5: the signal is present by t ~ 3 and an 18x slowdown starts just after 4.75.
   **Equal `picard_iters` in every arm**, or the comparison confounds accumulation with the
   second-order correction -- the same shape of error as the `momentum_tol` rigging in Gate 4.

The probe is `results/logs/t7_base.log` / `t7_patch.log`'s driver, which is NOT in the repo (it
lived in a scratchpad); it is ~60 lines and reproducible from those logs' header format.

## Loose end worth its own look

Both T7 arms slowed ~18x at t ~ 4.75 -- 2h43m at 100% CPU with no output, against 0.86 s/step
before. Not a hang: RSS identical to the byte throughout, and the main thread sampled into numpy
rather than `_sparsetools`, which points at the pressure solve's ITERATION COUNT exploding. Nobody
has instrumented it, and the case reaches it in about an hour.
