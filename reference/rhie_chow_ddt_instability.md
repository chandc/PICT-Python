# The `ddt_corr` transient term diverges, and Rhie–Chow is not to blame

**Status: worked around, not fixed.** `run_square_cylinder.py` runs with `ddt_corr=False`. The
term itself still has an unbounded recurrence on this case.

## The failure

Square cylinder, Re = 100, 63,280 cells, dt = 0.01, BDF2, Dong outflow. The probe trace is
smooth right up to the edge and then gone in five steps:

| step | t | v at (2D, 0.5D) |
|---|---|---|
| 445 | 4.45 | −1.669e-01 |
| 450 | 4.50 | +1.955e+00 |
| 451 | 4.51 | +5.112e+01 |
| 453 | 4.53 | +2.376e+05 |
| 455 | 4.55 | **nan** |

Growth of ~10² per step. Nothing in the preceding 445 steps hints at it.

## What it is not

The first visible symptom is `RuntimeWarning: divide by zero` at `coef_g = Jg / rowsum`, which
is misleading — it is *downstream* of the failure. `min|rowsum|` sits at 1.362e+04, exactly
1.5·J_min/dt, so a zero row sum requires the velocity to blow up first. There are **0 zero rows
of 63,280** at t = 0.

Also ruled out, each by its own run: the symmetry-breaking inlet pulse (stable 150 steps with it
on), the pulse switching **off** (that is step 400; the blow-up is at 450), and the grid (the
GCL and seam invariants both hold at ~1e-15 — see `reentrant_corner_gcl.md`).

## What it is

Isolated on a coarse grid (5,360 cells, same case, reproduces in ~2 minutes at step 311), with
everything else held fixed:

| rhie_chow | persistent_flux | ddt_corr | outcome |
|---|---|---|---|
| True | True | True | DIVERGED, step 311 |
| True | True | **False** | **stable, 2000 steps** |
| True | False | **False** | **stable, 2000 steps** |
| True | False | True | DIVERGED, step 324 |

**`ddt_corr` alone decides it.** `persistent_flux` makes no difference in either direction, and
Rhie–Chow by itself is stable. Instrumenting the correction magnitude shows the mechanism
directly — the correction outgrows the flux it is correcting:

    step 200   |RC|/|F| = 0.184
    step 311   |RC|/|F| = 1.818     -> diverged

## Why the existing limiter is not enough

`piso_multiblock.py` already carries OpenFOAM's `fvcDdtPhiCoeff`, and its comment explains
exactly why it must: with SIMPLEC (Γ = J/rowsum(A)) and conservative central convection and
diffusion — both zero row sum — `rowsum(A) = J/dt` exactly, so **Γ/dt is 1.0 to machine
precision** and the `F_prev` recurrence has **unit gain**. The limiter

    lim = 1 - min(|dF| / (|F_prev| + SMALL), 1)

goes to zero only where the face flux and the interpolated cell flux disagree *most*. Where they
agree, `lim → 1` and the gain stays at exactly 1. A marginally stable recurrence accumulates any
inconsistency without bound; on the backward-facing step that showed up at step 439 with the raw
form, and here it survives the limiter and reappears at step 311/455.

This is the second independent case — Re = 389 BFS diverged at step 1714 — so it is a property of
the term, not of one mesh.

## Why turning it off is acceptable here, and what it costs

`ddt_corr` exists because Γ ~ Δt, so plain Rhie–Chow damping vanishes as Δt → 0. The question is
whether the damping that remains is enough. Measured after 1500 steps on the coarse grid:

| config | worst flip fraction | amplitude | max abs(p) |
|---|---|---|---|
| RC off | 0.975 | 1.438e-01 | 3.15 |
| RC on, **persist**, ddt_corr off | **0.469** | **3.484e-03** | 2.16 |
| RC on, no persist, ddt_corr off | 0.510 | 5.803e-03 | 2.04 |

Rhie–Chow still does its job without the transient term — **41× smaller checkerboard
amplitude**, and the flip fraction drops out of the "genuine node-to-node mode" range. So at
dt = 0.01 the trade is clearly worth taking.

`persistent_flux` is worth keeping on. It has no bearing on STABILITY -- the ablation above
shows it changes nothing in either direction -- but it does damp better, 3.48e-03 against
5.80e-03, a 1.7x edge for free. So the shipped configuration is RC on, persist on,
ddt_corr off. At much smaller dt the argument for `ddt_corr`
returns and this workaround would need re-examining.

## What a real fix would need

The gain must be strictly less than 1. Options, none measured yet:

1. **Cap the coefficient** — multiply by a safety factor `< 1`, the crude but reliable route.
2. **Normalise the limiter globally** rather than pointwise, closer to what OpenFOAM does when
   it forms `ddtPhiCoeff` from mesh-wide flux magnitudes instead of per-face ones.
3. **Break the unit gain at its source** — Γ/dt = 1 exactly *because* the convection and
   diffusion row sums both vanish. A formulation that does not put `rowsum(A) = J/dt` on the
   nose would not sit on the stability boundary in the first place.

Reproducer: `scratchpad/rc_matrix.py` pattern, coarse grid, ~2 min per configuration.

The square-cylinder run this was found on went on to shed correctly with `ddt_corr=False` --
St = 0.1467 at Re = 100, see `square_cylinder_vortex_street.md`. So dropping the term costs
nothing measurable at dt = 0.01. At much smaller dt the argument for it returns, because
Gamma ~ dt and plain Rhie-Chow damping weakens; this workaround would need re-examining there.
