# The Re_tau = 180 channel LES: result withdrawn, rig sound

> **Superseded, 2026-09-07.** The "MECHANISM FOUND" section below identified the pressure
> scheme by an A/B that was measuring a bug: the Picard loop double-accumulated `p_flux` under
> incremental/rotational (chorin was immune). With the fix, rotational tracks chorin --
> 5.1e-6 of the seed at t = 10 against 5.348x for the buggy run, on live turbulence.
> See `channel_checkerboard_remediation.md` sections 2 and 5. Items (1)-(4) below are retained
> as the record of how the wrong conclusion was reached.

**Bottom line: there is no validated wall-bounded LES in this repo.** The Re_tau=180 channel run
completed, produced plausible statistics, and those statistics are a numerical artefact. They
are withdrawn. What the exercise DID establish is the rig, and that stands.

## What was withdrawn, and why

A 30-time-unit run sampled from t=12 to t=28 and reported:

| | DNS | LES | diff |
|---|---|---|---|
| u_tau | 1.013 | 0.973 | -4.0% |
| U+ centreline | 18.411 | 19.475 | +5.8% |
| u'+ peak | 2.845 | 3.098 | +8.9% |
| v'+ max | 0.867 | 0.753 | -13.2% |
| w'+ max | 1.043 | 0.892 | -14.5% |

Read as physics, that is a coherent story: streamwise fluctuation too strong with its peak
pushed outward, cross-stream components too weak -- the classic signature of an under-resolved
near-wall cycle. It was reported as such.

It is not that. A wall-parallel contour plot at y+ = 12.8 shows regular cells where streaks
should be, and the spanwise spectrum gives the number:

    y+   4.5:  peak lambda_x+ = 47.1   (99.4% of fluctuation energy)
    y+  12.8:  peak lambda_x+ = 47.1   (99.3%)
    y+  31.1:  peak lambda_x+ = 47.1   (98.3%)
    y+  90.1:  peak lambda_x+ = 47.1   (87.3%)

lambda_x+ = 47.1 is exactly 2*dx+ (dx+ = 23.6). **Over 99% of the near-wall fluctuation energy
is a grid-scale checkerboard.** u'+ is high because it IS the mode; v'+ and w'+ are low because
the real turbulence has largely gone. The mean profile survives better -- a sign-alternating
mode largely cancels in a plane average -- but it was produced by a field that is not turbulent.

The field is clean at t=4 (0.09%) and in the interpolated DNS initial condition (0.0%), so it
developed during the run.

## What IS established, and survives

  * **The rig is correct against an exact solution.** Started from the analytic Poiseuille
    parabola the solver leaves it alone to 0.009% of centreline over 50 steps, and wall stress on
    the stretched mesh returns u_tau = 0.9986 against an exact 1. That pins the CPG forcing, the
    no-slip wall and the wall-normal viscous operator on a non-uniform mesh simultaneously --
    none of which the Taylor-Green study exercised. `test_channel_laminar.py`, 4/4.
  * **The DNS interpolation is sound.** The t=0 field tracks the DNS mean profile closely to
    y+ ~ 30 with 0.0% grid-mode content.
  * **van Driest damping works.** Undamped Smagorinsky gives nu_t/nu = 0.99 across the viscous
    sublayer and 0.92 AT the wall -- it doubles the viscosity exactly where the flow is laminar,
    and makes u_tau meaningless as a diagnostic since sqrt(nu du/dy) then omits most of the
    stress. Damping cuts the sublayer mean to 0.0097, puts nu_t exactly to zero at the wall, and
    moves the peak from y+ = 5 to y+ = 77. Note it fixes the MAGNITUDE, not the exponent: van
    Driest on the mixing length gives nu_t ~ y^4, not the correct y^3.
  * **The infrastructure works** -- checkpoint/restart, the CFL guard (which fired correctly and
    named the right dt), and the new grid-mode monitor.

## What is NOT established

Whether this solver can sustain wall-bounded turbulence at dx+ = 23.6 / dz+ = 8.0 is now an open
question, not an answered one. Grid convergence was never attempted, so even a clean run would be
single-grid. And the Taylor-Green results, though validated against DNS on dissipation history,
have never been checked for grid-mode content -- that check should be run before repeating any
general claim that the code does LES.

## MECHANISM FOUND: the pressure scheme accumulates the mode

Same field, same mesh, same everything except the pressure scheme, run from t=4 to t=12:

| t | rotational (amp) | chorin (amp) |
|---|-----------------:|-------------:|
| 5.0  |  0.273x | 0.002x |
| 8.0  |  1.120x | 0.000x |
| 10.0 |  5.348x | 0.000x |
| 12.0 | **15.292x** | **0.001x** |

**A factor of 15,000 between the two schemes.** Rotational grows; chorin decays to nothing. The
growth shape also matches the production run -- flat-to-decaying until t ~ 6, then accelerating
(0.69, 1.12, 2.24, 5.35, 15.29), which extrapolates to dominance around t = 20-28, exactly where
the production run reached 99.3%.

This confirms the prediction in `diag_checkerboard.py` and `pressure_checkerboard.md`, written
before this investigation began:

> chorin sets p = phi each step, straight from the compact solve -> CLEAN.
> incremental/rotational ACCUMULATE p += phi -> the projection must absorb, every step, the
> discrepancy between the pressure the predictor actually felt and the p it is credited with.
> The mode is regenerated in phi and accumulates in p.

## Why the periodic box is clean and the channel is not

The scheme alone is not sufficient -- the Taylor-Green box runs `rotational` and stays at 0.03%.
What the channel has and the box does not is BOUNDARIES, and the Rhie-Chow damping is switched
OFF at them:

    faces with EXACTLY ZERO Rhie-Chow damping, Re=100 cylinder:  4,096 of 533,568 (0.8%)
      wall / body surface   2,048  (50%)
      interior                  0  ( 0%)
      outer boundary        2,048  (50%)

The wide half of compact-minus-wide needs a central `np.gradient` at every cell a face touches;
at a boundary the stencil is one-sided and the code sets the whole correction to zero.

So the mechanism is a COMBINATION, and it explains every observation:

  1. the rotational scheme regenerates a checkerboard in phi and accumulates it in p;
  2. Rhie-Chow damps it in the interior, so a periodic box stays clean indefinitely;
  3. at walls and outflows there is NO damping, so the mode grows there;
  4. once seeded at the boundary it spreads inward and eventually dominates.

That also predicts the cylinder's unexplained far-field failure -- an odd-even pressure
oscillation on the Dong arc, concentrated at the mixed Dirichlet-Neumann junctions -- which is
the same mechanism at the same kind of face.

**Status: (1) and (2) are demonstrated. (3) and (4) are inference from the measured
zero-damping faces, not yet a demonstration.** The decisive test is to give the boundary faces a
one-sided-but-valid wide stencil, or fall back to the compact term rather than zeroing the whole
correction, and re-run the rotational case.

## Hypotheses tested and eliminated

**The block seam: REFUTED.** One block and four blocks produced bit-identical output at every
reported step (2dx energy 3.1358e-02, amp 0.350x, total 9.925e+02 at step 4000 for both). The
decomposition is irrelevant to the mode -- which is also independent confirmation that the
multi-block machinery is exact, consistent with the DD port's bitwise results.

**The CPG forcing: REFUTED.** Forced and unforced runs tracked each other to three digits
(1.315x vs 1.316x).

**Rhie-Chow: not the cause, possibly a contributor.** Disabling it DAMPS the mode harder
(0.718x vs 0.908x over 400 steps), which is the opposite of its intended role. Suggestive, not
conclusive.

## What is known about the growth

    t= 5.00  amp 0.273x     t= 8.00  amp 1.120x
    t= 6.00  amp 0.688x     t= 9.00  amp 2.236x
    t= 7.00  amp 0.769x     t=10.00  amp 5.348x

The mode genuinely GROWS -- it is not residue left behind by decaying turbulence, and total
fluctuation energy stays healthy (~400-1500) throughout. Growth from t=8 to t=10 is 4.8x in
energy, about 0.39 per time unit in amplitude, which extrapolates to dominance by t ~ 20-28 and
matches the production run.

**Critically, it DECAYS for the first ~2 time units before turning around.** Any probe shorter
than that reads the sign backwards, which is how the first investigation concluded the mode was
damped in every configuration.

## Where to look next

The mechanism is unknown and the most plausible hypothesis is eliminated. Candidates not yet
tested: the wall-normal stretching (the mode is streamwise, but the stretching is the one thing
that distinguishes this case from the periodic boxes that never showed it), the BDF2 +
persistent-flux interaction, and the SGS model's response at the grid scale. `pressure_checkerboard.md`
records the repo's earlier encounter with odd-even decoupling and is the place to start.

`channel_gridmode_probe.py` is the tool: guarded against hangs, long enough to see growth, and
set up for dose-response comparison across variants.
