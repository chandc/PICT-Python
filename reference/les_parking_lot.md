# LES checkerboard -- CLOSED

**Superseded by commit 822c468.** The root cause was found in a parallel session and is not what
this investigation proposed. Full record in `channel_checkerboard_remediation.md`; this file is
kept only for what it got right, what it got wrong, and one trap worth carrying forward.

## The actual cause

`_step_impl` restored `u`, `p` and `u_prev` before each Picard repeat but **not `p_flux` or
`F_prev`**. So `p_flux += phi_tot` ran once per SWEEP while `p += phi_tot` ran once per STEP, and
the projection pressure the Rhie-Chow term reads drifted without bound from the pressure the
predictor feels: |p - p_flux| reached 7.5 in ten steps against a legitimate 6.6e-3.

Fixed probes give a mode share at or below 0.001% on live turbulence, against 99.3% at the
production death. **Both pressure schemes are exonerated.**

## What this investigation got right

* **The Rhie-Chow boundary defect was real, and is fixed on its own merits.** Two implementations
  of the same operator disagreed by 5.7e+01 relative on a wall-bounded mesh while agreeing to
  2e-16 on a periodic one; every physical boundary had a cell layer with exactly zero damping.
  Verified six ways (`rhie_chow_boundary.md`). It explains the CYLINDERS' far-field oscillation.
* **Ruling the boundary defect OUT for the channel.** Three independent lines -- the zero-damping
  faces were wall-normal only while the mode is streamwise; the cell-Reynolds table explains all
  five boundary observations without the channel; T7 ran 4750 steps with both arms identical to
  three digits. Had this not been established, the boundary fix would have been miscredited with
  a cure it did not deliver.

## What it got wrong, and the trap in it

The successor hypothesis -- that `p = p + phi` is intrinsically an integrator, so the fault lies
in the fractional-step formulation itself, and the remedy is to adopt upstream PICT's and
OpenFOAM's segregated PISO (neither accumulates; PICT has no `p += ...` anywhere) -- was **wrong**.
Accumulation was implicated, but the pathology was a missing state restore, not the formulation.

**THE DISCRIMINATING EXPERIMENT WOULD HAVE FALSELY CONFIRMED IT.** The plan was to run an
`incremental` arm, on the reasoning that accumulation predicts it amplifies while a
rotational-term-specific cause predicts it stays clean. But `incremental` accumulates `p_flux`
through the same unrestored path, so it would have amplified -- and the conclusion drawn would
have been "the projection formulation is at fault", leading to an expensive rewrite of the
pressure coupling to match upstream.

The two hypotheses predicted the SAME outcome for the one arm chosen to separate them. A test is
only discriminating if the candidates disagree about its result, and that has to be checked
against each candidate's actual mechanism rather than its label. The 15,000x "scheme dependence"
that motivated the whole line was itself measuring chorin's immunity to a state bug -- chorin
replaces `p_flux` each sweep -- and not a property of the schemes at all.

## Still open, from this side

Both T7 arms slowed ~18x at t ~ 4.75 -- 2h43m at 100% CPU with no output, against 0.86 s/step
before. Not a hang: RSS identical to the byte throughout, and the main thread sampled into numpy
rather than `_sparsetools`, pointing at the pressure solve's iteration count exploding. Measured
on the PRE-FIX code, so it may simply be the checkerboard making the system ill-conditioned and
may already be gone. Worth one instrumented re-run now that the cause is fixed.
