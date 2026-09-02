# Measurement traps — how a correct solver produces a wrong number

Collected on 2026-09-01, when eleven things went wrong in one day and **nine of them were in the
measurement, not in the code being measured**. Every entry here is a case where the solver was
right and the instrument was wrong, or where a check could not have failed. They are recorded
together because they rhyme: each one produced a plausible number, none of them raised an error,
and most were caught only because a second, independent quantity disagreed.

---

## 1. A bar taken from the wrong regime

**The log-law constant.** The van Driest channel gate demanded the textbook `B = 5.0-5.2` and
measured 4.271, so it failed. But at `Re_tau = 180` there is barely a log layer -- `y+ = 150` is
`y/delta = 0.83`, deep in the wake -- and the van Driest reference ITSELF gives 4.232 over the
same window:

| Re_tau | y+ 30-100 | y+ 50-150 |
|---|---|---|
| 180 | 4.459 | 4.232 |
| 590 | 4.992 | 4.980 |
| 5200 | 5.190 | 5.249 |

The solver agreed with its own target to 0.04 and was called a failure by comparison against a
different Reynolds number. **Compare like for like, at the same conditions and the same window.**

**The onset Reynolds number.** `sigma = k(Re - Re_c)` is a NEAR-ONSET expansion. Fitted across
Re = 55, 65, 80, 95 it gives `Re_c = 25.7` with `R^2 = 0.96`; the two lowest points alone give
39.4, against a published 45-47. Each higher point drags the intercept down because a straight
line through curved data crosses zero too early.

| fit range | Re_c | R^2 |
|---|---|---|
| 55, 65 | 39.4 | 1.00000 |
| 55, 65, 80 | 33.2 | 0.98153 |
| 55, 65, 80, 95 | 25.7 | 0.95825 |

**A high R^2 says the line fits, not that the model is right.** The residuals ran -, +, - --
curvature, not scatter -- and every one of those fits looked healthy.

---

## 2. An estimator that averages across regimes

The growth rate took **three** attempts, and the first two were biased by the same thing from
opposite sides.

Fitting the whole post-kick record of a saturated run returns almost nothing -- 0.0019 against a
true 0.09 -- because 250 of its 300 time units are a limit cycle at constant amplitude and least
squares averages the flat part into the slope.

Cutting at a fixed fraction of the maximum envelope fixes that and breaks the opposite case: a
run deliberately stopped BEFORE saturation is still growing at its last sample, so its maximum
IS its final value, and the cut discards the best-conditioned 60% of the data.

The local slope is not constant across a run at all:

```
Re = 55   -0.018  -0.001  +0.009  +0.028  +0.033  +0.032  +0.030  +0.028
Re = 65   +0.004  +0.027  +0.044  +0.055  +0.048  +0.038  +0.028  +0.018
```

It RISES while the kick's stable components decay and the unstable eigenmode takes over,
PLATEAUS -- that plateau is sigma -- then FALLS as the amplitude becomes nonlinear. **Find the
regime you want and fit inside it**; the plateau estimator changed sigma at Re = 55 from 0.0154
to 0.0342 and the implied Re_c from 23 to 39.

---

## 3. A check that cannot fail

**Seeding a minimum with the bar being tested.** 5c.10 asserted the diffusion operator is
positive semi-definite by accumulating `worst = min(worst, v.Dv)` starting from `worst = 0.0`.
It reported "worst = 0.000e+00" on a matrix whose true minimum is positive. The check could not
fail.

**Random probing for a localised mode.** Fixing that revealed a second problem: a `nu_t` with a
NEGATIVE patch still gave `min v.Dv = +4.9e+04` over 200 random vectors, because the unstable
direction is localised and a random vector barely overlaps it. The smallest EIGENVALUE finds a
SINGLE negative cell immediately, at -1.35e+02.

**A control with no discriminating power.** 6.3's missing-transpose control gave only a 1.0%
error, which a 1e-2 bar would pass on a technicality. The reason is structural: the momentum
matrix carries `J/dt` on the diagonal and its only non-symmetric part is convection, so in a
diffusion-dominated corner the transpose barely moves an inner product. Sweeping the cell Peclet
number and dt makes the discriminating power visible -- 1.1% at Pe 1.7 and dt 0.02, 19.6% at
Pe 16.7 and dt 2.0 -- and the bar belongs at the convective corner.

**Every mangle test is this idea made deliberate.** Zero `F_prev`, `p_flux`, `u_prev` or the
Dong pressure in the BACKWARD only and require the gradient to change. A backward pass that
silently ignores carried state produces a plausible gradient and nothing errors.

---

## 4. Measuring the wrong quantity entirely

**The divergence column.** `run_square_cylinder.py` reported
`divergence(face_fluxes(u, v, w))` -- the flux re-interpolated from the CELL velocities. On a
collocated grid that is not the object the projection makes solenoidal; Rhie-Chow makes the two
differ by construction. It read 4.22e+00 on a perfectly healthy run and could never have
distinguished a failing pressure solve from a grid working exactly as designed. The projected
flux on the same case and step: **5.92e-14**.

**A threshold relative to the wrong scale.** "How far does the wake reach" using 5% of peak
vorticity reported 9.3 D while coherent vortices were plainly visible at 25 D -- because the
peak lives in the boundary layer on the body and a fraction of it measures near-body dominance,
not the wake. An absolute level, `|omega| > 0.1`, answers the question asked.

**Normalising by a near-zero.** FD-vs-adjoint comparisons divided by the per-cell gradient
reported 3.2e-05 on a cell whose `dL/dS` was near zero, while every large entry agreed to
1e-08. Normalise to `max|g|` -- the scale a training step actually sees -- and sample where the
signal is.

---

## 5. A test blind to the thing most likely to be wrong

5c.1 checks that a CONSTANT `nu_t` array reproduces the scalar path. It passes exactly, and it
**cannot** detect a wrong face interpolation: interpolating a constant is exact however you do
it. Only a `nu` that varies exercises the interpolation, which is what MMS with a varying `nu`
is for -- and that came out at order 1.96-1.99, where a first-order or cell-valued face
coefficient would show as a rate near 1.

The same shape appears in the network: `TinySGSNet` and `SGSNet` use `padding_mode="circular"`,
which is right for one periodic box and wrong at a multi-block seam. Applied per block, interior
planes agree to 1.1e-16 and the four planes adjacent to a seam are wrong by **O(1)**, 146% of
the output scale. None of the 38 adjoint gates would have caught it, because every one drives
the solver with a raw parameter vector and never a network.

---

## 6. The optimiser will reduce numerical error if that is cheapest

Fitting van Driest's two constants against a target from a fine CONTINUOUS integration, with a
97-point Picard solve as the forward model:

```
oracle, true constants        L = 1.760e-04
fitted                        L = 2.583e-06     68x BETTER
recovered kappa 0.36913       true 0.41         -10.0%
recovered A+    22.5714       true 26.0         -13.2%
```

**Beating the oracle is the diagnostic.** Target and model are different functions, so the
optimiser moved the physical constants to absorb the DISCRETISATION error. The loss improved 68x
while the physics got 10% worse.

The control removes the freedom -- regenerate the target with the same discrete solver at the
true constants, so only the two numbers separate target from model -- and recovery is then
exact: kappa 0.409875 (0.031%), A+ 25.98744 (0.048%), loss 3.0e-10.

This is the same hazard as `src/forces.py`'s viscous normal stress, which is 1.7% of C_D while
the genuine friction drag is also 1.7%: **a network told to reduce C_D can reduce the
discretisation error instead**, and unlike kappa and A+ its parameters have no known right
answer, so nothing would look wrong. Keep a small parametric fit in the loop as a permanent
diagnostic: if its constants drift when the grid changes, the loss is measuring the mesh.

---

## 7. Not every quantity is recoverable, however good the solver

Learning `nu_t(y)` from a target velocity profile recovers it to **1.5% of peak** where
`5 < y+ < 108` and fails completely near the centreline. That is not a training failure and more
data will not fix it: `dU/dy` and the total stress vanish together there, so the velocity is
insensitive to `nu_t` and the inverse problem has no unique answer.

Stage 2 recorded the same limit in a different form -- only the SOLENOIDAL part of a momentum
source is identifiable from velocity data, so its loop matched the velocity to 8.9e-04 while the
recovered source was 18% wrong, and BOTH were correct. **A recovery test must target an
identifiable quantity**, or it will fail a working method.

---

## 8. The checks that actually caught things

Not instruments, but the pattern worth copying. In every case the catch came from a SECOND
quantity that had to agree:

* **The wrong flux** was found because 5.92e-14 and 4.22e+00 cannot both be the divergence.
* **The asymmetric far-field BC** was found because a picture looked asymmetric and the mirror
  test then measured 0.751 where the grid was symmetric to 0 nodes.
* **The leaking outflow arc** was found because mass was conserved and the inflow was exact,
  yet the near field ran at 0.74 U -- three facts that cannot all be innocent.
* **The unstable BDF2 chain** was found because the adjoint norm grew, and the cause was a
  matrix assembled with `bdf2=False` against a right-hand side carrying BDF2 coefficients.
* **The swapped C_p** was found because the front stagnation value must be +1 and read -0.34.

The common structure is a **redundant invariant** -- something that must hold if everything is
right, and that nothing in the code is trying to satisfy.

---

## 9. Every reported diagnostic was healthy while the run died

Added 2026-09-02, after the cylinder run diverged.

`run_cylinder.py` reported four things every 500 steps: the probe velocity, the amplitude
envelope over the last 500 samples, the step time, and once shedding began, `C_D` and `C_L` rms.
All four stayed plausible for 200 time units. `C_D` settled at 1.13, `St` at 0.144, and both were
close enough to the published 1.33 and 0.164 to be written down as a slightly-under-resolved
result rather than a warning.

They were measured on a field with `max|u| = 3.13` in the far field at `t = 200`, 4.97 at 215, and
NaN shortly after. The disturbance lived at `r > 10`, where the cell Peclet number reached 218 --
central differencing needs it near 2 -- and **not one reported quantity looked there.** The probe
sits in the near wake, the forces integrate over the body, and the envelope is a statistic of the
probe. The instrument was pointed entirely at the region that was still fine.

The general form: **a diagnostic suite assembled to measure the answer will not see the solution
being destroyed somewhere it is not looking.** The fix is not a better version of the existing
columns, it is one column over a region where the correct value is known a priori -- here
`max |u - U_inf|` outside the wake, which must be small because there is nothing out there -- and
an abort when it is not. That check crosses its threshold at `t = 20` on the old solution.

This differs from every other entry above. Those are instruments that returned the wrong number.
This is a set of instruments that all returned the right number, for the quantity each was
measuring, while the run was already lost.
