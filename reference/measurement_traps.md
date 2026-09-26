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

**THE SAME TRAP CAUGHT US TWICE MORE ON 2026-09-03, INSIDE THIS REPO, AFTER THIS SECTION WAS
WRITTEN.** Both were found only when a real reference at the matched configuration was finally
pulled -- Sohankar, Norberg & Davidson, IJNMF 26:39-56 (1998) -- rather than a band remembered
from somewhere.

*The onset Reynolds number, again.* `sqcyl_onset.py` carried `RE_C_REF = (45, 47)`. That is the
**zero-blockage experimental** estimate. Our grid is 5% blockage, where the same paper computes
`Re_cr = 51.2 +/- 1.0` and states that the critical Reynolds number RISES with blockage. Our
measured 51.31 was being read as 10% high when it is **0.2% off the right reference**.

*The stagnation pressure.* Several notes in this repo, and several statements made while
reading these results aloud, called `C_p = +1` at the front stagnation "the one number with an
analytic answer independent of the solver". It is exact only at ZERO blockage. At finite
blockage the flow accelerates past the body, so the stagnation coefficient exceeds one: the
reference gives 1.052 at 5% and 1.083 at 2.5%. Our 1.0725 was being called a 7% error; against
the right value at the right blockage it is **+1.9%**.

The pattern in both is the same and it is worth stating flatly: **a reference value is a
measurement of a CONFIGURATION, not of a quantity.** "The critical Reynolds number of a square
cylinder" and "the stagnation pressure coefficient" are not numbers; they are functions of
blockage, incidence and Reynolds number, and quoting one without its configuration is quoting
nothing. The cost here was not a wrong result -- both of ours were right -- but weeks of
apparent disagreement that would have driven a search for a bug that does not exist.

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


## 10. A guard that measures a constant instead of the quantity

The Re_tau=180 channel driver printed a CFL number at launch and never again. The line was

    f"CFL_y {3.0*a.dt/np.diff(y).min():.2f}"

with `v_max = 3.0` written in rather than measured. It printed a comfortable 0.54, which
happened to be right for the initial field, and then went stale. v_max grew 2.80 -> 3.49 -> 3.81
-> 4.60 as the constant-pressure-gradient forcing spun the flow up, CFL_y crossed 1 just after
t = 4.95, and the solution went from v_max 4.65 to 158 in fifty steps. Two and a half hours of
compute, and 79 minutes of a Krylov solver grinding on an already-destroyed field afterwards,
because the failure presented as a hang rather than a divergence.

**A start-up estimate cannot police a limit that moves.** The wall-normal CFL TIGHTENS as a
channel develops, so it is exactly the wrong quantity to check once. It is now measured from the
field in all three directions and re-checked every 25 steps, and on breach the run saves the
field and exits naming the dt that would have worked.

## 11. Divergence reports perfect health on a meaningless field

The same channel run then completed 48,000 steps reporting u_tau 0.95-0.98, U+_c ~19, nu_t/nu
~0.19 and interior divergence 3e-14 -- all steady, all plausible -- while the resolved field
became **99.3% grid-scale checkerboard**, peak streamwise wavelength 47.1 wall units against a
mesh spacing of 23.6, i.e. exactly 2*dx.

Divergence is the worst offender because it gives POSITIVE reassurance: a 2*dx velocity mode can
be exactly divergence-free, so the projection is doing its job perfectly on a field that is
physically nonsense. u_tau and U+_c are nearly as bad -- a mode alternating sign cell to cell is
largely invisible to a plane average -- so the mean profile stayed credible and the Reynolds
stresses, 99% artefact, looked like a recognisable under-resolution signature. They were
published as such before the field was ever plotted.

**It was caught only by looking at a FIELD rather than an integral.** A wall-parallel contour
plot showed regular cells where streaks should be, and a spanwise spectrum settled it in one
measurement. The Nyquist mode is now monitored directly and aborts above 25%.

## 12. Two paths that must agree, and a diff that proves they do

The distributed solver failed its equivalence criterion by 2.8x in iteration count, which
matched a failure mode the plan had explicitly predicted -- block-Jacobi being
partition-dependent by construction. That explanation was recorded in two commits before anyone
checked it.

The data contradicted it the whole time. Partition dependence should hurt the ELLIPTIC, globally
coupled PRESSURE solve most; instead pressure was flat at 1.08x while the diagonally dominant
MOMENTUM system inflated 2.8x. Backwards for a preconditioner effect, and exactly what a dropped
initial guess looks like: `_petsc_solve_mpi` did not take `x0` as an argument at all, so every
distributed solve restarted from zero. The pressure solve passes no guess, so it never noticed;
the momentum solve passes the previous Picard iterate, so it paid the entire cost.

**A correct prior made a wrong explanation feel like a confirmed prediction.** The plan's warning
was good and the mechanism was real -- it just was not what was happening. Two lessons:

  * When an explanation is available before the measurement, it is worth asking what the data
    would look like if the explanation were FALSE. Here it would have looked exactly as it did.
  * Duplicated configuration is how two paths come to disagree. Fixing the convergence criterion
    in the serial path changed nothing distributed, because `_petsc_solve_mpi` is a second copy
    and the copies had drifted. A programmatic diff of every PETSc configuration call in the two
    functions is now the check; it is cheap and it is exhaustive.

## 13. A test whose own instrument is confounded

To decide whether the remaining discrepancy came from the preconditioner, the distributed run
was switched to `PCREDUNDANT`, which applies the full factorisation on every rank and is
therefore partition-independent. The gap did not close (1.8e-12), which looked like evidence
that the preconditioner was innocent.

It was not evidence of anything. **PCREDUNDANT defaults to a direct LU**, so the test compared a
DIRECT distributed solve against an ITERATIVE serial one, and measured the solver difference
rather than the partitioning. The tell was in the same output: momentum iterations dropped to 6,
about one per solve, which no iterative method does.

The clean experiment was to stop distributing the momentum system at all -- making it
bit-identical by construction -- and it closed the gap immediately, 1.0e-12 -> 8.2e-14.

## 14. Mangles that silently stop mangling

`test_halo_content.py` and `test_mpi_equivalence.py` between them carry five deliberate defects
that must be detected. When slab extraction moved behind the `Comm` abstraction, all of them
kept hooking the OLD routing point and stopped intercepting anything. The halo test dropped to
4/5 with its corruption undetected; the others reported PASS while testing nothing.

This is the failure mode mangles exist to prevent, landing on the mangles themselves. A mangle
is only alive while it is hooked to the code path actually in use, and moving an abstraction
silently unhooks it.

**Two mangles were also wrong on their own terms.** One routed `fetch_padded_coords` through
`fetch_padded_field` and could never fail, because at that gate the two methods had identical
bodies -- the distinction they guard lives in the callers. Another reversed layer order in every
exchanged slab and produced a perfectly PASSING run, because the field halo and the coordinate
halo were corrupted identically and the errors cancelled: the test compared a padded field
against f evaluated at equally-corrupted coordinates. That blind spot is closed by checking the
coordinates independently against closed form.

## 15. `mpirun -n 1` pins to one core

A single-threaded scaling run showed 1 rank at 10.6 s/step against 2 ranks at 5.4 -- an apparent
halving of SOLVE time, which is impossible when the solve is replicated. It reproduced across
runs, so it was not noise. It was `mpirun -n 1` binding to a single core on a machine with 12
performance and 4 efficiency cores; `--bind-to none` brought 1 rank to 6.9 s/step and the
speed-up vanished.

The figure was self-consistent, reproducible and flattering, and was contradicted only by knowing
what the code actually distributes. **Any scaling number must state its binding and thread
settings, or it means nothing.** Absolute timings also vary by 40% with machine load, so
configurations must be measured back to back rather than against a stored table.

## 16. A probe sampling the wrong window

The 2*dx mode had to grow from 0.1% to 99% of fluctuation energy over 24 time units, which is an
amplitude growth rate of about 0.14 per unit. The first probe seeded the mode and ran 0.3 time
units -- over which the predicted growth is a factor of **1.04**, swamped by the seed's own
projection transient. It measured decay in all four configurations and concluded the mode was
damped, the opposite of the truth.

Worse, the mode DECAYS for its first ~2 time units and only then turns around, so a short probe
does not merely lack resolution: it reads the sign backwards. The corrected probe runs 8 time
units and reproduces growth to 5.35x.

**Before running a probe, compute what the effect size will be over the window sampled.** If it
is comparable to the transient, the probe cannot answer the question however clean its output.

## 17. Two sessions, one Dropbox, same output paths

R2/R3 ran twice in parallel -- on the Mac and in a second session's sandbox -- both writing
`results/logs/rotational_fixed.log` under the same tag into the same Dropbox-synced tree. Mid-run,
Dropbox replaced the Mac's log files with the sandbox's copies: the Mac processes kept appending
to the orphaned inodes (`lsof` showed fd 1 on a 7 KB inode while the directory entry pointed at a
25 KB file with a different `s/step`), and their remaining output was silently lost. The
overwritten logs even carried a plausible-looking `DONE` line from the OTHER run's wrapper -- a
finished-looking log for a run that had not finished.

By luck the two runs were the same deterministic code on the same seed, so the trajectories were
bitwise-identical and nothing scientific was lost; with different arms the collision would have
spliced two experiments into one file with no visible seam.

**A tag names an experiment, not a machine.** When more than one machine (or session) can run the
same experiment into a synced tree, the tag must include the machine, or the results directory
must not sync. And a log's `DONE` line proves nothing about the process you launched -- only the
process table and the artefacts it wrote (checkpoint mtimes at the expected cadence) do.

## 18. A discriminating experiment validated by label, not mechanism

To separate "the fractional-step pressure coupling is intrinsically an integrator" from other
causes of the channel checkerboard, the chosen discriminating arm was the `incremental` scheme --
OpenFOAM's coupling, the candidate remedy. It would have amplified the mode exactly as the
hypothesis predicted, and the conclusion would have been to rewrite the pressure coupling.

It would have been wrong. The actual cause was a state-restoration bug (the Picard loop restored
`u`, `p`, `u_prev` between sweeps but not `p_flux`), and `incremental` accumulates `p_flux`
through the SAME unrestored path as `rotational`. Both hypotheses predicted the same outcome for
the one arm chosen to separate them; the arm was selected because its LABEL differed from the
suspect ("this is the other formulation"), not because its MECHANISM did. Only chorin -- which
replaces `p_flux` each sweep -- was mechanically immune, and that immunity was the whole of the
measured 15,000x "scheme dependence".

**Before running a discriminating experiment, trace each arm through the code path under
suspicion and confirm the rival hypotheses predict DIFFERENT outcomes for it.** A discriminator
whose arms differ in name but share the mechanism is not a discriminator. (The fix's R3 run --
incremental, clean to t = 12 with zero drift -- is what the test would have looked like once the
mechanism was actually different; see `channel_checkerboard_remediation.md` section 5.)

## 19. Bisecting through a silent fallback

Six AmgX rebuilds -- across source revisions, CUDA toolkits and configs -- all "failed the
same way": banner printed, then silence at 100% CPU with the GPU idle. The bisection treated
each arm as a test of the library. It was not: `_amgx_solve` caught every init failure and
silently fell back to scipy, so most arms were measuring the FALLBACK (a 6 s/step CPU solve
whose first log line takes 50 minutes), and the actual failure -- a process-wide config
refusing the second tolerance, momentum 1e-9 vs pressure 1e-6 -- was invisible in all of them.
Standalone smoke tests passed the whole time because they created ONE solver.

The bisection produced real findings (the release tag genuinely cannot run on CUDA 13), but it
burned hours attributing one bug's symptoms to another. **Before bisecting a failure, make
every failure path in the harness LOUD; a fallback that silently substitutes a working slow
path makes all arms of the bisection measure the substitute.** One printed traceback located
in one minute what six builds could not. Related: trap 17's lesson that a DONE line proves
nothing -- here, the absence of an error line proved even less.

## 20. The profiler contended with the thing it profiled

Every cylinder performance probe -- cProfile, the solve-stats runs, the preconditioner
shootout -- ran on the same GPU as the live production arm, and for several hours THREE
processes were solving concurrently: the arm, a leftover arm from the previous configuration
take, and an orphaned profiling container. Two mechanics put them there: `docker ps -q
--filter ancestor=X | head -1` kills an arbitrary one of several matching containers (probes
and arms share the image), and a remote `timeout` kills the docker CLIENT while the container
runs on. The contended numbers were internally consistent and wrong: a 59 ms pressure solve
read as 449 ms, which spawned a coarse-operator-refresh theory, an (exact, harmless, useless)
skip-identical-upload patch, and an hour of misdirected analysis. The stale cumulative s/step
column in the driver log reinforced the wrong number.

**Before profiling, enumerate everything sharing the resource (`docker ps` in full, not
head -1 of a filter), and measure throughput by wall-clock arrival of progress markers, not
by in-process averages that integrate over a dirty past.** Kill containers by exact ID; a
timeout on `docker run` orphans, not stops.

## 21. The fix that was never deployed

R10's relaunch "with all three solver fixes" ground for two hours exactly like the
run before it -- because the rsync to Spark had never carried the fixed files. The
failure was then almost attributed to the fixes not working. The tell, found only
after a py-spy stack sample and a binding diff: `md5sum` of the touched files on the
two ends disagreed.

**A remote failure indicts the fix only if the remote provably runs the fix. Checksum
every touched file on both ends as part of the launch, not as a post-mortem.**

## 22. A log that is alive and says nothing

The same relaunch logged through a plain redirect without `-u`. Python block-buffers
redirected stdout; the C library's own prints (unbuffered fd writes) got through. The
result was the worst combination: a log whose mtime advanced and which contained the
AmgX banner -- looking alive -- while every Python-side progress row sat in a buffer
for two hours.

**`python -u` for every containerised run. A log's liveness is proved by progress
markers arriving, not by its timestamp.**

## 23. Every readout the solver library offers can lie at once

AMGX_solver_solve's return code reports API health, not convergence: a diverged solve
returns rc 0 and garbage. The status handle knows about divergence but measures
convergence RELATIVE TO THE INITIAL residual -- a good warm start makes the target
sub-machine-precision, so exact solutions read "not converged". And the residual it
monitors is the PRECONDITIONED one, so a broken preconditioner reads "converged" on a
solution whose true residual is 1e9. All three lies were load-bearing in one week.

**The only convergence statement worth trusting is one you compute yourself:
||b - Ax|| <= rtol ||b||, one SpMV, in the caller, on every solve.**

## 24. np.ascontiguousarray is not a copy

The binding passed the caller's warm start straight to AMGX_vector_download through
`np.ascontiguousarray`, which returns the SAME array when it is already contiguous
float64. Every failed solve therefore overwrote the caller's x0 -- a slice of live
solver state -- with divergence garbage, in place. One bad solve then poisoned every
retry, every fresh solver object, and a full library re-initialisation, perfectly
imitating permanent process corruption; a fresh process "fixed" it, deepening the
illusion. Found only when a dump recorded |x0|=2e5 for a solve whose entry trace had
just printed |x0|=2.29.

**Any buffer a C library writes into must be an explicit `np.array(..., copy=True)`.
`ascontiguousarray` is a cast that sometimes copies, never a guarantee.**
