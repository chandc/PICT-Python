# The channel's streamwise checkerboard: remediation plan, and the OpenFOAM cross-reference

**Status (2026-09-06, evening): one defect found and fixed; the decisive runs are in progress.**
Sections 1-4 are the diagnosis and the plan as written before the runs finished. Section 5 is
filled in as measurements arrive and supersedes anything above it that it contradicts.

Companion to `channel_les_status.md` (what was withdrawn and why), `rhie_chow_boundary.md`
(T1-T7, the boundary gap -- fixed, and shown NOT to be the channel's problem), and
`pressure_checkerboard.md` (the pressure-mode family, which this is not).

---

## 1. What the evidence actually establishes, re-read

`nyquist_share` in the probe takes the Nyquist bin of the streamwise FFT of the CELL VELOCITY
`u` at y+ ~ 12. The mode is in velocity. That constrains the mechanism more than the previous
write-up allowed for:

* Every place pressure reaches velocity in this solver is a WIDE gradient (`Domain.gradient`,
  `compute_gradient`, both `np.gradient`). The wide operator annihilates the exact-Nyquist mode
  identically: it can neither create nor remove `(-1)^i a(y,z)` in `u`. The projection is blind
  to it for the same reason (wide divergence of the mode is zero). So `p += phi` accumulating a
  pressure checkerboard -- the `pressure_checkerboard.md` mechanism -- cannot be what puts the
  mode in `u`, under any scheme.
* The only operator that sees the mode is the compact diffusion, which damps it at
  ~4 nu_eff/dx^2 ~ 1.3 per time unit at dx+ = 23.6. The SGS models cannot help: Smagorinsky,
  WALE, Vreman and sigma all form |S| or g from wide gradients, so nu_t is exactly zero FOR the
  mode.
* The convection operator, however, is the advective form with the CELL velocity as the
  convecting velocity (`build_momentum_matrix`: row P gets a_P (u_E - u_W)/2h). If `u` carries
  the mode, a_P = ubar_P + (-1)^P a, the transported difference is mode-free, and the operator
  produces the source term **(-1)^P a du/dx** -- O(1), sign set by the local streamwise strain.
  The envelope then obeys roughly da/dt + u.grad(a) = -a du/dx + nu_eff lap(a): exponential
  growth wherever the flow converges in x. Near-wall streaks have du/dx of order tens in outer
  units, so intermittent growth beats a damping of 1.3 on average even though <du/dx> = 0. This
  is odd-even decoupling of central advective schemes at high cell Reynolds number, and
  Re_cell = 363 in x is where it lives.

So the scheme A/B (rotational 15.292x vs chorin 0.001x) is real, but it had to be acting through
the SMOOTH field, and section 2 is what it was acting through.

The A/B was also not controlled for turbulence health. In `results/logs/cb_chorin.log` the
`total` column -- streamwise fluctuation energy at y+ = 12 -- goes 1495 -> 747 -> 141 -> 104 ->
338 -> 1045; the 2dx energy of 5.8e-10 at t = 8 coincides with total = 141. An arm whose
near-wall turbulence dropped 10x has no mode for reasons unrelated to pressure. Every report line
of the probe now carries u_tau, e_u/e_v/e_w, the Nyquist share of p, |p - p_flux| and the wide
divergence of the cell velocity, and the comparison is read on the RATIO of mode to turbulence.

## 2. The defect: the Picard loop double-accumulated p_flux

`MultiBlockPISO._step_impl` (and `PISOSolver._step_impl`) restore `u, v, w, p, u_prev` before
the second Picard sweep and then call `_step_once` again. They did NOT restore `p_flux` or
`F_prev`. So with `picard_iters=2`, `p_flux += phi_tot` ran once per SWEEP while `p += phi_tot`
ran once per STEP -- the projection pressure that the Rhie-Chow term reads drifted from the
pressure the predictor felt by one first-sweep increment per step, without bound.

Measured on `results/fields/chan_re180_t4.npz`, ten steps, dt = 5e-4:

| scheme | picard_iters | max \|p - p_flux\| after 10 steps |
|---|---|---|
| incremental | 1 | 0.000e+00 |
| incremental | 2 | **7.516e+00** |
| rotational | 1 | 6.562e-03  (the legitimate -nu div(u*) sum) |
| rotational | 2 | **7.512e+00** |

against a pressure range of ~19. The t = 4 checkpoint itself carries |p - p_flux| = 6.3 from the
production run. `chorin` REPLACES `p_flux` every sweep and is immune -- which is exactly the
split the A/B measured, and why `rotational` vs `chorin` looked like a formulation property.

Why it can feed the velocity mode: the drift is a sum of first-sweep pressure increments, which
carry grid-scale content (that is what a projection increment is). The Rhie-Chow term
`Gamma (grad_c - I grad_w) p_flux` is precisely a high-wavenumber detector, so the flux acquired
a growing grid-scale perturbation every step, the Poisson solve absorbed it into phi, and the
velocity received `-Gamma grad_w phi`: not the Nyquist mode itself, but its near-Nyquist
neighbours and a steadily roughened du/dx field for the section-1 source term to work on. It is
also why feeding `p` rather than `p_flux` into Rhie-Chow diverged (`pressure_checkerboard.md`):
the two had already drifted apart.

**Fix.** Save `p_flux` and `F_prev` with `u`, `p`, `u_prev` at the top of the Picard loop and
restore all of them before each repeat, in both solvers. With the fix, picard_iters=2 gives
0.000e+00 (incremental) and 6.566e-03 (rotational). `test_channel_laminar.py` 4/4 and
`test_multiblock.py` 55/55 unchanged.

**Blast radius.** Every multi-block run with `picard_iters=2` and `rhie_chow=True` under an
accumulating scheme: `run_channel_les.py`, `run_square_cylinder.py` (the St = 0.1467 result),
`run_cylinder.py`, `run_tgv_les.py`, `run_les.py`. `run_armaly_bfs5.py` uses `picard_iters=1`
and is unaffected. The TGV LES stayed clean (0.03% Nyquist) despite the bug, which says the
drift alone does not manufacture the mode -- it needs the section-1 source, i.e. walls and
streaks -- but the Rhie-Chow term in those runs was not the term the documentation describes, and
the square-cylinder Strouhal number should be re-measured with the fix before it is quoted again.

## 3. What OpenFOAM's PISO is, in this repo's vocabulary

`pisoFoam` solves `UEqn == -grad(p^n)`, forms `HbyA = rAU*H(U)` (pressure stripped), and sets
`U = HbyA - rAU*grad(p^{n+1})`. Since `a_P u* = H - V grad(p^n)`, `HbyA = u* + rAU grad_w(p^n)`,
hence `U = u* - rAU grad_w(p^{n+1} - p^n)`. That is this repo's **`incremental`** scheme:
predictor feels `grad_w p^n`, correction by the increment, pressure carried forward -- with

1. `Gamma = 1/a_P` (the DIAGONAL, which includes the diffusion contribution), not
   `J/rowsum(A)`. That is why OpenFOAM's `ddtCorr` coefficient `rAU/dt` is strictly below 1
   (~0.7 in the first cell at dy+ = 1, dt = 1e-3) where ours was exactly 1 -- option 3 of
   `rhie_chow_ddt_instability.md`, "break the unit gain at its source".
2. No rotational term. OpenFOAM never adds `-nu div(u*)` to p.
3. No `p_flux`. The Rhie-Chow wide half is built from `interpolate(HbyA)`, so it uses the
   pressure the predictor actually felt, by construction. There is nothing to drift.
4. Convection is `div(phi, U)`: conservative form with the FACE FLUX `phi` -- RC-corrected,
   solenoidal -- as the convecting velocity, and `linear` (averaged) U at faces. Work through
   the section-1 algebra with that operator: the Nyquist mode of U appears at faces only as
   its envelope gradient (a_P - a_N)/2 = O(h da/dx), and the net production is O(h^2). Neither
   transported nor produced; it just diffuses. This, not the pressure scheme, is why a
   collocated central PISO can run `channel395` at dx+ ~ 40 without the field turning into a
   checkerboard (it still shows mild 2-delta wiggles, which is why `LUST` and
   `filteredLinear2` exist).

The scheme A/B ran the two arms that are NOT OpenFOAM. The incremental arm is in section 5.

## 4. The plan, in order

Cheap and decisive first.

| # | action | reads | decides |
|---|---|---|---|
| R1 | Picard-loop fix (section 2), both solvers | `test_channel_laminar` 4/4, `test_multiblock` 55/55, drift table | **done** |
| R2 | probe: `rotational` from t=4 WITH the fix, p reset to p_flux | 2dx amp vs `ab4882b`'s 15.292x at t=12, with health columns | whether the bug IS the scheme dependence |
| R3 | probe: `incremental` (the OpenFOAM-equivalent arm), same seed | same | whether accumulation per se matters once the bug is gone |
| R4 | if R2 still grows: `rotational` with `picard_iters=1` and with molecular nu only in the rotational term | same | isolates the variable-nu_eff rotational term |
| R5 | re-run the production channel (`run_channel_les.py`) from the DNS IC with the fix | grid-mode monitor, u_tau, U+, u'v'w' vs DNS | whether the withdrawn result comes back |
| R6 | `convection='flux'`: conservative form on the persistent face flux F (section 3.4) | `test_energy_conservation`, `test_channel_laminar`, Gate 0 periodic, then R5 again | removes the section-1 source term structurally |
| R7 | re-measure the square cylinder St with the fix | St vs 0.145-0.150 | whether the headline number survives |

R2/R3 run from the same `chan_re180_t4.npz` with every arm's `p` reset to `p_flux` (the saved
`p` carries 6.3 of drift). Report every 500 steps; `--save-every 4000` so any arm can be
post-processed.

## 5. Results

*(filled in as the runs report; see `results/logs/rotational_fixed.log`,
`results/logs/incremental_fixed.log`)*

### R2 / R3 -- the fixed arms, t = 4 -> 10. THE BUG WAS THE SCHEME DEPENDENCE.

Both arms from `chan_re180_t4.npz`, p reset to p_flux, dt = 5e-4, 12,000 steps, Smagorinsky +
van Driest, same mesh and flags as `ab4882b`. Amplification is 2dx energy of u at y+ = 12 over
its t = 4 value (8.9698e-02); `total` is the streamwise fluctuation energy at that plane.
Buggy-rotational and chorin columns are the earlier runs (`results/logs/cb_2x2.log`,
`cb_chorin.log`).

| t | rotational, buggy | chorin | **rotational_fixed** | **incremental_fixed** | total (rot_fixed) | u_tau | e_u / e_v / e_w |
|---|---|---|---|---|---|---|---|
| 4.25 | -- | -- | 0.289x | 0.289x | 180 | 0.980 | 4.55 / 0.32 / 0.88 |
| 5.00 | 0.273x | 0.002x | 0.002x | 0.002x | 1119 | 1.086 | 3.78 / 0.65 / 0.84 |
| 5.25 | -- | -- | 1.527x | 1.527x | 969 | 1.080 | 3.14 / 0.55 / 0.77 |
| 5.50 | -- | 0.572x | 0.168x | 0.169x | 1533 | 1.043 | 2.96 / 0.51 / 0.66 |
| 6.00 | 0.688x | 0.018x | 0.007x | 0.007x | 898 | 1.007 | 2.35 / 0.32 / 0.51 |
| 7.00 | 0.769x | 0.001x | 0.000x (5.7e-7) | 0.000x (4.8e-7) | 411 | 0.955 | 2.12 / 0.17 / 0.34 |
| 7.75 | -- | 0.000x | 0.000x | 0.000x | 127 | 0.931 | 2.48 / 0.15 / 0.28 |
| **8.00** | **1.120x** | 0.000x | **0.001x** | **0.001x** | 181 | 0.948 | 3.13 / 0.18 / 0.31 |
| 8.50 | -- | 0.000x | 0.092x | 1.760x | 643 | 0.970 | 3.12 / 0.26 / 0.53 |
| 9.00 | 2.236x | 0.000x | 0.041x | 0.069x | 842 | 1.023 | 3.65 / 0.38 / 0.72 |
| 9.50 | -- | 0.000x | 0.002x | 0.171x | 669 | 1.040 | 3.28 / 0.49 / 0.65 |
| **10.00** | **5.348x** | 0.000x | **0.000x (5.1e-6)** | **0.081x** | 529 | 0.993 | 2.89 / 0.35 / 0.54 |

Three readings.

1. **The fix removes the growth.** Fixed rotational is 5.1e-6 of its seed at t = 10 against
   5.348x for the buggy run -- six orders of magnitude -- and it tracks chorin's trajectory
   throughout. The Nyquist SHARE of u never exceeds 0.022% in either fixed arm; the pressure's
   own Nyquist share stays below 0.01%; |p - p_flux| holds at 2-4e-2 (the honest rotational
   term) for the whole run instead of climbing past 7 in ten steps.
2. **The turbulence is real in both fixed arms.** `total` cycles 127 -> 1533, e_v 0.15 -> 0.65,
   u_tau 0.93 -> 1.09 -- the intermittent minimal-box cycle, with the same quiet phase at
   t ~ 7-8 that the chorin arm showed and the same recovery after it. This is not a
   relaminarised win. The buggy arm's `total` (400-1500) is in the same range, so the mode was
   growing ON TOP of live turbulence there, which is what made it look like physics.
3. **Rotational and incremental agree to three digits until t ~ 8.25**, then separate as two
   chaotic trajectories must; the rotational term itself does nothing to the mode. Incremental
   shows brief excursions (1.53x at t = 5.25, 1.76x at t = 8.5) that decay within half a time
   unit and never carry more than 0.02% of the energy -- the transient response of a Nyquist
   mode to a burst, damped as section 1 says it should be at this Re_cell.

So `channel_les_status.md`'s "MECHANISM FOUND: the pressure scheme accumulates the mode" should
now read: the Picard loop double-accumulated `p_flux` (section 2); the pressure scheme was the
carrier, not the cause. The section-1 source term (cell-velocity advective convection) and the
variable-nu_eff rotational term remain as second-order concerns and are NOT needed to explain
the observations; R6 is deferred until R5 gives a reason to revisit it.

Logs: `results/logs/rotational_fixed.log`, `results/logs/incremental_fixed.log` (sliced and
resumed -- the PAUSED/resumed lines are the slice boundaries; read the `step` lines as one
series). Checkpoints every 2000 steps in `results/fields/probe_<arm>_*.npz` were kept on the
machine that ran them, not committed.

### Independent replication on the Mac, extended to t = 12

A second pair of R2/R3 arms ran concurrently on the Mac (same seed, same flags, 16,000 steps,
0.67 s/step, 2026-09-06 23:11 -> 01:59): trajectory bitwise-identical to the sandbox pair
until chaos separated them (~t = 7), and the verdict holds for two further time units. At
t = 12: rotational 2dx share 0.0001% (energy 8.0e-04, total 681, u_tau 0.983); incremental
share 0.0000% (energy 6.6e-10, total 313, u_tau 0.954). Checkpoints every 4000 steps in
`results/fields/probe_{rotational,incremental}_fixed_*.npz` (through `_016000`); final
numbers computed from the 016000 files. The Mac pair's LOGS are gone -- both sessions used
the same tags, and the sandbox logs synced over the Mac ones through Dropbox mid-run (the
Mac milestones survive in that session's transcript). Two machines writing one Dropbox
`results/` clobber each other: use distinct tags per machine.

### Addendum: the T7 18x slowdown, closed by measurement

The pre-fix T7 arms ground 18x at t ~ 4.75 (`rhie_chow_boundary.md`). Post-fix, on the same
start checkpoint: R2/R3 (dt = 5e-4) crossed the window at declining cumulative cost (0.75 ->
0.70 s/step), and a dedicated dt = 1e-3 probe (`burst_dt1e3_mac`) ran flat at 0.687 s/step
until the CFL guard stopped it at t = 4.60 (CFL_y 0.92, v_max 5.14). The burst is this seed
trajectory's spin-up transient (R5's production run tripped its guard at t = 4.750 and 5.41 on
an independent trajectory); the 18x cost was the bug's accumulated p_flux drift poisoning the
pressure conditioning on top of it, and it does not survive the fix in any configuration.

### R7 VERDICT (2026-09-08 06:12): the Strouhal number survives the fix

`sqcyl_r7_spark`, the full protocol on the rebuilt 82k-cell/ratio-1.10 grid at nz = 8
(164,192 cells), tol 1e-6, AmgX (see src/amgx/CONFIG.md for the restored stack), post-fix
code: settle to t = 80, sinuous kick, shedding to t = 380.

    St  = 0.1488 +- 0.0001   (32 periods over t = 162-377, period 6.7216 +- 0.0036)
    C_D = 1.427              (literature 1.4-1.5 at ~5% blockage)
    C_L rms = 0.163

Against the pre-fix headline 0.1467 +- 0.0067: inside its error bar, inside the published
CFD band 0.145-0.150, measured on a cleaner limit cycle (period scatter 0.05% vs the old
4.6%) and on the rebuilt grid the truncated v3 run never got to judge. The last
blast-radius item clears: every number the p_flux bug touched has now been re-measured
and stands. figures/R7_shedding_t380_vorticity.png is the street; the post-fix BASE flow
far field is clean to x = 25D (figures/R7_postfix_base_Re100_vorticity.png), with the
boundary-fix attribution pair (R8, cyl_postfix_spark vs cyl_legacybc_spark) running next.

### Next: R5 and R7

R5 is the production channel from the DNS initial condition with the fix, exactly as
`run_channel_les.py` runs it, watched by the grid-mode monitor and compared to DNS at the end.
R7 is the square-cylinder St with the fix. Both belong on the Mac or Spark: at 2 s/step in the
sandbox used for R2/R3, R5 alone is ~17 hours of slices.
