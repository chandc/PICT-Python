# Vortex street behind a square cylinder — and the one setting that decided it

**Result at Re = 100, 5% blockage, against Sohankar, Norberg & Davidson (1998) at the same
blockage:** `St = 0.1488` (+1.9%), `C_D = 1.4529` (−0.5%), stagnation `C_p = +1.073` (+1.9%),
base `C_p = −0.705` (−6.6%), `C_L` rms `= 0.1722` (+23.9%, against a reference whose own three
grids spread 12%), and `Re_c` bracketed 52–55 with a near-onset fit of **51.31 against a
published 51.2 ± 1.0**.

The drag oscillates at exactly twice the lift frequency, shown from the fields rather than from
a spectrum: `C_L(t+T/2) = −C_L(t)` to 6e-06 of the lift rms and `C_D(t+T/2) = +C_D(t)` to 6e-05
of the drag's own peak-to-peak.

**Not yet validated:** grid convergence has never been run — every saved result shares one
82,096-cell fingerprint — so all of the above is agreement, not convergence. Section 6.3.

## 1. What actually made it shed

At `tol=1e-4` this case converges **bitwise steady** — the probe froze at machine epsilon
(max |dv/step| = 5.6e-17) and stayed identical for thousands of steps. At `tol=1e-6`, with
nothing else changed, it sheds.

| tol | outcome |
|---|---|
| 1e-4 | bitwise steady, no shedding |
| 1e-6 | **St = 0.1467**, amplitude 0.62 |

**The mechanism was already written down.** `reference/linear_solver_tolerance.md` records that at
1e-4 the pressure correction is under-resolved (~48 Krylov iterations), so *each velocity update
is damped*. That note validated 1e-4 on the **steady** five-domain BFS, where a damped iteration
still converges to the same fixed point — it just gets there differently. Nobody asked what
per-step damping does to a **marginally unstable** flow. It suppresses the growing mode entirely.

**The failure is silent and total.** It does not shift St by a few percent; it deletes the
physics and returns something that looks like a beautifully converged steady solution, with
divergence at 1e-8 and a residual that flatlines. There is no warning anywhere.

---

## 2. Prerequisites — needed to run, but not the reason it was steady

Four separate defects had to be fixed before the case would run at all. None of them caused the
missing shedding; all were found while chasing it.

| # | defect | how it showed | fix |
|---|---|---|---|
| 1 | obstacle was **1.203 × 1.107, off-centre by 0.048** | `validate()` clean throughout — it checks seam compatibility, and the seams were fine | give each body face's node line to the block whose face is the *wall*; seams here are non-duplicating |
| 2 | **32 nodes free inside solid material** | `wall_mask()` derives from face types, and an obstacle corner belongs to a block whose faces there are both *seams* | `pin_obstacle_corners` in `square_cylinder_bc.py` |
| 3 | **GCL violated at 1.81** at reentrant corners | uniform flow acquired divergence; ~0.5% spurious velocity out of an undisturbed freestream | grid builder supplies the `background` node distribution — see `reentrant_corner_gcl.md` |
| 4 | **NaN at step 455** | `ddt_corr` unit-gain recurrence; \|RC\|/\|F\| 0.184 → 1.818 | `ddt_corr=False`; Rhie–Chow itself was innocent — see `rhie_chow_ddt_instability.md` |

Defect 1 is the instructive one: the grid was wrong, every existing check passed, and it was
caught only because the boundary-condition classifier refuses to place a face it cannot identify
geometrically. Write the check that fails loudly on the unknown case.

---

## 3. Hypotheses that were wrong, and what killed each

Recorded because each cost real time and each *looked* right.

| hypothesis | evidence for it | test | result |
|---|---|---|---|
| shear layer under-resolved | L_r/D = 0.89 vs ~1.5–2 expected; only 4.5 cells across the layer | clustering `beta` 2.0 / 2.5 / 3.0 → 4.5 / 9.7 / 21.5 cells | **all identical**, ~5e-16 |
| effective Re far below nominal | short bubble *and* St ≈ 0.03 both point that way | run at Re 200 and **Re 300** | **both steady** |
| Rhie–Chow dissipation | RC is deliberate added damping | RC off | still steady |
| first-order upwind convection | would explain everything at once | check the scheme | already central |

**Re = 300 is the test that should have run first.** A square cylinder must shed well above
Re ≈ 45–50, so a steady result there falsifies every dissipation story in one run, for the cost
of one run. It went fourth, after a long detour into grid refinement driven by L_r/D = 0.89 and
St ≈ 0.03 — numbers that were real, and pointed the wrong way.

The general lesson: when two independent measurements agree on a diagnosis, that is *not*
confirmation if both are downstream of the same unknown cause.

---

## 3a. The second silent failure — the kick excited a stable mode

Found on 2026-09-01, rebuilding the grid. It is the same shape of failure as the tolerance one
and cost about as much, so it is recorded at the same length.

**Symptom.** On the rebuilt grid the case would not shed, no matter how it was perturbed.
`sqcyl_spark2` ran 45,000 steps to t = 450 and was bitwise steady from t ≈ 105 (amp/500 ~ 5e-8);
`sqcyl_spark3` restarted that checkpoint, kicked it again, and decayed to 1e-16 a second time.
The reading at the time was "the rebuilt grid has lost the instability" — the grid had just
changed, so the grid was the suspect.

**Cause.** The perturbation was built from `np.sign(y)`, i.e. transverse velocity **odd in y**.
That is the **varicose** mode: the wake breathing symmetrically about the centreline, and it is
**stable** at Re = 100. The von Karman mode is **sinuous** — the wake meanders bodily sideways,
so v has the SAME sign right across it, **even in y**. Every kick to that point had been
projecting onto the wrong mode, and a stable mode decays on every grid at every resolution. The
decay was evidence about the perturbation, not about the grid.

A second error compounded it: the kick was applied at t = 0, in undisturbed parallel flow with
no wake to perturb. It convects downstream and is gone before the recirculation region forms.
The fix is two-stage — settle to the base flow, then perturb THAT (`--settle`, then `--kick`).

**Confirmation.** `sqcyl_v3`, 82,096 cells, settled 8,000 steps to the base flow and given a 2%
sinuous kick, grew as a clean exponential at **sigma = 0.083 per time unit** (measured over 2.1 e-folds,
t = 83 to 109, by `sqcyl_onset.growth_rate`; a first coarse estimate over 10-unit windows
gave 0.070 because it included windows that were already saturating) and saturated:

| window (t) | peak-to-peak | ratio |
|---|---|---|
| 130-140 | 0.5556 | 1.264 |
| 140-150 | 0.5948 | 1.071 |
| 150-160 | 0.6203 | 1.043 |
| 160-170 | 0.6263 | 1.010 |
| 170-180 | 0.6301 | 1.006 |

**0.6301 against 0.62 on the original grid** — the same limit cycle. So the grid rebuild
(spacing ratio 1.15 -> 1.10) cost nothing, and the instability had never been lost.

**Why it is worth a section.** Both of this case's expensive failures return a *plausible*
answer with no error anywhere: the tolerance one converges to the unstable steady solution, and
this one decays back to it. In both, "it converged to steady" is the output, and in neither is
steady the right answer. A stability calculation is not finished when the residual is small; it
is finished when the perturbation that was applied is the one the physics amplifies.

`figures/sqcyl_spark2_streamlines.png` is what the wrong answer looks like — symmetric twin
vortices, L_r/D = 7.23, entirely clean.

---

## 4. Reproducing it

Grid — `square_cylinder_grid.py`, 8-block H-grid, 63,280 cells at `nz=4`, 5% blockage,
wake resolved to 18.6 D.

```python
m = MultiBlockPISO(d, U_INF*D/100.0, dt=0.01, 2, tol=1e-6,
                   time_scheme="bdf2", scheme="rotational", picard_iters=2,
                   rhie_chow=True, persistent_flux=True, ddt_corr=False)
```

| setting | value | why |
|---|---|---|
| `tol` | **1e-6** | 1e-4 kills the instability outright (§1) |
| `ddt_corr` | **False** | diverges at step 455 otherwise |
| `rhie_chow` | True | 41× smaller checkerboard than off |
| `persistent_flux` | True | no stability effect; 1.7× better damping, free |
| `nz` | 4 | Re = 100 is 2D; cost is exactly linear in nz |
| `dt` | 0.01 | CFL ≈ 0.46; 682 steps per shedding period |

Boundary conditions — `square_cylinder_bc.py` classifies all 48 faces by geometry and raises on
anything it cannot place. Laterals are **prescribed freestream, not no-slip** (measured global
flux imbalance 2.0e-06 against 1.9e-04). Outlet is Dong.

**Two things about starting it.** The configuration and the discretisation are both symmetric
about y = 0, so a symmetric initial condition stays symmetric forever — it converges to the
*unstable* steady solution and never sheds. It needs a deliberate perturbation. And restarting
from the converged steady state is worth doing: it is the base flow whose stability is in
question, so growth from there is the instability itself rather than a startup transient. A 1%
antisymmetric nudge in the near wake is enough.

Run: 30,000 steps (t = 40 → 340) at 0.299 s/step on the GB10 with AmgX.

---

## 5. How well the outflow held up

Vortices crossing the boundary is the hard test for an outflow condition. Dong passed it.

| check | measured |
|---|---|
| global mass balance | 1.08e-06 relative to inlet — *is* the solver tolerance |
| largest non-harmonic spectral peak | 5.09e-03 of the fundamental |
| non-harmonic energy fraction | 2.4e-05 |
| limit-cycle envelope drift | **+0.1%** over 28 periods |
| rms(v) one cell from the outlet | 0.16 — vortices leave at strength |

Reflections would appear as sidebands or an incommensurate peak, and a leaking outlet as slow
envelope modulation. Neither is present. The harmonic ladder is clean and monotonic:
1.000 / 0.0643 / 0.0264 / 0.0042 / 0.0012.

Attenuation beyond x ≈ 15 D is the **mesh**, not the boundary — the grid deliberately resolves
the wake only to 18.6 D.

---

## 6. What this does and does not validate

Updated 2026-09-03, when the run reached t = 535 and the forces, the onset sweep and the far
field had all been measured.

### 6.1 What is measured, against data at the SAME blockage

Reference: **Sohankar, Norberg & Davidson, IJNMF 26:39-56 (1998)**, Table III case 5 — Re = 100,
zero incidence, **5% blockage**, which is exactly this grid. That match matters: the same paper
shows base suction changing 7.6% between 5% and 2.5% blockage, so comparing against a
"low-blockage band" is not a comparison.

| quantity | ours | reference | error | their own grid-to-grid spread |
|---|---|---|---|---|
| `C_D` | **1.4529** | 1.460 | **−0.5%** | 1.2% |
| `St` | **0.1488** | 0.146 | **+1.9%** | 0.0% |
| `C_ps` stagnation | **+1.0725** | +1.052 | **+1.9%** | 0.7% |
| `C_pb` base | **−0.7045** | −0.661 | −6.6% | 2.6% |
| `L_r` recirculation | **1.887** | 2.20 | −14.3% | — |
| `C_L` rms | **0.1722** | 0.139 | **+23.9%** | 12.2% |
| `Re_c` | 52 < Re_c < 55, fit **51.3** | **51.2 ± 1.0** | **+0.2%** | — |

`C_D`, `St` and the stagnation pressure land inside or beside the reference's own spread.
`Re_c` is the standout: 51.31 from the near-onset fit against a published 51.2 ± 1.0, and the
sign bracket 52–55 straddles it.

**`C_L` rms at +23.9% is the real disagreement.** It is also the quantity the reference itself
cannot pin down — their three grids give 0.139, 0.156, 0.153, a 12% spread with no sign of
convergence, and their outlet study produces 0.024 for one case, an order of magnitude off the
rest. So the discrepancy is partly ours and partly theirs, and nothing here separates the two.
`L_r` at −14.3% is second, and is a near-wake length, hence sensitive to the same resolution.

**Two corrections this comparison forced.**

`C_p = +1` at the front stagnation is **not** exact at finite blockage. The flow accelerates past
the body, so the stagnation coefficient exceeds 1: the reference gives 1.052 at 5% blockage and
1.083 at 2.5%. Earlier notes in this file treated +1 as the exact answer and read our 1.0725 as
a 7% error; against the right value at the right blockage it is **+1.9%**.

`sqcyl_onset.py` carried `RE_C_REF = (45, 47)`, which is the **zero-blockage experimental**
estimate. At 5% blockage the reference is 51.2 ± 1.0, and the paper states explicitly that
`Re_c` rises with blockage. Comparing a 5%-blockage result against a zero-blockage number made a
correct answer look 10% wrong — section 1 of `measurement_traps.md`, committed inside the file
that documents it.

### 6.2 St is 60x sharper than the spectrum suggests, and this was quoted wrongly

An FFT of the 35-time-unit force record gives St = 0.1429 with a **bin width of 0.0286**, i.e.
±19%, and that bin was quoted for a while as the dominant uncertainty. It is not the dominant
uncertainty; it is an artefact of asking for the answer in the wrong form.

The information is in the TIMING of the zero crossings, not in the spectral resolution. Eleven
crossings over 35 time units give

    T = 6.719914 ± 0.000002   ->   St = 0.148811 ± 0.000002

The ten half-periods in the record are 3.359956, 3.359962, 3.359955, ... -- a scatter of
3.6e-06. **Fit the feature, do not bin the spectrum**, whenever the signal is a clean limit
cycle.

**But that is REPEATABILITY, not accuracy.** The limit cycle is converged to six figures, so the
period is known to six figures; St as a PHYSICAL number is limited by the grid, the 5% blockage
and the domain length, and none of those has been varied. The right quotation is St = 0.1488
with the systematic uncertainty stated as unmeasured -- see 6.3. Reporting +/- 0.000002 as
though it were an error bar on the physics would be the most misleading number in this file.

### 6.3 What is still not validated

**Grid convergence has never been run.** Every saved square result shares one grid fingerprint,
`95b3efa8...`, 82,096 cells. St at 1.5x and 2x resolution should move well under 0.0005, and
until that is done a single well-placed number is agreement, not convergence.

**The far wake past x ≈ 18 is not resolved and its decay cannot be apportioned.** The wake
plateau holds dx = 0.15 to x = 14 and then stretches to 0.94. Peak vorticity decays smoothly and
decelerating -- station-to-station ratios 0.79, 0.82, 0.86, 0.89, 0.92 over 3 D steps -- and
then drops abruptly to 0.70 exactly where dx jumps, and reads 1.000 for the next interval, which
is not physically possible and shows the measurement itself has failed. The reason is the same
either way:

    x     viscous core radius    cell dx    cells per core radius
   15            0.797            0.154            5.2
   18            0.881            0.328            2.7
   21            0.958            0.524            1.8
   24            1.029            0.701            1.5

A core spreading as sqrt(4 nu t) reaches 1.0 D by x = 24 while the cell reaches 0.7 D. Physical
diffusion is genuinely large at Re = 100 over that convection time, but at 1.5 cells per core
radius a vortex cannot be represented at all, so physical and numerical diffusion cannot be
separated from one run. Grid refinement is the only way to apportion them, which is a second
reason to do the study in 6.3.

**None of this touches the quoted numbers.** St comes from a probe at x = 2 and the forces from
the body surface, both deep inside the plateau.

### 6.4 The far field is clean, and that is a positive result

Same metric applied to the square and to the circular cylinder -- max |u - U_inf| over cells away
from the wake:

| | t | away from wake | outflow face, \|y\|>4 | lateral boundary |
|---|---|---|---|---|
| square `v3` | 380 | 0.0795 | 0.0417 | 0.0000 |
| square `v3_forces` | 415 | **0.0757** | 0.0429 | 0.0000 |
| cylinder | 60 | 0.126 | | |
| cylinder | 80 | **0.187**, still climbing | | |

The square's deviation is FLAT -- slightly down over 35 time units, on a run that reached 535 --
while the cylinder trebles in 50 and does not stop. Same solver, same Dong outflow, same
tolerance. What differs is the boundary topology: the square's outlet is a whole face with the
inlet and sides as separate boundaries, whereas the cylinder's perimeter is CLOSED with 94%
pinned to the free stream and 6% carrying the entire mass balance. See `reference/` notes on the
cylinder for that investigation.

### 6.5 The pictures were wrong before they were right

`plot_farfield.py` interpolated u and v onto a display grid and then took `np.gradient` of that.
`LinearNDInterpolator` is piecewise linear, so its derivative is piecewise CONSTANT and jumps at
every mesh cell edge: differentiating it paints one band per cell. The far wake therefore came
out in vertical stripes past x ≈ 18, widening downstream exactly as the cells widen, and those
stripes were briefly read as a flow feature.

They were not. Native vorticity along y = 1 for x > 16, differentiated on the mesh with the real
metrics, has a node-to-node alternating component of 1.7% of its own rms. Mean
|d(omega)/d(pixel)| past x = 18 falls from 0.00813 to 0.00295 when the order is corrected, and
peak |omega| rises from 26.19 to 31.00 -- the old order was also smoothing the near-body
gradients where the mesh is finest.

**Interpolating a scalar for a picture is fine; differentiating an interpolant is not.**
`plot_vorticity_pipeline.py` is the side-by-side demonstration, kept because the failure is
invisible unless both orders are drawn from the same data.

### 6.6 Time-averaged surface pressure

From the mean field accumulated over 3500 samples (5.2 periods), with the rms taken from eight
snapshots spaced uniformly over one period. The two are cross-checked: the eight-phase mean
reproduces the 3500-sample mean to **0.0039**.

| face | s | mean C_p | rms C_p over the cycle |
|---|---|---|---|
| front | 3.5–4.0, 0–0.5 | **+0.7965** | 0.0202 |
| top | 0.5–1.5 | −1.1501 | 0.0813 |
| base | 1.5–2.5 | **−0.7045** | 0.0473 |
| bottom | 2.5–3.5 | −1.1460 | 0.0813 |

Two checks come free with this table. The front stagnation reads **+1.0725 against the exact
+1**, so the pressure field and its reference are good to 7%. And the front-minus-base
difference is **+1.5010** against a C_D of **1.4529** measured independently by surface
integration -- they agree to 3.3%, which is the residue of friction drag and of the spurious
viscous normal stress documented in `src/forces.py`. The top and bottom faces carry equal mean
C_p to 0.4%, as a time-averaged symmetric configuration must.

The rms is four times larger on the side faces (0.081) than on the front (0.020): the shedding
modulates the separated shear layers, not the stagnation region.

### 6.7 The figures

All under `figures/`, all regenerable from the scripts named beside them.

| figure | what it shows | script |
|---|---|---|
| `sqcyl_vs_data.png` | surface `C_p` and the integral quantities against Sohankar et al. at matched 5% blockage, with their own grid spread drawn as the band | `plot_utility/plot_square_vs_data.py` |
| `sqcyl_cp_spectra.png` | time-averaged `C_p` with cycle rms, the force records, and the `C_L`/`C_D` spectra with the FFT bin drawn honestly | `plot_utility/plot_square_cp_spectra.py` |
| `sqcyl_half_period_symmetry.png` | `C_L(t+T/2) = -C_L(t)` and `C_D(t+T/2) = +C_D(t)` over 3200 samples, residuals 6e-06 and 6e-05 | `plot_utility/plot_square_half_period.py` |
| `sqcyl_shedding_cycle.png` | one period at eight phases, with `C_L` and `C_D` integrated from each panel's own field | `plot_utility/plot_square_shedding_cycle.py` |
| `sqcyl_v3_forces_nearfield.png` | near-field vorticity over pressure — the shear layers rolling up, and each shed core as a suction minimum | `plot_utility/plot_square_nearfield.py` |
| `sqcyl_v3_forces_farfield.png` | the whole domain, corrected vorticity pipeline | `plot_utility/plot_farfield.py` |
| `sqcyl_vorticity_pipeline.png` | the stripe artefact, both orders of operation on identical data | `plot_utility/plot_vorticity_pipeline.py` |

Data behind them: `results/sq_surface_cp.npz`, `results/sq_force_spectra.npz`, and the nine
phase checkpoints `results/fields/sqph_00..08.npz`. Reference values with provenance live in
`reference_data.py`.

The older `sqcyl_phase01-05` files are superseded — 63,280 cells, no grid fingerprint, spaced 3
time units apart, which is not a fraction of `T = 6.72`.

## 7. Consequences for the rest of the repo

**Any marginally-unstable case run at `tol=1e-4` is suspect.** The BFS results are probably
sound — those were validated as genuinely steady flows — but *"it converged to steady"* is no
longer evidence that steady was the right answer. That inference is exactly what failed here.

`linear_solver_tolerance.md` now carries the caveat: **1e-4 for steady, 1e-6 or tighter for
anything that oscillates.** If you do not know which you have, use 1e-6 — a steady run merely
gets slower, whereas an unsteady run at 1e-4 returns a plausible wrong answer.

Tight tolerances are affordable with AMG, whose iteration count is O(1); Jacobi's goes from 48 to
~1568 between 1e-4 and 1e-8, which is why this trade looked far worse than it is.

Artifacts: `figures/sqcyl_spark_vortex_street.png`, `results/sqcyl_spark_history.npy`,
`results/fields/sqcyl_spark.npz`, logs in `results/logs/`.
