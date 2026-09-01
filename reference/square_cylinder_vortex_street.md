# Vortex street behind a square cylinder — and the one setting that decided it

**Result: St = 0.1467 ± 0.0067 at Re = 100**, against a computational consensus of 0.145–0.150
at ~5% blockage. Saturated limit cycle, amplitude 0.62, stationary to +0.1% over 28 shedding
periods.

**The whole difference between a vortex street and a steady solution was the linear solver
tolerance: 1e-4 → 1e-6.** Everything else in this document is either a prerequisite for the run
to work at all, or a wrong hypothesis worth recording so nobody repeats the detour.

---

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

Landing at 0.1467 with 5% blockage is real evidence the solver, grid and outflow all behave. But
**one Strouhal number at one Reynolds number is weak validation** — it is a single scalar and
several errors could cancel into it.

Note the reference value is blockage-dependent, which makes "the expected St" ambiguous:
~0.145–0.150 for 2D simulations at low blockage, 0.13–0.14 for experiments and higher blockage.
Quoting 0.13 against a 5%-blockage grid — which this project did for a while — makes a good
result look 13% wrong.

Stronger checks, none of them run yet:

* **St vs Re over 60–200** — the curve shape is a far harder target than one point
* **grid convergence** — St at 1.5× and 2× resolution should move well under the 0.0067 FFT bin
* **C_D ≈ 1.4–1.5** at Re = 100 — an independent number testing surface forces rather than a
  wake probe

The FFT bin width (0.0067) currently dominates the uncertainty, and that is only run length:
2× longer halves it.

---

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
