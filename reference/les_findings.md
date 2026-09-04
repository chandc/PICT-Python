# LES: what has been learned, and at what cost

Running log of measured findings, 2026-09-03. Every number here came out of this repo or out of
the reference DNS it is compared against; nothing is quoted from memory.

---

## 1. The reference

`~/Dropbox/Apple_MLX_CFD/sem_demo/scratch/tgv_diag_re800_88.npz` — Taylor–Green at **Re = 800**,
spectral-element, 11×11 elements at order 8 with Nz = 88 (~88³ effective), 2216 samples to
t = 4π.

It self-certifies: **`−dE/dt / 2νΩ = 1.0000`** throughout, i.e. the scheme removes no energy of
its own. That is the property that makes it usable as a reference at all, and it is worth more
than the resolution number.

Volume-averaged (their file stores integrals; divide by (2π)³ = 248.05):

| t | E | 2νZ |
|---|---|---|
| 0 | 0.125000 | 0.000938 |
| 2 | 0.122849 | 0.001394 |
| 4 | 0.118303 | 0.003644 |
| 6 | 0.106061 | 0.008074 |
| 8 | 0.087377 | 0.011327 |
| **8.93** | — | **0.012299 (peak)** |
| 12 | 0.047202 | 0.007139 |

---

## 2. Two bugs the reference caught immediately

### The box was the unit cube, not [0, 2π]³

```
  reference and analytic:  E(0) = 0.125000   Z(0) = 0.375000
  ours:                    E(0) = 0.125000   Z(0) = 14.058875
```

Writing TGV as `sin(2πx)` on a unit box gives the same velocities at wavenumber 2π instead of 1.
**The energy is identical, so an energy check alone passes**; enstrophy and dissipation are
(2π)² = 39.5 times too large. A whole sweep was run and analysed before this was noticed, and it
was noticed only because a reference existed to compare the INITIAL STATE against.

### And a second, inherited, unit-box assumption

Scaling the box exposed `Block`'s default `period=(1,1,1)`, which is what `pad_coords` adds when
wrapping a periodic ghost. Coordinates scaled, period left at 1 → the wrapped ghost landed a
whole box short, Jacobian 1/(2π)³, E(0) reading 0.147 with an error **that did not converge under
refinement** — the signature of a geometry bug rather than a discretisation one.

Now exact: E(0) to 1e-14 at every resolution, Z(0) converging at second order (2.24× and 1.78×
against 2.25 and 1.78).

**Lesson.** Check the initial state against an exact value before running anything. It costs one
evaluation and it caught two bugs that a converged-looking sweep did not.

---

## 3. A diagnosis I got wrong twice

I attributed the first sweep's failure to a cell Péclet number of 157, computed with
`dx = 2π/n` when the actual spacing was `1/n`. **The real Pe was 25** and the failure was the
39.5× excess dissipation.

With the geometry fixed the Péclet numbers apply, and they are now confirmed:

| grid | dx | Pe at Re = 800 | outcome |
|---|---|---|---|
| 32³ | 0.196 | **157** | all three models blew up |
| 48³ | 0.131 | 105 | stable except WALE |
| 64³ | 0.098 | 79 | stable so far |

So Pe ≈ 157 is where this scheme stops working on this flow — a measured limit, arrived at
after two wrong statements about it.

---

## 4. The scheme's own dissipation, which bounds everything else

`test_numerical_dissipation.py`, TGV at Re = 800:

```
  grid    -dE/dt      2 nu Z     numerical    share
  24^3   8.641e-04   9.185e-04  -5.442e-05   -6.3%
  32^3   8.964e-04   9.280e-04  -3.157e-05   -3.5%
  40^3   9.119e-04   9.324e-04  -2.050e-05   -2.2%

  24->32 falls 1.79x (second order predicts 1.78x);  32->40 falls 1.57x (1.56x)
```

**2.2% at 40³, converging at second order.** Any model appearing to supply ~2% of the
dissipation is supplying the discretisation's error, not physics. This number must be quoted
beside any model comparison.

---

## 5. Model behaviour, measured

At 48³ against the reference:

| t | reference E | no model | err | WALE | err |
|---|---|---|---|---|---|
| 1 | 0.124034 | 0.123775 | −0.21% | — | — |
| 2 | 0.122849 | 0.123338 | **+0.40%** | 0.121526 | **−1.08%** |
| 4 | 0.118303 | 0.121420 | **+2.63%** | 0.114759 | **−3.00%** |

**The unmodelled run does exactly what an under-resolved LES should** — holds too much energy,
with the error growing steadily, because it cannot represent the cascade that carries the
dissipation.

**WALE over-corrects, then destabilises.** At t = 4 it was −3.00% with `ν_t/ν = 1.1` — an eddy
viscosity exceeding the molecular one while the flow had not yet broken down. At t = 5 the
energy ROSE, with dissipation −0.0188. At t = 6 it was NaN. The reported `ν_t/ν = 2.6e12` is the
consequence, not the cause.

**Why, and it was derived before it was observed.** WALE does not vanish in solid-body rotation:
`S^d` is the traceless symmetric part of `g·g`, which for `u = ω × r` is `diag(−1,−1,2) ω²/3`,
so `S^d:S^d = (2/3) ω⁴` while `S:S = 0`. Transitional TGV is full of coherent vortices, so WALE
produces eddy viscosity where there is no subgrid content to model. A first draft of the
docstring claimed WALE vanishes in rotation; the analytic test written to check that claim
failed, and the corrected test now asserts the exact non-zero value.

Smagorinsky is quieter (`ν_t/ν = 0.427` at t = 3) and stable so far — unsurprising, since
Lilly's `C_s ≈ 0.17` was derived by matching the subgrid dissipation to a Kolmogorov cascade,
making homogeneous isotropic turbulence its home ground.

---

## 6. What was added because of this

`--model {none, sigma, smagorinsky, vreman, wale}`, driven off the `MODELS` registry.

**Vreman (2004)** and the **σ-model (Nicoud et al. 2011)** were added specifically to switch off
when there is nothing to model. Both are algebraic, need only the velocity gradient already
computed, and need **no test filter** — unlike a dynamic procedure, which would require spatial
filtering across multiblock seams.

`test_sgs_models.py`, 7/7, every target derived rather than tabulated:

```
  pure shear a = 2.5     vreman 0.00e+00,  sigma 0.00e+00   (Smagorinsky 5.02e-04)
  solid rotation         sigma  0.00e+00                    (WALE non-zero by design)
  solid rotation         vreman 5.843452e-04  vs analytic c D^2 omega/sqrt(2) = 5.843452e-04
  filter width           nu_t falls exactly 4.0000x when the cell halves
  WALE near-wall         exponent 2.997 against the required 3
  1 block vs 4 blocks    difference 0.00e+00
```

---

## 7. Open, and how it would be closed

* **The verdict window is t = 6–10**, around the reference's dissipation peak of 0.012299 at
  t = 8.93. Everything above is from t ≤ 5, before the flow breaks down, so "Smagorinsky looks
  best" may only mean "least active during a phase when inactivity is correct".
* **Vreman and σ may fail the opposite way** — a model that vanishes in shear and rotation might
  also produce too little at the peak. Same number decides it.
* **64³ is the resolution that matters** and is slowest; 32³ is unusable at this Re.
* **A dynamic procedure** is the textbook answer to transition-plus-turbulence and is not
  implemented. It needs a test filter, which is the multiblock halo problem — already solved for
  convolutions in `src/sgs_net.py` and reusable.

Full citations: `reference/bibliography.md`.
