# LES model and resolution study — Taylor–Green at Re = 400 against DNS

Completed 2026-09-04. Ten runs: two grids × five configurations, judged against an in-house
spectral-element DNS. Every number below is measured.

---

## 1. The reference, and why it can be trusted

`results/tgv_diag_re400.npz` — 6×6 spectral elements at order 8 with Nz = 48, i.e. **48 degrees
of freedom per direction**, 1313 samples to t = 15.

It carries **`−dE/dt / 2νΩ = 1.0000`** throughout: the scheme removes no energy of its own. That
property, not the resolution number, is what makes it usable as a reference.

**Our 48³ is NOT comparable to its 48³.** An order-8 spectral element resolves roughly 2.5–3×
more per direction than second-order central differences, so our 48³ sits well *below* the
reference in resolving power — nearer an equivalent 16–20³. That is the correct setup for an LES
test, a coarse run judged against a resolved one, but quoting "48³ vs 48³" would imply a parity
that does not exist.

---

## 2. Results

| grid | model | outcome | mean \|err\| | max |
|---|---|---|---|---|
| 48³ | **Smagorinsky** | **reached t = 15** | **1.58%** | 4.16% |
| 48³ | WALE | diverged t = 6 | — | — |
| 48³ | Vreman | diverged t = 8 | — | — |
| 48³ | σ | diverged t = 8 | — | — |
| 48³ | none | diverged t = 5 | — | — |
| 64³ | **σ** | **reached t = 15** | **0.94%** | 3.35% |
| 64³ | **Vreman** | reached t = 14 | 1.06% | 3.92% |
| 64³ | **Smagorinsky** | reached t = 14 | 1.25% | 4.00% |
| 64³ | WALE | reached t = 15 | 3.39% | 9.59% |
| 64³ | none | **diverged t = 6** | — | — |

---

## 3. Four findings

### 3.1 The model is mandatory, not an accuracy refinement

**The unmodelled run diverges at BOTH resolutions** — t = 5 at 48³ and **t = 6 at 64³, exactly
at the DNS dissipation peak.** It reaches +12.4% energy with negative dissipation: it cannot
produce the dissipation the peak demands, grid-scale energy accumulates, and the run is lost.

The gap between having a closure (≈1%) and not having one (divergence) is far larger than the
gap between the best and worst working closure (0.94% vs 3.39%).

### 3.2 Robustness and accuracy rank OPPOSITELY, and resolution decides which matters

| model | 48³ | 64³ |
|---|---|---|
| Smagorinsky | **only survivor** | mid-pack, 1.25% |
| σ | diverged t = 8 | **best, 0.94%** |
| Vreman | diverged t = 8 | 1.06% |

**σ is the worst model at 48³ and the best at 64³.** A single-resolution study would have drawn
the wrong conclusion whichever grid it used.

The mechanism is visible in the resolved enstrophy as each 48³ run failed:

```
   t   smagorinsky        vreman            sigma
      nu_t/nu  2nuZ    nu_t/nu  2nuZ    nu_t/nu  2nuZ
   6   0.290  0.00810   0.287  0.00889   0.282  0.00899
   8   0.294  0.00805   0.333  0.01708   0.305  0.01264
   9   0.272  0.00660   0.369  0.02572   0.332  0.02398   (DNS: 0.01098)
```

Smagorinsky's resolved enstrophy **turns over and decays with the flow**. Vreman's and σ's climb
monotonically to more than twice the DNS, while their `ν_t` barely responds.

**`ν_t = (C_sΔ)²|S|` has nothing that switches it off**: grid-scale energy steepens gradients,
`|S|` rises, damping rises in proportion. WALE, Vreman and σ are all built to be *selective* —
to vanish in pure shear, rotation, two-dimensional flow. That selectivity is exactly what removes
the safety net, because grid-scale oscillations often have locally simple structure. **Selectivity
trades robustness**, and the price is paid only when the grid is too coarse.

### 3.3 WALE over-dissipates, for a reason derived before it was observed

WALE is the outlier at both grids: **3.39% at 64³ against 0.94–1.25% for the others**, and the
first to diverge at 48³. Its `ν_t/ν` is 0.53 at t = 3 where the others are 0.21.

**WALE does not vanish in solid-body rotation.** `S^d` is the traceless symmetric part of `g·g`,
which for `u = ω × r` is `diag(−1,−1,2) ω²/3`, so `S^d:S^d = (2/3)ω⁴` while `S:S = 0`. Responding
to the rotation rate is the entire reason it is built from `g·g` rather than from `S`. Transitional
TGV is full of coherent vortices, so WALE produces eddy viscosity where there is no subgrid
content to model. It overshoots the peak, spends the energy early, and then runs **below** the DNS
through t = 9–12.

A first draft of this repo's WALE docstring claimed it vanishes in rotation; the analytic test
written to check that claim failed, and the corrected test now asserts the exact non-zero value.

### 3.4 All algebraic models under-dissipate on the post-peak plateau

The DNS holds ε ≈ 0.0110 from t = 7 to t = 9 and then rolls off. Our runs decay monotonically
from t = 7, opening a deficit of **−17% at t = 9** that closes again by t = 11.

```
    t    DNS eps    sigma tot   resolved    model   deficit
    7    0.01140     0.01181     0.00855   0.00326    +3.6%
    9    0.01098     0.00912     0.00689   0.00223   -17.0%
   11    0.00788     0.00765     0.00562   0.00203    -3.0%

  from t=7 to t=9:  resolved -19%,  model -32%,  DNS total -4%
```

**All three surviving models do this, within a few percent of each other** — so it is structural,
not a constant needing tuning.

**It is a non-equilibrium failure.** An algebraic eddy viscosity is a LOCAL-EQUILIBRIUM closure:
`ν_t ∝ Δ²|S|` assumes the subgrid dissipation equals the instantaneous cascade rate set by the
resolved strain. The plateau is precisely where that fails — the large scales have stopped feeding
the cascade, but energy already IN the small scales keeps dissipating. The subgrid field has
memory and outlives the strain that created it. The DNS carries that memory explicitly; an
algebraic model cannot, so `ν_t` falls with the resolved strain (σ: 0.151 → 0.133 across the
plateau) when the true subgrid dissipation does not.

**This is the argument for a one-equation model** carrying a transport equation for the subgrid
kinetic energy `k_sgs`, which would give the closure exactly the memory it lacks.

---

## 4. What this does and does not establish

**Established.** The code runs an LES of a transitional, periodic, moderate-Re flow and
reproduces integral quantities against a resolved DNS to **1–3.4%**, with four different closures
and at two resolutions. The scheme's own numerical dissipation is separately quantified at 2.2%
at 40³ and converging at second order (`test_numerical_dissipation.py`), so the models are not
fitting the discretisation's error.

**Not established, roughly in order of importance:**

* **No wall-bounded case.** Every LES run is a periodic box. No wall model, no near-wall
  resolution study, no channel. Walls are where LES is hardest and where most practical LES lives.
* **This is not turbulence.** `Re_λ ≈ 45` at peak — transitional, with no inertial range. The
  models have been tested on whether they switch on at the right *moment*, not on sustaining a
  cascade.
* **Only integral quantities compared.** E(t) and ε(t), never a spectrum. The total energy is
  right; whether it sits in the right *scales* is unknown.
* **The stability envelope is narrow.** Four of ten configurations diverged. Re = 800 failed at
  every resolution tried.
* **A hard compute ceiling.** 64³ is 9.7 s/step on one core; a 10⁵-step channel at 10⁶ cells is
  45 days serial, 10 days on the GPU.

**The single most informative next test is a wall-bounded channel at Re_τ = 180** against the
Kim–Moin–Moser DNS: it exercises walls, sustained turbulence and near-wall model behaviour at
once, and has a published reference. Second is an E(k) diagnostic on the existing TGV runs — an
hour's work, and it would show whether the energy is in the right places.

Figure: `figures/tgv400_vs_dns.png`. Full citations: `reference/bibliography.md`.
