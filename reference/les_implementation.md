# Large-eddy simulation — what was already here, what was added, what is proven

Written 2026-09-03. Every number is measured by `test_sgs_models.py` or `run_les.py` at this
commit.

---

## 1. The two hard prerequisites were already satisfied

An LES needs two things from its host solver that are easy to get wrong and hard to notice.

**A variable-viscosity operator that is actually second order.** `nu` may be a per-block array
throughout `MultiBlockPISO`; the four sites that previously assumed a scalar — the momentum
matrix, the deferred cross term, the rotational pressure update and the Dong outflow — are all
field-aware. Validated by MMS with a spatially varying `nu` at **order 1.96–1.99**, and against
a manufactured van Driest channel to **0.126%**. A first-order or cell-valued face coefficient
would show as a rate near 1, which is exactly what a constant-`nu` test cannot detect:
interpolating a constant is exact however you do it (`measurement_traps.md` §5).

**A convective operator that conserves kinetic energy.** For inviscid periodic flow
`dE/dt = 0`, which discretely requires the convective operator to be **skew-symmetric** in the
discrete inner product. `test_energy_conservation.py` measures
`P = sum_c u_c . (C u_c) dV` and requires it to vanish — 6/6. This is the property that decides
whether a scheme can carry a cascade or merely bleeds energy through its own truncation error,
in which case the numerics compete with the SGS model they are supposed to be calibrating. Every
other test in the repo — MMS, Poiseuille, the duct, Ghia, Stokes — is linear or steady and
cannot see it.

**Neither of these was built for LES.** They came from other work and happened to be the right
foundations, which is why the remaining gap was smaller than expected.

---

## 2. What `src/sgs.py` adds

| function | what it does |
|---|---|
| `velocity_gradient(d, u, v, w)` | `[block][i][j] = du_i/dx_j`, seams resolved by `MultiBlock.gradient` |
| `strain_rate(g)` | `S_ij = ½(g_ij + g_ji)` |
| `strain_magnitude(g)` | `\|S\| = sqrt(2 S_ij S_ij)` |
| `filter_width(d, b)` | `Δ = (cell volume)^(1/3)` |
| `smagorinsky(...)` | `nu_t = (C_s Δ)² \|S\|`, optional van Driest damping |
| `wale(...)` | Nicoud & Ducros (1999) |
| `effective_viscosity(...)` | `nu_eff = nu_mol + nu_t` as the dict the solver takes |

`MultiBlockPISO.set_nu()` is the supported way to replace the viscosity between steps. Assigning
`self.nu` directly does not work: `_nu_field` is decided at construction from the *type* of `nu`,
so a solver built with a scalar would treat a dict as a scalar and multiply an array by it.

**Seams are handled by the domain, not by this file.** Every derivative goes through
`MultiBlock.gradient`, the same exchange the solver's own operators use and the same one a
distributed version would send over MPI. `src/sgs_net.py` was changed for the same reason: its
`pad_mode="none"` consumes a caller-supplied halo instead of inventing a circular boundary.

### The models

```
Smagorinsky   nu_t = (C_s Δ)² |S|                                    C_s = 0.17

WALE          g² = g·g,   S^d = ½(g² + g²ᵀ) − ⅓ tr(g²) I
              nu_t = (C_w Δ)² (S^d:S^d)^{3/2} / ( (S:S)^{5/2} + (S^d:S^d)^{5/4} )
                                                                     C_w = 0.55
```

---

## 3. Two things I got wrong, and how they were caught

Both were found by analytic test targets, not by reading the code, which is the argument for
writing the targets first.

### The filter width is not `J^(1/3)`

`block_metrics` returns `J` as the ratio of physical to **computational** volume, and the
computational grid is normalised, so on a uniform box `J = 1` at every resolution. Using
`Δ = J^(1/3)` made `nu_t` **resolution-independent** — refining 8 → 16 left it unchanged where
it must fall by 4. The physical cell volume is `J · h_ξ h_η h_ζ`. After the fix the measured
ratio is **4.0000**.

This matters far more on a stretched grid than on the uniform box that exposed it: the cylinder
O-grid spans a factor of 47 in cell size between the wall and the far field, and a hardwired
uniform `Δ` would be silently wrong everywhere.

### WALE does not vanish in solid-body rotation

The first draft of the docstring claimed it did, and the test encoded that claim and failed. For
`u = ω × r`:

```
  S = 0                     rotation has no strain
  g² = diag(−ω², −ω², 0),   tr = −2ω²
  S^d = diag(−1, −1, 2) ω²/3
  S^d:S^d = (2/3) ω⁴        NOT zero
  nu_t = (C_w Δ)² (S^d:S^d)^{1/4}
```

Responding to the **rotation rate** is the entire reason the model is built from `g·g` rather
than from `S`. The test now asserts that exact value — `3.226926e-03` measured against
`3.226926e-03` analytic — which is a stronger check than the wrong assertion of zero it
replaced.

---

## 4. Test cases and results

`test_sgs_models.py`, **6/6**. Every check has an answer known in advance.

| # | case | target | measured |
|---|---|---|---|
| 1 | pure shear `u = (a y, 0, 0)`, `a = 2.5` | `\|S\| = a` exactly | error **1.78e-15** |
| 2 | solid rotation `ω = 1.7` | Smagorinsky `nu_t = 0` | **3.6e-19** |
| 2 | same | WALE `= (C_w Δ)² ((2/3)ω⁴)^{1/4}` | **3.226926e-03** vs 3.226926e-03 |
| 3 | random field | `nu_t ≥ 0`, finite, both models | min **7.0e-04** / **6.5e-06** |
| 4 | refine 8 → 16 | `nu_t` ratio `= 4` (`Δ²`) | **4.0000** |
| 5 | WALE near a wall | exponent **3** | **2.997** |
| 6 | one block vs four | identical | **0.00e+00** |

Check 2 is the one that separates a model built on `\|S\|` from one built on `\|grad u\|`.
Check 6 is the one that would fail if the gradient did not resolve the seam.

Check 5 is worth stating precisely, because this repo got the neighbouring fact wrong once:
WALE gives `nu_t ~ y³` approaching a wall, which is the correct asymptotic and is why it needs
no van Driest damping. **Van Driest damping applied to the mixing length gives `y⁴`, not `y³`**
— measured elsewhere in this repo at 3.993 — so "van Driest fixes the near-wall scaling" is
false as it is usually said.

---

## 5. The runnable

`run_les.py` — Taylor–Green in a periodic box, `nu_eff` rebuilt every step from the current
field.

```
  python run_les.py --model wale --n 16 --blocks 2 --steps 200

  model         final E     nu_t/nu mean
  none          0.097987     --
  smagorinsky   0.091459     ~0.4
  wale          0.084668     0.592

  seam invariance of the WHOLE LES step:  1 block 0.103021,  4 blocks 0.103021
  projected divergence 3.11e-15
```

**The viscosity ratio is reported alongside the energy on purpose.** A decaying energy curve on
its own cannot distinguish a working SGS model from a numerically dissipative scheme; `nu_t/nu`
says whether the closure is doing anything — near zero and it is decoration, enormous and it has
swamped the physics.

The driver refuses a non-positive `nu_eff` rather than letting the momentum solve diverge in a
way that looks like a physical instability. Both closures are non-negative by construction and
check 3 verifies it, but the guard costs nothing and the failure mode it prevents is one this
repo has spent days misdiagnosing before.

---

## 6. What this does and does not establish

**Established:** the pieces compose, the closures are correct against analytic targets, the
filter width follows the grid, the whole LES step is seam-invariant to the last digit, and the
models order sensibly by dissipation.

**Not established: this is not a validated LES of anything.** Taylor–Green at 16³ is a
transitional flow, not a developed cascade, and the run is short. Nothing here has been compared
against turbulence data.

**What would make it validated**, in increasing cost:

* **Decaying isotropic turbulence** against the Comte-Bellot & Corrsin spectra — the standard
  first LES benchmark, and the one that tests whether the model dissipates at the right *rate*
  rather than merely dissipating.
* **Turbulent channel at `Re_τ = 180`** against DNS. The repo already has the channel geometry,
  the van Driest reference profile and a validated variable-`nu` operator; what it lacks is
  spanwise extent and resolution. Present cases run `nz = 4`, which is not LES.
* **A dynamic procedure** (Germano–Lilly), which needs a test filter — a spatial filter at
  multi-block seams, i.e. the same halo problem, already solved for convolutions in
  `src/sgs_net.py` and reusable.

**And the eddy-viscosity network does not exist yet.** `TinySGSNet` outputs a momentum *source*,
not `nu_t`. An eddy-viscosity closure needs a non-negative `nu_t` entering the *matrix*, which is
a different path — and the reason Stage 5c cared about differentiating through `A`. The
machinery for it is now all present; the network head is not.

---

**Figures.** `figures/tgv400_vs_dns.png` (models and resolutions against DNS) and
`figures/tgv400_dissipation_split.png` (the post-peak non-equilibrium deficit), both described in
`reference/les_model_study.md`.

Full citations for every source named here: `reference/bibliography.md`.
