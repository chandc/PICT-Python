# Handling skew unstructured grids — literature survey and what it did to our solver

Five parallel literature searches, run 2026-09-21, against the open T5 Stokes failure. Every
number attributed to "measured" below was produced by this solver on this machine; everything
else is cited. Where a source is contradicted by our own measurement, the measurement wins and
the contradiction is recorded rather than smoothed over.

## 0. The problem this was commissioned to explain

Manufactured steady Stokes, `u = sin(pi x)cos(pi y)`, `v = -cos(pi x)sin(pi y)`,
`p = cos(pi x)cos(pi y)`, nu = 1, velocity Dirichlet everywhere, pressure all-Neumann.

| n | L2(u) | order |
|---|---|---|
| 16 | 1.2802e-02 | -- |
| 32 | 1.2227e-02 | 0.07 |
| 64 | 1.2080e-02 | 0.02 |

The fixed point is genuine: it satisfies its own discrete momentum equation to 8e-11 and has
`max|div F|` = 1e-15. It is simply the wrong fixed point, by a mesh-independent amount. In
isolation every operator is second order -- momentum with the exact pressure 1.99, Poisson 1.99.
Only the coupled system is zeroth order.

## 1. Bugs the survey found, all fixed

| # | bug | how it was wrong | effect |
|---|---|---|---|
| 1 | SIMPLE correction coefficient | `u' = -(V/a_P) grad p'` uses the bare diagonal; a viscous operator has zero row sum so its contribution cancels out of the true response while inflating `a_P`. Measured `a_P` = 6.06 against a row sum of 0.058594 = `a_t V` exactly, so D was **92x too small** | diverged ~32x/step for any nu >= 0.1; SIMPLEC (`a_P - sum a_N`, the row sum) made it stable from nu = 0.01 to 10 |
| 2 | `_decompose` split | docstring said OVER-RELAXED and justified it with "\|E_f\| GROWS with skewness", but the formula `E_f = d(d.S)/(d.d)` is MINIMUM-CORRECTION, where \|E_f\| shrinks. Deferred contraction factor is then **tan(theta)**, not sin(theta) | divergent by construction beyond 45 deg. Dormant while every test mesh was near-orthogonal; NaN on the first wall-clustered cavity run (worst face 69 deg, tan = 2.65) |
| 3 | Rhie-Chow compact vs wide | `(p_N-p_P)/\|d\|` is `grad p . dhat`; `gpf . nhat` is `grad p . nhat`. Different directional derivatives, equal only where d \|\| S | O(1) contamination on any non-orthogonal face. Invisible at cluster=0; fixed to the over-relaxed form per Liu et al. (2024) Eq. 30-33, verified bit-identical on the orthogonal mesh |
| 4 | face interpolation weight | `w = dn/(do+dn)` using raw distances to the face centre. OpenFOAM and code_saturne both use the NORMAL-PROJECTED form; the raw version appears in OpenFOAM only as the degenerate-face fallback | also inconsistent with the skewness correction, which assumes the interpolation targets the centroid-line/face intersection. `max\|w_new - w_old\|` = 0.000 uniform, 0.088 perturbed, 0.035 clustered |
| 5 | k-exact LSQ conditioning | quadratic basis mixes O(h) and O(h^2) columns, so the condition number doubled per refinement (43.5 / 87.1 / 174.1) | column equilibration fixes it (1.766 at every level). Results came back BIT-IDENTICAL, so it was never the cause of anything |

Fix 1 is the only one that changed the solver's behaviour qualitatively. Fixes 2-4 are correctness
issues that were dormant on uniform meshes and become load-bearing the moment the mesh is
clustered or skewed -- i.e. exactly when we go after boundary layers or the cylinder.

## 2. Hypotheses killed by measurement

Recorded so they are not re-tried. Each of these was stated with some confidence at some point.

* **Gradient accuracy.** The face-neighbour WLSQ gradient is first order (measured 0.97, 0.98,
  0.99, interior cells included). **This is the correct, published answer, not a bug.** Syrakos et
  al., Phys. Fluids 29, 127103 (2017), Table 5: LS is order 1 on arbitrary grids. Their Eq. 35 and
  Appendix A show second order arises only by cancellation, needing either opposite-direction
  neighbour pairs or equal angles with **more than three** neighbours -- a cell-centred triangle
  has exactly three at ~120 deg, the one configuration where both provably fail.
  Our own check: analytic vs discrete LSQ `grad p` in the momentum source gave 1.47e-03 vs
  1.52e-03, **both order 1.99**. That is supraconvergence (Kreiss, Manteuffel, Swartz, Wendroff &
  White, Math. Comp. 47 (1986) 537) and it is evidence the solver is healthy, not sick.
* **Wider stencil.** A LINEAR fit on the wider vertex stencil is *worse* (1.17e-01 vs 4.90e-02)
  and has NEGATIVE order in the coupled solver. Width alone buys nothing; only polynomial degree
  helps. Corroborated: Sozer et al. measured compact-LSQ error below extended-LSQ; Diskin et al.
  measured that face-node stencil expansion "does not improve gradient approximation".
* **k-exact gradient as the fix.** Quadratic k-exact on the vertex stencil (order 1.75) cut T5
  error 4.7x and raised order 0.07 -> 0.80, but **diverged at n=64** and never approached order 2.
  The instability is structural, not a bug: a wider gradient is *blinder* to checkerboard, so
  feeding it the pressure correction builds the wide-stencil Laplacian whose kernel contains the
  checkerboard mode. Syrakos §6.1: "the gradient field can be smooth even though phi oscillates
  from cell to cell". Stable in the momentum deferred iteration (explicit source, damped by the
  diagonal), divergent in the elliptic pressure loop (no such damping).
* **Conditioning.** Real flaw, fixed, bit-identical results. Not the cause.
* **Rhie-Chow damping magnitude.** Error scales as ~1/D (3.79e-02 / 1.28e-02 / 5.98e-03 at
  rc_scale 0.1/1/3, diverging at 10) but no value restores order. Syrakos & Goulas describe the
  same knife-edge: too small fails to suppress oscillations, too large "divergence occurs".
* **The damping needing a smoothing gradient.** Keeping the narrow gradient in `dp_wide` still
  diverged at n=64.
* **Corrector count.** n_corr 1, 2, 5, 20, 50 gave a **bit-identical** divergent trajectory.
* **Cell shape at walls.** All 256 wall cells are the identical right triangle, area ratio 1.000.
* **SKEWNESS.** See §4 -- the decisive experiment, and it came back negative.
* **The checkerboard itself.** See §3 -- theory says it contributes only O(h) to velocity.

## 3. The error decomposition that reframes the problem

From the collocated-checkerboard search, the cleanest available analysis. Write the converged
fixed point as `A e_u + G e_p = -tau_M`, `D e_u + B e_p = -tau_C`, with `tau_C` the residual of
inserting the EXACT fields into our assembled operators and `B` the Rhie-Chow damping. Eliminating
`e_u` with `S = -D A^-1 G`:

* **Smooth modes (kh << 1):** `B` is negligible by construction, so `e_p ~ nu tau_C` and
  `e_u ~ A^-1 G e_p ~ (nu k^2)^-1 k (nu tau_C) = tau_C / k`. **The viscosity cancels.**
  `||e_u|| ~ ||tau_C||/k` -- mesh-independent if `tau_C` is mesh-independent.
* **Checkerboard mode (kh ~ pi):** `S ~ 0`, so `e_p^cb ~ B^-1 tau_C^cb`, and feeding back gives
  `e_u^cb ~ O(h) tau_C^cb`, because the viscous operator has eigenvalue nu/h^2 there.

**Consequence: the checkerboard cannot be causing the zeroth-order velocity error, and no amount
of Rhie-Chow tuning will move 1.2802e-02.** The driver is the SMOOTH part of `tau_C`. This is the
single most useful reframing the survey produced, and it retrospectively explains why the
rc_scale sweep improved the error without ever restoring the order.

It also means the right checkerboard metric is not our face correlation but Hopman et al.
(JCP 521:113537, arXiv:2408.06821) Eq. 27, `C_cb = p^T (L - L_c) p / (p^T L p)` -- global,
dimensionless, datum-independent, and computable on the EXACT pressure as a pure operator check.

## 4. The decisive experiment, and its negative result

Eymard, Herbin & Latche, ESAIM M2AN 40 (2006) 501, Remarks 2.1 and 4.4, is the theorem-level
statement of our symptom: the convergence proof requires the mass-balance flux to be a second-order
approximation, which holds **only when the segment joining the two collocation points crosses the
shared face at its barycentre**. "This considerably reduces the generality of the possible
meshings... the second order estimate for the residual associated to the divergence term seems to
be necessary to obtain a first order convergence rate."

Our mesh violates this by exactly h/6 -- and, critically, **h/6 is a fixed fraction that does not
shrink under refinement**, measured identical at n = 16, 32, 64. That is precisely the
non-vanishing skewness Syrakos Eq. 14 identifies as the route to an O(1) truncation error.

The alternating diagonal is what creates it. A same-diagonal split has **exactly zero** skewness
on every face family (measured 0.000000 at all three refinements). So we tested it:

| n | diag=alt (skew h/6) | order | diag=same (skew 0) | order |
|---|---|---|---|---|
| 16 | 1.2465e-02 | -- | 3.1557e-02 | -- |
| 32 | 1.2127e-02 | 0.04 | 3.1271e-02 | **0.01** |

**Zero skewness is 2.5x WORSE and plateaus just as hard.** Skewness is not the cause. The Eymard
condition is necessary for their proof but is not what is binding here, and the h/6 offset -- which
several independent threads of this investigation converged on as the prime suspect -- is
exonerated by direct experiment.

## 5. Literature findings worth keeping regardless

**Our deferred over-relaxed correction is a two-point flux approximation with a lagged patch**,
and TPFA is known-inconsistent on non-orthogonal grids. Diskin & Thomas call the two-point form
the "thin-shear-layer" scheme and state that its "accuracy is zeroth order on general grids"; they
use it only as a defect-correction driver, never as a discretisation. Aavatsmark (Comput. Geosci.
11:199, 2007) shows any two-point stencil is a multipoint stencil for *some* effective tensor --
i.e. on a skew grid it consistently solves a different problem. At our clustered worst face
(81 deg) the lagged term is **6.3x larger than the implicit one**.

**Nobody has published convergence data at our clustered skewness.** Bonaventura & Della Rocca's
"strongly non-orthogonal" test meshes top out at 66.5 deg. The ubiquitous mesh-quality thresholds
("non-orthogonality > 80 won't converge") are vendor folklore; the only quantitative statements
with analysis attached are Diskin's CFL-vs-skew-angle curve and Bonaventura's condition (64).

**Stretched triangles are the wrong way to resolve a boundary layer, and the geometry is
irreducible.** Split an aspect-ratio-A wall quad on its diagonal: the diagonal face normal sits at
arctan(A) from the x-axis while the centroid-to-centroid line runs nearly along the diagonal, so
the non-orthogonality angle -> 90 deg as A -> infinity, *independent of refinement in either
direction*. A quad layer has theta = 0 on both wall-parallel and wall-normal faces at any A. Our
own measurement agrees: cluster beta = 2.0 buys 9 cells in the layer but takes the worst face from
27 deg to 81 deg. Diskin et al.'s grid types (II)/(III) -- on which cell-centred WLSQ face
gradients go **O(1)** -- are *defined* as quads split by a diagonal; the unsplit quad type (I) is
O(h^2) for every scheme they tested. Katz & Sankaran (JCP 230:7670): "for stretched meshes,
cell-centered schemes are favored, with cell-centered prismatic approaches in particular showing
the lowest levels of error."

**The alpha-damping face gradient (Nishikawa)** is the most attractive replacement for our
deferred correction:
`grad u|_f = (grad u_j + grad u_k)/2 + [alpha / |(x_k - x_j).nhat|] (u_R - u_L) nhat`
with `u_L`, `u_R` reconstructed from the cell gradients. The first term is consistent (averaged
physical-space gradients, no two-point stencil); the second **vanishes for linear fields**, so it
cannot touch the order. It is a LINEAR operator in the cell unknowns, so it assembles once as a
fixed sparse matrix -- no lagged iteration, no contraction factor, and an exact adjoint, which our
design requires. Use the edge-midpoint variant (`x_a = (x_j+x_k)/2`), whose leading O(h^2) term
vanishes identically. As skew increases, `1/(nhat . ehat)` *increases damping* rather than
amplifying stiffness.

**Scheme families, ranked for our constraints** (2D, cell-centred collocated, PISO + Rhie-Chow,
every operator an explicit sparse matrix, no solution-dependent switching):

| family | restores 2nd order on skew meshes? | verdict for us |
|---|---|---|
| quad/prism wall layers | n/a -- removes the skew instead | **do this first**, zero numerical risk |
| alpha-damping face gradient | order-preserving by construction | **days of work**, adjoint-exact |
| wall-distance-mapped LSQ (FAMLSQ) or Diskin compact stencil (B) | yes, measured at A = 1e5 | geometry-only, adjoint-safe |
| MPFA-O (eta = 1/3 on simplices) | 1.97 potential / ~1.3 flux on triangles | weeks; vertex-ring stencil; monotonicity not guaranteed; the nonlinear fixes would destroy the adjoint |
| FCFV (face-centred) | strongest measured evidence on exactly our failure modes | months -- replaces Rhie-Chow and PISO with a hybridised saddle-point solve |
| DDFV / diamond | L2 yes, H1 falls to 0.5 on degenerate triangles | steal the gradient formula, not the framework; triples unknowns, Stokes variants proved only 1st order, not collocated |
| mimetic FD / VEM | yes on polygons | no mature collocated pressure-correction formulation |
| circumcentre collocation | **no** -- actively counterproductive | wall clustering *creates* the obtuse triangles that put the circumcentre outside the cell; Domelevo & Omnes measured VF4/TPFA **not converging** on flat triangles |

A useful corrective from FVCA5 (Herbin & Hubert), which benchmarked ~20 of these schemes:
"a first remarkable feature of the benchmark is the relative homogeneity of the results... it is
difficult to rate the schemes, and we did not attempt to do so." The exotic machinery buys little
until the mesh is genuinely pathological.

## 6. Disagreements in the literature, flagged

* **Non-orthogonal correction INSIDE the Rhie-Chow term.** Liu et al. (arXiv:2405.20629, Eq. 30-33)
  do it and say it is "required to provide second-order accuracy"; Bartholomew/Denner/van Wachem
  (JCP 375:177 and the arXiv versions) use the bare `(P_Q - P_P)/dx` with no decomposition, while
  applying deferred correction to the shear-stress term. We follow Liu.
* **`O(h^2)` vs `O(h^3/2)` for the Rhie-Chow filter.** Bartholomew's `O(h^2)` is a Taylor argument
  on aligned stencils. Negrini, Parolini & Verani (IJNMF 96:1160, arXiv:2308.01059) prove only
  `O(h^3/2)`, and only on a circumcentric dual. There is no published `O(h^2)` proof for a
  barycentre-collocated general triangulation.
* **"Rhie-Chow ensures the discrete inf-sup condition"** is asserted casually across the applied
  literature. The only place it is actually addressed is Negrini et al., where it is
  **Conjecture 4.2** supported by numerically computed inf-sup constants -- not a theorem.
* **Node-centred vs cell-centred.** Diskin & Thomas: "comparable at equivalent DOF". Katz &
  Sankaran: node-centred better isotropic, cell-centred better stretched. Neither supports
  "switch to node-centred to fix skewness".
* **"CCFV fails on triangles"** (Cortellessa et al., arXiv:2501.00450, measuring simpleFoam
  stagnating at 1e-1 on triangles -- strikingly close to our own symptom) is a statement about one
  code's gradient and interpolation choices, not a theorem about cell-centred FV. Diskin's
  CC-FAMLSQ is second order on far worse grids.
* **Truncation vs discretisation order.** Syrakos says an O(1) truncation error kills the solution
  order; Diskin & Thomas show repeatedly that it does not for conservative divergence-form
  residuals. Both are right about different slots: a flux-difference residual telescopes and can
  supraconverge, a source term does not. Bakhvalov & Surnachev (arXiv:2404.04157) give the sharp
  criterion -- the truncation error must have zero mean over the mesh period.

## 7. What is still open

The zeroth-order coupled error is **not** explained. Excluded by measurement: momentum path,
Poisson path, gradient accuracy, stencil width, conditioning, damping magnitude, corrector count,
cell shape, dt, and now skewness. The error decomposition in §3 says the driver is the smooth part
of `tau_C`, the continuity residual on the exact fields.

Untested diagnostics, in order of expected value:

1. **Measure `tau_C` directly** -- assemble `(1/V_P) sum_f F_f(u*, p*)` with our own flux routine
   on the exact manufactured fields at n = 16/32/64, then split it into smooth and grid-scale
   parts. The analysis predicts the smooth part is `~ k ||e_u|| ~ 1.2e-02 k`. If it does not
   decrease, that is the whole explanation and nothing in the pressure solver will fix it.
2. **Check `G = -D^T` numerically** (random vectors; `u^T G p + p^T D u` should be ~1e-15). Eymard
   et al. build the discrete gradient as the exact transpose of the divergence, which forces a
   *swapped* face weighting for pressure, `(d_L/d)p_L + (d_K/d)p_K` -- the opposite of the usual
   interpolation weight -- and call this "of crucial importance in the analysis of the stability
   of the scheme". If our `G` and `D` are not adjoint, there is no discrete Helmholtz
   orthogonality at all. **This is the cheapest untested item and it is structural.**
3. **Compute Hopman's `C_cb`** on the exact pressure as an operator-consistency check independent
   of any flow solve.
4. **Boundary faces carry no Rhie-Chow damping** (the flux is prescribed from the Dirichlet data),
   so the stabilisation is systematically weaker in boundary cells. With velocity Dirichlet
   everywhere, that is the entire boundary. Acknowledged as unresolved in MOOSE issue #16442,
   which asks whether the `a_P` coefficients need boundary corrections "to maintain second-order
   convergence on non-orthogonal meshes".
5. Test the LSQ weight exponent **q = 3/2** -- Syrakos finds it is the only exponent recovering
   second order via opposite-pair cancellation. One-line change.
6. Our k-exact used `1/d^2` weights; Diskin & Thomas specifically recommend **unweighted** LSQ
   with a quadratic fit, which is the configuration where 2.0 was measured. Likely explains our
   1.75 rather than 2.0.

## 8. Primary sources

* Eymard, Herbin & Latche, *On a stabilized colocated finite volume scheme for the Stokes
  problem*, ESAIM M2AN 40 (2006) 501-527 — http://www.numdam.org/item/M2AN_2006__40_3_501_0/
* Eymard, Herbin, Latche & Piar, *A class of collocated finite volume schemes for incompressible
  flow problems* — https://arxiv.org/abs/2003.05786
* Syrakos, Varchanis, Dimakopoulos, Goulas & Tsamopoulos, *A critical analysis of some popular
  methods for the discretisation of the gradient operator*, Phys. Fluids 29 (2017) 127103 —
  https://arxiv.org/abs/1606.05556
* Syrakos & Goulas, *Estimate of the truncation error of FV discretisation of the Navier-Stokes
  equations on colocated grids*, IJNMF 50 (2006) 103 — https://arxiv.org/abs/1508.02733
* Negrini, Parolini & Verani, *On the convergence of the Rhie-Chow stabilized Box method for the
  Stokes problem*, IJNMF 96 (2024) 1160 — https://arxiv.org/abs/2308.01059
* Hopman, Santos, Alsalti-Baldellou, Rigola & Trias, *Quantifying the checkerboard problem to
  reduce numerical dissipation*, JCP 521 (2025) 113537 — https://arxiv.org/abs/2408.06821
* Bartholomew, Denner, Abdol-Azis, Marquis & van Wachem, *Unified formulation of the
  momentum-weighted interpolation*, JCP 375 (2018) 177 — doi:10.1016/j.jcp.2018.08.030
* Liu et al., *Center-to-face momentum interpolation and face-to-center flux reconstruction* —
  https://arxiv.org/abs/2405.20629
* Diskin, Thomas, Nielsen, Nishikawa & White, *Comparison of node-centered and cell-centered
  unstructured finite-volume discretizations*, AIAA 2009-0597 —
  https://fun3d.larc.nasa.gov/papers/AIAA-2009-0597.pdf
* Diskin & Thomas, *Notes on accuracy of finite-volume discretization schemes on irregular grids*,
  Appl. Numer. Math. 60 (2010) 224
* Sozer, Brehm & Kiris, *Gradient calculation methods on arbitrary polyhedral unstructured meshes*,
  AIAA 2014-1440 — https://ntrs.nasa.gov/api/citations/20140011550/downloads/20140011550.pdf
* Cortellessa, Giacomini & Huerta, *An OpenFOAM face-centred solver for incompressible flows robust
  to mesh distortion* — https://arxiv.org/abs/2501.00450
* Giacomini & Sevilla, *A second-order face-centred finite volume method on general meshes* —
  https://arxiv.org/abs/2005.01663
* Bonaventura & Della Rocca, *Convergence analysis of a cell centred FV diffusion operator on
  non-orthogonal polyhedral meshes* — https://arxiv.org/abs/1806.09180
* Domelevo & Omnes, *A finite volume method for the Laplace equation on almost arbitrary
  two-dimensional grids*, M2AN 39 (2005) 1203 — https://www.numdam.org/item/M2AN_2005__39_6_1203_0/
* Nordbotten & Keilegavlen, *An introduction to multi-point flux (MPFA) and stress (MPSA) finite
  volume methods* — https://arxiv.org/abs/2001.01990
* Aavatsmark, *Interpretation of a two-point flux stencil for skew parallelogram grids*,
  Comput. Geosci. 11 (2007) 199
* Katz & Sankaran, *Mesh quality effects on the accuracy of CFD solutions on unstructured meshes*,
  JCP 230 (2011) 7670
* Bakhvalov & Surnachev, *On the order of accuracy of finite volume schemes on unstructured
  meshes* — https://arxiv.org/abs/2404.04157
* Cubero & Fueyo, *A compact momentum interpolation procedure for unsteady flows and relaxation*,
  Numer. Heat Transfer B 52 (2007) 543
* Jasak, *Error analysis and estimation for the finite volume method*, PhD thesis, Imperial 1996
* Kreiss, Manteuffel, Swartz, Wendroff & White, *Supra-convergent schemes on irregular grids*,
  Math. Comp. 47 (1986) 537
* SU2 PR #2812, pressure-based solver — https://github.com/su2code/SU2/pull/2812
* MOOSE issue #16442, Rhie-Chow coefficients on boundary faces —
  https://github.com/idaholab/moose/issues/16442

## 9. Measured corroboration: wall clustering on the Ghia cavity

The survey's strongest practical recommendation -- resolve the wall layer, and prefer quad layers
to stretched triangles -- is borne out at the same cell count. Re = 1000, 64x64 = 8192 cells,
first cell 0.01562 (uniform) vs 0.00882 (beta = 1.0), wall layer ~ Re^-1/2 = 0.0316:

| mesh | u max\|err\| | u rms | v max\|err\| | v rms | u min | v min | v max |
|---|---|---|---|---|---|---|---|
| uniform, beta = 0 | 0.1138 | 0.0581 | 0.0620 | 0.0351 | -0.4041 | -0.5206 | +0.3873 |
| clustered, beta = 1.0 | **0.0908** | **0.0292** | **0.0463** | **0.0172** | -0.3903 | -0.5241 | +0.3761 |
| Ghia reference | -- | -- | -- | -- | -0.3829 | -0.5155 | +0.3710 |

**Both rms errors halve for free.** Extrema error falls from +5.5 / +1.0 / +4.4 % to
+1.9 / +1.7 / +1.4 %. Note this is at beta = 1.0, where the worst face is 49 deg -- mild enough
that the over-relaxed fix (§1 bug 2) keeps the deferred correction contracting (sin = 0.76 against
the old tan = 1.16, which would have diverged). The uniform mesh was under-resolving the layer
with ~3 cells; beta = 1.0 gives 4 and buys a 2x.

This is the cheapest accuracy available to us and it is orthogonal to the unresolved T5 defect.

## 10. Follow-up diagnostics (run after the survey)

### 10.1 `G` and `D` are not adjoint, and they cannot both be fixed

Test: `sum_P V_P u_P.(G p)_P + sum_P V_P p_P (D u)_P`, which the continuous identity makes zero
when `u.n = 0` on the wall. Random fields, velocity zeroed in boundary cells:

| mesh | T1 | T2 | residual | relative |
|---|---|---|---|---|
| uniform n=32 | -8.8738e-01 | 1.0101e+00 | 1.2267e-01 | **12%** |
| perturbed n=32 | 2.2286e-01 | -5.8475e-01 | -3.6189e-01 | **62%** |
| clustered n=32 | -2.1346e+00 | -4.5329e-01 | -2.5879e+00 | **121%** |
| same-diagonal n=32 | 2.6615e+00 | -2.2204e+00 | 4.4119e-01 | 17% |

So Eymard's duality requirement is violated outright. **But imposing it is not available to us.**
Constructing the exact adjoint `G_adj = -diag(1/V) D^T diag(V)` and testing it on a linear field:

| mesh | LSQ gradient error | `G_adj` error |
|---|---|---|
| uniform n=32 | 2.84e-14 | **2.67e+00** |
| perturbed n=32 | 2.84e-14 | **4.75e+00** |
| clustered n=32 | 1.14e-13 | **6.40e+00** |

on a gradient of magnitude 3.6. The exact adjoint of our plain-interpolation divergence is not a
consistent gradient at all. **Duality and consistency are in direct conflict for this operator
pair**, and Eymard et al. obtain both only because on a circumcentric orthogonal mesh the plain
interpolation is already the adjoint of the natural gradient. This is a structural property of the
mesh/collocation choice, not a coding error, and it cannot be patched by changing weights.

### 10.2 The residuals, measured

`tau_C` -- our own Rhie-Chow flux routine on the exact fields, then divergence:

| n | \|\|tau_C\|\| | order | smooth part | order | grid-scale | order |
|---|---|---|---|---|---|---|
| 16 | 9.6806e-02 | -- | 1.1416e-02 | -- | 1.0141e-01 | -- |
| 32 | 4.8451e-02 | 1.00 | 4.0407e-03 | 1.50 | 4.9616e-02 | 1.03 |
| 64 | 2.4231e-02 | 1.00 | 1.4290e-03 | 1.50 | 2.4524e-02 | 1.02 |

`tau_M` -- momentum residual of the exact fields in our own discrete operator:

| n | `tau_M` (LSQ grad p) | order | `tau_M` (analytic grad p) | order |
|---|---|---|---|---|
| 16 | 3.1337e+00 | -- | 3.1338e+00 | -- |
| 32 | 3.2136e+00 | **-0.04** | 3.2136e+00 | -0.04 |
| 64 | 3.2522e+00 | **-0.02** | 3.2522e+00 | -0.02 |

**`tau_M` is O(1) and does not converge.** By itself that is expected and benign -- it is exactly
the Diskin & Thomas situation ("the truncation errors remain O(1) and the discretization errors
converge with 2nd-order"), and our own momentum-only test with the exact pressure does give 1.99,
so it supraconverges when the pressure is GIVEN.

**The open question is now sharp.** The error decomposition of §3 predicts `||e_u|| ~ ||tau_C||/k`,
which with `tau_C` at order 1.00 (smooth part 1.50) would give a first-order velocity error. We
measure 0.02. So the decomposition is incomplete for this solver: something prevents the O(1)
`tau_M` from supraconverging once the pressure becomes a free variable determined by continuity
rather than a given field.

That is the definition of a **non-pressure-robust** discretisation -- one whose velocity error
depends on the pressure. It is the next literature to read (Linke; John, Linke, Merdon, Neilan &
Rebholz, *On the divergence constraint in mixed finite element methods for incompressible flows*,
SIAM Review 59 (2017) 492), and it is consistent with every measurement here: momentum alone is
clean, continuity alone is clean, the coupling is not, and the failure is worst exactly where the
pressure gradient is largest relative to the viscous term.

## 11. Wall-clustering sweep, completed

Full sweep at fixed cell count (8192), Re = 1000, against Ghia. `worst` is the largest face
non-orthogonality angle; `BL` counts cells inside the Re^-1/2 = 0.0316 wall layer.

| beta | orth min | worst | BL | u max\|err\| | u rms | v max\|err\| | v rms | u_min | v_min | v_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.894 | 26.6 | 3 | 0.1138 | 0.0581 | 0.0620 | 0.0351 | -0.4041 | -0.5206 | +0.3873 |
| 1.0 | 0.652 | 49.3 | 4 | 0.0908 | 0.0292 | 0.0463 | 0.0172 | -0.3903 | -0.5241 | +0.3761 |
| 1.5 | 0.353 | 69.3 | 6 | 0.0914 | 0.0260 | 0.0409 | 0.0142 | **-0.3807** | **-0.5157** | +0.3665 |
| 2.0 | 0.149 | **81.4** | 9 | **0.0577** | **0.0213** | **0.0244** | **0.0141** | -0.3740 | -0.5100 | +0.3586 |
| Ghia | | | | | | | | -0.3829 | -0.5155 | +0.3710 |

**Monotone improvement all the way to 81.4 degrees of non-orthogonality.** u rms improves 2.7x and
v rms 2.5x over the uniform mesh at identical cell count. At beta = 1.5 the extrema are essentially
exact -- v_min lands within **0.04%** of Ghia, u_min within 0.6%, v_max within 1.2% -- against
+5.5 / +1.0 / +4.4 % for the uniform mesh. Beyond that, beta = 2.0 buys the best rms but begins to
undershoot all three extrema (-2.3 / -1.1 / -3.3 %), so the useful range is beta ~ 1.5-2.0.

Two things worth recording:

* **None of this would have run before the `_decompose` fix (§1 bug 2).** At beta = 1.0 the old
  minimum-correction contraction factor was tan(49.3) = 1.16 -- already divergent -- and at
  beta = 1.5 it was 2.65. The over-relaxed factor sin(theta) is 0.76 / 0.94 / 0.99 across the
  sweep: marginal at the top end but convergent, and it held.
* **This is beyond where the literature has data.** Bonaventura & Della Rocca's "strongly
  non-orthogonal" meshes top out at 66.5 degrees, and the survey found no published convergence
  study at 81 degrees for a scheme of this type. The result is ours to report, not to cite.

The practical conclusion stands unchanged and is now quantified: resolving the wall layer is the
cheapest accuracy available to this solver, it is orthogonal to the unresolved T5 defect, and the
quad-layer route (§5) should do better still by removing the non-orthogonality rather than
tolerating it.

## 12. The controlling variable is nu, NOT the momentum-diagonal ratio

A sequence of five refinement studies on the same manufactured field, same field-L2 norm, finally
separating viscosity from the `a_P/(a_t V)` ratio that had been the leading suspect all session.

**Step 1 -- the cavity really does behave differently from T5.** Not a metric or sample-count
artefact:

| config | n=16 | n=32 | n=64 | order |
|---|---|---|---|---|
| Stokes, nu=1, dt=0.05 (T5) | 1.2465e-02 | 1.2127e-02 | 1.2047e-02 | 0.04 / 0.01 |
| NS MMS, nu=1e-3, dt=0.005 (cavity) | 8.2776e-02 | 3.0902e-02 | 1.8758e-02 | 1.42 / 0.72 |

**Step 2 -- isolation, one factor at a time.** nu rescues it; dt does nothing; convection is
irrelevant:

| changed | nu | dt | convect | ratio 16->64 | order |
|---|---|---|---|---|---|
| baseline | 1 | 0.05 | no | 103 -> 1639 | 0.04 / 0.01 |
| dt only | 1 | 0.005 | no | 11.2 -> 165 | 0.06 / 0.02 |
| **nu only** | 1e-3 | 0.05 | no | 1.10 -> 2.64 | **1.43 / 0.90** |
| nu and dt | 1e-3 | 0.005 | no | 1.01 -> 1.16 | 0.94 / 0.79 |
| + convection | 1e-3 | 0.005 | yes | ~1.16 | 1.42 / 0.72 |

**Step 3 -- does the ratio GROWING matter?** No. Refining `dt ~ h^2` holds it at 103.40 at every
level; the result is indistinguishable from letting it grow 16x:

| | n=16 | n=32 | n=64 | order |
|---|---|---|---|---|
| dt ~ h^2, ratio held at 103.40 | 1.2465e-02 | 1.2147e-02 | 1.2073e-02 | 0.04 / 0.01 |
| dt fixed, ratio 103 -> 411 -> 1639 | 1.2465e-02 | 1.2127e-02 | 1.2047e-02 | 0.04 / 0.01 |

**Step 4 -- decoupling nu from the ratio.** Choose `dt = 1.5 V (R-1)/visc` to force any target
ratio R at any nu. This is the experiment that settles it:

| arm | nu | ratio | n=16 | n=32 | n=64 | order |
|---|---|---|---|---|---|---|
| low nu, HIGH ratio | 1e-3 | 103 | 8.329e-02 | 2.590e-02 | 1.444e-02 | **1.68 / 0.84** |
| low nu, low ratio | 1e-3 | 1.16 | 8.076e-02 | 3.436e-02 | 2.598e-02 | 1.23 / 0.40 |
| HIGH nu, low ratio | 1 | 1.16 | 2.410e-02 | 2.407e-02 | -- | **0.00** |

A 94x change in the ratio moves nothing, in either direction. A 1000x change in nu flips
convergent to flat. Note the high-ratio arm converges BETTER than the low-ratio control, and that
forcing the "good" ratio at nu=1 makes the error WORSE than the original T5 configuration
(2.41e-02 against 1.25e-02) while remaining perfectly flat.

**What this exonerates.** Every coefficient-based explanation pursued this session is a statement
about the ratio: `V/a_P` versus SIMPLEC, the diffusion number, the damping magnitude, the
Bartholomew/Syrakos coefficient constructions, the SU2 transient-term-removal fix. None of them
can be the cause of the plateau. The SIMPLEC change made earlier remains correct and necessary --
it took the solver from diverging at ~32x/step to stable across four decades of nu -- but it is a
STABILITY fix, not an accuracy one, and it is not on the path to this defect.

**What this leaves.** An error term in the velocity proportional to the pressure and inversely
proportional to nu: dominant at nu=1, invisible at nu=1e-3, independent of how `a_P` is composed.
That is the non-pressure-robustness signature (John, Linke, Merdon, Neilan & Rebholz, *On the
divergence constraint in mixed finite element methods for incompressible flows*, SIAM Review 59
(2017) 492). It is now the only hypothesis standing, and it has not yet been tested.

**Practical consequence.** The cylinder target runs at Re=100, nu=0.01 -- far closer to the cavity
than to T5. If the error scales as 1/nu, the term is ~100x smaller there than in the gate that is
failing. T5 at nu=1 is the hardest configuration this solver will ever be asked to run, not a
representative one. That does NOT make the gate passable as written, and it should not be quietly
relaxed; but it does mean T9 is not blocked on T5 to the degree previously assumed.

**Hypotheses killed, with the test that killed each** (all had been stated with confidence):
skewness (same-diagonal zero-skew mesh is 2.5x WORSE and equally flat); gradient accuracy
(k-exact improves magnitude 8x, order not at all, and destabilises the pressure loop); stencil
width (wider linear fit has NEGATIVE order); conditioning (fixed, results bit-identical);
ratio growth (control arm indistinguishable); ratio magnitude (94x change, no effect).

## 13. CORRECTION to section 12 — nu is NOT the controlling variable

Section 12 is left above as written, because the way it failed is the useful part.

Its conclusion ("the controlling variable is nu") was drawn from the FIRST order in each refinement
pair while ignoring that every order was DECAYING: 1.68->0.84, 1.43->0.90, 1.42->0.72, 1.23->0.40.
That is the signature of approaching a floor, not of convergence. Extending the low-nu runs one
more level settles it:

| ratio | nu=1 (flat from the start) | nu=1e-3 (descending) |
|---|---|---|
| **103** | 1.2465e-02, 1.2147e-02, **1.2073e-02** (ord 0.01) | 2.590e-02, 1.444e-02, **1.2536e-02** (ord 0.84 -> **0.20**) |
| **1.16** | 2.4098e-02, 2.4067e-02, **2.4049e-02** (ord 0.00) | 3.436e-02, 2.598e-02 (ord 0.40, still descending) |

At n=128 the nu=1e-3 run reaches 1.2536e-02 against the nu=1 floor of 1.2073e-02 -- **3.8% apart**
-- with its order collapsed to 0.20. The two viscosities land on the SAME floor at the SAME ratio.

**Corrected reading.** There is one mesh-independent error floor. Its value is set by
`a_P/(a_t V)`; nu determines only how far above it a given mesh starts. Every configuration
descends at a decaying rate and then stops. The cavity is not a case that works -- it is a case
that had not yet arrived.

**The floor depends on the ratio in the direction opposite to every coefficient argument.** The
"good" SIMPLEC-like ratio 1.16 has the WORSE floor, 2.40e-02, and the "bad" ratio 103 the better
one, 1.21e-02 -- a factor of 2. So changing the momentum-diagonal ratio does move the floor, but
making it more "correct" by the SIMPLEC/Bartholomew reasoning makes the answer worse.

**What this changes about the practical outlook.** Section 12 concluded the cylinder at Re=100
(nu=0.01) would be ~100x less affected and T9 was therefore not badly blocked. That is now wrong
in the way that matters: under a universal floor, a Re=100 run starts above the floor on a coarse
mesh and descends onto it as the mesh refines, so **refinement stops buying accuracy at some
resolution**. The correct statement is that T9's accuracy will be capped by this floor, and the
cap has to be measured on the cylinder mesh rather than assumed small.

**Method note, which is the real lesson.** Three separate times in this investigation an order
computed from two mesh levels was misleading, and each time the error was mine for reporting it:
the original T5 "order 1.1" from 400-step runs; the cavity "order 0.67-1.28" from two levels; and
section 12. A decaying order is not a converging scheme. Every order in this suite should be
quoted with at least three levels AND checked against the known flat value of the corresponding
nu=1 case before any conclusion is drawn from it.

## 14. The floor, decomposed — it is entirely the pressure error, seen through G

The single most useful measurement of the investigation. Using the actual discrete momentum
operator `A` (Laplacian plus its deferred cross term, homogeneous Dirichlet), the velocity error
of the converged T5 solution obeys exactly

    A e_u = -V G(e_p) - tau_M

so it splits into `e_u = x + y` with `A x = -V G(e_p)` (driven by the pressure error) and
`A y = -tau_M` (driven by the momentum truncation error). Solving each separately on the saved
converged fields:

| n | \|e_u\| | \|x\| pressure-driven | order | \|y\| tau_M-driven | order | \|x+y-e_u\| |
|---|---|---|---|---|---|---|
| 16 | 1.2465e-02 | 1.1878e-02 | -- | 1.1997e-03 | -- | 4.2e-11 |
| 32 | 1.2127e-02 | 1.1977e-02 | -0.01 | 3.0695e-04 | **1.97** | 7.1e-11 |
| 64 | 1.2047e-02 | 1.2009e-02 | -0.00 | 7.7731e-05 | **1.98** | 1.5e-10 |

The decomposition is exact to 1e-10. Two conclusions follow immediately:

* **The O(1), non-converging `tau_M` of section 10.2 is completely benign.** Its velocity
  footprint `y` is clean second order, 1.97/1.98 -- the Diskin & Thomas supraconvergence holds
  in the COUPLED solve, not just with the pressure given. Every worry about the momentum operator,
  its gradient, its boundary treatment and its skewness is closed by this row.
* **The floor is `x`, and `x` is 98% of the error.** The velocity error is the Stokes response,
  through the momentum gradient operator `G`, to a pressure error `e_p` that does not converge.
  Nothing else contributes at a measurable level.

So the fix target is now precise: **`G(e_p)` must go to zero**, which means either the pressure
error itself must converge, or `G` must not see the part of it that does not. Section 12/13's
observation that the floor depends on the momentum-diagonal ratio is consistent with this -- the
ratio enters the fixed point only through the Rhie-Chow flux, which is what determines `e_p`.

It also explains, retroactively, the one intervention that ever moved the floor. The k-exact
gradient placed in the momentum source (`grad_p`) also feeds Rhie-Chow's `dp_wide`, since
`_rhie_chow` receives that same `gp`. It therefore improved the Rhie-Chow bracket, reduced `e_p`,
and cut the floor 8x. The k-exact gradient placed ONLY inside the pressure-correction loop (the
`gpp` and Poisson deferred terms) was the configuration that diverged -- the wide-stencil
Laplacian problem -- and it is not needed there.

**Error map** (`figures/t5_error_map.png`): the velocity error is a smooth four-fold ring with
lobes on the centrelines at (0.5, 0.2), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5), zero at the centre
and the walls, with IDENTICAL shape and amplitude at n = 16, 32, 64. The pressure error is a
checkerboard organised along the diagonals with a smooth quadrant-alternating envelope. A direct
correlation between `e_u` and `grad(e_p_smooth)` is zero (R^2 = 0.000) -- which is expected, since
in Stokes the velocity responds through the inverse Laplacian, not the gradient; that naive test
was wrong and is recorded so it is not repeated. The operator-based decomposition above is the
correct one.

Fixed-band split of the velocity error (distance from nearest wall, physical units) is flat in
EVERY band: <0.06: 6.16e-03 / 5.74e-03 / 5.69e-03; 0.06-0.15: 1.61e-02 / 1.54e-02 / 1.49e-02;
0.15-0.30: 2.34e-02 / 2.29e-02 / 2.25e-02; core >0.30: 1.77e-02 / 1.83e-02 / 1.83e-02. An
earlier h-relative banding suggested the error was "migrating inward"; that was an artefact of
the bands themselves shrinking, and is withdrawn. The defect is global, not a boundary skin --
the boundary-skin wiggles measured on the cavity are a separate and milder effect.

## 15. The mechanism — the LSQ gradient transmits the checkerboard into the velocity

Splitting the pressure error into a smooth part (four passes of face-neighbour averaging) and the
checkerboard remainder, and solving `A x = -V G(e_p_part)` for each:

| n | \|e_p smooth\| | \|e_p cb\| | \|x_smooth\| | \|x_cb\| | \|G e_p_cb\| / \|G e_p_smooth\| |
|---|---|---|---|---|---|
| 16 | 2.751e-02 | 1.093e-01 | 1.21e-03 | **1.2127e-02** | **14.6** |
| 32 | 3.051e-02 | 6.540e-02 | 1.42e-03 | **1.2447e-02** | **14.5** |
| 64 | 3.134e-02 | 4.737e-02 | 1.48e-03 | **1.2538e-02** | **19.5** |

**The floor is the checkerboard's velocity footprint, not the smooth envelope's -- 98% of it.**
The checkerboard amplitude itself DOES shrink (order ~0.6), but the face-neighbour LSQ gradient
sees it 15-20x more strongly than it sees the smooth field, and a checkerboard's gradient scales
as amplitude/h. Net: `G(e_p_cb)` is mesh-independent, and so is `A^-1(V G e_p_cb)`.

**This overturns section 3.** The literature-derived claim there -- that the wide gradient is
"blind" to the checkerboard, so it contributes only O(h) to velocity -- is true only if the wide
gradient is a face-AVERAGED Green-Gauss operator, which is what OpenFOAM's `fvc::grad(p)` is and
what Hopman et al. and Syrakos have in mind. Ours is a least-squares gradient: exact for linear
fields, and for that very reason NOT blind to an alternating field. Choosing the "consistent"
gradient for the pressure term was the root cause, in two ways at once:

1. **Transmission.** `G(p)` in the momentum source carries the checkerboard into `u`.
2. **Self-cancelling damping.** Rhie-Chow's `dp_wide = interp(G p) . S` uses the SAME gradient.
   A gradient that sees the checkerboard makes `dp_wide` track `dp_compact` on that mode, so
   `(compact - wide)` partially cancels and the damping is weak -- which is why the checkerboard
   only decays at order 0.6 instead of being suppressed, and why `rc_scale` helped (it scaled up
   what little damping survived) but could not be pushed past ~3x.

**Fix under test:** a Green-Gauss face-averaged gradient (`ugrad.GGGradient`) for `grad(p)` in the
momentum source and in `dp_wide`, with LSQ retained everywhere else (Laplacian cross terms,
pressure-correction gradient, Poisson deferred correction -- where the wide-stencil-Laplacian
instability lives and where linear-exactness matters). Prediction: `x_cb` collapses, the floor
drops toward the `x_smooth` level of ~1.4e-03 or below, and the checkerboard itself decays faster
because the damping stops cancelling.

Known cost: Green-Gauss is zeroth-order INCONSISTENT on irregular triangles (Sozer 2014, Syrakos
2017 -- their Table 5 gives order 0/0 on arbitrary grids). On the uniform test mesh that is masked;
on `medium.msh` it will not be. If the fix works here, the follow-up is a gradient that is BOTH
checkerboard-blind and linear-exact -- Green-Gauss with skewness-corrected or iteratively
corrected face values (Syrakos's `grad^{d-inf}`, OpenFOAM `iterativeGauss`) is the standard
answer, and it is what the literature survey (sections 5-6) already recommended for other reasons.

## 16. THE FIX — discrete duality, not consistency. T5 passes.

Replace the pressure gradient used in the momentum source (and hence in Rhie-Chow's `dp_wide`,
which receives the same `gp`) with a Green-Gauss face-averaged gradient, keeping LSQ everywhere
else. `PISO.grad_p = GGGradient(mesh)`. Uniform mesh, T5:

| n | baseline (LSQ) | Green-Gauss `grad_p` | order |
|---|---|---|---|
| 16 | 1.2465e-02 | **6.5577e-04** | -- |
| 32 | 1.2127e-02 | **1.6634e-04** | **1.98** |
| 64 | 1.2047e-02 | **4.1905e-05** | **1.99** |

**T5 PASSES.** Three levels, 1.98 / 1.99 -- the design order, on the coupled problem, for the first
time in this solver. 288x below the baseline floor at n=64. Pressure converges at order 1.00 in L2
(9.85e-02, 4.92e-02, 2.46e-02): that is the rate Eymard et al. prove for stabilised collocated FV
Stokes, and it means a residual checkerboard persists in p even though it is no longer transmitted
to u. Relevant for pressure-integrated forces (T9 drag/lift) and recorded as an open item, not a
failure of this gate. The pressure error follows: smooth part 3.05e-02 -> 2.42e-04 (126x); the
checkerboard is only mildly reduced (6.54e-02 -> 4.92e-02) and is no longer transmitted.

**Why it works -- and why section 10.1 was wrong.** The mechanism is the discrete duality
`G = -D^T` between the momentum pressure gradient and the continuity divergence. Measured:

| gradient | duality residual, uniform | perturbed 0.25 | clustered 1.5 | clustered 2.0 |
|---|---|---|---|---|
| face-LSQ | 1.2e-01 | 6.2e-01 | 1.2e+00 | -- |
| Green-Gauss, ordinary weights | **2.2e-15** | 1.1e+00 | 2.4e-02 | 1.4e-01 |
| Green-Gauss, SWAPPED weights (`GGDualGradient`) | **2.2e-16** | **7.6e-16** | **3.7e-16** | **1.8e-16** |

Ordinary Green-Gauss is dual only where `w = 1/2` -- i.e. only on the uniform mesh, which is why
it happened to work there. The exact adjoint on any mesh needs the pressure face weights SWAPPED,
`p_f = (1-w) p_O + w p_N`, so that every face contributes `(u_O p_O - u_N p_N).S_f` and the sum
telescopes by cell closure. That is eq. (11) of Eymard, Herbin & Latche (M2AN 40, 2006), relayed
by the literature search in section 2 of the collocated-analysis report and NOT acted on.

Section 10.1 built exactly this adjoint, found it was not linear-exact (error 2.67 on a linear
field against 1e-14 for LSQ), and concluded "duality and consistency are in direct conflict for
this operator pair, and we cannot simply impose G = -D^T." The premise was right and the
conclusion was wrong: **duality wins**. The pointwise consistency of the pressure gradient is
irrelevant to the coupled solution -- which had ALREADY been measured in section 2 ("analytic vs
LSQ grad p in the momentum source: 1.47e-03 vs 1.52e-03, both order 1.99"). The gradient's
STRUCTURE is what matters, and the structure that matters is being the transpose of the
divergence. I tested consistency, saw it fail, and stopped one experiment short.

**The mechanism in one paragraph.** Rhie-Chow leaves a residual checkerboard in the pressure
(it decays at order ~0.6, not 2). The face-neighbour LSQ gradient of that checkerboard is 22%
SMOOTH -- a spurious smooth pressure-gradient mode, the quadrant-alternating envelope visible in
`figures/t5_error_map.png`. That mode enters momentum as a body force; the Stokes inverse
Laplacian amplifies it into the O(1), mesh-independent four-lobe velocity error; and it feeds
back through continuity into a smooth pressure error of 3e-02. A divergence-form gradient
telescopes, so its gradient of the checkerboard is only 4.4% smooth (measured) and the loop is
broken. The dual form telescopes EXACTLY, on any mesh.

**Falsified alongside, cleanly.** Raising the Rhie-Chow bracket to O(h^3) -- the unweighted
quadratic gradient achieved bracket order 2.93/2.92 exactly as predicted -- moved the floor only
from 1.25e-02 to 5.42e-03 / 4.31e-03 / 3.98e-03, order 0.33 / 0.11 -- flat at three levels. So the bracket order was never the variable;
the 8x improvement seen from the k-exact gradient came from it being a somewhat more
divergence-like operator, not from its accuracy.

**Cost and caveat.** Green-Gauss is zeroth-order inconsistent on irregular triangles (linear-field
error 1.00 measured here, matching Sozer 2014 and Syrakos 2017). That inconsistency is harmless
for the pressure gradient in momentum (supraconvergence, measured) but would NOT be harmless
elsewhere -- it must not replace LSQ in the Laplacian cross terms, the pressure-correction
velocity update, or the Poisson deferred correction, where linear-exactness is needed and where
the wide-stencil-Laplacian instability lives. The hooks `grad_p` / `grad_rc` / `grad` keep those
separate. Whether ordinary or dual GG is needed on the clustered/perturbed meshes is the test
running now.

## 17. Regression of the fix on the cavity — a real trade-off, not a clean win

`grad_p = GGDualGradient`, Re = 1000, 64x64, compared with the LSQ results of sections 9-11:

| | beta = 0, LSQ | beta = 0, dual GG | beta = 1.5, LSQ | beta = 1.5, dual GG | Ghia |
|---|---|---|---|---|---|
| u rms | 0.0581 | **0.0403** | 0.0260 | 0.0247 | |
| v rms | 0.0351 | **0.0275** | **0.0142** | 0.0155 | |
| u_min | -0.4041 (+5.5%) | -0.3740 (-2.3%) | -0.3807 (-0.6%) | -0.3734 (-2.5%) | -0.3829 |
| v_min | -0.5206 (+1.0%) | -0.4997 (-3.1%) | **-0.5157 (+0.04%)** | -0.5088 (-1.3%) | -0.5155 |
| v_max | +0.3873 (+4.4%) | +0.3602 (-2.9%) | +0.3665 (-1.2%) | +0.3608 (-2.7%) | +0.3710 |
| core vorticity | -2.514 (+22.7%) | **-2.132 (+4.0%)** | **-2.052 (+0.1%)** | -1.945 (-5.1%) | -2.0497 |

**Uniform mesh: clear gain** -- both rms errors improve (1.44x, 1.28x), core vorticity 5.7x closer.
**Clustered mesh: wash on rms, slightly WORSE on every extremum and on core vorticity.** The LSQ
beta = 1.5 result was the best number of the day and dual GG gives it back.

Two things explain the difference, both already in the survey:
* The cavity sits at `a_P/(a_t V)` ~ 1.16, where the T5 defect is weak; the floor it removes is
  small against the discretisation error that remains, so there is little to gain.
* Green-Gauss is zeroth-order INCONSISTENT on non-uniform triangles (Sozer 2014 Eq. 48-50: the
  gradient converges to `C grad(phi)`, `C != 1`). On the uniform mesh that bias is small; on the
  stretched wall cells of the clustered mesh it is not, and it shows up as every extremum
  undershooting by 1-5% -- a systematic multiplicative error, not noise. This is the cost the
  fix was flagged as carrying in section 16, now measured.

**Conclusion.** The divergence-dual structure is necessary -- it is the only thing that has ever
moved the T5 floor, and it moved it 288x. But the plain Green-Gauss operator that provides it is
not accurate enough on stretched cells to be the final answer. The next step is the one the
survey already named: a gradient that is BOTH divergence-form and linear-exact -- Green-Gauss with
skewness-corrected or iteratively corrected face values (Syrakos `grad^{d-inf}`, OpenFOAM
`iterativeGauss`). Applying the correction to the FACE VALUE keeps the operator in divergence
form, so it telescopes and stays dual to a divergence built from the same corrected face values.
That would recover the LSQ-quality extrema on clustered meshes while keeping the T5 fix.

Caveat on these numbers: the GG cavity runs reported a relative change of 6.7e-02 over their
final 4000 steps at t = 40, larger than the LSQ runs appeared to show at a different report
interval. The comparison may be slightly unfair to GG on convergence; the sign of the clustered
result is not in doubt but the exact percentages should not be over-read.

`PISO` defaults are UNCHANGED. `GGDualGradient` is available via `grad_p` and should be used for
viscous-dominated problems now; it is not yet the right default for boundary-layer meshes.

## 18. T4 temporal order — FAILS: first order, then a floor

First run of the temporal gate. Unsteady Stokes, decaying Taylor-Green `u = U(x,y) e^{-lam t}`,
`lam = 2 pi^2 nu`, nu = 0.1, T = 1 (decay to 0.139). This field is an EXACT unsteady-Stokes solution
with a time-independent forcing `f = grad p`, because `du/dt = -lam u` and `-nu lap u = +lam u`
cancel identically -- so only the Dirichlet boundary velocity is time-dependent. Fixed uniform
n = 32 mesh, dual-GG `grad_p` (steady spatial error there 1.66e-04), exact initial condition, BC
set at the BDF implicit level `t^{n+1}` before each step.

| dt | steps | L2(u) at T | order |
|---|---|---|---|
| 0.1 | 10 | 3.0522e-03 | -- |
| 0.05 | 20 | 1.6193e-03 | 0.91 |
| 0.025 | 40 | 8.5999e-04 | 0.91 |
| 0.0125 | 80 | 6.4725e-04 | 0.41 |
| 0.00625 | 160 | 5.7904e-04 | 0.16 |

**Two defects, not one.** The slope is 0.91 where BDF2 should give 2, and it then flattens onto a
floor of ~5.8e-04 that is 3.5x the steady spatial error on this mesh -- so the floor is not (only)
spatial. Suspects, each being isolated in a separate run: (a) the Choi time-step-independent
Rhie-Chow term `(D/dt)(F_old - Fbar_old)`, which is first-order in time by construction and whose
coefficient `D/dt -> 1/1.5` at small dt, leaving an O(1)-coefficient persistent term; (b) PISO
splitting error at n_corr = 2; (c) spatial floor -- rerun at n = 64. Note this is the first time
the solver's time-dependent boundary-condition path has been exercised at all.

Diagnostics, one change each from the failing baseline:

| variant | dt = 0.1 / 0.05 / 0.025 / 0.0125 / 0.00625 | orders |
|---|---|---|
| (a) transient Rhie-Chow term OFF | 3.05e-03 / 1.62e-03 / 8.56e-04 / 6.40e-04 / 5.67e-04 | 0.92 / 0.92 / 0.42 / 0.18 |
| (b) n_corr = 4 | 3.05e-03 / 1.54e-03 / 8.10e-04 / 6.21e-04 / 5.51e-04 | 0.98 / 0.93 / 0.38 / 0.17 |
| (c) mesh n = 64 | 2.93e-03 / 1.52e-03 / 7.73e-04 / 3.47e-04 / 2.10e-04 | 0.95 / 0.98 / 1.15 / 0.73 |

(a) and (b) are indistinguishable from the baseline: neither the Choi term nor PISO splitting is
involved. (c) drops the floor 2.8x and it is still falling, so **the floor is spatial and benign**
-- the unsteady spatial error is simply ~3.5x the steady one on this mesh. What survives all three
is the slope: **~1 everywhere BDF2 should give 2.** One defect, in the time integration itself.

Remaining suspect, specific: the Laplacian's non-orthogonal cross term is DEFERRED -- built from
the previous time level's gradient. At a steady fixed point the lag is invisible, which is why
every steady gate passed with it in place. In an unsteady problem a one-step lag in a flux is O(dt).
On the uniform mesh the cross term vanishes on interior faces (orthogonal) but not on the 26.6-deg
boundary faces -- precisely where the time-dependent Dirichlet data enter. Under test by zeroing
the boundary cross term (spatial operator changes, so only the slope is meaningful).

**Lag test — confirmed.** Zeroing the boundary cross term (diagnostic only; it changes the spatial
operator) against the lagged baseline, n = 32:

| dt | lagged (as-is) | boundary cross term zeroed |
|---|---|---|
| 0.1 | 3.052e-03 | **7.440e-04** |
| 0.05 | 1.619e-03 | 5.704e-04 |
| 0.025 | 8.600e-04 | 6.084e-04 |
| 0.0125 | 6.472e-04 | 5.812e-04 |

Without the lagged term the error is at the spatial floor (~5.8e-04) from dt = 0.1 onward. The
lagged term was therefore contributing ~2.3e-03 / 1.0e-03 / 2.8e-04 / 0.7e-04 -- halving with dt.
**That is the O(dt) term.** The slope-2 recovery is hidden only because the temporal error fell
straight under the spatial floor.

Fix: `PISO(n_inner=k)` iterates the momentum deferred correction k times within each step,
rebuilding the cross term (and the deferred central convection) from the CURRENT iterate. The
steady operator is unchanged. Zeroing the term is not a fix -- section 1 showed the boundary
non-orthogonal correction is what makes the boundary Laplacian second order (dropping it gave
Poisson order 1.54). Under test on n = 64 with dt chosen to keep the temporal error above the
floor and an exact `u(-dt)` so BDF2 runs from the first step.

**T4 with the inner iteration — second order in time.** n = 64, nu = 0.05, T = 1.6, exact `u(-dt)`
so BDF2 runs from step 1, dt chosen so the temporal error sits above the spatial floor (~2.7e-04
on this mesh for this problem):

| dt | steps | n_inner = 1 (lagged) | order | n_inner = 3 | order |
|---|---|---|---|---|---|
| 0.4 | 4 | 7.4968e-03 | -- | 4.8696e-03 | -- |
| 0.2 | 8 | 4.1485e-03 | 0.85 | 9.5741e-04 | **2.35** |
| 0.1 | 16 | 2.1888e-03 | 0.92 | 2.8521e-04 | **1.75** |
| 0.05 | 32 | 1.1630e-03 | 0.91 | 2.6249e-04 | floor |
| 0.025 | 64 | 5.8005e-04 | 1.00 | 2.8509e-04 | floor |

Lagged: cleanly first order, 0.85-1.00 at every level. Iterated: 2.35 / 1.75 -- second order --
until the spatial floor. Two points above the floor is thin evidence on its own; the contrast with
the lagged column at identical dt is not. A clean three-point demonstration on n = 128 (floor
~7e-05) and a check that `n_inner` leaves the STEADY T5 answer untouched are running.

Note on what this means for the design: the deferred non-orthogonal correction is a spatial
device that is exact only at a fixed point. Any solver that defers it and then runs unsteady is
first order in time on every non-orthogonal face, whatever its time scheme claims. On a
wall-clustered or cylinder mesh that is every face, not just the boundary. `n_inner >= 2` is
therefore not optional for T8/T9; it is part of being second order at all.

**T4 clean demonstration — PASSES.** n = 128 (spatial floor ~7e-05), nu = 0.05, T = 1.6, `n_inner = 3`,
exact BDF2 startup:

| dt | steps | L2(u) | order |
|---|---|---|---|
| 0.8 | 2 | 2.4379e-02 | -- |
| 0.4 | 4 | 4.9039e-03 | **2.31** |
| 0.2 | 8 | 9.7209e-04 | **2.33** |
| 0.1 | 16 | 1.9883e-04 | **2.29** |
| 0.05 | 32 | 6.4340e-05 | 1.63 (floor) |

Three consecutive second-order points before the spatial floor. BDF2 is second order in time once
the deferred correction is converged within the step.

**Steady regression.** T5, n = 16, 2500 steps: `n_inner = 1` -> 6.557685e-04, `n_inner = 3` ->
6.557685e-04. **Bit-identical.** The inner iteration changes the path to the fixed point and not
the fixed point, exactly as the argument said -- and now it is a measurement. Nothing steady that
passed today can have moved.

Cost: `n_inner - 1` extra momentum solves per component per step through the cached solver. Whether
`n_inner = 2` already suffices (the minimum that removes the lag) is being measured before any
default is changed.

**`n_inner = 2` suffices.** Same n = 128 setup: 3.9767e-03 / 6.1084e-04 / 1.5179e-04, orders
**2.70 / 2.01** -- second order, with absolute errors slightly BELOW `n_inner = 3` at every dt. One
extra momentum solve per component per step is the whole cost, through the cached solver, and the
steady fixed point is unchanged (measured bit-identical at `n_inner = 3`). `PISO`'s default is now
`n_inner = 2`. The lagged behaviour remains available as `n_inner = 1` for comparison only.

## 19. T3 with the fix, and the default configuration diverging on the clustered mesh

**T3 — Navier-Stokes MMS**, convection ON (implicit upwind + deferred central), nu = 1e-3,
dt = 0.005, uniform mesh, `grad_p = GGDualGradient`:

| n | LSQ (section 12) | dual GG | order |
|---|---|---|---|
| 16 | 8.2776e-02 | **2.1875e-02** | -- |
| 32 | 3.0902e-02 | **8.4244e-03** | **1.38** |
| 64 | 1.8758e-02 | **3.6686e-03** | **1.20** |

4-5x lower error, and the order now HOLDS at ~1.2-1.4 instead of decaying 1.42 -> 0.72 toward a
floor. But it is not second order. Whatever limits T3 is in the convective path, not the pressure
coupling, and it is a separate item: candidates are the deferred central correction's dependence
on the first-order LSQ gradient in the skewness term, and the convective flux being carried by the
Rhie-Chow `Ff`, which still holds a first-order checkerboard residue. Also note the cost: 43,500
to 64,000 pseudo-time steps to stagnate at nu = 1e-3 -- the transient march is a poor steady solver
in this regime.

**T5 clustered (cluster = 1.5, worst face 69 deg), LSQ default:** 1.4498e-02 / 1.4175e-02, order
0.03 -- the floor, as on every other mesh -- and then **NaN at n = 64**. The first wall cell there
has diffusion number ~8700 and the deferred contraction factor is sin(69 deg) = 0.94; the default
configuration is not merely inaccurate on a strongly clustered viscous-dominated problem, it is
unstable. The GG arms are being run separately (the original script had no per-arm guard and the
LSQ divergence killed them).

**Decay rate against the exact exponent.** The T4 L2 error bounds the rate error but does not
isolate it. Fitting the log-slope of kinetic energy and of the modal amplitude `a(t) = <u,U>/<U,U>`
over t in [0.2, 1.6], dual GG, `n_inner = 2`, against `lam = 2 pi^2 nu = 0.986960`:

| n | dt | lam (KE) | rel err | lam (mode) | rel err | L2 at T |
|---|---|---|---|---|---|---|
| 64 | 0.1 | 0.986761 | -2.0e-04 | 0.986762 | -2.0e-04 | 2.95e-04 |
| 64 | 0.05 | 0.986794 | -1.7e-04 | 0.986795 | -1.7e-04 | 2.93e-04 |
| 64 | 0.025 | 0.987545 | +5.9e-04 | 0.987546 | +5.9e-04 | 2.81e-04 |
| 128 | 0.05 | 0.986726 | -2.4e-04 | 0.986726 | -2.4e-04 | 1.27e-04 |

Rate correct to ~2e-04 relative; the two extractions agree to six digits. Sign negative: the
solver is very slightly UNDER-dissipative, the right direction for a non-upwinded scheme. The
dt = 0.025 sign flip is the fit reaching the n = 64 spatial floor, not a trend -- n = 128 at the
same dt sits at -2.4e-04.

**Relevance to T8.** Orr-Sommerfeld's growth rate is 2.23e-03 per unit time and the plan's
standing warning is that numerical damping of that size flips its sign. The rate error measured
here is ~2e-04 absolute -- an order of magnitude below what T8 must resolve. A necessary condition,
not a prediction: T8 is Re = 7500 on a periodic channel not yet built, and the structured solver
needed 48x401 to reach 1.2%. But it is the first direct measurement that the dissipation budget is
in the right place, and it is the methodology T8 will use.

**Correction: the -2e-04 is fit resolution, not a bias.** Varying the fit window, run length and
inner-iteration count on the n = 64, dt = 0.05 case moves the fitted rate through -1.7e-04,
+2.2e-04, +5.0e-04 and +1.6e-04 -- sign and magnitude both change. The transient Rhie-Chow term is
not involved (identical with it off). The numerical solution is not a pure exponential: it carries
a non-modal component at the spatial floor, and the log-slope of that mixture wanders by ~2e-04
depending on where it is fitted. The statement "under-dissipative, the right sign" made above is
withdrawn.

Correct statement: at dt <= 0.1 the decay rate is right to within the measurement's own resolution
of +-2e-04 at n = 64, and that resolution is set by SPATIAL error, not the time step. The genuinely
dt-driven figure is dt = 0.2, where the temporal rate error is +1.5e-03 (consistent with dt^2 from
dt = 0.05). To resolve the rate more finely, refine h; n = 128 has a floor 4x lower.

## 20. Pressure order, and the regime dependence that survives the GG fix

**Pressure decomposed** (T5 uniform, dual GG, `rc_scale` 1 and 2):

| n | L2(p) | smooth part | order | checkerboard | order |
|---|---|---|---|---|---|
| 16 | 9.851e-02 | 8.205e-04 | -- | 9.842e-02 | -- |
| 32 | 4.922e-02 | 2.419e-04 | **1.76** | 4.920e-02 | 1.00 |
| 64 | 2.462e-02 | 7.213e-05 | **1.75** | 2.461e-02 | 1.00 |

The pressure error is 99.9% checkerboard. Once that is removed, the pressure converges at ~1.75.
`rc_scale = 2` halves the checkerboard amplitude (9.84e-02 -> 5.53e-02) at identical order 1.00
and slightly worsens velocity (6.56e-04 -> 8.27e-04): damping scales amplitude, never order.
(A naive 4-pass smoothing of the computed p against the exact p gives 1.31 / 1.10 -- but that
filter also smooths the true field; comparing filtered-to-filtered is the fair test and was not
done. The smooth-part order of 1.75 is the meaningful number.)

**The regime dependence is still there with GG.** Stokes at T3's exact parameters (nu = 1e-3,
dt = 0.005, ratio ~1.16), no convection, dual GG: 3.012e-02 / 1.361e-02, **order 1.15** -- against
1.98 at ratio 103. So the T3 limit is NOT convection: T3 with convection gave 1.38, Stokes in the
same regime gives 1.15. It is the transient-dominated regime, again.

**Duality alone does not rescue the perturbed mesh.** Dual GG there: 2.867e-03 / 1.873e-03,
order 0.61 -- indistinguishable from plain GG (2.948e-03 / 1.887e-03, 0.64). A floor at ~2e-03.
**Plain GG on the clustered mesh** likewise: 6.872e-03 / 6.300e-03, order 0.13 -- a floor, 2x below
LSQ's but a floor. Dual arm pending.

**One hypothesis covers all three.** The Rhie-Chow bracket's wide term is now built from the GG
gradient, whose linear-field error is 1.00 on the uniform mesh and worse on irregular cells. An
inconsistent gradient in the bracket makes it O(h)·O(1) = O(h): a first-order spurious flux,
scaled by D. At ratio 103, D ~ 3e-04 and the term is masked; at ratio 1.16, D is 10x larger and
it shows as order ~1.15; on irregular meshes the GG error is larger and it shows as a floor. The
velocity in T5-uniform was second order only because D was small enough there to hide it. And it
is the same term that makes the pressure checkerboard first order everywhere.

Test running: `grad_p = GGDualGradient` (no transmission) with `grad_rc = Gradient` (linear-exact
bracket), at the T3 regime and on the perturbed mesh. If both recover order, one hook setting
closes T3, the pressure order, and the perturbed/clustered floors together.

## 21. Where the hours went — the linear solver

Every run in this record used `backend="scipy"`: Jacobi-preconditioned CG on the pressure Poisson
to rtol 1e-12 and Jacobi-BiCGStab on momentum to 1e-10, at every corrector of every step. A
cached sparse-LU path existed in `SolveCache` but was reachable only as an AmgX-fallback refuge.
Exposed as `backend="splu"` (pins DOF 0 on the singular all-Neumann system, then mean-subtracts,
matching the SciPy/AmgX normalisation). T5 n = 64, 500 steps, identical configuration otherwise:

| backend | wall | s/step | L2(u) at step 500 |
|---|---|---|---|
| scipy | 46.1 s | 0.0923 | 6.120375e-05 |
| **splu** | **11.6 s** | **0.0231** | 6.120285e-05 |

4x, same answer to five digits. The operators in the verification suite are fixed across steps
(steady Stokes at constant dt), so the factorisation amortises to nothing. The residual 0.023 s/step
is Python-side assembly, not the solve. `PISO`'s default backend is now `splu`; `scipy`, `petsc`
and `amgx` remain selectable for the cylinder-scale and GPU work where they were chosen.

## 22. The bracket-consistency hypothesis is falsified; the pressure checkerboard is intrinsic

Section 20's unifying hypothesis -- that the GG gradient's inconsistency inside the Rhie-Chow
bracket produces an O(h) spurious flux behind the first-order pressure, the T3-regime velocity
order and the perturbed-mesh floor -- was tested with the `grad_rc` hook and is dead.

**S1, uniform mesh, nu = 1, `grad_p` = dual GG throughout:**

| bracket gradient | u orders | p orders | checkerboard, n = 16 / 32 / 64 |
|---|---|---|---|
| GG | 1.98 / 1.99 | 1.00 / 1.00 | 9.84e-02 / 4.92e-02 / 2.46e-02 |
| face-LSQ, linear-exact | 1.98 / 1.99 | 1.00 / 1.00 | 9.98e-02 / 4.99e-02 / 2.49e-02 |
| unweighted quadratic | 1.98 / 1.99 | 1.00 / 1.00 | 7.70e-02 / 3.84e-02 / 1.92e-02 |

Bracket accuracy does not touch the order of anything. The quadratic bracket trims the
checkerboard AMPLITUDE by 22%, the order stays exactly 1.00 -- the same amplitude-not-order
behaviour `rc_scale` showed.

**S3, perturbed mesh, nu = 1:** GG-dual bracket vs LSQ bracket identical to four digits
(2.8673e-03 vs 2.8655e-03 at n = 16). Two findings that matter more than the falsification:

* Velocity is NOT on a floor there: 2.867e-03 / 1.873e-03 / 1.000e-03, order 0.61 -> **0.91** and
  rising. Slowly converging. Whatever limits it on irregular cells is plausibly GG's own `C != 1`
  bias in the momentum term (Sozer), which the skewness-corrected dual gradient targets.
* Pressure is FLAT: 2.651e-01 / 2.710e-01 / 2.757e-01, order -0.03, with a checkerboard of
  **0.24 -- a quarter of the pressure scale, mesh-independent.** On the uniform mesh the
  checkerboard decays at order 1; on the perturbed mesh it does not decay at all.

**What this establishes about the pressure.** The checkerboard is a genuine near-null mode of the
collocated discretisation. It is harmless to velocity because the divergence-form gradient does
not transmit it (that is the whole content of the T5 fix). It is real for the pressure field
itself, and on irregular meshes its amplitude is set by the mesh, not by h. No choice of gradient
in the bracket, and no scaling of the damping, changes its order -- those change amplitude only.
The clustered cavity's 13%-of-max|p| oscillation (section 9) is the same object.

Routes that remain, in order of cost:
1. **Filter for output.** The smooth part converges at ~1.75 on the uniform mesh (section 20).
   A fixed sparse projection removing the near-null mode, applied to reported pressure, is
   adjoint-friendly and standard. Not yet measured on the perturbed mesh.
2. **Forces may not need it.** A boundary integral of pressure over many faces cancels an
   alternating mode to leading order. Whether cylinder drag/lift are second order with the
   checkerboard present is a direct measurement to make on T9, before any filtering.
3. **Suppress it in the solve** -- Hopman's negative-feedback predictor `theta_p = 1 - C_cb`, or a
   different pressure-space construction. Structural; only if 1 and 2 are insufficient.

Not pursued further today. S2 (T3 regime) and S4 (clustered) are the remaining arms; S2's
question -- what limits the transient-dominated regime -- is now open again, since the bracket
was not it.

**S3 complete, S5 — the smooth pressure bias on irregular meshes.** Perturbed mesh, dual GG:

| n | u | order | p total | smooth part | order | checkerboard |
|---|---|---|---|---|---|---|
| 16 | 2.867e-03 | -- | 2.651e-01 | 6.007e-02 | -- | 2.351e-01 |
| 32 | 1.873e-03 | 0.61 | 2.710e-01 | 6.549e-02 | **-0.12** | 2.386e-01 |
| 64 | 1.000e-03 | 0.91 | 2.757e-01 | -- | -- | 2.433e-01 |

**Even the smooth part of the pressure error does not converge on the perturbed mesh** -- ~6% of
scale, mesh-independent -- where the uniform mesh gives 1.75. So route 1 of section 22 (filter the
checkerboard for output) is NOT viable on irregular meshes: the filtered pressure is still wrong by
6%. This is Sozer's `C != 1` inconsistency of plain Green-Gauss surfacing in the pressure: the
momentum equation sees `C grad p`, so the pressure that yields the right velocity is the wrong
pressure by an O(1) factor locally. On the uniform mesh the bias alternates between the two
triangle orientations and averages out; on the perturbed mesh it is random and does not.
Velocity converges there only slowly (0.61 -> 0.91) for the same reason.

Bracket choice remains irrelevant to four digits at every level (LSQ bracket: 0.61 / 0.90).
Clustered: GG plain n = 64 = 6.204e-03, a floor confirmed; GG dual at n = 16 carries a checkerboard
of **32%** of the pressure scale.

**Fix under test (S6): `GGSkewGradient`** -- dual-weighted Green-Gauss with skewness-corrected
face values, `p_f = (1-w) p_O + w p_N + grad_p_f . (x_f - x_int)` using the LSQ cell gradient for
the correction. Linear-exact (the correction moves the interpolated value from the centroid-line
crossing to the face centroid, exactly for linear fields) AND divergence-form (still telescopes,
still blind to the checkerboard in velocity). Approximately dual: the interpolation term is
exactly dual, the O(h)-relative correction term is not. One fixed sparse matrix. Targets: uniform
T5 unchanged at 1.98 / 1.99; perturbed velocity order toward 2; perturbed smooth-pressure order
from -0.12 to convergent.

**S5 complete, S4 dual GG on the clustered mesh.** Perturbed smooth pressure at three levels:
6.007e-02 / 6.549e-02 / 6.272e-02, orders -0.12 / +0.06 -- flat. The ~6% smooth pressure bias
on irregular meshes with plain-face-value GG is confirmed mesh-independent.

Clustered, dual GG: 7.045e-03 / 6.352e-03 / (n = 64 pending), order 0.15 -- the same floor as plain
GG (6.204e-03 at n = 64). **Exact duality does not help on the clustered mesh.** Together with S1/S3
(bracket irrelevant) this isolates the irregular-mesh floor to one thing: the `C != 1`
inconsistency of plain-face-value Green-Gauss in the MOMENTUM term. That is exactly what
`GGSkewGradient` removes, and it is linear-exact to 1e-13 on all four test meshes (a first version
corrected from the ordinary crossing point instead of the dual one, was exact only on the uniform
mesh where the two coincide, and was caught by the three-mesh exactness check before any solver
run was trusted). S6 uniform arm so far: 6.903e-04 / 1.760e-04, order 1.97 -- second order retained.

## 23. S6 — the skew-corrected gradient does NOT lift the irregular-mesh floors

`GGSkewGradient` as `grad_p` (linear-exact to 1e-13 on all four meshes, divergence-form):

| mesh | n=16 | n=32 | n=64 | orders | GG dual for comparison |
|---|---|---|---|---|---|
| uniform | 6.903e-04 | 1.760e-04 | 4.443e-05 | **1.97 / 1.99** | 1.98 / 1.99 |
| perturbed 0.25 | 2.993e-03 | 2.024e-03 | 1.175e-03 | 0.56 / 0.78 | 0.61 / 0.91 |
| clustered 1.5 | 7.274e-03 | 6.394e-03 | 6.227e-03 | 0.19 / 0.04 | 0.15 / floor 6.2e-03 |

Second order retained on the uniform mesh; **no change** on the irregular ones. The `C != 1`
inconsistency hypothesis of sections 17/20/22 for the irregular-mesh floors is falsified -- the
third hypothesis this suite has killed. Excluded now as the cause on irregular meshes: gradient
consistency in momentum (S6), gradient choice in the bracket (S1/S2/S3/S4), exact duality (S4
plain vs dual). S2 completed: T3-regime velocity 6.352e-03 at n = 64, order 1.10 -- ~1.1-1.15 at
three levels; its pressure converges at 1.71 there (better than velocity, note), and the LSQ
bracket arm is identical (1.18).

What still separates uniform from irregular and is not excluded: the pressure checkerboard is
MESH-INDEPENDENT on irregular meshes (0.24 perturbed, 0.30 clustered) where it decays at order 1 on
the uniform mesh; and the velocity is blind to it only to the extent the divergence-form gradient
is. Measured next: the smooth fraction of `G(checkerboard)` on the perturbed mesh against the
uniform mesh's 4.4%, and the velocity footprint `A^-1(V G cb)` against the measured floor.

## 24. The irregular-mesh floor, diagnosed quantitatively — and the gradient side closed

**The velocity floor on an irregular mesh IS the checkerboard's footprint through the divergence-form
gradient**, and it can be predicted from the converged pressure error alone. Solving
`A x = -V G_GG(cb)` with the momentum operator (homogeneous Dirichlet):

| mesh | checkerboard | GG smooth fraction (non-blindness) | predicted footprint | measured velocity floor |
|---|---|---|---|---|
| uniform | 4.920e-02 | **0.5%** | 1.795e-04 | 1.663e-04 |
| perturbed 0.25 | 2.386e-01 | **16.5%** | 1.522e-03 | 1.873e-03 |

Prediction within 10-20% on both meshes. Two things multiply on the irregular mesh: the
checkerboard is 5x larger and does not decay (section 22), and the GG gradient is 33x less blind
to it. These share a root -- the damping is `D (compact - wide)`, so a non-blind `wide` term
partially cancels the compact one, the same self-cancellation that broke LSQ on the uniform mesh.

**Blindness is a property of the mesh, not the gradient.** Smooth fraction of `G(cb)` for the
converged near-null mode:

| mesh | GG dual | GG skew-corrected | GG with w = 1/2 everywhere | LSQ |
|---|---|---|---|---|
| uniform | 0.5% | 0.5% | 0.5% | 31.9% |
| perturbed 0.25 | 16.5% | 16.3% | **16.6%** | 21.9% |
| clustered 1.5 | 4.4% | 4.5% | 4.5% | 5.8% |

Every divergence-form variant is identically non-blind on the perturbed mesh -- including
`w = 1/2`, which is exactly blind to any two-colour alternating field by construction. So the
near-null mode of an irregular triangulation is NOT a two-colour pattern, and no face-averaged
gradient cancels it. This is why S6 (linear-exact) changed nothing: linear-exactness and
blindness are independent properties, and only blindness matters here. **The gradient side is
closed.** The remaining lever is the checkerboard AMPLITUDE on irregular meshes -- i.e. why the
Rhie-Chow damping leaves it mesh-independent there when it decays at order 1 on the uniform mesh.

Note the clustered mesh is only 4.4% non-blind yet has the HIGHER floor (6.2e-03): its checkerboard
is larger (0.30) and its stretched cells change how `A^-1` amplifies the transmitted mode. The
footprint calculation, not the blindness number alone, is the predictor.

S7 (running): `rc_scale` 1 / 2 / 3 on the perturbed mesh -- does damping strength move the
checkerboard amplitude and the floor there as it did on the uniform mesh?

**S7 — damping strength on the perturbed mesh is a weak lever.** `rc_scale` 1 / 2 / 3, dual GG:

| rc_scale | checkerboard n=16 / 32 / 64 | velocity n=16 / 32 / 64 | orders |
|---|---|---|---|
| 1 | 0.235 / 0.239 / 0.243 | 2.87e-03 / 1.87e-03 / 1.00e-03 | 0.61 / 0.91 |
| 2 | 0.173 / 0.185 / 0.190 | 2.62e-03 / 1.66e-03 / 8.63e-04 | 0.65 / 0.95 |
| 3 | 0.143 / 0.156 / 0.160 | 2.48e-03 / 1.54e-03 / 7.82e-04 | 0.69 / 0.98 |

Tripling the damping cuts the checkerboard by 1.5x and the velocity floor by 1.3x; the
checkerboard stays mesh-independent at every setting, and the order creeps toward 1, not 2.
Amplitude scales roughly as `rc_scale^-0.35`. On the uniform mesh the same knob halved the
checkerboard and left order 1 -- there the mode decays with h; here it does not, at any strength
within the stable window (10x diverged earlier today).

**Conclusion for irregular meshes.** The checkerboard is a genuine near-null mode of the discrete
system whose amplitude is set by the mesh, weakly responsive to damping, not two-colourable, and
therefore not removable by any face-averaged gradient. It induces a velocity floor of ~1e-03 on the
perturbed mesh and ~6e-03 on the clustered one, predicted from the pressure error to 10-20%. This
is not a coefficient problem and will not be tuned away. The remaining routes are structural:

1. **Quad layers at the walls.** On Cartesian quads the mode IS two-colourable, `w = 1/2` exactly,
   and GG is exactly dual and exactly blind -- the uniform-mesh result (1.98 / 1.99) is the
   demonstration. Removes the problem where it matters most, rather than solving it in general.
   Meshing work: ~60 lines in `umesh.py` (`Mesh.__init__`, `_build_faces`), operators untouched.
2. **A different pressure stabilisation** -- Hopman's negative-feedback predictor, or a
   stabilisation built on the actual near-null mode rather than on a compact-minus-wide
   difference. New algorithm; adjoint implications to be assessed.
3. **Accept the floor and measure what matters.** ~1e-03 velocity error on a moderately irregular
   mesh may be acceptable for T9 if drag and lift -- boundary integrals that cancel an
   alternating mode to leading order -- come out second order. Direct measurement on the cylinder.

Not pursued further in this session. The uniform-mesh gates stand; the irregular-mesh limitation
is diagnosed, quantified, predictive, and recorded.

## 25. Speed — where a step goes, and the dt lever

Profile of 40 PISO steps at n = 64 (8192 cells), defaults splu + dual GG + n_inner = 2, before
today's speed work: **18.4 ms/step**, of which the linear solves were 22% and Python-side assembly
78%. The single largest item was the Laplacian's deferred cross-term RHS (`rhs_fn`) at 30% -- four
calls per momentum step and one per Poisson outer pass. Converting its `np.add.at` scatter to
`np.bincount` changed it from 0.223 s to 0.220 s: the scatter was never the cost, the mask indexing
on `(nface, 2)` arrays was. The cross term is linear in the cell gradient, so it is now assembled
ONCE as two sparse matrices and applied as matvecs -- also what the adjoint wants. The pressure
Laplacian, previously rebuilt every corrector although `gam` is constant for fixed dt, is cached.

**The steady march converges in FEWER steps at SMALLER dt.** T5, n = 32, to stagnation:

| dt | steps | wall | L2(u) |
|---|---|---|---|
| 5.0 | 40000 (cap) | 220 s | 1.6694e-04 |
| 0.5 | 31400 | 172 s | 1.6640e-04 |
| 0.05 | 3800 | 20.5 s | 1.6633e-04 |
| 0.02 | 1800 | 9.6 s | 1.6623e-04 |
| 0.01 | 1100 | 5.7 s | 1.6606e-04 |
| **0.005** | **700** | **3.7 s** | 1.6577e-04 |

At large dt the march is not tracking physics; it is a pressure-coupling iteration whose
contraction weakens as `V/a_C = dt/1.5` grows. At dt = 0.005 the steady state is reached in 3.5
time units, about 70 viscous times -- near the physical floor. The answer is dt-independent to
four digits throughout. `test_ustokes.py` now uses dt = 0.005 and runs to stagnation of the error
rather than a fixed step count (the fixed-count version reported the "order 1.1" of section 13 that
was an unconverged transient over a flat floor). This is a 5.5x that costs nothing.

Not done, in order of value: precompute the index arrays and constant coefficients in
`_rhie_chow` and `_pressure_flux` (together ~20% of a step, same mask-indexing pattern); a true
steady solver for the verification gates -- for linear Stokes the converged PISO fixed point is
ONE sparse saddle-point solve, which would replace the march entirely; numba/Cython on what remains.

## 26. Route 1 — polygon meshes, and quad wall layers

`Mesh` now accepts triangles, quads or any mix (`cells` as an (ncell, k) array or a ragged list;
stored padded with vertex counts in `nvert`; `tris` survives as an alias only for all-triangle
meshes). Shoelace area and area-centroid; every polygon oriented counter-clockwise; the face
builder enumerates each cell's `nvert` edges cell-major, so an all-triangle mesh reproduces the
old face ordering exactly. `rect_mesh(cells="tri"|"quad"|"hybrid", wall_layers=k)`: hybrid keeps
the outermost k rows/columns as quads and splits the core.

Verification before any solver run was trusted:

| mesh (32x32) | cells | audit | lap(linear) | GG-dual grad(linear) | duality | orth min |
|---|---|---|---|---|---|---|
| triangles (regression) | 2048 tri | CLEAN | -- | -- | -- | 0.894 |
| quad uniform | 1024 quad | CLEAN | **0.0** | **0.0** | 1.4e-13 | **1.0000** |
| quad clustered 1.5 | 1024 quad | CLEAN | 8.9e-15 | 2.7e-01 | **9.4e-16** | **1.0000** |
| hybrid, 4 quad layers, clustered 1.5 | 448 quad + 1152 tri | CLEAN | 8.9e-15 | 2.9e+00 | 3.8e-16 | 0.655 |

Triangle T5 n = 16: 6.557684008e-04, identical to nine digits. Three things the table shows:

* **Stretched Cartesian quads have ZERO non-orthogonality.** `orth min = 1.0000` at cluster 1.5,
  where the same clustering on triangles gave 69 deg (tan = 2.6, divergent before the
  over-relaxed fix). The irreducible-skewness argument of section 5 is confirmed from the other
  side: the diagonal was the whole problem.
* On uniform quads the dual Green-Gauss gradient is EXACT for linear fields (0.0), and dual to the
  divergence -- both properties at once, which no triangulation achieved. On stretched quads it is
  dual (9.4e-16) but not linear-exact (0.27; `w != 1/2`) -- S6 showed the latter does not matter.
* The hybrid mesh's worst face (0.655) is in the triangle CORE, not at the quad/triangle interface
  -- interface faces share one node pair and are ordinary manifold edges.

S8 (running): T5 on quad uniform, quad clustered 1.5 and 2.0, and the hybrid. Prediction from the
mechanism of sections 22-24: the clustered QUAD meshes should be second order -- no skewness, no
non-orthogonality, a two-colourable near-null mode, exact duality -- where clustered triangles
sat on a 6.2e-03 floor.

## 27. S8 — on quads, BOTH velocity and pressure are second order; the checkerboard is gone

T5 Stokes, nu = 1, dt = 0.005, defaults (dual GG, `n_inner` = 2, splu):

| mesh | u: n=16 / 32 / 64 | u orders | p orders | checkerboard orders | steps |
|---|---|---|---|---|---|
| quad uniform | 2.169e-03 / 5.445e-04 / 1.363e-04 | **1.99 / 2.00** | **1.92 / 1.98** | **3.48 / 3.52** | 200-400 |
| quad clustered 1.5 | 2.629e-03 / 6.726e-04 / 1.689e-04 | **1.97 / 1.99** | **2.03 / 2.01** | 3.76 / 3.94 | 200-900 |
| quad clustered 2.0 | 3.940e-03 / 1.025e-03 / 2.582e-04 | **1.94 / 1.99** | **2.05 / 2.02** | 3.63 / 3.91 | 300-1300 |
| triangles, uniform (ref) | 6.50e-04 / 1.66e-04 / 4.19e-05 | 1.97 / 1.99 | 1.02 / 1.01 | 1.00 / 1.00 | 1500-14500 |
| triangles, clustered 1.5 (ref) | 7.05e-03 / 6.35e-03 / 6.20e-03 | floor | flat | flat | 5000-40000 |

**Everything the triangular meshes could not do, the quads do at design order, clustered or not.**
Pressure second order; the checkerboard decays at order ~3.5-4 instead of order 1 (uniform tri) or
not at all (irregular tri); convergence to steady state 10-40x fewer steps. The velocity constant is
~3x larger than on the uniform triangles (fewer cells per unit area at equal n), but the order is
the design order on every quad mesh.

This confirms the mechanism of sections 22-24 from the constructive side: on Cartesian quads,
stretched or not, `w = 1/2` exactly, faces are exactly orthogonal (measured `orth min = 1.0000` at
cluster 2.0), the near-null mode is two-colourable so the divergence-form gradient is exactly blind
to it, and the gradient is the exact adjoint of the divergence. The pressure-order problem, the
irregular-mesh velocity floor and the non-decaying checkerboard were never properties of the
scheme; they were properties of triangulations.

**The hybrid mesh (4 quad wall layers, triangle core) FAILS**: n = 16 3.152e-03, n = 32 5.259e-03
-- order **-0.74** -- with a checkerboard of 0.28, WORSE than pure triangles at either resolution.
The quad/triangle interface generates the mode rather than suppressing it: the two-colour pattern
of the quads and the alternating-diagonal pattern of the triangles do not match, and the mismatch
line acts as a source. (Also note `wall_layers` was fixed at 4 cells, so the quad band thins under
refinement -- not a clean refinement family; but a negative order is not a refinement artefact.)
n = 64 pending. Consequence for the cylinder: an all-quad mesh where possible -- a quad O-ring around
the body and quads outward -- rather than a thin quad skin on a triangular core.

## 28. The HydroGym reference mesh, regenerated from source

`medium.msh` on this machine was a 130-byte git-lfs pointer and the HydroGym copy is an sdist with
no `.git`, so `lfs pull` was impossible. But `medium.geo` -- the Gmsh source, i.e. the spec itself
-- ships alongside it. Regenerated with gmsh 4.15.2 (`-2 -format msh2`) into
`meshes/cylinder_medium.msh`: 868,345 bytes against the pointer's 868,324, and exactly the 230
boundary edges the P0 record lists (16 Inlet, 60 Freestream, 42 Outlet, 112 Cylinder). Cell/face
counts and area in the line above. Worst face 29.1 deg; all triangles.

The `.geo` confirms the case geometry the plan assumed: cylinder radius 0.5 at the origin, domain
x in [-5, 15], y in [-5, 5], size 1/35 on the body, 1/20 at the outlet centreline, 1/1.5 at the far
corners, two surfaces split along y = 0. Having the source rather than the file also means a
quad-recombined variant is a one-line change -- which, after section 27, is the mesh the cylinder
should run on.

Regenerated `meshes/cylinder_medium.msh`: **17258 cells, 26002 faces, 230 boundary, area 199.2150,
audit CLEAN** -- every P0 number reproduced exactly. T9 is unblocked.

**Quad-recombined variant** (`meshes/cylinder_medium_quad.geo`: Blossom recombination, Frontal-
Delaunay-for-quads): 7413 cells = 7409 quads + 4 triangles (99.9%), 14940 faces, 232 boundary,
audit CLEAN, area 199.2150. Two cautions before it is used: worst face **54.9 deg** off-normal
(recombined Delaunay quads are irregular, not Cartesian -- the triangle mesh's worst was 29.1), and
the 4 residual triangles sit at r = 0.51, ON the cylinder wall, exactly where section 27 says a
quad/triangle interface must not be. Whether the section-27 property survives IRREGULAR quads is
therefore the deciding question for route 3's mesh, and is tested next on gmsh-recombined quads of
the unit square with the T5 manufactured solution. If it does not survive, the cylinder needs a
structured quad O-ring (transfinite) rather than recombination.

**Hybrid, final.** n = 64: 5.977e-03, order -0.18 -- the same floor as pure clustered triangles
(6.20e-03). A thin quad skin on a triangle core buys nothing: the floor is set by the triangle
region, and the interface at least does not help. The mesh choice is binary -- quads, or the floor.

## 29. S9 — irregular (recombined) quads are intermediate; the cylinder needs a structured O-grid

gmsh Blossom-recombined quads of the unit square, graded lc..2lc so the quads are irregular
(worst face 11.5 / 19.0 / 27.8 deg at the three levels), T5 Stokes:

| level | cells | u | order | p | order | checkerboard | order |
|---|---|---|---|---|---|---|---|
| H=16 | 146 quads | 4.511e-03 | -- | 7.98e-02 | -- | 7.61e-02 | -- |
| H=32 | 561 quads | 1.463e-03 | **1.62** | 1.17e-01 | -0.56 | 1.15e-01 | -0.59 |
| H=64 | 2018 quads | 4.564e-04 | **1.68** | 7.92e-02 | +0.57 | 7.72e-02 | +0.57 |

Velocity recovers to ~1.65 -- far above irregular triangles (0.6-0.9), below Cartesian quads (2.0).
Pressure does NOT converge: the checkerboard is back at 0.08-0.11 (three times weaker than on
perturbed triangles, but present and wandering). Recombined quads are not two-colourable (odd
vertex degrees), so the divergence-form gradient is only partially blind to the near-null mode.

**Consequence for T9.** The cylinder mesh should be a structured transfinite butterfly O-grid:
curvilinear but structured quads -- two-colourable, near-orthogonal, exactly the S8 properties --
around the body and Cartesian blocks outward. Not gmsh recombination (S9, plus the 4 residual
triangles it leaves ON the wall). The butterfly topology is the one the structured phase could not
tile because of its indexing parity obstruction; in gmsh the node-count constraints on shared
curves are the tool's problem and every seam is just a face. The triangle `medium.msh` remains the
HydroGym-conformant reference; running both meshes on the same solver and comparing St and C_D is
the cleanest possible test of whether the triangle floor reaches the forces.

## 30. The structured butterfly O-grid for the cylinder

`meshes/cylinder_butterfly.geo`, generated programmatically (a shared-edge helper returns a signed
line id for a reversed request, so every seam is one line -- the duplication that defeated the
structured phase cannot occur). Topology: four transfinite quad blocks between the quarter arcs
(cylinder points at 45/135/225/315 deg) and the sides of an inner square of half-width 1.5, then
eight transfinite Cartesian blocks filling [-5,15] x [-5,5]. Counts Nt = 29 nodes per quarter arc
(-> 112 cylinder edges, matching `medium.msh`), Nr = 24 radial with progression 1.08 outward from
1/35 at the wall, NL = NB = 13 with progression 1.20 outward, NR = 61 downstream with progression
1.02 (wake). All node counts on shared curves match by construction.

Result, first attempt: **6992 cells, 100% quads, 0 triangles**, 14192 faces, 416 boundary, area
199.2150 (analytic 199.2146), audit CLEAN, HydroGym's five physical tags reproduced. Worst face
43.1 deg (the inner blocks shear where a quarter arc maps onto a straight side); 1078 faces below
26 deg, none below cos 0.7 -- against 55 deg for recombined quads and 69-81 deg for clustered
triangles. Structured, hence two-colourable around the body (4 x 28 = 112 cells, even). Minimum
cell 0.0217 at the wall against the spec's 0.0286. A coarse variant (all counts halved) is emitted
by scaling the transfinite counts.

S10 (running): the Stokes MMS on this domain -- wavelength 20 so the far-field coarseness does not
dominate -- on coarse and medium butterfly and on the HydroGym triangle mesh. Same solver, same
problem, three meshes: the direct test of whether the triangle floor and non-decaying checkerboard
of sections 22-24 appear on the reference mesh, and whether the butterfly avoids them.

## 31. Route 1 deliverable — the cavity on all-quad meshes

Re = 1000, 64x64, defaults (dual GG, `n_inner` = 2, splu). NOTE: 64x64 quads is 4096 cells against
8192 triangles at the same n -- half the resolution.

| mesh | cells | u rms | v rms | u_min | v_min | v_max | core vorticity | p checkerboard |
|---|---|---|---|---|---|---|---|---|
| tri, uniform (LSQ, section 9) | 8192 | 0.0581 | 0.0351 | +5.5% | +1.0% | +4.4% | +22.7% | 10.2% |
| tri, beta 1.5 (LSQ, best tri) | 8192 | **0.0260** | **0.0142** | -0.6% | **+0.04%** | -1.2% | **+0.1%** | ~13% |
| quad, uniform | 4096 | 0.0413 | 0.0245 | -5.2% | -4.8% | -5.0% | -4.2% | 19.7% |
| **quad, beta 1.5** | **4096** | **0.0260** | 0.0177 | -1.1% | **-0.1%** | -0.9% | -1.2% | **17.9%** |
| Ghia | | | | -0.3829 | -0.5155 | +0.3710 | -2.0497 | |

Clustered quads match the best triangle result on u rms and v_min with HALF the cells, and are
within 1.2% on every other measure. Uniform quads at 4096 cells beat uniform triangles at 8192 on
every measure. A fair equal-cell comparison (91x91 quads, ~8281 cells) is running.

**The pressure checkerboard is LARGER on quads (18-20%) than on triangles (10-13%).** This is the
flip side of the property that fixed T5: on a two-colourable mesh the divergence-form gradient is
EXACTLY blind to the near-null mode, so the velocity dynamics provide no damping of it at all --
only Rhie-Chow does -- and the lid-corner singularity sources it at every resolution. On S8's
smooth Stokes problem the source was tiny and the mode decayed at order 3.5; here the source is
O(1). Harmless to the velocity (that is the whole point), but pressure output must be filtered.
On a two-colourable mesh that filter is EXACT: the mode is a pure +-1 colouring, one fixed vector,
and projecting it out is one inner product -- adjoint-trivial. Being tested on the saved fields.

**Correction to the paragraph above.** The exact two-colour filter was applied to the saved fields.
The mesh IS two-colourable (BFS confirms), but the global +-1 mode has amplitude **8e-05 (beta 0) and
1e-05 (beta 1.5)** -- essentially zero -- and projecting it out changes nothing (19.7% -> 19.7%).
So the 18-20% figure is NOT the near-null checkerboard. Decomposing the residual oscillation by
position: near-lid max 0.133 / 0.226, **interior max 0.0040 / 0.0028** -- a 30-80x contrast. It is
the lid-corner singularity's imprint, confined to the two or three cell rows under the lid, which
every discretisation of this problem carries. The quad cavity's INTERIOR pressure oscillation is
0.3-0.4% of max|p| -- cleaner than the triangles' 0.5% (section 9). The near-null mode that
plagued every triangular mesh is absent on the quads, in the cavity as in S8. The "flip side of
blindness" argument above was wrong as an explanation of this number and is withdrawn; the
underlying point -- that on a two-colourable mesh the global mode, if present, is removable
exactly by one projection -- stands, and the projection is now written and tested.

**S10, first two levels** (`figures/cylinder_meshes.png` shows both meshes). Stokes MMS on the
cylinder domain, wavelength 20:

| mesh | cells | h_mean | worst face | u | p | checkerboard (% of \|p\|) |
|---|---|---|---|---|---|---|
| butterfly coarse | 1776 | 0.335 | 41.9 deg | 3.603e-03 | 4.749e-02 | 4.410e-02 (8.8%) |
| butterfly medium | 6992 | 0.169 | 43.1 deg | 1.669e-03 | 2.961e-02 | 2.727e-02 (5.5%) |
| orders per halving of h | | | | **1.12** | 0.69 | 0.70 |

NOT the Cartesian-quad result (S8: u 2.0, checkerboard decaying at 3.5). Better than irregular
triangles (0.6-0.9, checkerboard flat), roughly the recombined-quad regime (S9: 1.65) or below. Two
coarse levels are thin evidence -- the coarse mesh has ~15 far-field cells per wavelength -- and a
fine level (all counts doubled, ~28k cells) is running for a real order. But the likely reason is
already visible in the mesh: the inner blocks shear to 43 deg where a quarter arc maps onto a
straight square side, and S8's property rested on near-orthogonality as much as on
two-colourability. A rounder inner boundary (an outer CIRCLE rather than a square, i.e. a true
O-grid annulus, then Cartesian blocks outside a larger square) would keep the inner cells close to
orthogonal. Also visible in the figure: the HydroGym triangle mesh carries a wake refinement band
to the outlet (the `.geo`'s 1/20 at the outlet centreline) that the butterfly's uniform right
column does not reproduce -- a separate design gap to close before T9.

**O-grid variant and the transition collar.** `meshes/cylinder_ogrid.geo`: a polar annulus
r = 0.5 -> 1.35 (four transfinite blocks), then four transition blocks from the outer circle to the
square, then the eight Cartesian blocks. 6992 cells, all quads, audit CLEAN, 112 wall edges. The
annulus is **exactly orthogonal (0.0 deg)** -- the whole boundary-layer region is a perfect polar
grid -- but the global worst face is unchanged at 43.1 deg, now in the transition blocks' corner
kites at r ~ 1.4-2.1 instead of against the wall. A sweep of the collar geometry, (a, Ro) from
(1.5, 1.35) through (3.0, 1.5) to (2.5, 2.0): worst transition angle 42.3-43.1 deg in every case.
**The shear is intrinsic to mapping a smooth arc onto a 90-degree square corner with diagonal seam
lines; it is scale-free and not tunable by proportions.** Whether it matters -- S8's property rested
on near-orthogonality as well as two-colourability -- is what S11 (O-grid vs butterfly at identical
cell count) measures. Smoothing probe below.

**S10 complete -- the HydroGym triangle mesh, same MMS, same solver:**

| mesh | cells | h_mean | worst face | u | p | checkerboard (% of \|p\|) | steps / wall |
|---|---|---|---|---|---|---|---|
| butterfly coarse | 1776 | 0.335 | 41.9 | 3.603e-03 | 4.749e-02 | 8.8% | 5800 / 30 s |
| butterfly medium | 6992 | 0.169 | 43.1 | 1.669e-03 | 2.961e-02 | 5.5% | 5800 / 98 s |
| **HydroGym `medium.msh` (triangles)** | **17258** | 0.107 | 29.1 | 1.723e-03 | 2.731e-02 | 5.2% | 6400 / 241 s |

**The butterfly at 6992 cells matches the HydroGym reference at 17258 on velocity, pressure and
checkerboard -- 2.5x fewer cells, 2.5x less wall time, despite a coarser mean spacing.** That is
route 1's payoff on the target geometry, measured with the same solver and the same manufactured
problem. Neither mesh shows S8's Cartesian cleanliness (checkerboard decaying at 3.5): both carry a
~5% checkerboard, so the 43-deg transition shear in the butterfly and the 29-deg triangles both
cost something, and by about the same amount.

**Coarse O-grid diverged (NaN)**, and the S11 script had no per-arm guard so the medium O-grid did
not run -- the same mistake as the clustered run earlier today; relaunched guarded as S11b. The
coarse O-grid's cell aspect (5.3) is not far from the butterfly's (4.1), so aspect alone does not
explain it; the O-grid medium result decides whether the topology or the coarseness was at fault.
gmsh `Mesh.Smoothing = 20` on the O-grid made the worst face slightly WORSE (43.9) and destroyed the
annulus's exact orthogonality (0.0 -> 1.9 deg) -- not used.

**S11b -- O-grid medium vs butterfly medium, identical cell count (6992):** u 1.656e-03 vs
1.669e-03, p 2.930e-02 vs 2.961e-02, checkerboard 5.4% vs 5.5%. **Identical.** Relocating the 43-deg
shear from the wall to r ~ 1.4-2.1 changed nothing -- because the MMS error is not near the wall.
By region: annulus 2.85e-04, transition 9.10e-04, **far field 4.55e-03**. The coarse Cartesian
far-field blocks dominate 16:1. So S10's order 1.12 measures far-field coarseness against a
domain-scale wavelength, not near-body mesh quality; and the coarse O-grid's NaN was coarseness
(transition cells 0.04 thick at 1720 cells), not topology -- the medium O-grid ran cleanly.
Consequence: the cylinder-domain MMS is a poor discriminator of near-body mesh quality, and the
O-grid's exactly-orthogonal annulus is a real but so-far-unmeasured advantage for the boundary
layer. Both meshes are kept; T9 on both will discriminate where the MMS cannot.

**Cavity at EQUAL cell count -- route 1's result on the practical problem.** 91x91 quads = 8281 cells
against 8192 triangles, both clustered beta = 1.5, Re = 1000, 64x64-equivalent resolution:

| | u rms | v rms | u max\|err\| | v max\|err\| | u_min | v_min | v_max | core vorticity |
|---|---|---|---|---|---|---|---|---|
| best triangle (LSQ, beta 1.5) | 0.0260 | 0.0142 | 0.0914 | 0.0409 | -0.6% | +0.04% | -1.2% | +0.1% |
| **quad beta 1.5, 8281 cells** | **0.0100** | **0.0093** | **0.0379** | **0.0185** | **-0.2%** | +1.6% | **+0.4%** | **-0.5%** |

2.6x on u rms, 2.4x on u max, 2.2x on v max; every extremum within 1.6% of Ghia. At equal cost the
quad mesh is the better mesh by a wide margin, on the same solver with the same defaults. The
morning's clustering result (2.5x from wall resolution on triangles) and this one (2.6x from cell
shape at equal count) compound.

## 32. Route 3 preparation — a boundary-condition bug the cylinder would have hit

Rhie-Chow's `fixed_u` mask (no damping on faces whose flux is prescribed) required BOTH velocity
components to be Dirichlet. The cylinder's Freestream boundary is u-Neumann / v-Dirichlet = 0 on
y-normal faces -- a symmetry plane whose NORMAL flux is exactly prescribed -- and would therefore
have received damping, injecting spurious net mass through a symmetry plane: the mechanism of bug
(18), which broke all-Neumann compatibility and smeared error across the domain. Generalised: a
face is fixed when the component along its normal is Dirichlet (axis-aligned mixed faces handled;
oblique mixed faces are not a well-posed prescription and fall back to requiring both). T5 is
all-Dirichlet and unchanged by construction (regression above). None of the gates so far had a
mixed-component boundary, which is why this survived them all.

**S10b -- the butterfly PLATEAUS on the cylinder-domain MMS.** Fine level (27968 cells, h_mean
0.084, worst 43.7 deg): u 1.245e-03, p 2.744e-02, checkerboard 2.564e-02. Orders medium -> fine:
**u 0.42, p 0.11, cb 0.09.** Three levels: 3.60e-03 / 1.67e-03 / 1.25e-03. The butterfly does NOT
carry S8's Cartesian-quad property on this domain, and S11b located the error in the far-field
blocks, not near the body. Where the CHECKERBOARD lives (block seams? far field? the hole?) is
being measured on the medium butterfly. Three candidates, all absent from S8's single-block unit
square: the twelve block seams (cell-size and weight jumps); the annular topology around the body
(a near-null mode on a domain with a hole); and the far-field grading (progression 1.20).

## 33. T9 — first run

`run_ucylinder.py`: mixed BCs from the Gmsh tags (Inlet u = 1, v = 0; Freestream v = 0, u free;
Outlet p = 0, u and v free; Cylinder no-slip), wall forces `F = sum_f [p_f S_f - tau_f . S_f]` with
tau = nu(grad u + grad u^T) from the LSQ cell gradient at the wall cell, C_D = 2F_x, C_L = 2F_y, St
from the lift spectrum over the last 40% of the run. Sanity run, butterfly medium, 300 steps at
dt = 0.01: C_D 2.63 -> 1.56 -> 1.51 (impulsive-start transient decaying), |u|max 1.38, no NaN,
63 ms/step with convection on.

**Bug caught by the sanity run: C_L was identically 0.0000.** The symmetry-breaking perturbation
carried a `sign(y)` factor -- ODD in y, the same symmetry class as the base flow's v -- so it broke
nothing. Replaced with a y-EVEN blob of v in the near wake. Long runs launched (T = 150, dt = 0.01,
15000 steps) on the butterfly (6992 quads) and on the regenerated HydroGym triangle mesh (17258):
same solver, same defaults, same BCs, two meshes.

**Where the butterfly's checkerboard lives: everywhere.** L2 of the oscillatory pressure error by
region, medium butterfly, cylinder-domain MMS: inner four blocks 1.72e-02, far field 1.78e-02, left
column 2.30e-02, right column 1.39e-02, top/bottom rows 2.77e-02, within 0.3 of a block seam
1.63e-02, far from any seam 1.81e-02. Peak 7.35e-02 at (+1.45, -1.40), a transition-block corner
kite. **Uniform to within a factor of two across every region** -- not a seam artefact, not a
far-field artefact, not concentrated at the wall. A global near-null mode of the whole mesh, at a
level S8's single-block quads never showed (theirs decayed at order 3.5). The structural difference
that remains is the HOLE: the butterfly is a two-colourable mesh on a multiply-connected domain.
Two tests running: the same MMS on the same 20 x 10 box WITHOUT the cylinder (single-block Cartesian
quads, similar cell count -- if the mode vanishes, the hole is the cause), and the exact two-colour
projection on the butterfly's saved checkerboard (if it removes the mode, the filter of section 31
applies; if not, the mode is not the +-1 colouring).

## 34. The butterfly is NOT two-colourable — and that is the whole explanation

S12a: BFS two-colouring of the medium butterfly's cell-adjacency graph: **`two-colourable: False`**.
The global +-1 mode amplitude is 1e-04 and projecting it out removes nothing (1.780e-02 -> 1.780e-02).
The checkerboard on the butterfly is not the two-colour mode because the mesh has none.

Why: at each of the four inner-square corners FIVE blocks meet (two inner transition blocks and
three axis-aligned outer blocks -- middle row, middle column, corner). The five cells around that
vertex are pairwise adjacent in sequence, an odd cycle, and one odd cycle makes the whole graph
non-bipartite. A Cartesian grid has exactly four quads at every interior vertex, which is why S8's
single-block quads were bipartite and their checkerboard decayed at order 3.5. Every other vertex
of the butterfly is fine: seam-interior vertices have four quads, box-edge and cylinder vertices
are boundary. Four bad vertices frustrate the colouring everywhere -- which is why the
checkerboard was UNIFORM across regions (the paragraph above) and peaked at a square corner.

S12b, first level -- the same 20 x 10 box WITHOUT the cylinder, one Cartesian block, 1800 quads:
u 7.487e-04, checkerboard **3.45e-03 (0.7%)** -- against the 1776-cell butterfly's 3.603e-03 and
4.41e-02 (8.8%). Twelve times less checkerboard and five times less velocity error at the same cell
count. **The hole is not the cause; the block-corner valence is.** This is a general result for
body-fitted multi-block quad meshes: the near-null pressure mode is suppressed only if every
interior vertex has an even number of cells around it, and the standard butterfly's square
corners violate that.

**Fix: an 8-block O-grid whose outer region is four blocks with DIAGONAL seams from the square
corners to the box corners** (`meshes/cylinder_bf2.geo`). Every interior vertex then has exactly
four quads. Cost: transfinite blocks need opposite sides equal, so the four diagonals share one
node count, and the box sides carry the square's Nt -- the far field is coarser and the wake
refinement is limited by that shared count. Being verified for two-colourability and run on the
same MMS (S13).

**bf2 built and verified bipartite.** `meshes/cylinder_bf2.msh`: 7056 quads, audit CLEAN, 112 wall
edges, min wall cell 0.0217, area 199.2150, **two-colourable: True** at both the coarse (1736) and
medium levels -- against the 12-block butterfly's False. The price is shear: worst face **75.1 deg**
and 5772 faces beyond 26 deg (butterfly: 43.1 deg, 1078). The diagonal seams run at 45 deg on the
inlet side (square corner (-1.5, -1.5) to box corner (-5, -5)) and 14.5 deg on the outlet side, so
the top and bottom blocks twist between two very different seam angles. Two constraints of the
transfinite topology also bite: the four diagonals must share one node count, and the box sides
inherit the square's Nt, so the far field is coarse and the wake resolution is fixed by that
shared count. S13 (running) is the direct test of whether bipartiteness beats shear on the MMS;
if it does, shear can be reduced afterwards by moving the seam endpoints along the box edges
(any four-block partition of the frame keeps the corners 4-valent) and by grading the box-side
node distributions toward the steep seam. If it does not, the near-null mode is not the only thing
that matters and the 12-block butterfly's near-body accuracy (annulus error 2.85e-04) stands.

**Cavity contours, triangles vs quads at equal cell count** (`figures/ucavity_contours_quad_vs_tri.png`).
Velocity and streamlines near-identical: primary vortex, both bottom corner eddies, Ghia's Table III
centres inside them. Primary vortex centre: tri (0.5345, 0.5601), quad (0.5364, 0.5724), Ghia
(0.5313, 0.5625). Core vorticity at Ghia's centre: tri -2.052, quad -2.040, Ghia -2.050. **The
triangle mesh's pressure and vorticity isolines are visibly ragged through the core and along the
walls; the quad mesh's are smooth.** The near-null footprint measured all day, now visible in the
derivative fields on the practical problem.

**S13 — bipartite bf2 on the cylinder MMS: better ORDER, worse ACCURACY.** Coarse -> medium: u
1.936e-02 -> 5.989e-03 (order **1.67**, butterfly 1.12), checkerboard 32% -> 14% (order **1.33**,
butterfly 0.70). Bipartiteness does what the theory says. But at equal cell count the absolute
error is **3.6x worse** than the butterfly (5.99e-03 vs 1.67e-03) and the checkerboard 2.5x larger:
the 75-deg shear of the diagonal seams costs more, at these resolutions, than the four odd cycles
did. bf2 would overtake the butterfly only at much finer resolution. Not the T9 mesh as built; a
bipartite topology with low shear remains the target, and the two constraints are in tension --
4-valent square corners force the outer seams to the box corners, and the box's 20 x 10 aspect
with the body at x = 0 makes the inlet-side seams steep.

**S12b, second level -- the no-cylinder box is clean second order:** 120 x 60 = 7200 quads, u
1.833e-04 (**2.03**), p 2.192e-03 (1.83), checkerboard 7.02e-04 (**2.30**). The hole is fully
exonerated; the multi-block corner valence is the entire cause of the butterfly's plateau.

**T9 butterfly, shedding saturated.** C_L amplitude 0.09 (t = 20) -> 0.23 (30) -> 0.34 (60) ->
0.34 (70), C_D 1.452, pressure drag 1.106, |u|max 1.38. Within the expected range for a beta = 0.10
confined cylinder at Re = 100 (unconfined literature: C_D 1.33-1.35, C_L amplitude ~0.33; blockage
raises C_D). St from the final 40% of the run. Decision for T9 at this resolution: the 12-block
butterfly -- 3.6x more accurate than bf2, 2.5x fewer cells than the HydroGym triangles for equal
MMS error -- with its known non-bipartite plateau recorded as the reason its pressure field, like
the triangles', will need filtering for anything pointwise.

**S12b complete -- the body-free 20 x 10 box, three levels:**

| box | quads | h | u | order | p | order | checkerboard | order |
|---|---|---|---|---|---|---|---|---|
| 60x30 | 1800 | 0.333 | 7.487e-04 | -- | 7.790e-03 | -- | 3.451e-03 | -- |
| 120x60 | 7200 | 0.167 | 1.833e-04 | **2.03** | 2.192e-03 | 1.83 | 7.018e-04 | **2.30** |
| 240x120 | 28800 | 0.083 | 4.463e-05 | **2.04** | 6.962e-04 | 1.65 | 1.780e-04 | **1.98** |

Velocity second order at three levels; checkerboard decaying at ~2 and at 0.0% of |p| by the
finest level. Same domain, same MMS, same solver as the butterfly -- which plateaued at 0.42 with a
5.5% checkerboard. The difference is one thing: a single Cartesian block has four quads at every
interior vertex. **Design rule, now established from both sides on this solver:** the collocated
near-null pressure mode is suppressed if and only if the cell-adjacency graph is bipartite.
Cartesian quads (clustered or not): yes. Triangles: no. gmsh-recombined quads: no. 12-block
butterfly: no (five blocks at each inner-square corner). Diagonal-seam 8-block: yes, at a shear
cost that outweighs it at practical resolutions. Body-free box: yes. Pressure order on the box is
1.65-1.83 rather than 2. UNTESTED hypothesis: the wavelength-20 field on a 20 x 10 box has its
pressure extrema on the boundary, where the Neumann-effective boundary pressure (owner value) is
first order. Consistent with S8's uniform-square pressure order of 1.92-2.05 on a field whose
extrema are interior, but not measured; recorded as a hypothesis, not a conclusion.

## 35. T9 -- the cylinder at Re = 100: butterfly quads vs HydroGym triangles

`run_ucylinder.py MESH --T 150 --dt 0.01`, identical setup on both meshes (impulsive start, y-even
perturbation, window t = 90..150, St from linearly interpolated C_L zero-crossings; the FFT on the
same signals gives 0.1833 for both -- one 0.0167 bin -- and parabolic peak refinement 0.1751/0.1783,
neither adequate at the 1% level).

| quantity | butterfly, 6992 quads | HydroGym medium, 17258 tris | tri/quad | unconfined literature |
|---|---|---|---|---|
| **St** (10 periods, std 0.0000 both) | **0.1750** | **0.1775** | +1.4% | 0.164-0.167 |
| C_D mean | 1.4529 | 1.4762 | +1.6% | 1.33-1.35 |
| C_L amplitude | 0.3457 | 0.3755 | +8.6% | ~0.33 |
| pressure drag C_Dp | 1.1072 | 1.1095 | +0.2% | |
| wall time / step | 66 ms | 160 ms | 2.4x | |

Both are periodic to the fourth decimal from t ~ 60. All headline numbers sit 5-8% above the
unconfined values, in the direction and roughly the size that beta = 0.10 blockage produces;
HydroGym's own numbers on this mesh are not in hand (plan caveat stands).

The mesh-to-mesh comparison is route 3's answer. Pressure drag agrees to 0.2%, so the triangle
mesh's non-decaying pressure checkerboard does not reach the pressure force -- as the O(h)
alternating-sign argument of section 36 predicts. The differences are in the shear-dependent
quantities: viscous drag (0.367 vs 0.346, +6%), lift amplitude (+8.6%) and St (+1.4%). Which mesh
is closer is not decided by this pair: the butterfly has 2.5x fewer cells and no wake refinement
band, the triangles have the wall checkerboard (rms(p - neighbour mean) in the r < 0.6 layer
8.9e-3 vs 2.9e-3, max 6.7e-2 vs 1.2e-2) and first-order gradients at the wall where the shear is
read. A butterfly refinement (`cylinder_butterfly_fine`, 27968 quads) settles it.

Pressure fields at t = 150 (`figures/t9_pressure.png`) are smooth on both meshes at the outlet:
rms(p - neighbour mean) 4.4e-3 (quads, 0.5-wide cells) and 2.2e-3 (tris) for x > 14 -- no outlet
wiggles on the cylinder. The p = 0 Dirichlet outlet flattens a vortex core sitting at x ~ 14 on
the triangle mesh (p range -0.114..0.035 in the last unit), the usual fixed-pressure-outlet
footprint, identical in kind on both meshes.

## 36. Wall pressure wiggles on the triangle cavity (user observation, verified)

The ragged pressure contour along the east wall of the triangle cavity panel in
`figures/ucavity_contours_quad_vs_tri.png` is real, not a contouring artefact. Reading the cell
values of the 64 east-wall cells directly (`figures/ucavity_eastwall_pressure.png`): the pressure
alternates strictly (-+-+-+ over y = 0.75..0.95), with amplitude equal to the local h |dp/dy|.

| east wall, y in 0.8..0.95 | rms alternating component | rms h|dp/dy| | ratio |
|---|---|---|---|
| triangles, LSQ momentum gradient | 2.69e-2 | 2.32e-2 | 1.16 |
| triangles, GG-dual momentum gradient | 2.50e-2 | 2.33e-2 | 1.07 |
| quads 91x91 | 8.1e-4 (all one sign: curvature) | 1.69e-2 | 0.05 |

Properties: (i) first order -- amplitude ~ h times the tangential pressure gradient, largest where
the lid-driven jet decelerates into the corner (dp/dy ~ 2); (ii) confined to a boundary layer of
cells -- rms (p - neighbour mean) is 4.8e-3 in the first layer, 3.1e-3, 1.7e-3, 8.6e-4 outward,
and 6.0e-4 in the interior (the quad interior is 1.3e-4, pure curvature); (iii) independent of the
momentum gradient (LSQ and GG-dual within 10%), so it is the pressure equation's near-null mode,
not the gradient -- the same non-bipartite mechanism as section 34, excited at a Neumann wall
because every wall square contributes exactly one wall-edge triangle whose face pattern has period
two squares along the wall. This is the "boundary checkerboard" that external recommendation 3
(boundary Rhie-Chow damping) was aimed at, and the first direct measurement of it.

Consequence for forces: the mode alternates sign cell to cell with amplitude O(h), so its
contribution to sum p_f S_f cancels to O(h^2) against a smooth normal. Consistent with T9, where
the HydroGym triangle mesh and the butterfly quads give pressure drag within 0.3% (1.108 vs
1.107) while the shear-dependent parts differ by 1.5-8%.

## 37. What HydroGym actually publishes for the Re = 100 cylinder (looked up 2026-09-21)

Searched: the Nature paper (arXiv 2512.17534, incl. Supplementary Information), the L4DC 2025 paper
(PMLR v283, Lagemann et al.), the docs site, and the repository at commit 4ab9854 (2026-08-24).

1. **The SI validation tables do not contain Re = 100 in 2D.** Table SI 1 validates the 2D cylinder
   at Re = 200 / 1,000 / 3,900 only, and those runs are the m-AIA lattice-Boltzmann solver on a
   40D x 20D domain with periodic lateral boundaries (blockage 0.05) and 320 cells per diameter --
   not the Firedrake solver on `medium.msh`. Their 2D claim is "St within 1-3%, C_D within 5% of
   literature". Nothing in the paper is directly comparable to our T9 setup.
2. **The one hard number on our exact mesh and BCs** is the unit test `test/test_cyl.py::test_steady`:
   Newton steady state, `Cylinder(Re=100, mesh="medium")`, Taylor-Hood P2-P1, asserts
   **C_D = 1.2840 (C_L = 0) to 1e-3**. Force definition matches ours: C_D = 2 F_x, F = -sigma.n
   integrated over the cylinder, rho = U = D = 1. A second test (`test_steady_rotation`,
   RotaryCylinder omega = 0.1, BDF dt = 0.1, 40 steps from rest) asserts C_D = 1.49, C_L = -0.06032,
   but that is a coarse-dt transient at t = 4, not a flow quantity.
3. **The uncontrolled shedding numbers exist only as a figure**: L4DC 2025 Fig. 4(d,e), rotational
   cylinder at Re = 100 (medium mesh; element order not stated -- the paper's text favours the
   P1-P1 BDF variant and the PD-control examples set `velocity_order = 1`). Read off the plot:
   **C_D ~ 1.49 with oscillation +-0.01, C_L amplitude ~ 0.35**. The time axis is not in D/U
   (12 periods in 7 units), so St cannot be read from it.
4. The example docstrings quote "St ~ 0.165" as expected behaviour; that is the unconfined
   literature value, not a measurement on the mesh.

| | HydroGym (Firedrake, medium.msh) | ours, HydroGym tris 17258 | ours, butterfly 6992 quads |
|---|---|---|---|
| shedding C_D | ~1.49 (figure) | 1.4762 | 1.4529 |
| C_L amplitude | ~0.35 (figure) | 0.3755 | 0.3457 |
| St | not published | 0.1775 | 0.1750 |
| steady C_D (unstable branch) | **1.2840** (unit test) | -- (mesh not mirror-symmetric) | see below |

Our shedding C_D on their own mesh is 1% below the figure reading; C_L amplitude 7% above. The
steady branch is the clean comparison (deterministic, no window/frequency ambiguity): reached in
`run_ucylinder.py --steady` by projecting out the y-antisymmetric mode each step on the
mirror-symmetric butterfly family; results appended below when the runs finish.

**Steady branch, butterfly family** (`--steady`, dt = 0.01, T = 80; dCd/dt < 1e-5 by t = 70):

| mesh | cells | wall-cell h | steady C_D | C_Dp | vs HydroGym 1.2840 |
|---|---|---|---|---|---|
| butterfly coarse | 1776 | 0.062 | 1.2442 | 0.9851 | -3.1% |
| butterfly medium | 6992 | 0.024 | 1.2627 | 0.9537 | -1.7% |
| butterfly fine (corrected mesh) | 27968 | 0.012 | 1.2744 | 0.9540 | -0.7% |
| **HydroGym medium.msh (tris)**, started from the butterfly steady state, `--init` | 17258 | | **1.2702** (plateau t = 12..40, drift < 2e-6, C_L < 1e-3) | 0.9456 | **-1.1%** |
| extrapolated: observed order 0.65 / assumed 2nd / assumed 1st | | | 1.295 / 1.278 / 1.286 | | +0.9% / -0.5% / +0.2% |

Corrected reading (an earlier version of this section called 1.269 "mesh-converged" on the
strength of coarse->medium alone; the fine mesh moved another +0.0117, so that was wrong).
The butterfly sequence 1.2442 -> 1.2627 -> 1.2744 approaches HydroGym's 1.2840 from below with an
observed order of only 0.65 (differences 0.0185, 0.0117, ratio 1.57). Extrapolations bracket the
HydroGym value: 1.278 if the asymptotic order is 2, 1.286 if 1, 1.295 at the observed 0.65. The
fine mesh itself is 0.7% low; the like-for-like triangle mesh (their `medium.msh`, our scheme)
is 1.1% low and sits between our medium and fine butterflies, as its wall-cell size would predict.
The low observed order is the same signature as the T3 Navier-Stokes MMS (1.2-1.4): first-order
wall gradients and wall pressure entering the force integral. Conclusion: the steady drag is
consistent with HydroGym's to within our discretisation error, converging toward it, not away.

Mesh note: `cylinder_butterfly_fine.geo` had doubled every interval count but kept the medium
progressions (1.08, 1.20, 1.02), which stretches 1.08^46 = 34x instead of 5.9x and puts a 9:1
volume jump on the block seams at |x|,|y| = 1.5; the cylinder run blew up there within t = 1 at
both dt = 0.01 and 0.005 (the MMS in section 30 survived it only because that has Dirichlet data
everywhere). Progressions are now the square roots of the medium values; the regenerated mesh
has neighbour-volume ratio max 1.32 (medium 1.37) and is stable. The section 30 fine-mesh MMS
orders (0.42 / 0.11 / 0.09) were measured on the bad mesh and need rerunning.

Wall pressure taken as the owner value versus gradient-extrapolated to the face changes C_Dp by
0.0025 on the medium mesh (0.2%), so the wall-pressure reconstruction is not where the gap is.

## 38. Running HydroGym itself (Spark, 2026-09-22)

Container `hgcmp` on the Spark from `firedrakeproject/firedrake-vanilla-default:latest` (arm64,
Firedrake in the system Python 3.12, PETSc /opt/petsc), `pip install --no-deps hydrogym==1.0.0`
plus its deps by hand (the `[firedrake]` extra fails to resolve on aarch64 because there is no
gmsh wheel; gmsh is only needed to *generate* meshes). Two traps: the PyPI wheel ships the
`.msh` files as Git-LFS pointers ("File is not a valid Gmsh file, expecting $MeshFormat, not
version"), so our regenerated `cylinder_medium.msh` was copied in -- their `medium.geo` is
byte-identical to ours; and their `fine.geo` is **a different problem**, domain
x in [-60, 200], y in [-20, 20] (blockage 0.025), not a refinement of medium. Driver:
`tools/hydrogym_cmp/hg_cylinder.py` (Reynolds-ramped Newton, then 1e-3 random perturbation and
SemiImplicitBDF order 3, exactly the pattern of their `unsteady.py`; forces from
`flow.compute_forces()`).

**Steady branch, their solver.** `medium.msh`, P2-P1: **C_D = 1.284032** -- reproduces the unit
test value to all printed digits, so that number is confirmed and is what our 1.2702 / 1.2744
should be read against. Same mesh with P1-P1 (the RL examples' `velocity_order=1`): 1.272115.
Their `fine.msh` (other domain): 1.076565 -- the near-unconfined steady value (Fornberg's
Re = 100 steady drag ~ 1.06), which incidentally measures the beta = 0.10 confinement effect on
the steady branch at +19%.

**Their steady-branch mesh convergence, same domain** (medium.geo with n1,n2,n3 halved / as is /
doubled, generated with our gmsh 4.15.2):

| mesh | cells | P2-P1 C_D | P1-P1 C_D |
|---|---|---|---|
| mediumhalf | 4672 | 1.278390 | |
| medium | 17258 | 1.284032 | 1.272115 |
| medium2x | 64262 | 1.285557 | 1.298398 |
| Richardson (observed order 1.8) | | **1.2862** | |

Their P2-P1 converges from below too, observed order 1.8, to 1.286; the medium mesh they ship is
already within 0.2% of that. P1-P1 is not usable for this: it jumps by 2% between medium and 2x.
So the reference for the steady branch on this domain is **C_D = 1.286 +- 0.001**. Our fine
butterfly (1.2744) is 0.9% below it and our own extrapolation (1.278-1.295) brackets it; our
scheme's lower observed order (0.66 vs their 1.8) is the actual difference between the codes.

**Where the steady-drag gap lives: the wall shear, not the pressure.** HydroGym's split
(`tools/hydrogym_cmp/hg_split.py`, sigma = 2 nu eps(u) - p I as in `compute_forces`):

| mesh | C_Dp | C_Dv |
|---|---|---|
| mediumhalf | 0.948413 | 0.329977 |
| medium | 0.954386 | 0.329647 |
| medium2x | 0.955324 | 0.330233 |
| converged | ~0.9556 | ~0.3302 |

Ours on the saved steady fields, pressure part = owner value, viscous part by three wall-shear
reconstructions (s = wall-normal distance into the fluid, d_n = wall-cell centroid distance):
(a) the wall cell's cell-centre gradient, what `run_ucylinder.py` uses; (b) one-sided
u_P / d_n, which is also the solver's own discrete wall flux; (c) quadratic through
u(0) = 0, u(d_n) = u_P, u'(d_n) = cell gradient.

| mesh | C_Dp | C_Dv (a) cellgrad | (b) one-sided | (c) quadratic | C_D (a) | C_D (b) | C_D (c) |
|---|---|---|---|---|---|---|---|
| butterfly coarse 1776 | 0.9850 | 0.2591 | 0.3377 | 0.3978 | 1.2441 | 1.3227 | 1.3828 |
| butterfly medium 6992 | 0.9536 | 0.3090 | 0.3388 | 0.3620 | 1.2626 | 1.2924 | 1.3156 |
| butterfly fine 27968 | 0.9539 | 0.3204 | 0.3351 | 0.3466 | 1.2743 | 1.2891 | 1.3005 |
| HydroGym tris 17258 | 0.9456 | 0.3246 | 0.3447 | 0.3545 | 1.2702 | 1.2903 | 1.3001 |

Pressure drag on the fine butterfly is within 0.2% of HydroGym's converged value (0.9539 vs
0.9556); the whole gap is the viscous part: the cell-centre gradient reads the shear at the
wall-cell centroid, d_n/2 too far from the wall where the profile has already relaxed, and is
3% low at fine (converging from below, slowly). The one-sided value is within 1.5% at every
resolution and puts the total within 0.2-0.3% of 1.2861 on both the fine butterfly (1.2891) and
HydroGym's own triangles (1.2903). The quadratic overshoots and converges from above (order
~1.2). Recommendation: report forces with the solver's own discrete wall flux (one-sided), which
is also the momentum-consistent choice; the shedding runs need repeating with it since the
force is evaluated in the time loop.

**Shedding, their solver on their mesh** (perturbed steady state, BDF3, dt = 0.01, T = 200, window
120-200, St by C_L zero-crossings, 13-14 periods, period std 0.0000; 8 ranks, 60 ms/step for
P2-P1). Force histories in `results/hydrogym_cmp/*_forces.dat`, overlay `figures/t9_vs_hydrogym.png`.

| run | St | C_D | C_L amp | C_L rms | C_D amp |
|---|---|---|---|---|---|
| **HydroGym P2-P1, medium** | **0.1791** | **1.4862** | **0.3582** | 0.2532 | 0.0109 |
| HydroGym P1-P1, medium (RL examples' order) | 0.1785 | 1.4703 | 0.3541 | 0.2515 | 0.0124 |
| ours, HydroGym tris 17258 | 0.1775 (-0.9%) | 1.4762 (-0.7%) | 0.3755 (+4.8%) | 0.2639 (+4.2%) | 0.0120 |
| ours, butterfly 6992 quads | 0.1750 (-2.3%) | 1.4529 (-2.2%) | 0.3457 (-3.5%) | 0.2445 (-3.4%) | 0.0096 |

Reading: on their own mesh our St and C_D are within 1% of their P2-P1 answer and our C_L
amplitude 5% high; the butterfly is 2-3% low on everything, consistent with its unrefined wake
(section on the grid overlay). The L4DC Fig. 4 reading (C_D ~ 1.49, C_L amp ~ 0.35) is consistent
with both of their element orders. Our C_D numbers still carry the cell-gradient wall shear
(about -0.010 in C_D, see the steady split), so with the wall-flux force the triangle mesh would
read ~1.496 (+0.7%) and the butterfly ~1.463 (-1.6%); that is an estimate until the shedding runs
are repeated with the corrected force routine.

## 39. Acting on sections 37-38: wall-flux force routine and a wake-refined butterfly (2026-09-22)

`run_ucylinder.py` `forces()` now takes the wall shear from the solver's own discrete wall flux
(one-sided u_P/d_n along the wall normal; tangential derivatives vanish at a no-slip wall),
pressure still the owner value. Verified to reproduce the offline (b) numbers exactly on the saved
steady fields (fine butterfly 1.2891, HydroGym tris 1.2903). Both shedding cases rerun with it
into `results/t9/v2/`.

`meshes/cylinder_butterfly_wake.geo`: the medium butterfly with the downstream x-curves
(26, 28, 30, 32) at 121 nodes and the outer-row y-curves ({14, 16, 23, 27}, {19, 21, 24, 31}) at
25 nodes, progressions the square roots of the medium ones (transfinite consistency forces the
whole top and bottom rows, not just the wake blocks). 13952 quads (was 6992; HydroGym's
triangles 17258); wake cells for 2 < x < 12, |y| < 2.5 have sqrt(area) median 0.096, p90 0.123
(was 0.25 x 0.11). Mirror-symmetric to 4e-3 h, neighbour-volume ratio max 2.54 (at the x = 1.5
seam), stable at dt = 0.01, 111 ms/step.

**Results** (window 90-150, 10 periods each, period std 0.0000; HydroGym P2-P1 medium as reference):

| run | cells | St | C_D | C_L amp | C_L rms |
|---|---|---|---|---|---|
| HydroGym P2-P1 | 17258 tris | 0.1791 | 1.4862 | 0.3582 | 0.2532 |
| ours v1, butterfly (cell-gradient shear) | 6992 | 0.1750 (-2.3%) | 1.4529 (-2.2%) | 0.3457 (-3.5%) | 0.2445 |
| **ours v2, butterfly (wall-flux shear)** | 6992 | 0.1750 (-2.3%) | **1.4866 (+0.03%)** | 0.3519 (-1.8%) | 0.2488 |
| **ours v2, butterfly_wake** | 13952 | 0.1746 (-2.5%) | 1.4896 (+0.2%) | 0.3645 (+1.8%) | 0.2574 |
| ours v1, HydroGym tris | 17258 | 0.1775 (-0.9%) | 1.4762 (-0.7%) | 0.3755 (+4.8%) | 0.2639 |
| ours v2, HydroGym tris (wall-flux shear) | 17258 | 0.1775 (-0.9%) | 1.4992 (+0.9%) | 0.3798 (+6.0%) | 0.2669 |

What each change did. (i) The wall-flux force adds +0.034 to C_D on the butterfly and +0.023 on
the triangles (the steady analysis predicted +0.010 to +0.015 from the viscous part alone; the
shedding wall shear is larger and the cell-gradient deficit with it). It changes nothing else:
St and C_L amplitude are force-integration-independent to within the C_L's own +1.8%. The
butterfly now matches HydroGym's mean drag to 0.03%; the triangles overshoot by 0.9%. (ii) The
wake refinement raises C_L amplitude 3.6% (butterfly now +1.8% vs HydroGym), C_D +0.2%, and
leaves St unchanged (0.1750 -> 0.1746). So the remaining butterfly St deficit of 2.3-2.5% is not
wake resolution. The triangle mesh, which has 0.03-0.05 cells around the cylinder and along the
separating shear layers, gets St to -0.9%; the butterflies' near-body O-grid (wall cell 0.024,
radial growth 1.08) is the remaining suspect, together with our second-order time scheme against
their BDF3 at the same dt = 0.01. Not yet tested. (iii) The triangle mesh's C_L amplitude is 5-6%
high on both force routines, which is where its wall checkerboard (section 36) would show.

## 40. The triangle-mesh vorticity speckle: what it is and what moves it (2026-09-22)

Vorticity contours of our T9 runs (`figures/t9_vorticity_contours.png`) show a zero-contour
speckle across the whole free stream on HydroGym's triangle mesh and none on the butterflies.
Free-stream rms vorticity where it should vanish (x < -1, r > 1.5):

| field | rms vorticity |
|---|---|
| ours, tris, cell-gradient vorticity | 2.44e-2 |
| ours, tris, vertex-averaged vorticity | 8.3e-3 |
| ours, wake butterfly, cell-gradient | 1.01e-2 |
| HydroGym P2-P1 on the same tris, DG0 (cell) vorticity | 8.4e-3 |
| HydroGym P2-P1, CG1 projection (what their plots show) | 8.2e-3 |

So ~8e-3 is a common floor (same on both codes and both mesh types); what the triangle solution
carries on top is the two-colour mode of section 34, 3x the floor at cell level and removed by
one vertex averaging (`figures/t9_tri_vorticity_cell_vs_vertex.png`; HydroGym's own fields in
`figures/hydrogym_vorticity_fields.png`). For like-for-like pictures against HydroGym, plot the
vertex-averaged field; theirs is CG1-projected, i.e. the same operation.

Two solver-side attempts to shrink the mode itself:
1. Rhie-Chow damping scale 2x and 4x (`--rc`, PISO.rc_scale): both go NaN within 600 steps from
   the steady state on the triangle mesh. V/a_P is the largest stable damping; no room there.
2. Linear extrapolation of the pressure to Neumann faces in the momentum gradient
   (`PISO.p_neumann_extrap`, one fixed-point pass, boundary term only), the mechanism proposed
   for the wall excitation in section 36. Clustered 64x64 triangle cavity, Re = 1000, t = 40:
   east-wall alternating rms 2.50e-2 -> 2.05e-2 (-18%), first-layer rms(p - nbr mean)
   4.9e-3 -> 4.4e-3 (-10%), second layer and deeper unchanged, Ghia errors identical (u 0.0247,
   v 0.0155). Real but marginal; left off by default.

Standing conclusion: on a non-bipartite mesh the mode exists at the amplitude the compact
Rhie-Chow damping leaves it; it costs ~6% in C_L amplitude on the triangles and nothing visible in
St or C_D, and it does not exist on the quad meshes. The effective remedy remains the mesh
(bipartite quads near walls), not the scheme.

## 41. Field-level comparison with HydroGym's Firedrake solution (2026-09-22)

`plot_utility/plot_hydrogym_vs_ours_fields.py` -> `figures/hydrogym_vs_ours_fields.png` and
`_mesh.png` (same with every cell edge drawn). Columns: HydroGym P2-P1 on its medium mesh
(t = 200), ours on the identical mesh (t = 150), ours on the wake-refined butterfly (t = 150).
Rows: vorticity near and far, pressure, |u|. Vorticity is vertex-averaged for ours and
CG1-projected for theirs (the same operation, section 40). Snapshots are phase-labelled from
C_L (0.73 / 0.62 / 0.19 cycles after the last upward zero-crossing); the butterfly is mirrored in
y to bring it within a tenth of a cycle, using the half-period reflection symmetry of the wake.

Findings: wake structure identical on all three (core spacing ~2.7, same core strength on the
+-2 scale, same lateral spread at the outlet, same roll-up length); |u| indistinguishable;
pressure identical in the far wake, with our triangle-mesh run carrying a somewhat deeper
low-pressure lobe at x ~ 1.5 than either HydroGym or the butterfly, matching its +6% C_L
amplitude; near-body vorticity at the +-6 level identical including separation point and
boundary-layer thickness (hence C_D to 0.03%). The two-colour speckle appears only in our
triangle-mesh vorticity, in the 0.3-0.7 free-stream triangles; HydroGym's own field has a trace
of it just upstream of the cylinder. With the mesh drawn, HydroGym's far-wake vortices are seen to
be carried on 0.3-0.7 triangles above and below the 0.1 centreline band, and ours on the same mesh
reproduce them at the same strength; the butterfly carries them on 0.06-0.19 x 0.1 quads.

Figure index for T9: `t9_forces.png` (histories, spectra), `t9_pressure.png`, `t9_vorticity_grid.png`
(cell-flat vorticity with mesh, two butterflies vs tris), `t9_vorticity_contours.png` (near/far,
three v2 runs), `t9_tri_vorticity_cell_vs_vertex.png`, `hydrogym_vorticity_fields.png` (their DG0
vs CG1), `t9_vs_hydrogym.png` (force overlays), `hydrogym_vs_ours_fields(_mesh).png`.

## 42. Equal-order finite elements show the same speckle (2026-09-22)

HydroGym's P1-P1 case (velocity_order = 1, `stabilization="none"`, the element pair their RL
examples use) rerun with field export: St 0.1785, C_D 1.4702, C_L amp 0.3541 (unchanged from
section 38). `figures/vorticity_wiggles_compare.png`: raw cell-level vorticity (top) and
vertex-averaged / CG1 (bottom) for four solutions of the same problem.

| solution | cell-level free-stream rms | vertex-averaged |
|---|---|---|
| HydroGym P2-P1 (Taylor-Hood, inf-sup stable) | 8.4e-3 | 8.2e-3 |
| HydroGym P1-P1 (equal order, unstabilised) | 1.50e-2 | 8.9e-3 |
| ours, collocated FV, same mesh | 2.44e-2 | 9.7e-3 |
| ours, wake-refined butterfly quads | 1.01e-2 | 9.8e-3 |

The equal-order finite element on the same triangles carries a cell-level speckle of the same
kind as ours, 1.8x the Taylor-Hood floor against our 2.9x; vertex averaging takes all four to
the same 8-10e-3 floor. So the mode is a property of equal-order velocity-pressure collocation
on a non-bipartite mesh, present in Firedrake's P1-P1 as in our FV; Taylor-Hood does not have it
because its velocity space is richer than its pressure space (inf-sup), which is the FE way of
buying what a staggered or bipartite arrangement buys in FV. Our amplitude is 1.6x theirs, the
price of Rhie-Chow's compact damping against the FE's implicit consistency. Nothing in this
changes the force conclusions of sections 38-39.

## 43. Training HydroGym's jet cylinder as shipped (Spark, 2026-09-22): the objective rewards suction

Setup: `tools/hydrogym_cmp/train_sb3_firedrake.py` (their SB3 script; only addition `--max-steps`),
PPO defaults (n_steps 200, lr 3e-4, gamma 0.99, batch 64), `Cylinder` Re = 100 medium mesh,
dt = 0.01, one CFD step per action, action in [-0.1, 0.1] (one scalar, both jets), reward
-dt C_D, observations (C_L, C_D) (`--obs-type lift_drag`; the script's own default is 50 wake
pressure probes despite its help text), 5000-step episodes each restarting from HydroGym's
published developed-shedding checkpoint (HF dataset `dynamicslab/HydroGym-environments`,
`Cylinder_2D_Re100_medium_FD`, C_D 1.467 at load). GPU container `hgrl` (torch 2.11+cu128 on the
GB10, SB3 2.9.0); the MLP policy on the GPU is cosmetic, wall time is the Firedrake step at
~7 env steps/s, 100k steps ~ 4 h.

Episode returns: -65.3, -52.7, ... (mean C_D 1.31 then 1.05) within the first two episodes,
i.e. a "29% drag reduction" after 100 time units. Reference rollouts from the same checkpoint
(`baseline_rollout.py`, 50 or 30 time units, mean C_D over the last 60%):

| actuation | C_D | C_L rms | note |
|---|---|---|---|
| none | 1.4862 | 0.252 | matches section 38 |
| uniform random in [-0.1, 0.1] | 1.4832 | 0.247 | |
| random sign +-0.1 each step (untrained Gaussian policy after clipping) | 1.4727 | 0.242 | |
| **constant suction -0.1** | **1.0416** | **0.021** | shedding suppressed |
| constant blowing +0.05 | 1.8656 | 0.459 | |
| constant blowing +0.1 | 2.3154 | 0.714 | |

Steady maximal suction through both 10-degree slots at +-90 degrees (wall-normal velocity up to
R A u = 0.5 x 36 x 0.1 = 1.8 U_inf) suppresses shedding and cuts the surface-stress drag 30%.
The reward has no actuation cost, so this is the optimum the environment defines, and PPO finds
it in two episodes. It is not the 8-12% of the published references: Rabault's jets are
zero-net-mass-flux and opposed with a bounded mass-flow budget, and the L4DC rotary actuator
cannot remove fluid. The docs' "more than 20% drag reduction" for the cylinder environment is
consistent with this suction solution. The 100k-step run was left to finish to see whether PPO
converges to the constant -0.1 or finds anything else; evaluation with `eval_policy.py`.

**Result of the 100k-step run.** Per-episode returns -65.3 (exploration), then -52.7, -52.4,
-52.4, -52.2 and flat thereafter (20 episodes); PPO's running mean ended at -53.1. Evaluation
(`eval_policy.py`, 100 time units each from the checkpoint, frozen VecNormalize, deterministic
policy; `figures/hydrogym_rl_eval.png`, histories in `results/hydrogym_rl/`):

| | C_D (t = 40..100) | C_L rms | action |
|---|---|---|---|
| uncontrolled | 1.4863 | 0.2525 | 0 |
| PPO policy | **1.0316** | 0.0023 | **-0.1000 constant** (rms 0.0000) |
| constant suction reference | 1.0416 (t = 12..30) | 0.021 | -0.1 |

Drag reduction 30.6%, lift oscillation removed (C_L rms decays exponentially to 3e-4 by t = 100).
The trained policy is the constant -0.1: deterministic output pinned at the bound at every step,
independent of the (C_L, C_D) observation. So the "as shipped" reproduction is exactly what the
reference rollouts predicted after the first episode: HydroGym's jet Cylinder with its default
reward is solved by maximal steady suction, and the >20% figure in their docs is this solution.
It says nothing about feedback wake control and is not comparable to the 8-12% of Rabault or the
rotary L4DC case. A meaningful jet benchmark needs (a) zero-net-mass-flux opposed jets and (b)
an actuation cost or mass-flow budget in the reward, both one-line changes in `flow.py` /
`evaluate_objective`; that variant has not been run.

## 44. The zero-net-mass-flux variant (Spark, 2026-09-23)

`tools/hydrogym_cmp/hg_znmf.py`: `CylinderZNMF(hgym.Cylinder)` overriding only
`cyl_velocity_field` with `slot(+90) - slot(-90)` -- the top slot blows while the bottom sucks
for a > 0, net flux 4e-17 at a = 0.1 (shipped class: -0.2007), |flux| 0.2007 (0.1 per slot);
same 10-degree cosine slots, same single scalar in [-0.1, 0.1], same actuator lag, same reward
-dt C_D, same restart checkpoint (`..._00000690.ckpt`, selected with HydroGym's own glob rule so
both runs start from the identical state, C_L 0.233 / C_D 1.467). Training command identical to
section 43 apart from `--env cylinder_znmf`; run dir `PPO_Firedrake_cylinder_znmf_20260923_033943`.

Constant-actuation references (30 time units from the checkpoint, mean over the last 60%):

| ZNMF actuation | C_D | C_L rms | note |
|---|---|---|---|
| none | 1.4862 | 0.252 | |
| constant +0.1 (top blows, bottom sucks) | 1.409 | 0.281 | still settling: 1.515 / 1.410 / 1.400 per 10 units |
| constant -0.1 (top sucks, bottom blows) | 1.328 | 0.194 | still settling: 1.320 / 1.275 / 1.365 |

By reflection symmetry the two should agree once settled; the difference is the shedding phase of
the shared start and the short window. Steady asymmetric ZNMF actuation at the bound deflects the
wake and buys of order 5-11% of drag with the lift oscillation intact -- the "steer the wake"
solution seen earlier with our own opposed jets (hydrogym_jet_cylinder_plan.md). That is the
floor a ZNMF policy has to beat to be called feedback control.

**Fields of the trained shipped-jet control** (`rollout_fields.py`, 60 time units from the
checkpoint in HydroGym's Firedrake solver; `figures/hydrogym_control_fields_shipped.png`).
Uncontrolled: the Karman street, cores +-2, spacing 2.7, C_D 1.486, C_L +-0.36. Controlled
(constant suction -0.1 at both slots, which is the PPO policy): no vortex anywhere in the domain
-- the two separated shear layers stay parallel and thin to the outlet, the boundary layer is drawn
onto the surface at the +-90-degree slots and separation moves aft; C_D falls to 1.08 within two
time units and relaxes to 1.033, C_L decays as a damped oscillation to 0.007 rms by t = 24 and
keeps shrinking. It is the unstable symmetric base flow held by suction, with a thinner wake than
the natural base flow (steady C_D 1.284) because fluid is removed from it. The 30% is complete
suppression by steady mass removal, not weakened shedding.

Run archive (kept in full): `results/hydrogym_rl/runs/PPO_Firedrake_cylinder_20260922_225051/`
(models every 10k steps + final, VecNormalize statistics, TensorBoard events), `logs/` (SB3
stdout, evaluation, reference rollouts), `eval_ppo_jets_*.dat`, `baseline_*.dat`,
`fields_shipped_*.npz`; the 800-step pressure-probe false start is kept under its own name.

**Result of the ZNMF run** (100k steps, same PPO defaults; `figures/hydrogym_rl_training_znmf.png`,
`figures/hydrogym_control_fields_znmf.png`, run archive `results/hydrogym_rl/runs/PPO_Firedrake_cylinder_znmf_20260923_033943/`):

Per-episode mean C_D: 1.496, 1.504, 1.488, 1.488, 1.484, 1.564, 1.532, 1.556, 1.550, 1.538,
1.586, 1.502, 1.498, 1.466, 1.488, 1.484 -- never more than 1.4% below the uncontrolled 1.486,
and 5-7% above it for six episodes while the policy explored asymmetric actuation. Evaluation
(100 time units, deterministic): C_D 1.4865 vs 1.4863 uncontrolled (0.0%), C_L rms 11.9 (!),
action mean 0.0000, rms 0.1000: the policy chatters between +0.1 and -0.1 on alternate steps.
Through the actuator lag (TAU 0.0556 = 5.6 steps) that averages to near-zero jet flux, so the
wake is the natural Karman street and the drag is unchanged; the huge C_L rms is the impulsive
surface-pressure response at the slots to the reversing 1.8 U_inf jets, not a wake effect.

Reading: with HydroGym's default PPO, one CFD step per action and (C_L, C_D) as the only
observation, the opposed ZNMF jets learn nothing in 100k steps; the policy did not even find the
steady asymmetric deflection worth 5-11% that the constant references show. This is not evidence
against Rabault's 8%, whose setup differs in the three things that make the problem learnable:
151 velocity probes (state observability), an action held for 50 solver steps (no chattering, a
control period ~1/10 of the shedding period), and ~10x the training budget. Those are the next
knobs (`--obs-type velocity_probes`, `--num-substeps 50`, longer run) if the ZNMF comparison is
pursued. Meanwhile the contrast is the point: the shipped jets reach 30% by trivial mass removal
in two episodes; the physically constrained jets reach nothing in the same budget.
