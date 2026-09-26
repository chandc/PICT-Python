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

## 45. NACA0012 at Re = 100, alpha = 20 deg: the grid (2026-09-23)

Target chosen from HydroGym's airfoil family (all on their m-AIA lattice-Boltzmann backend, which
is closed source and amd64-only, so we build the case in our own solver). Their
`NACA0012Gust_2D_Re100_AOA20/environment_config.yaml` records the **unperturbed means C_D = 0.577,
C_L = 0.783** (Ma 0.1, U_inf 0.0577 lattice units, chord 8 cells at level 0, bounding box
[-64,-48]x[192,48] i.e. 32c x 12c with the airfoil 8c from the inlet, local refinement to
level 12 within 0.5c of the body, sponge at the far boundary). That is the reference number for
this case; whether their flow sheds is not recorded.

Grid, `meshes/gen_naca_cgrid.py` -> `meshes/naca0012_a20.msh`, `figures/naca0012_a20_grid.png`:
structured C-grid, all quads (bipartite, no checkerboard), chord 1, LE at the origin, airfoil
rotated 20 deg nose-up, free stream along +x. Domain: front arc R = 8 about the TE x, straight
top/bottom at y = +-8 to the outlet at x = 24 (32c x 16c). 121 surface points clustered at the
nose only (1 - cos), TE spacing 0.026 matched to the wake cut's first spacing 0.02; cut leaves the
TE along the chord and relaxes to horizontal over one chord (no kink); 121 cut points, stretch
1.030, spacing 0.21 at the outlet; 71 layers, first cell 0.005 on the airfoil (per-line geometric
ratio 1.006-1.071), first spacing on the cut 0.3x the local cut spacing (aspect ~3 instead of 400).
Far-boundary points uniform in arc length. Construction: transfinite interpolation, then 6000
Jacobi iterations of the Winslow equations with control functions taken from the algebraic grid
(TTM/Sorenson), which keeps the wall clustering and removes the fanning. Two rejected routes are
in the file's comments: plain TFI gave 10-deg cells at the nose; marching along normals folded at
the 172-deg wedges the cut makes at the sharp TE (244 and 1819 inverted cells).

Result: 25200 quads, 25510 nodes, 120 wall faces, 0 inverted cells, min angle 28 deg (p1 29),
aspect max 11 (at the nose), neighbour-volume ratio p99 1.33 / max 8.1, wall cells 0.001-0.013.
Boundary tags: Inlet (arc + top + bottom, Dirichlet u = 1, v = 0), Outlet (x = 24, p = 0),
Airfoil (no-slip). Driver `run_uairfoil.py` (cylinder driver with the tag map and no symmetry
planes). Not yet run.

**Grid revisions before the first run.** (i) Trailing edge: nose-only clustering left 0.026 cells at
the TE beside 0.005 wall layers and the smoother opened a sparse fan behind the tip -> two-sided
arc-length distribution per side (nose 8e-4 growing at 1.051 over the front 65% of the arc, TE
0.008 growing at 1.078 over the rear 35%, spacings meeting at 0.033; 97 points per side, 193 in
all; cut first spacing 0.012). (ii) Far field: uniformly spaced far points against the geometric
wake cut sheared the far-field cells to 22 degrees around x = 11-15, |y| = 3-4, and the first flow
probe (dt = 0.005) blew up exactly there -- pressure to 47 at t = 0.45, then velocity to 3 at
(12.3, -3.4) -- with the near-body region still clean. Sliding far points during the smoothing did
not cure it (the control functions carried the TFI shear). Fix: far points directly above/below the
cut points (vertical eta-lines over the wake) and a geometric distribution around the front arc
starting from the junction spacing (0.012 growing at 1.040 to 0.49 at the front). Cost: a visible
band of fine lines over the TE spanning the domain height and aspect 46 in the far field there.

Final grid: 33040 quads, 192 wall faces, min angle 48.3 deg, aspect p99 25 / max 46 (far field
above the TE), neighbour-volume ratio max 1.71, wall cells 0.0005-0.012. Probe at dt = 0.005:
stable through 600 steps, C_D 1.37 -> 0.635 and C_L 1.56 -> 1.005 by t = 3 and still falling
toward HydroGym's 0.577 / 0.783; 300 ms/step. Production run: T = 100, dt = 0.005 (CFL ~9 on the
0.0008 nose spacing, implicit), impulsive start, ~100 min; `results/naca/naca0012_a20_re100.npz`.

**Result (T = 100, dt = 0.005, 274 ms/step, 91 min).** The flow is steady: C_L rms 1e-8 over
t = 60-100, dC_D/dt 3e-12 at the end, drag within 1e-4 of its final value by t = 17 after the
impulsive start (`figures/naca0012_a20_result.png`, fields `figures/naca0012_a20_field_t10.png`
and `_t100.png`).

| | ours (FV, C-grid 33k quads) | HydroGym (m-AIA LBM, config) | diff |
|---|---|---|---|
| C_D | **0.5703** | 0.577 | -1.2% |
| C_L | **0.7729** | 0.783 | -1.3% |
| L/D | 1.355 | 1.357 | -0.1% |
| pressure / viscous drag | 0.318 / 0.252 | not given | |
| C_p min | -1.615 at x/c 0.014 | | |
| suction-side C_f sign changes | x/c 0.333 (separation), 0.923 | | |

Flow: attached over the front third of the suction side, separation at x/c = 0.33, one large
recirculation bubble to half a chord behind the TE with two weak counter-rotating cells inside and
a small secondary reversal at x/c 0.92-1; C_p plateau -0.9 to -0.4 over the bubble (hence pressure
drag 56% of the total); stagnation just under the nose, peak speed 1.23. No shedding: the
height-based Reynolds number is 34. The two shear layers leave smoothly and decay; the free stream
is clean at the +-2 level (all-quad grid, no speckle). The starting vortices of the impulsive
start are at x = 8-10 at t = 10 and gone by t = 100.

Reading: both coefficients 1.2-1.3% below HydroGym's, L/D identical. Their number carries their
own discretisation (LBM at Ma 0.1 with an immersed STL boundary, 32c x 12c, sponge far field) and
ours the far-field Dirichlet at 8 chords (circulation-induced velocity there ~0.8% of U_inf, which
lowers the lift at about that level) -- the gap is of the size the two far-field treatments alone
would produce. A far-field sensitivity (`--R 16`) and a refinement (`--d0 0.0025 --ns 289 --neta 101`)
are the two checks to close it; neither has been run.

## 46. NACA0012 at Re = 100, alpha = 40 deg: shedding (2026-09-23)

Same generator and parameters with `--alpha 40` (`meshes/naca0012_a40.msh`, 33040 quads, min angle
43.1 deg, aspect max 49 far field, nbr-volume ratio max 2.8; `figures/naca0012_a40_grid.png`). Same
BCs and driver; probe stable at dt = 0.005; production T = 150, 30000 steps, ~290 ms/step, 2.4 h.
Reference: HydroGym `NACA0012Gust_2D_Re100_AOA40/environment_config.yaml` unperturbed means
C_D 1.081, C_L 1.027 (no St recorded).

| window t = 90-150 | ours | HydroGym | diff |
|---|---|---|---|
| C_D mean | **1.0679** | 1.081 | -1.2% |
| C_L mean | **1.0104** | 1.027 | -1.6% |
| L/D | 0.946 | 0.950 | -0.4% |
| C_L amplitude / rms | 0.1368 / 0.0968 | | |
| C_D rms | 0.0252 | | |
| **St (chord) / St (projected height c sin 40)** | **0.2331 / 0.1498** | not recorded | |
| pressure / viscous drag | 0.882 / 0.186 | | |

Shedding: 13 periods in the window, period 4.2898 +- 0.0000 (limit cycle exact to four digits from
t ~ 20; lift amplitude 0.137 constant from t = 20 to 150). The height-based Strouhal 0.150 is the
bluff-body value at Re_h = 64, as expected for a stalled plate. Flow (`figures/naca0012_a40_field_t20.png`,
`_t150.png`, `naca0012_a40_result.png`): both shear layers roll up within a chord of the TE, alternately
-- a leading-edge vortex detaching around x ~ 1.5 and a compact trailing-edge vortex just behind the
tip -- into a Karman street deflected downward with ~2-chord spacing, cores still above +-1.2 at
x = 10; recirculation bubble ~1.5 chords long that pinches off each cycle; C_p min -2.2 at the nose,
suction side a chain of low-pressure cores riding the shed vortices; suction-side C_f sign changes at
x/c 0.05 and 0.60 in the final snapshot (instantaneous). Both coefficients again 1.2-1.6% below
HydroGym's, the same offset and sign as the 20-deg case, consistent with the far-field difference
noted in section 45. Contrast with 20 deg: drag 1.87x, lift 1.31x, steady -> periodic, exactly the
transition the alpha_1 ~ Re^-0.65 scaling puts near 33 deg at Re = 100.

## 47. Periodic faces, T6 Poiseuille, T8 Orr-Sommerfeld (2026-09-23)

**Periodic seams** (`Mesh.make_periodic(tag_a, tag_b, shift)`): the a-faces and b-faces are paired
by centre (`fcentre_a + shift = fcentre_b`), each pair becomes one interior face owned by a's cell
with b's owner as neighbour, and the b-face is deleted. The cell-to-cell vector of the merged face is
`centroid_b - shift - centroid_a` -- b's owner mapped back beside a -- so `dcc` points forward
across the seam (first attempt had the sign reversed: dcc +1.75 and orth = -1 on a unit domain;
now -0.25, wf 0.5, orth 1, audit clean). Two operators read `centroid[neigh]` directly and were
changed to `centroid[owner] + dcc` (convection skewness point in `uops.laplacian`/`convection`,
`GGSkewGradient`); the LSQ gradient already used `dcc`. Check: LSQ gradient of sin(2 pi x) y^2 on a
16x16 periodic quad mesh has the same max error on seam cells as in the interior (0.112 both).

**T6 plane Poiseuille** (`test_upoiseuille.py`: 4 x ny quads, walls Dirichlet, periodic x, body
force, nu = 0.1, dt = 0.5, marched to |du| < 1e-11 in ~42 steps):

| ny | A: parabola L2 error | B: sin(pi y) L2 error | rate |
|---|---|---|---|
| 8 | 1.56e-2 | 9.16e-3 | |
| 16 | 3.91e-3 | 2.28e-3 | 2.01 |
| 32 | 9.77e-4 | 5.68e-4 | 2.00 |
| 64 | | 1.42e-4 | 2.00 |

Second order at the walls on both cases, |v| at round-off. The parabola is not reproduced exactly:
the cell-centred half-cell wall flux leaves a uniform offset G h^2/(8 nu) (u_max = 1.000000 at every
ny against the centroid-exact 0.984/0.996/0.999), which is second order, not the "exact by
construction" of the node-based structured scheme -- the expected difference between cell-centred
and vertex-centred wall treatments. PASS.

**T8 Orr-Sommerfeld, Re = 7500, alpha = 1** (`test_uorr_sommerfeld.py`: 48 x ny quads on
[0, 2 pi] x [-1, 1], periodic x, no-slip walls, f_x = 2 nu, least-stable eigenmode seeded at 1e-4,
BDF2, central convection, growth and phase from the first streamwise Fourier mode fitted over
t = 30-100; reference growth 0.00223497, phase 0.24989154):

| grid | cells | dt | growth | error | phase speed | error | time |
|---|---|---|---|---|---|---|---|
| 48 x 100 | 4800 | 0.05 | 0.001865 | -16.6% | 0.24445 | -2.18% | 1.1 min |
| 48 x 200 | 9600 | 0.05 | 0.002214 | -0.9% | 0.24572 | -1.67% | 2.0 min |
| 48 x 400 | 19200 | 0.05 | 0.002266 | +1.4% | 0.24603 | -1.55% | 4.0 min |
| structured code, 48 x 401 | | 0.05 | 0.002209 | -1.2% | 0.24614 | -1.50% | 80 min |

The disturbance GROWS on every grid -- the correct sign of stability, which is the substantive
result; a first-order scheme's damping would flip it. The growth error changes sign between
ny = 200 and 400 (order 4.1 then -0.55), the fingerprint of two errors of opposite sign: spatial
over-damping that vanishes with refinement and a temporal error that does not. The phase speed
is flat at -1.6% across the grids, exactly as in the structured code at the same dt, i.e.
temporal-limited. The unstructured code reaches the structured code's 48 x 401 accuracy in 4
minutes instead of 80 (sparse direct solves against the structured code's iterative ones).

Time-step check at 48 x 400, dt = 0.025 (7.8 min): growth 0.002256 (+0.9%), phase 0.24739
(-1.00%). Both move toward the reference with dt alone, so the residuals at dt = 0.05 are mostly
temporal. Splitting each error into a dt-independent part and a part scaling as dt^2 from the two
runs: growth = +0.7% (spatial, at ny = 400) + 0.7% (temporal at dt 0.05); phase = -0.8% (spatial)
- 0.7% (temporal at dt 0.05). The structured code's 48 x 401 / dt 0.025 gave 12.5% / 0.94% at
ny = 201 and was never run at 401 with the smaller step. T8: PASS -- growth of the correct sign
on every grid, within 1% of Streett's rate at 48 x 400 / dt 0.025, phase within 1%.

**Orders, one variable at a time** (`figures/t8_os_growth.png`, `plot_utility/plot_t8_growth.py`;
all fits on t = 30-100 from the energy |a|^2 of the seeded mode):

| grid | dt | sigma | err | phase c | err |
|---|---|---|---|---|---|
| 48x100 | 0.0125 | 0.001838 | -17.8% | 0.24648 | -1.36% |
| 48x200 | 0.0125 | 0.002193 | -1.9% | 0.24776 | -0.85% |
| 48x400 | 0.0125 | 0.002245 | +0.5% | 0.24807 | -0.73% |
| 48x400 | 0.025 | 0.002256 | +0.9% | 0.24739 | -1.00% |
| 48x400 | 0.05 | 0.002266 | +1.4% | 0.24603 | -1.55% |
| 48x400 | 0.1 | 0.002241 | +0.3% | 0.24331 | -2.64% |

Spatial (dt = 0.0125, temporal error ~1/16 of its dt = 0.05 size): growth-rate error vs Streett
3.24 then 2.05 per halving; Richardson on the raw values 2.77 (sigma) and 2.02 (phase). SECOND
ORDER IN SPACE for both. Temporal (48x400, four dt): the phase-speed error halves per dt halving,
-2.64 / -1.55 / -1.00 / -0.73%, Richardson order 1.00 on both triplets -- FIRST ORDER IN TIME with
convection present, against 2.3-2.7 on the unsteady Stokes gate (T4) where there is no convection.
The growth rate is within +-1% at 48x400 for every dt and non-monotone in dt (0.1 gives +0.3%,
0.05 gives +1.4%): its temporal error is below the level at which an order can be read.
Hypothesis for the first-order phase: `_momentum` assembles the convection matrix with `self.Ff`,
the mass flux of the PREVIOUS time step -- a first-order-in-time linearisation of the convecting
velocity that Stokes never exercises. Test: extrapolate the flux to n+1 (2 F^n - F^{n-1}).

**Hypothesis confirmed, fix adopted.** `PISO.conv_flux_extrap`: the convection matrix is
assembled with 2 F^n - F^{n-1} instead of F^n. On 48x200:

| dt | lagged flux F^n: growth / phase | extrapolated 2F^n - F^{n-1}: growth / phase |
|---|---|---|
| 0.05 | -0.9% / -1.67% | -1.9% / -0.60% |
| 0.025 | -- | -2.5% / -0.58% |
| 0.0125 | -1.9% / -0.85% | -2.6% / -0.58% |

With the extrapolation both quantities are dt-independent to the third digit: the phase error that
halved with dt is gone, and what remains (-0.58% phase, -2.6% growth at 48x200) is the spatial
error at that grid. The lagged flux had been partially cancelling the spatial growth error
(hence the misleading -0.9% at dt 0.05). The extrapolation is now the DEFAULT; every T9/T10
number recorded above was produced with the lagged flux and is therefore first-order in time
(at their dt = 0.01 / 0.005 the effect is small; measured on the butterfly cylinder below).

**Effect on the cylinder** (butterfly 6992 quads, dt = 0.01, T = 150, wall-flux force, window 90-150):

| convecting flux | St | C_D | C_L amp | C_L rms |
|---|---|---|---|---|
| F^n (lagged, all earlier T9 results) | 0.1750 | 1.4866 | 0.3519 | 0.2488 |
| 2F^n - F^{n-1} (new default) | 0.1755 | 1.4801 | 0.3331 | 0.2357 |
| HydroGym P2-P1 BDF3, same dt | 0.1791 | 1.4862 | 0.3582 | 0.2532 |

The second-order flux moves St by +0.3%, C_D by -0.4% and the lift amplitude by -5.3%, so the
first-order lag had been flattering the lift comparison (-1.8% -> -7.0% vs HydroGym) and the
drag agreement (+0.03% -> -0.4%). Which is the time-converged answer is being settled by the
same pair at dt = 0.005 (below).

**Streamwise resolution.** The phase-speed residual that survived dt and wall-normal refinement
(-0.47% at 48x400) is the streamwise dispersion of second-order central differencing at 48 cells
per wavelength, (kh)^2/6 = 0.29%: doubling nx gives phase -0.60% -> -0.29% at ny = 200 and
-0.47% -> -0.16% at ny = 400 (growth unchanged, -2.1% / +0.3%: the growth rate is wall-normal
limited). 96x400 at dt 0.05: growth +0.3%, phase -0.16%.

**T8 summary.** Second order in space in both directions (wall-normal: sigma error 3.25 / 1.84
per halving, Richardson 2.74 sigma / 2.03 phase; streamwise: the dispersion residual falls 3.7x
per doubling), second order in time once the convecting flux is extrapolated (dt-independent to
the third digit), and the three error sources are separated and each quantified. Best point
96x400, dt 0.05: growth within 0.3% and phase within 0.16% of Streett's eigenvalue. The
convecting-flux fix is the one solver change to come out of T8; the record's earlier T9/T10 numbers
carry the first-order lag (quantified on the cylinder above).

**dt = 0.005 pair, butterfly 6992 quads** (T = 150, window 90-150):

| convecting flux | dt | St | C_D | C_L amp |
|---|---|---|---|---|
| 2F^n - F^{n-1} | 0.01 | 0.1755 | 1.4801 | 0.3331 |
| 2F^n - F^{n-1} | 0.005 | 0.1755 | 1.4793 | 0.3328 |
| F^n (lagged) | 0.01 | 0.1750 | 1.4866 | 0.3519 |
| F^n (lagged) | 0.005 | 0.1753 | 1.4826 | 0.3422 |
| HydroGym P2-P1 BDF3 | 0.01 | 0.1791 | 1.4862 | 0.3582 |

The extrapolated flux is time-converged at dt = 0.01 (all three quantities move < 0.1% on halving
dt); the lagged flux converges toward it at exactly first order (lift-amplitude gap 0.019 -> 0.0094).
So the TIME-CONVERGED butterfly answer at this grid is St 0.1755 (-2.0% vs HydroGym), C_D 1.479
(-0.45%), C_L amplitude 0.333 (-7.1%). The earlier -1.8% lift agreement was the first-order lag
cancelling a spatial deficit. Both remaining deficits point at near-body resolution on the 6992-cell
butterfly (wall cell 0.024): the fine butterfly (27968 quads) shedding run and a rerun of
HydroGym's triangle mesh with the new default are the next two runs; neither done here.

**All meshes with the corrected flux** (v3, T = 150, dt = 0.01, window 90-150, wall-flux force;
`results/t9/v3/`, `figures/t9_vs_hydrogym.png`):

| mesh | cells | St | C_D | C_L amp | C_L rms |
|---|---|---|---|---|---|
| HydroGym P2-P1 BDF3 (reference) | 17258 tris | 0.1791 | 1.4862 | 0.3582 | 0.2532 |
| butterfly | 6992 | 0.1755 (-2.0%) | 1.4801 (-0.4%) | 0.3331 (-7.0%) | 0.2357 |
| wake-refined butterfly | 13952 | 0.1750 (-2.3%) | 1.4829 (-0.2%) | 0.3453 (-3.6%) | 0.2441 |
| **fine butterfly** | 27968 | **0.1782 (-0.5%)** | **1.4878 (+0.1%)** | **0.3520 (-1.7%)** | 0.2492 |
| **HydroGym's tris, our solver** | 17258 | **0.1780 (-0.6%)** | **1.4927 (+0.4%)** | **0.3606 (+0.7%)** | 0.2547 |

Reading. (1) Near-body refinement is what the butterfly needed: coarse -> fine takes St from
-2.0% to -0.5% and the lift amplitude from -7.0% to -1.7%, while the wake refinement alone (same
near-body grid) left St untouched. The 0.024 wall cell / 1.08 radial growth of the 6992-cell
O-grid under-resolves the separating shear layers; at 0.012 it does not. (2) On HydroGym's own
mesh our solver now reproduces their Taylor-Hood result to 0.7% on all three quantities; the
earlier +6% lift amplitude on the triangles was the first-order flux lag, not the wall
checkerboard (which is still there, at the +-0.1 vorticity level, and evidently costs < 1% in
forces). (3) The two remaining differences are of opposite sign on the two mesh families
(butterfly lift -1.7%, tris +0.7%; C_D +0.1% / +0.4%) and bracket the reference, which is what
two second-order discretisations converging to the same answer look like. T9 is closed at the
1% level; the open item is the 2% St deficit that persists on butterflies coarser than 0.012 at
the wall, a resolution requirement now quantified rather than a scheme defect.

**Fine triangle mesh** (`meshes/cylinder_medium_fine.geo`: HydroGym's generator with n1 35 -> 70,
n2/n3 unchanged; 25376 tris, 220 wall faces, wall cell 0.009, wake band 0.025, far field 0.31;
matched to the fine butterfly's 27968 quads / 224 faces / 0.012; `figures/cylinder_meshes_fine_tri_vs_butterfly.png`).
T = 150, dt = 0.01, corrected flux, 240 ms/step:

| mesh | cells | St | C_D | C_L amp |
|---|---|---|---|---|
| HydroGym P2-P1 (reference) | 17258 tris | 0.1791 | 1.4862 | 0.3582 |
| fine butterfly | 27968 quads | 0.1782 (-0.5%) | 1.4878 (+0.1%) | 0.3520 (-1.7%) |
| HydroGym tris | 17258 | 0.1780 (-0.6%) | 1.4927 (+0.4%) | 0.3606 (+0.7%) |
| **fine tris** | 25376 | **0.1785 (-0.3%)** | **1.4851 (-0.07%)** | **0.3611 (+0.8%)** |

Refining the wall from 0.019 to 0.009 on the triangles moves St +0.3%, C_D -0.5% and C_L amplitude
+0.1%: the triangle family is converged at the medium mesh already for the lift and needed the
refinement only for drag and St. At matched wall resolution and cell count the two mesh families
now agree with each other to 0.2% in St and C_D and 2.6% in lift amplitude, and both sit within
0.8% of HydroGym on St and C_D; the lift amplitude straddles the reference (quads -1.7%, tris
+0.8%). Field (`figures/t9_trifine_field_mesh.png`): the same street; the two-colour speckle is
present at cell level as on every triangle mesh and absent after vertex averaging.

**Quads vs triangles, coarse and fine, side by side** (`figures/t9_quad_vs_tri_coarse_fine.png`,
`plot_utility/plot_quad_vs_tri_coarse_fine.py`; grids in `figures/cylinder_meshes_medium2x.png` and
`figures/cylinder_meshes_fine_tri_vs_butterfly.png`):

| | coarse quads | coarse tris | fine quads | fine tris |
|---|---|---|---|---|
| cells / wall faces | 6992 / 112 | 17258 / 112 | 27968 / 224 | 25376 / 220 |
| wall cell | 0.024 | 0.019 | 0.012 | 0.009 |
| St vs HydroGym | -2.0% | -0.6% | -0.5% | -0.3% |
| C_D | -0.4% | +0.4% | +0.1% | -0.1% |
| C_L amp | -7.0% | +0.7% | -1.7% | +0.8% |

The butterfly spends its cells in a radial O-ring and coarsens abruptly at the block boundary at
1.5; HydroGym's generator grades continuously from the wall into a fine centreline band the length
of the wake, so at equal cell count the triangles have the finer wake and the quads the more
orthogonal wall layer. The coarse butterfly's diffuse shear layers and elongated wake vortices are
its -7% / -2%; both fine meshes resolve the layers with ~10 cells across and carry compact cores to
the outlet. Both families converge toward the reference; the fine pair agrees to 0.2% in St and
C_D and brackets the lift amplitude (-1.7% / +0.8%). The two-colour speckle is a cell-level feature
of every triangle mesh and does not survive vertex averaging.

## 48. Inviscid kinetic-energy conservation (T11, 2026-09-24)

Port of the structured code's `test_energy_conservation.py` -> `test_uenergy.py`: fully periodic
unit box (both seams), nu = 0. (1) OPERATOR: production P = sum u.(C u) of the assembled convection
operator (implicit upwind part + deferred central/skewness correction, as the solver applies it) on
a solver-consistent field (one 1e-4 step so div F = 0 to round-off), as eps = P/E per turnover
(1/max|u|). (2) SOLVER: E(T)/E(0) through full inviscid steps on the 2D Taylor-Green vortex, which
is an exact STEADY Euler solution, so any change is numerical dissipation.

**Operator** (eps per turnover; positive = dissipative):

| field | mesh | n | central | upwind |
|---|---|---|---|---|
| Taylor-Green | quads | 16 / 32 / 64 | -3e-16 / -1e-16 / -2e-16 | 1.77 / 0.89 / 0.44 |
| Taylor-Green | tris | 16 / 32 / 64 | -2e-16 / -1e-16 / -4e-17 | 1.18 / 0.57 / 0.29 |
| random solenoidal | quads | 16 / 32 / 64 | +1e-17 / +3e-17 / +4e-17 | 3.39 / 1.75 / 0.89 |
| random solenoidal | tris | 16 / 32 / 64 | -4.1e-3 / -1.1e-3 / -2.7e-4 | 2.08 / 1.09 / 0.54 |

Central is skew-symmetric to round-off on quads (w = 1/2: sum_f F (u_O^2 - u_N^2)/2 telescopes to
sum_P u_P^2 div F = 0), for both fields, and on triangles for the symmetric Taylor-Green field. On
triangles with a general field the skewness correction breaks the symmetry by O(h^2), order 1.9
per halving, and the sign is anti-dissipative: 0.03% of E per turnover at n = 64. First-order
upwind removes 44-89% of E per turnover at n = 64, halving per refinement (the structured code's
second-order upwind removed ~10%); it is what upwinding is, and it is why `central` is not a
preference.

**Full inviscid steps**, Taylor-Green, T = 0.1, loss rate (1 - E(T)/E(0))/T per turnover:

| mesh | n | dt 0.02 | 0.01 | 0.005 | 0.0025 | 0.00125 |
|---|---|---|---|---|---|---|
| quads | 32 | 0.315 | 0.092 | 0.030 | 0.011 | 0.0046 |
| quads | 64 | 0.307 | 0.081 | 0.022 | 0.0063 | 0.0020 |
| tris | 32 | 0.360 | 0.117 | 0.050 | 0.034 | 0.029 |
| tris | 64 | 0.332 | 0.082 | 0.020 | 0.0069 | 0.0057 |
| upwind, quads 64, dt 0.005 | | | | 0.436 | | |

The full-step loss is TEMPORAL: on 64^2 quads it falls 3.7x, 3.5x, 3.1x per dt halving (second
order in dt) with no visible spatial floor, so the 2.2% per turnover at dt = 0.005 that matches the
structured code's ~2% is time integration and splitting, not the operator. Triangles show a spatial
floor (2.9% per turnover at n = 32, 0.6% at 64, falling ~4x per refinement) -- the skewness/
checkerboard price on non-bipartite meshes, here at the energy level. Over a full turnover on 64^2
quads at dt = 0.005 the rate is steady (0.24% per 0.1, 0.44% total), a genuine dissipation rate, not
a transient. With the LAGGED convecting flux (pre-section-47 default) the loss is 3-4x smaller
(0.029 / 0.0086 / 0.0030 at dt 0.01 / 0.005 / 0.0025): the first-order lag is slightly
anti-dissipative, which is another reason its lift-amplitude agreement on the cylinder was
accidental.

**Where the full-step loss comes from** (64^2 quads, central, T = 0.1, loss rate per turnover):

| dt | Rhie-Chow on, n_corr 2 | Rhie-Chow OFF | RC on, n_corr 4 |
|---|---|---|---|
| 0.01 | 0.0808 | 0.0778 | 0.0807 |
| 0.005 | 0.0218 | 0.0196 | 0.0217 |
| 0.0025 | 0.0063 | 0.0049 | 0.0063 |

Switching the Rhie-Chow damping off removes only 4-22% of the loss (and the pressure stays clean
over the short window), and doubling the correctors changes nothing, so the dissipation is the
time integration itself -- BDF2 plus the PISO velocity-pressure splitting -- at O(dt^2). That is
the same conclusion the structured code reached ("discretisation error rather than a systematic
energy source"), here with the split made explicit. T11: PASS -- operator conserving to round-off
on quads; full step dissipation second order in dt and 0.2% per turnover at dt = 0.00125.

## 49. L0 of the LES plan: what dissipates in a full inviscid step (2026-09-24)

`PISO.n_outer` added: PIMPLE-style outer iterations within a time step -- convection re-linearised
about the corrected flux, momentum re-solved with the corrected pressure, time level held fixed;
`n_outer = 1` is classic PISO and reproduces T6 and T8 bit-for-bit. The iterate change falls to
1e-15 by the fourth pass on the inviscid Taylor-Green.

**Steady Taylor-Green, 64^2 quads, T = 0.1, loss per turnover:**

| dt | PISO (n_outer 1) | converged coupling | converged, Rhie-Chow OFF |
|---|---|---|---|
| 0.01 | 8.08e-2 | 3.35e-3 | -7e-9 |
| 0.005 | 2.18e-2 | 2.32e-3 | -7e-11 |
| 0.0025 | 6.30e-3 | 1.44e-3 | -3e-13 |
| 32^2, dt 0.005 | 2.96e-2 | 1.10e-2 | -3e-11 |

Three findings. (1) The velocity-pressure SPLITTING is 77-89% of the loss; iterating it out
leaves the rest. (2) What remains is the Rhie-Chow damping, entirely: with it off and the
coupling converged the energy is conserved to round-off. That residual does NOT vanish with dt
(order 0.5-0.7 in dt) and falls ~4x from 32^2 to 64^2: an O(h^2) dissipation inherent to the
collocated scheme, 0.14-0.33% per turnover at 64^2. (3) The STEADY Taylor-Green cannot see the
time integrator at all -- a steady exact solution satisfies BDF2 with zero temporal error -- so
the "BDF2 dissipation" of section 48 was misattributed: it was splitting plus Rhie-Chow. The
integrator is measured on the advected Taylor-Green below.

**Advected Taylor-Green** (TGV + uniform stream (1.0, 0.7): exact solution u = TGV(x - U0 t) + U0,
unsteady in the grid frame; 64^2 quads, T = 0.1, CFL ~0.4 at dt 0.005; fluctuation-energy loss per
turnover and L2 error against the exact solution):

| dt | PISO | converged coupling, RC on | converged coupling, RC off | L2 err: PISO / converged |
|---|---|---|---|---|
| 0.01 | 1.89e-1 | 9.09e-2 | 8.86e-2 | 3.3e-2 / 5.0e-3 |
| 0.005 | 4.94e-2 | 2.40e-2 | 2.22e-2 | 1.6e-2 / 1.7e-3 |
| 0.0025 | 1.31e-2 | 6.75e-3 | 5.54e-3 | 7.9e-3 / 1.0e-3 |
| 0.00125 | 3.72e-3 | 2.12e-3 | 1.38e-3 | 4.3e-3 / 9.0e-4 |

The BDF2-only loss (converged coupling, RC off) is second order in dt to three digits (1.99, 2.01,
2.00): 2.2% per turnover at dt 0.005, 0.14% at 0.00125. The splitting adds as much again (PISO
4.9e-2 vs 2.2e-2 at dt 0.005) and dominates the SOLUTION error: 1.6e-2 against 1.7e-3, which is
already the spatial floor of the 64^2 grid (the converged error flattens at 9e-4). Rhie-Chow adds
~10%.

**G0 verdict.** Splitting and BDF2 contribute comparably (each ~2% per turnover at dt 0.005 on
64^2), Rhie-Chow is a ~10% / O(h^2) floor. Per the plan's decision rule this selects **L1a: RK3
with per-stage projection** -- it removes the linearisation lag of the splitting by construction
and cuts the integrator's own dissipation from O(dt^2) to O(dt^3). Converged BDF2 coupling
(`n_outer` >= 4) stays available as the reference integrator; it already lowers the solution
error 9x at dt 0.005 for 4x the cost per step.

## 50. L1a: RK3 with per-stage projection (2026-09-24)

`PISO.time_scheme = "rk3"` selects the Le & Moin (1991) RK3 / Crank-Nicolson fractional step
(coefficients gamma = 8/15, 5/12, 3/4; zeta = 0, -17/60, -5/12; alpha = beta = 4/15, 1/15, 1/6):
per stage an implicit CN viscous solve with the convection explicit on the stage's own
divergence-free flux (no linearisation lag, no deferred convecting flux), Rhie-Chow face flux,
one pressure correction, cell velocity and face flux corrected, `p += pp`. Three momentum
matrices and three pressure operators, factorised once and cached. Cost 15-20 ms/step against
BDF2-PISO's 29 ms at 64^2 (three Poisson solves, but no corrector loop). Explicit convection
puts a CFL limit of about sqrt(3) on the stage; the cylinder runs at dt 0.005 (CFL ~0.2 at the wall).
`run_ucylinder.py --time-scheme rk3`, `OS_SCHEME=rk3` for T8, `test_utemporal.py` for T4.

**Three things went wrong before it worked, in order.** (1) `p += pp/(alpha+beta)`: the cell
diffusivity `D = (alpha+beta) V / a_C` already carries the stage step, so pp IS a pressure; the
rescaling ran away. (2) Choi's dt-independent Rhie-Chow term `(D/dt)(F_old - Fbar_old)` inside
the stages. With the predictor's interpolated velocity as `Fbar_old` it was unstable; with the
matched pair (corrected flux, plain interpolation of the corrected velocity) it was stable but
left a **dt-independent** loss of 1.2-1.7%/turnover at 64^2, whether the current or the previous
stage step sat in its denominator. The term cancels the previous damping only when consecutive
steps are equal; the RK stage steps are 8/15, 2/15, 5/15 of dt. **Decision: no transient term
inside RK3.** The per-stage damping is then `D_k = dt_k V/a_C`, O(dt), and the checkerboard
indicator on the 64^2 / 32^2 Taylor-Green is unchanged (0.010 / 0.038 over T = 1).
(3) `Ff = 0` at the start of EVERY run: the first step convected nothing. A one-off O(dt) error
with an O(|u.grad u|) coefficient, invisible in windowed statistics (T8, T9) and in runs from
rest (T5, T6, cavity), but it capped every temporal-order test on a non-trivial initial field at
first order -- BDF2 and RK3 alike showed `err = 1.36 dt` on the advected Taylor-Green,
mesh-independent (32^2-128^2), which is what finally gave it away (a spatial residual would have
fallen 4x per refinement). `PISO.init_flux()` now builds Ff by interpolation of (u, v) on the
first step when Ff is identically zero and the velocity is not. T6 rates unchanged (2.01/2.00/2.00).

**T4 (temporal order), `test_utemporal.py`.** Decaying Taylor-Green, fully periodic 64^2, nu 0.01,
K = 2 pi, T = 1 (exact NS solution, decay to 0.454); error against the same scheme at
dt_ref = dt_min/4 isolates the temporal part from the O(h^2) floor (1.7e-4 / 2.0e-4).

| dt | BDF2-PISO vs ref (order) | RK3, Rhie-Chow on (order) | RK3, Rhie-Chow off (order) |
|---|---|---|---|
| 0.04 | 9.02e-3 | 1.32e-4 | 1.09e-5 |
| 0.02 | 2.42e-3 (1.90) | 8.29e-5 (0.67) | 2.71e-6 (2.01) |
| 0.01 | 6.78e-4 (1.84) | 4.67e-5 (0.83) | 6.74e-7 (2.01) |
| 0.005 | 2.17e-4 (1.65) | 2.38e-5 (0.97) | 1.66e-7 (2.02) |
| 0.0025 | 8.07e-5 (1.43) | 1.07e-5 (1.15) | 3.96e-8 (2.07) |

Inviscid advected Taylor-Green (U0 = (1, 0.7), T = 0.25, 64^2, floor 2.2e-3), the convective order:

| dt | BDF2-PISO (order) | RK3, Rhie-Chow on (order) | RK3, Rhie-Chow off (order) |
|---|---|---|---|
| 0.02 | 8.54e-2 | 5.40e-2 | 5.40e-2 |
| 0.01 | 1.09e-2 (2.97) | 1.50e-4 (8.5) | 6.78e-5 (9.6) |
| 0.005 | 2.74e-3 (1.99) | 5.60e-5 (1.42) | 8.70e-6 (2.96) |
| 0.0025 | 6.56e-4 (2.06) | 2.29e-5 (1.29) | 1.17e-6 (2.89) |

Reading: the integrator alone is **third order in convection (2.9-3.0) and second order in
diffusion (2.0, Crank-Nicolson, by construction)** -- the plan's "order >= 2.8" holds for the
convective part only; no CN-viscous scheme can give it for the viscous part, and the constant is
1000x below BDF2's (1.1e-5 against 1.3e-2 at dt 0.04). With Rhie-Chow on, the O(dt) per-stage
damping becomes the leading dt-dependence once the integrator error drops below it: a
first-order term with a ~1e-2 coefficient (5.6e-5 at dt 0.005, still 50x below BDF2), spatially
O(h^2). The exact-solution error sits on the spatial floor for every dt <= 0.04 in both cases,
which is what one wants from a time scheme.

**T8 (Orr-Sommerfeld, 48 x 400, `OS_SCHEME=rk3`):**

| dt | growth (error) | phase speed (error) | wall time |
|---|---|---|---|
| 0.05 | 0.002230 (-0.21%) | 0.248753 (-0.456%) | 2.7 min |
| 0.025 | 0.002231 (-0.20%) | 0.248752 (-0.456%) | 5.4 min |
| 0.0125 | 0.002231 (-0.19%) | 0.248752 (-0.456%) | 10.9 min |

Phase speed dt-independent to 4e-4% (criterion 0.1%), growth within 0.2% (criterion 1%). The
remaining -0.46% in the phase speed is the 48-cell streamwise dispersion, (kh)^2/6 = 0.29% with
kh = 2 pi/48 plus the wall-normal part; BDF2 at 48 x 200 gave -0.58% to -0.60% for all three dt
(section 47), so the temporal part of both schemes is already below the spatial one here. BDF2-PISO at
48 x 400, dt 0.05, like for like: growth 0.002247 (+0.5%), phase 0.248715 (-0.47%), 3.9 min against
RK3's 2.7 min -- same phase speed to 0.015%, growth error 2.5x larger, and slower.

**T11 (energy, 64^2 quads, T = 0.1, loss per turnover, initial flux from the velocity):**

| dt | steady TGV: BDF2-PISO | RK3 | advected TGV: BDF2-PISO | RK3 | RK3, Rhie-Chow off |
|---|---|---|---|---|---|
| 0.01 | 6.16e-2 | 6.74e-4 | 1.48e-1 | 1.22e-3 | 5.39e-4 |
| 0.005 | 1.69e-2 | 3.45e-4 | 3.87e-2 | 4.13e-4 | 6.76e-5 |
| 0.0025 | 5.10e-3 | 1.74e-4 | 1.04e-2 | 1.83e-4 | 8.48e-6 |
| 32^2, dt 0.005 | 4.54e-2 | 1.50e-3 | | | |

(The BDF2 numbers differ from sections 48-49 because the first step now convects; the
advected-TGV loss at dt 0.005 was 4.94e-2 with the zero initial flux and is 3.87e-2 with it.)
RK3 at dt 0.005: **0.04%/turnover, 94x below BDF2-PISO and 5x inside the 0.2% criterion**; over
T = 1 it is 0.042%. The integrator alone (Rhie-Chow off) loses at exactly dt^3 (orders 3.0, 3.0).
With Rhie-Chow on, the loss is ~dt (steady: 6.7e-4 -> 3.4e-4 -> 1.7e-4), which is the per-stage
damping `D_k ~ dt_k`, not the integrator -- so the plan's "and ∝ dt^3" holds for the time scheme
and not for the full step; the full step's dt-dependence is the pressure smoothing. This is the
trade made in (2) above: a dt-independent Rhie-Chow gives a dt-independent 1.2-1.7% floor; an
O(dt) one gives 0.04% at the working step and vanishes with it. On 32^2 the loss is 0.15%, so the
term is O(h^2) as before.

**T9 (fine butterfly, 27968 quads, dt 0.005, `--time-scheme rk3`):**

| | St | Cd mean | Cl amp | Cl rms | Cd pressure |
|---|---|---|---|---|---|
| BDF2-PISO, dt 0.005 (section 45) | 0.1782 | 1.4878 | 0.3520 | 0.2492 | 1.1112 |
| RK3, dt 0.005 | 0.1781 (-0.06%) | 1.4845 (-0.22%) | 0.3495 (-0.71%) | 0.2470 | 1.1094 |
| HydroGym P2-P1 | 0.1791 | 1.4862 | 0.3582 | | |

St and Cd inside the 0.3% band; the lift amplitude moves -0.7%, outside it, and away from
HydroGym (-2.4% against -1.7%). The lift amplitude was already the quantity most sensitive to the
time discretisation (5% between dt 0.01 and 0.005 with the lagged flux, section 45), so a 0.7%
shift between two second/third-order integrators at the same dt is the temporal error of ONE of
them showing. The dt 0.0025 pair (run 2026-09-24) says which -- and that it is not temporal:

| | St | Cd mean | Cl amp |
|---|---|---|---|
| BDF2-PISO dt 0.005 | 0.1782 | 1.4878 | 0.3520 |
| BDF2-PISO dt 0.0025 | 0.1781 | 1.4874 | 0.3518 |
| RK3 dt 0.005 | 0.1781 | 1.4845 | 0.3495 |
| RK3 dt 0.0025 | 0.1780 | 1.4829 | 0.3477 |

BDF2 is dt-converged (0.06%). RK3 moves 0.5% in Cl amp and 0.1% in Cd when dt halves -- AWAY from
BDF2, to a gap of 1.2% / 0.3%. The two schemes share every spatial operator except the Rhie-Chow
damping: BDF2's Choi term is dt-independent, RK3's per-stage damping is O(dt) and vanishes as dt
-> 0. So the dt -> 0 limits differ by the Rhie-Chow damping's effect on the shedding amplitude,
about 1.5% of Cl amp on this mesh -- the price of the collocated pressure smoothing, which BDF2
carries at every dt and RK3 sheds. Neither is "right"; the spatial limit is the undamped one, and
RK3 is closer to it. The G1 criterion (within 0.3% of BDF2) is therefore mis-posed for Cl amp
and is recorded as such; St and Cd meet it. Same wall
time per step as BDF2-PISO here (139 against ~140 ms: the three Poisson solves cost what the two
correctors did), 12 shedding periods in the window, log `results/logs/t9_rk3_butterfly_fine.log`,
fields `results/t9/rk3/butterfly_fine_re100.npz`.

**G1 verdict.** T8 met on both counts; T11 met on magnitude (0.04% against 0.2%), and ∝ dt^3 for
the integrator with the Rhie-Chow damping ∝ dt as the leading full-step term; T4 met for
convection (order 3), diffusion second order by construction (Crank-Nicolson) at a constant
1000x below BDF2; T9 met for St and Cd (-0.06%, -0.22%); the lift-amplitude gap (-0.7% at dt 0.005, -1.2% at 0.0025) is the dt-independent Rhie-Chow damping that BDF2 keeps and RK3 sheds, not a temporal error (table above). Why no dt-independent Rhie-Chow inside RK3: Choi's term makes the damping dt-independent BY
DESIGN, and a dt-independent damping is a dt-independent dissipation -- section 49 already
measured it in BDF2 with the coupling converged (0.14-0.33%/turnover at 64^2, order 0.5-0.7 in
dt). Inside RK3 the same term, with the per-stage matched pair and either stage step in its
denominator, gave 1.2-1.7%: the same floor, larger because three projections per step each
carry it. The O(dt) stage damping trades that floor for a term that vanishes with dt, which is
the right trade for an integrator whose own error is dt^3. BDF2-PISO stays the default (`time_scheme = "bdf2"`); RK3 is the LES integrator.

## 51. L2: the 2.5D solver -- unstructured plane, Fourier span (2026-09-24)

`src/upiso25.py`, class `PISO25(mesh, nz, Lz, nu, dt, bc_u, bc_v, bc_w, bc_p)`. The plane is the
existing cell-centred collocated FV scheme; the span is periodic and spectral: `nz` planes,
`nk = nz/2 + 1` modes from `rfft`, Nyquist held at zero. Per RK3 stage (the L1a integrator):

* nonlinear term in physical space on `M = 3nz/2` planes (3/2 rule): the in-plane part is the
  face sum `sum_f F_f phi_f` with the central + skewness face value (what the 2D deferred central
  converges to; no upwind matrix is needed for an explicit term), plus `V d(w phi)/dz` spectral;
  the products are truncated back to `nk` modes -- dealiased in z by construction;
* one momentum solve per mode and component: `(V/dt) - beta (L_2D - nu k^2 V)`, Crank-Nicolson in
  both directions, the deferred cross-diffusion iterated `n_inner` times; real factorisation,
  complex right-hand side as two columns;
* Rhie-Chow per mode with `D = dt_k V / a_P(k)` (no transient term, section 50), all modes at
  once as (nface, nk) complex arrays;
* pressure per mode, `[Lap(gam_k) - k^2 D_k V] pp = V div F* + i k V w*`, mode 0 pinned at one
  cell with the mean removed (exact for the compatible right-hand side), `u -= D grad pp`,
  `w -= D i k pp`, `F -= F_pp`, `p += pp`.

Factorisations: `3 x nk x (3 + 1)` (stages x modes x (u, v, w, p)), cached for the run since
`a_C(k)` depends only on dt, nu and k. Cost at 32^2 x 32 modes: 100 ms/step; 6992-cell butterfly
x 4 planes: 160 ms/step (three 2D problems per plane-mode; the 2D RK3 on the same mesh is 35 ms).

**Mode 0 IS the 2D solver.** The 2D RK3 (`PISO.step_rk3`) and `PISO25` with `nz = 4` on a
z-independent field: `max|u_25 - u_2D|` 1.8e-15, `|p|` 3e-14, `|F|` 6e-17, `w` identically zero,
over five steps. Same mesh, same factorisations up to the solver's own round-off. This is the check
that makes the rest of the 2D validation ladder (T1-T11) carry over to the plane of the 2.5D solver
without rerunning it.

**G2-A, energy balance. 3D Taylor-Green Re 100, (2 pi)^3, 32^2 x 32 modes, dt 0.02, T = 10.**
Two references for `-dE/dt`: `nu * int |omega|^2` with the vorticity from the least-squares cell
gradient (the plan's wording), and the scheme's OWN discrete dissipation
`eps_d = -sum phi . L_3D phi` with the operators the solver applies (in-plane orthogonal + cross,
spectral z). The two differ by the gradient's truncation error, which is not a solver defect:

| t | E | nu Z (gradient) | eps_d (discrete) | -dE/dt | -dE/dt / nu Z | -dE/dt / eps_d |
|---|---|---|---|---|---|---|
| 0 | 31.0063 (pi^3 exact) | 1.8445 (1.8604 exact) | 1.8564 | | | |
| 1.0 | 29.1440 | 1.9016 | 1.9187 | 1.9213 | 1.0103 | 1.0013 |
| 3.0 | 24.5743 | 2.6217 | 2.7192 | 2.7225 | 1.0385 | 1.0012 |
| 5.0 | 18.4756 | 3.0522 | 3.2277 | 3.2342 | 1.0596 | 1.0020 |
| 7.0 | 12.3892 | 2.4294 | 2.6679 | 2.6734 | 1.1004 | 1.0021 |
| 9.0 | 8.0568 | 1.5327 | 1.6831 | 1.6852 | 1.0995 | 1.0013 |

Against the discrete dissipation the balance holds to **0.11-0.22% at every sample** (criterion
0.5%): that residual is the numerical dissipation of the full step (the O(dt) Rhie-Chow damping
of section 50, the 32^2 in-plane truncation). Against the gradient-based enstrophy the "error"
grows to 10% by t = 7, when the flow has cascaded to scales the 32^2 least-squares gradient
under-resolves by 10% -- the same O(h^2) under-estimate the T11 enstrophy showed; at 64^2 the
gradient gap falls to 1.1-2.4% while the discrete balance tightens to 0.02-0.06%. dt 0.01 at 32^2:
0.06-0.10%, half the dt 0.02 residual -- the O(dt) stage damping of section 50, as expected.

| variant | -dE/dt / eps_d - 1, range over t = 0.5..9.5 | -dE/dt / nu Z - 1 at t = 7 |
|---|---|---|
| 32^2 x 32, dt 0.02 | 0.11-0.22% | 10.0% |
| 32^2 x 32, dt 0.01 | 0.06-0.10% | 10.0% |
| 64^2 x 32, dt 0.02 | 0.02-0.06% | 2.4% |

The gradient gap is h-dependent and dt-independent; the discrete residual is both, and the plan's
criterion is met with margin in every variant. Peak dissipation eps_d = 3.23 at t = 5.0
(nu Z: 3.05 at t = 4.9); Brachet et al. (1983) Re 100: peak at t ~ 5 (see V1 for the
quantitative comparison at Re 1600).

**G2-B, spanwise convergence (32^2 plane, dt 0.02, T = 1, error against nz = 32):**
nz 4: 1.84; nz 8: 1.18e-1; nz 16: 4.99e-4. Geometric (150x then 240x per doubling): spectral.

**G2-C, in-plane convergence (nz = 8, dt 0.01, T = 1, cell-averaged against n = 128):**
n 16: 1.63e-1; n 32: 3.95e-2 (order 2.04); n 64: 7.93e-3 (order 2.32, the reference's own error
entering). Second order in the plane.

**G2-D, periodic-span cylinder Re 100 (coarse butterfly, 6992 cells x 4 planes, Lz = 4, dt 0.005,
w perturbation 1e-3 in the near wake):**

| | St | Cd mean | Cl amp | Cl rms | Cd pressure | ms/step |
|---|---|---|---|---|---|---|
| 2D RK3, dt 0.005 | 0.1753 | 1.4704 | 0.3236 | 0.2289 | 1.0978 | 33 |
| 2.5D, 4 planes, w perturbation 1e-3 | 0.1753 | 1.4704 | 0.3236 | 0.2289 | 1.0978 | 133 |

Identical to every printed digit (criterion 0.1%). The spanwise energy E_3d fell from 1.3e-7 at
t = 0 to 3.5e-27 at t = 150, exponentially at 0.31 per unit time (a factor 50 per 10 time units,
steady from t = 40): the periodic-span perturbation at Re 100 is damped, as it must be (mode A
onsets near Re 190), and the 3D machinery -- per-mode Dirichlet zero at inlet and wall, Neumann
outlet, the k >= 1 Helmholtz solves, the spectral w-coupling -- ran for 30000 steps without
feeding anything back into mode 0. `results/t9/rk3/butterfly_25d_nz4.npz`,
`results/logs/t9_25d_butterfly_nz4.log`. **G2 met on all four counts.**

The plan's "spanwise-inclined TGV" is replaced by B and C on the standard 3D TGV, which has the
z-dependence (cos z) and the in-plane structure the inclined one would have tested.

## 52. L3: eddy viscosity on the 2.5D solver (2026-09-24)

`src/usgs.py`: the velocity-gradient tensor on (ncell, nz) fields (in-plane from the LSQ cell
gradient with the solver's boundary values, spanwise spectral), the cell filter width
`Delta = (V dz)^(1/3)`, and the closures of `src/sgs.py` (Smagorinsky, WALE) applied to that
tensor. `PISO25.sgs_model = "wale" | "smagorinsky" | "none"`; `nu_t` (ncell, nz) kept for reporting.

**The eddy-viscosity term is explicit.** `div(nu_t (grad u + grad u^T))` is evaluated with the
convection on the 3/2-padded planes -- `nu_t x grad u` is a product and is dealiased with the
rest -- and carried by the RK3 gamma/zeta weights; the molecular viscosity stays Crank-Nicolson
implicit. A z-varying coefficient cannot sit inside the per-mode implicit solve, and the explicit
limit `nu_t dt/h^2` is not binding for a wall-resolved LES. In-plane faces use the Laplacian's own
split, `nu_f [E_f/d (phi_N - phi_P) + T_f . grad_f phi]`, plus the transpose
`nu_f (d u_j/d x_i)_f S_j`; spanwise `V d/dz[nu_t (d u_i/dz + d w/d x_i)]` spectral.

**G3 checks (`test_usgs.py`, all pass):**

| check | result |
|---|---|
| solid rotation u = omega x r (Dirichlet box, omega 1.7) | \|S\| 5e-15, Smagorinsky nu_t 2e-18, WALE / analytic (C_w Delta)^2 ((2/3) omega^4)^(1/4) - 1 = 1e-15 |
| filter width follows the cell (n 8x8x4 -> 16x16x8, shear flow) | nu_t ratio 4.0000 (Delta^2) |
| WALE near a no-slip wall (u = y, v = y^2/2, 192 cells across, fit y in 0.02-0.12) | exponent 2.970 (3 +- 0.25), through the mesh gradient path |
| manufactured variable-nu operator, (2 pi)^3 periodic, nu = 1 + 0.5 sin x cos y sin z, three-component field with div u != 0 | L2 orders 1.98/1.99/1.97 (16 -> 32) and 1.99/2.00/1.99 (32 -> 64) in x/y/z |

**Dynamic check, TGV Re 1600, 64^2 x 64 modes, dt 0.02 (the V1 preview; V1 proper follows G3):**
| run | -dE/dt peak (per unit volume) | at t | resolved eps_d peak | numerical + model share at the peak | <nu_t>/nu (mean / max at the peak) | ms/step |
|---|---|---|---|---|---|---|
| DNS (les_findings.md: SEM reference) | 0.012299 | 8.93 | | | | |
| no model | 0.01303 (+5.9%) | 8.4 | 0.01163 (-5.5%) | 11.1% | 0 | 731 |
| WALE | 0.01188 (-3.4%) | 8.5 | 0.00442 | 63.4% | 1.32 / 24 | 964 |

`figures/utgv1600_dissipation.png`. Read carefully: (1) with no model the 64^3-equivalent grid
over-dissipates the peak by 6% and 11% of the peak is numerical (the O(dt) Rhie-Chow damping
plus the in-plane truncation, section 51); the LSQ-gradient enstrophy sits 40% low at the peak,
which is the resolution, not the balance. (2) WALE brings the peak value inside 5% but the model
does 63% of the dissipating and is active from t = 0 -- <nu_t>/nu 0.8 on the laminar initial
field (WALE responds to g.g, and S^d is not zero for the Taylor-Green vortex), rising to 1.3
with a maximum of 24 nu -- and the curve's shape is wrong before the peak: a shoulder at 0.009
from t = 4.5 to 7 where the DNS rises smoothly (the structured 48^3 study saw the same class of
behaviour, `les_model_study.md`). (3) Both peaks land 0.4-0.5 early (t 8.4-8.5 against 8.93,
-5%), outside V1's 3% window; the 0.1 sampling is not the cause. V1 proper (after G3) needs the
64^2 x 64 against 96^2 x 96 pair to separate resolution from model; this preview says the
implicit run is already within 6% at the peak, and the WALE constant at this Delta is too
active for this transitional case -- as WALE is known to be on laminar-to-turbulent Taylor-Green.

## 53. L5: forcing, statistics, restart on the 2.5D solver (2026-09-24)

* `PISO25.set_mass_flow(U_bulk)`: uniform body force adjusted after every step,
  `f += (U_target - U_b)/dt`; `f_bulk` is the mean pressure gradient. On the laminar channel
  (periodic x, walls y = +-1, nu 0.05, U_b 1, T = 20) the force settles on `3 nu` with the
  wall-cell's one-sided flux error: -2.6% / -0.61% / -0.20% at ny 8 / 16 / 32, and the profile
  converges to Poiseuille at order 1.94 / 1.99. The bulk is held to 1e-4 .. 4e-8 while the
  profile is still relaxing (the controller is exact only once the wall stress is steady).
* `src/ustats.py` `Stats(s, coord)`: bins cells by a centroid coordinate (unique rounded values,
  so any y-distribution works), accumulates volume-weighted U, V, W, second moments and uv over
  cells, planes and samples. One sample equals the instantaneous bin mean to 0.0; fifty samples
  equal the mean of the instantaneous profiles to 0.0; the fluctuations of a z-uniform flow are
  4e-15.
* `PISO25.save / load`: u, v, w, p, Ff, time, step, f_bulk. RK3 carries no history, so a restart
  needs nothing else: **100 steps after a load are bitwise identical** (u, v, w, p, Ff, f_bulk all
  `array_equal`) to the uninterrupted run, on a 3D-perturbed forced channel.
* `rect_mesh(..., cluster_y=beta)`: wall-normal-only tanh clustering for the channel (the existing
  `cluster` stretches toward all four walls). Cells stay rectangular, `orth` = 1.

`test_uchannel_laminar.py`, all pass. G5 met: restart lossless; statistics reproduce the analytic
mean up to the scheme's O(h^2) Poiseuille error (the criterion's "to round-off" holds for the
statistics machinery against the solution, not for the solution against Poiseuille, whose wall
flux is one-sided -- section 38's force finding, seen from the other side).

## 54. V2: turbulent channel Re_tau 180 on the 2.5D solver (2026-09-24)

`run_uchannel25.py`: the structured minimal-channel setup exactly (`run_channel_les.py`) --
delta = u_tau = 1, nu = 1/180, Lx = pi (565 wall units), Lz = 0.34 pi (192), y in [0, 2];
constant pressure gradient f_x = 1, so u_tau = 1 by construction and the measured wall stress is
a check; initial condition the in-house SEM DNS field at t = 18 (`results/minchan_re180_field.npz`,
Delaunay in the plane, periodic linear in z); WALE. Plane 24 x 80 quads with tanh wall
clustering (dx+ 23.6, dy+ 1.02 at the wall and 8.3 at the centre), 32 Fourier modes (dz+ 6.0),
dt 0.002 (CFL 0.3-0.5), T = 30, statistics over t = 10..30 -- 20 time units, 99 bulk
flow-throughs, 10001 samples. 0.2 s/step, 100 minutes on one core.

**The reference is the SEM FOSLS DNS run02** (first-order-system least-squares spectral element,
6 x 18 elements N = 8 x 32 modes, dx+ 11.8, dz+ 6.0, dt 0.0008, the same box and forcing;
`sem_demo/scratch/_dns_drive`, a local copy of `Google Drive/My Drive/lssem_dns`; its own table
puts it within 0.1-3.5% of the MKM / LM / VK / Torroja / AKM databases). Window t = 5.2..30, 3101
samples, u_tau 1.0025, built as the difference of its cumulative statistics files into
`results/fosls_chan180_stats_t5.2_30.npz`. The K- and E-path minimal-channel runs in
`sem_demo/results` are fractional-step SEM runs and are NOT used as reference (the K-path field
at t = 18 remains the initial condition only; its u' peak, 2.845, is 5% above every database).

| quantity | this run | FOSLS DNS run02 | criterion |
|---|---|---|---|
| u_tau from the one-sided wall flux, window mean | 0.9950, **Re_tau 179.1 (-0.5%)** | 1.0025 (Re_tau 180.5) | 2% |
| U+ in the log region 30 < y+ < 100 | +0.15 u_tau mean, 0.26 max (**+1.0% of U+**) | | 3% of U+ |
| U+ vs the log law 0.41 / 5.2 | +0.64 | (the DNS sits above it too) | |
| U_c+ | 18.27 | 18.84 | |
| u_rms+ peak | 2.697 at y+ 15.7 | 2.707 at y+ 13.7: **-0.3%** | 5% |
| v_rms+ max | 0.805 | 0.857 (-6.1%) | |
| w_rms+ max | 0.980 | 1.054 (-7.0%) | |
| -<u'v'>+ max | 0.720 | 0.729 (-1.2%) | |
| <nu_t>/nu over the window (max in the field) | 0.194 (3-5) | | reported (G3) |
| **pressure two-colour mode** | **0.00% of p_rms** (1e-4 at every report) | | **< 1%** |

`figures/uchannel_re180_profiles.png`. The wall stress wandered between 0.92 and 1.12 over the
run (Re_tau 166-202 instantaneous), as a minimal channel does; the bulk velocity stayed within
15.39-15.84 of the DNS's 15.63 without being forced to. Turbulence sustained throughout: no
relaminarisation, no growth of anything at the grid scale, the face high-pass share of the
pressure steady at 0.12-0.20.

**Verdict: V2 met on every criterion** -- wall stress, mean profile, u_rms peak (-0.3% against
the FOSLS DNS; the earlier "-5.2%" was against the fractional-step K-path field, whose own peak is
5% high), shear stress, and the pressure-mode criterion at 0.00% over 99 flow-throughs. The
cross-stream rms are 6-7% low, the usual signature of dx+ 24 (the DNS has 11.8); not a criterion,
recorded.

**Companions (same mesh, dt, window; scored against the FOSLS window):**

| run | Re_tau | U+ log region | u_rms+ peak | v_rms+ | w_rms+ | -<u'v'>+ | U_b | <nu_t>/nu | p two-colour |
|---|---|---|---|---|---|---|---|---|---|
| WALE, constant pressure gradient (the run above) | 179.1 | +1.0% | 2.697 (-0.3%) | -6.1% | -7.0% | 0.720 (-1.2%) | 15.62 | 0.19 | 0.00% |
| WALE, constant mass flow U_b 15.63 | 178.9 (-0.6%) | +1.6% | 2.771 (+2.4%) | -4.6% | -11.7% | 0.724 (-0.6%) | 15.63 (held) | 0.19 | 0.00% |
| no model (implicit) | 181.1 (+0.6%) | -4.9% | 2.500 (-7.6%) | +0.9% | +0.7% | 0.740 (+1.6%) | 15.00 | 0 | 0.01% |

The constant-mass-flow criterion (Re_tau within 2%) is met at -0.6%. Without a model the mean
profile drops 5% in the log region and the streamwise peak 8% while the cross-stream components
come up to the DNS: the 24 x 80 x 32 grid under-resolves the streaks and the model's 0.19 nu
is doing its job on the mean. All three keep the pressure two-colour mode at 0.00-0.01%.

**Near-wall instantaneous planes (`plot_utility/plot_uchannel_nearwall.py`, y+ ~ 12,
`figures/uchannel_re180_nearwall_yp12.png`).** Pressure, streamwise velocity and streamwise vorticity
fluctuations about the plane mean, in wall units, for our field at t = 30 and the FOSLS DNS
checkpoint at t = 30 (`_dns_drive/checkpoint_0037500.npz`: 14 split-real fields u v w ox oy oz p
per spanwise mode; omega_x is a FOSLS primary unknown and is plotted as such). Our planes are
refined spectrally in z for plotting and both columns use bilinear shading in x; the LES cells are
dx+ 23.6, the DNS is sampled at 5.9.

| plane y+ ~ 12, one instant, t = 30 both | 2.5D LES | FOSLS DNS |
|---|---|---|
| p' rms / u_tau^2 | 1.25 | 1.54 |
| u' rms / u_tau | 2.12 | 2.66 |
| omega_x rms nu/u_tau^2 | 0.139 | 0.126 |
| u' spanwise energy share, modes 1 / 2 / 3 (lambda_z+ 192 / 96 / 64) | 0.35 / 0.33 / 0.18 | (single instant, not tabulated) |
| u' energy at the streamwise period-2 mode | 0.000 | 0.000 |

Reading: the same objects in both -- a low-speed streak meandering across the box with high-speed
regions beside it, pressure in patches of O(1-4) u_tau^2 at the streak scale, streamwise vorticity in
elongated streaks between the velocity streaks -- ours at the LES resolution (smoother, 20% lower u'
on this plane, pressure 19% lower in rms, vorticity rms 10% higher). No energy at the period-2 grid
mode in either. Single-plane, single-instant numbers are noisy (24 x 32 cells, strongly correlated);
the windowed profiles above are the measure. `Stats` now carries p and p'^2 so the next runs give
p_rms(y) over the window. (An earlier version of this comparison used the fractional-step K-path
field at t = 18 and the E-path state at t = 15.95 and found their pressure levels differ by 2x from
each other; neither is the reference and that finding is withdrawn from the comparison.)

## 55. The channel on triangles (2026-09-24)

Asked for: the V2 channel on a triangular unstructured grid. Three meshes, `run_uchannel25.py
--cells tri | --cells hybrid --wall-layers-y N | --mesh meshes/channel_tri_graded.msh`.

**1. Every quad split (`rect_mesh(cells="tri")`, 3840 cells).** Wall cells are dy+ 0.5 by dx+ 24:
47:1 triangles, orthogonality 0.085. Diverged at step 11. Not a candidate: the deferred
non-orthogonal correction contracts at sin(theta) per pass (over-relaxed split |T|/|E| = sin theta)
and at orth 0.085 that is 0.996 -- the cross term is effectively lagged a whole stage.

**2. Hybrid, quad layers at the walls and triangles in the core (`wall_layers_y`).** Rows of the tanh
distribution below the switch stay quads:

| wall layers | triangles from y+ | orth min | dt 0.002 | dt 0.001 |
|---|---|---|---|---|
| 12 | 20 | 0.214 | diverged step 63 (t 0.13) | |
| 16 | 32 | 0.281 | diverged step 169 (t 0.34) | diverged step 106 (t 0.11) |
| 28 | 90 | 0.516 | stable to t 0.8 (400 steps), triangle-pair two-colour 10 -> 19% of p_rms, high-pass share 0.21 -> 0.39 | |
| 32 | 118 | 0.579 | stable to t 0.8, pair two-colour 10 -> 23% | |

The divergence time does not move with dt: it is not the explicit CFL limit but the same lagged
cross term as (1), on triangles whose aspect ratio (dx+ 24 over dy+ 4-7) makes the diagonal face
skew. And where the hybrid mesh does hold (layers 28/32), the two-colour pressure content of the
triangle pairs -- the half-difference of the two halves of a split quad -- is 10-23% of p_rms and
rising, against 0.00% on the quad cells of the same mesh: the non-bipartite mechanism of section
36/41 (the 2D cavity and cylinder speckle), now in a turbulent field. The long layers-28 run (T = 30, statistics t 10-30, against the FOSLS window) is the record of what
a triangle core does to the channel:

| | Re_tau | U+ log region vs DNS | u_rms+ peak | w_rms+ max | -<u'v'>+ | p two-colour quad / tri pairs |
|---|---|---|---|---|---|---|
| all quads (section 54) | 179.1 | +0.15 u_tau | 2.697 (-0.3%) | 0.980 (-7%) | 0.720 | 0.00% / -- |
| hybrid, triangles from y+ 90 | **170.9 (-5.0%)** | **+1.02 u_tau** | **3.028 (+12%)** | **1.412 (+34%)** | 0.687 (-6%) | 0.61% / **14.4%** |

Same wall layers, same dt, same model; the only change is the core. The wall stress falls 5%,
the log region rises a full u_tau, the fluctuations inflate, and the triangle pairs carry 14% of
p_rms in the two-colour mode while the quad cells of the SAME mesh stay at 0.6% -- with the face
high-pass share of the pressure at 0.37-0.61 against 0.13-0.20 on quads. The V2 pressure-mode
criterion fails on the triangles and passes on the quads of one mesh.

**3. Isotropic graded triangles (gmsh, `meshes/channel_tri_graded.geo`).** Size h = max(2, min(16,
54 y)) wall units (h = max(hw, min(hmax, 0.3 d_wall))), frontal-Delaunay, periodic x by gmsh's
matched-node constraint, 10278 triangles, orthogonality min 0.87 (5th percentile 0.99), wall
cells h+ 1.8, centre h+ 15, 3032 cells below y+ 10. This is what "an unstructured triangle mesh"
means in practice: isotropic cells, so the wall layer costs 5x the cells of the stretched quads
for the same wall spacing. Statistics are binned into the 80 tanh intervals of the quad run.
At dt 0.002 it diverged at step 6: the explicit stage convection on the buffer-layer cells
(h+ 2-3 where u+ is 5-10) is at CFL ~1 per component, over the RK3 limit. At dt 0.0005 it runs
(CFL 0.08 by the sqrt(V) measure), 1.2 s/step; dt 0.001 also holds (400 steps, CFL 0.16, 1.13 s/step) and is the step of the long run (T = 20, statistics from t = 8, ~6 h). With WALE the mean eddy viscosity is 0.03-0.05 nu against 0.15-0.20 nu on the quads: Delta = (V dz)^(1/3) is the cell size, and the isotropic wall triangles are 5-10x smaller in volume than the stretched quads at the same height. Its
face high-pass pressure share is 0.56-0.68 from the first steps, against 0.13-0.20 on quads:
the triangle pressure is 3-4x rougher cell to cell, before any turbulence statistics exist.

dt 0.001 diverged at step 855 (t = 0.86) in the long run; the probe at dt 0.0005 ran through
t = 1.2 (2400 steps) without incident (CFL 0.08-0.10 by sqrt(V), high-pass share 0.70-0.81,
u_tau drifting to 0.92). So the divergence IS step-size dependent on this mesh, unlike the hybrid
one -- the explicit stage convection plus the explicit eddy-viscosity term on h+ 2 cells with
u+ 5-10 -- and the graded triangles need dt 0.0005: 40000 steps at 1.15 s, ~13 h for T = 20.
Launched (`uchan_trigraded_32_wale_cpg_dt5e-4`, statistics t 8-20).

The long run (dt 0.0005, T = 20, statistics t 8-20, 24000 samples, 13 h on one core;
`figures/uchannel_re180_quad_vs_tri.png` with the quad and hybrid runs):

| | Re_tau | U+ log region vs DNS | u_rms+ peak | v_rms+ | w_rms+ | -<u'v'>+ | U_b start -> end | face high-pass share of p |
|---|---|---|---|---|---|---|---|---|
| quads 24 x 80 (section 54) | 179.1 | +0.15 u_tau | 2.697 (-0.3%) | 0.805 | 0.980 | 0.720 | 15.7 -> 15.6 | 0.13-0.20 |
| graded triangles 10278 | 183.7 (+2.1%) | **+1.22 u_tau (max 1.29)** | **3.037 (+12%)**, at y+ 24 | **0.624 (-27%)** | 0.888 (-16%) | 0.739 (+1.4%) | **15.7 -> 17.8** | **0.59-0.87** |

Stable for 20 time units, turbulent throughout, and wrong in a way the quads are not: the bulk
velocity climbs 14% over the run while the wall stress averages 1.04 x the forcing -- with a
constant pressure gradient the two together are impossible (dU_b/dt = f - tau_w/delta should be
negative), so the discrete momentum balance does not close on the triangles: there is a spurious
streamwise momentum source of about 10% of the applied gradient. The face high-pass share of the
pressure sits at 0.6-0.9 against 0.13-0.20 on quads, the two-colour pressure of a non-bipartite
mesh (the pair indicator reads 0 here only because a gmsh mesh has no quad pairs to difference);
the eddy viscosity is 0.036 nu (isotropic wall cells are tiny, so Delta is), leaving the pressure
mode undamped. The mean profile is 1.2 u_tau high in the log region, the streamwise fluctuation
12% high with its peak pushed out to y+ 24, the cross-stream ones 16-27% low: the signature of the
2D triangle results (sections 36, 41) carried into turbulence.

The figure adds one thing the table cannot: on the hybrid mesh the Reynolds shear stress
profile ZIGZAGS from bin to bin across the triangle core (y+ > 90), alternating +-0.1 around the DNS
curve, while the same run's quad layers below y+ 90 lie on it -- the two-colour mode is in the
velocity statistics, not only in the pressure -- and the hybrid's w_rms climbs to 1.4 u_tau in the
core against 0.6 for the DNS. On the isotropic triangles the shear stress is smooth but falls below
the DNS from y+ 50 outward, which with the +14% bulk drift is the momentum imbalance seen twice.

**Verdict on triangles for LES:** the V2 criteria fail on every triangle configuration tried, and
the mechanism is the same one the 2D cavity and cylinder showed -- the two-colour pressure mode of
a non-bipartite mesh -- now with a momentum-balance consequence. Quads (or a quad wall layer with a
quad core) are the mesh for wall-bounded LES with this scheme; triangles stay where they were
validated, laminar external flow with a vertex-averaged post-processing. The plan's exclusion of
triangle meshes for LES stands, now measured.


## 56. HydroGym's NACA0012 control task, decoded, and its rebuild on our solver (2026-09-24)

HydroGym's airfoil environments run only on m-AIA (closed, amd64). What the environment IS comes from
its published files (`dynamicslab/HydroGym-environments`, `NACA0012Gust_2D_Re100_AOA40/`) and the open
wrapper (`hydrogym/maia/envs/naca0012.py`, `env_core.py`):

* **Actuation: three synthetic jets around the leading edge** (`lbNoJets 3`, `lbObjectClass 1` flat
  plate, `lbJetCenter`, `lbJetAngles`, `lbJetRanges`). Placed on their STL (chord 8 cells, AoA 40):
  jet 1 at x/c 0.084 on the upper (suction) surface, jet 2 at the nose (x/c 0.000), jet 3 at x/c 0.088
  on the lower surface; slot widths 0.050 / 0.037 / 0.050 c; jet directions 95 / 140 / 185 deg,
  i.e. along the local surface normal (dot 0.83-1.00). Action a in [-1, 1]^3 scaled by
  `max_control 0.03` lattice units = **0.52 U_inf** (U_inf 0.0577), tanh-ramped over 95% of the
  action interval (`lbUseControlRamping`, `lbRampPercentage 0.95`).
* **Cadence:** 75 LBM steps per action; with c = 8 cells and U_inf 0.0577 cells/step a convective
  time is 139 steps, so one action per **0.54 c/U**; 200 actions per episode (108 c/U).
* **Observation:** `observation_type [u, v]` at the probe(s) `pp_probeCoordinates [-1.0, 0.0]`: the
  velocity one chord upstream. (Not the forces.)
* **Reward, gust task:** `-|C_L - 1.027| - 0.25 |C_D - 1.081|` (omega 0.25, the unperturbed means of
  their solver), under a **gust**: inlet velocity x2 at the centre (`lbGustFactor 2.0`), 16 cells = 2c
  wide (`lbGustRanges`), 4000 LBM steps = 29 c/U long (`lbGustDuration`). The plain `NACA0012` task
  (L/D reward) exists at Re 100 only for AoA 20 and at Re 1000 for AoA 40.

**Rebuild:** `src/ujets.py` (`JetSet`: slot faces from chord fraction and side, nose slot by distance
from the LE point, Dirichlet wall velocity along the outward normal, tanh ramp, jet mass flux) and
`naca_env.py` (`NACAJetEnv`: 2D BDF2-PISO, coarse 8260-quad C-grid `meshes/naca0012_a40_coarse.msh`
for cost, dt 0.01, 54 substeps per action, probe (u, v) [+ forces], gust or L/D reward with OUR
unperturbed means, reset from stored shedding snapshots). On the coarse mesh the slots are 4 / 9 / 5
wall faces (0.045 / 0.041 / 0.060 c). Costs: 52 ms/step here, so ~9 min per 200-action episode per
environment; the fine 33k mesh is 200-250 ms/step and out of reach for RL.

**Coarse-mesh baseline and the training launch (2026-09-24).** `meshes/naca0012_a40_coarse.msh`
(`gen_naca_cgrid.py --alpha 40 --ns 97 --nw 71 --neta 36 --d0 0.012`): 8260 quads, 96 wall faces,
min angle 40 deg, orth min 0.66. Baseline shedding, dt 0.01, window t 72-120:

| | St_c | C_D | C_L mean | C_L amp |
|---|---|---|---|---|
| coarse 8260 | 0.2270 | 1.0344 | 0.9762 | 0.121 |
| fine 33040 (section 46) | 0.2331 | 1.0679 | 1.010 | 0.137 |
| HydroGym m-AIA config | | 1.081 | 1.027 | |

The coarse grid is 2.6-3.3% low on the forces and 12% low on the lift amplitude: adequate for a
controller whose reward is a force deviation measured against the SAME solver's baseline (C_D0
1.0344, C_L0 0.9762 in the reward), not for a force benchmark. Episodes start from one of 10
phases of the developed shedding (`make_naca_snapshots.py`, `results/naca/a40_coarse_snapshots.npz`).
Environment check: zero action from a snapshot gives reward -0.12 (the phase-dependent departure
from the means); jet 1 at +1 for one action raises C_L 0.87 -> 1.22, jet 3 at -1 raises both forces
and costs -0.81; 2.7 s per action here.

Launched on the Spark (`hgrl`, GB10 for the policy, 8 `SubprocVecEnv` environments on the CPU):
`train_naca_ppo.py --task gust --obs probe --n-envs 8 --total-steps 40000`, PPO MLP 64x64,
n_steps 200 (one episode per environment per rollout), lr 3e-4; logs and checkpoints in
`~/hydrogym_cmp/rl_logs/naca40_gust_probe/` on the Spark. Expected ~8 min per rollout of 1600
steps, ~3.5 h for 40k. The container's GPU had to be recovered first (`docker restart hgrl`: NVML
had failed after two days up; the image and container are intact).

**The do-nothing baseline of the gust episode** (jets off, one episode of 200 actions from a
shedding snapshot, `results/naca/a40_gust_zero_action.npz`): return **-64.3**. The gust
(free stream x2 at the centre for 29 c/U) triples the loads -- peak C_L 2.96 and C_D 2.73 against
the means 0.98 / 1.03 -- and the mean |C_L - C_L0| is 0.72 during the gust and 0.10 after it, so
the return is 70% gust, 30% the shedding itself (which the reward also penalises, at the 0.12
lift amplitude). The untrained policy's first rollout scored -172: random jets at up to 0.52
U_inf are worse than doing nothing, as they should be. A learned policy has to beat -64.

## 57. L4 step 1: the per-mode families solved as one block (2026-09-24)

Every implicit system of the 2.5D solver is `(A + s_k D) x_k = b_k` over the modes k: one shared
sparse matrix, a positive diagonal, a per-mode shift s_k = k_z^2 (times beta nu for momentum).
`src/umodesolve.ModeFamily` solves the whole family at once: block PCG on the (N, nk) right-hand
side (each column its own scalars, stop when every column is below rtol), preconditioned by ONE
algebraic-multigrid hierarchy built from the k = 0 matrix whose level operators are shifted per
mode exactly, `A_l(k) = A_l + s_k D_l` with `D_l = R_l D_{l-1} P_l` precomputed; damped Jacobi
with the per-mode diagonal as the smoother (vectorised over modes), a batched dense solve at the
coarsest level, complex right-hand sides as two real columns. `PISO25(solver="amg")` selects it for
momentum and pressure (default stays `"lu"`).

**The hierarchy decides it, not the smoother.** On the wall-clustered channel pressure operator
(96 x 160, cell aspect 6:1 at the wall, 32 modes, rtol 1e-8): smoothed aggregation with block Jacobi
55 iterations (k = 0) falling to 22 (k = 16); pyamg's own smoothed aggregation with Gauss-Seidel 35;
**Ruge-Stuben classical coarsening 11 with Gauss-Seidel and 11-13 with block Jacobi**. The
G4 criterion "pressure to 1e-8 in < 30 iterations on the channel operator" is met with the classical
hierarchy:

| pressure family, 32 modes, rtol 1e-8 | N | RS + Jacobi 1+1 | RS + Jacobi 2+2 | LU (17 solves) | LU factor |
|---|---|---|---|---|---|
| channel 24 x 80 | 1,920 | 8 it, 12 ms | 6 it, 14 ms | 2 ms | 0.04 s |
| TGV box 64^2 (all-Neumann, singular k = 0) | 4,096 | 7 it, 18 ms | 5 it, 19 ms | 6 ms | 0.17 s |
| channel 96 x 160 | 15,360 | 13 it, 120 ms | 11 it, 147 ms | 22 ms | 0.59 s |
| channel 192 x 320 | 61,440 | 17 it, 800 ms | 14 it, 912 ms | 159 ms | 5.1 s |
| channel 384 x 640 | 245,760 | 18 it, 4.4 s | 15 it, 4.8 s | (not factorised) | |

Iterations grow 8 -> 18 over 128x in N (the hierarchy has 6-9 levels); the answers agree with LU to
4e-9 relative. **On one CPU core the cached LU solves stay 5-6x faster per step** up to 6e4 cells --
the factorisation is paid once per run -- so `"lu"` remains the default there. What the block
form buys is (i) memory and setup at V3 size (the 245k-cell factorisations were not attempted;
the AMG setup is 0.19 s), and (ii) the shape that moves to a GPU: the whole solve is sparse
matrix times dense block, diagonal scaling, and column reductions, which is what CuPy and AmgX
do well; the Python-level cost per cell-mode-iteration measured here is 58 ns, memory-bound numpy.

**Integrated (`PISO25(solver="amg")`, momentum and pressure).** Against the LU path on the 3D
Taylor-Green (20 steps): max|du| 9e-9 / 3e-9, max|dp| 1e-6 / 5e-7 at 32^2 x 16 / 64^2 x 32, energies
equal to 9 digits; the perturbed-Poiseuille channel 24 x 80 x 32 agrees to 6 digits in E and 7 in
max|u|. Iterations per stage: momentum 5 (the second inner pass 0 -- the lagged cross-diffusion
correction is below rtol), pressure 8. Cost on one core: 100 vs 46 ms/step (32^2 x 16), 642 vs 357
(64^2 x 32), 349 vs 148 (channel) -- the block path is 2x slower than cached LU here, as the table
above predicts, and is kept for what comes next. One sign slip on the way: the family is the
negated operator, so the pressure correction is `fam.solve(-(rhs - corr))`, not minus that.

**Against the G4 bar on one CPU core** (`bench_modesolve.py cpu 32`: the channel pressure family,
17 rfft modes as 34 real columns, rtol 1e-8, pyamg hierarchy built once):

| plane | N | iterations | solve | per 1e5 cell-modes |
|---|---|---|---|---|
| 24 x 80 | 1,920 | 8 | 24 ms | 36 ms |
| 96 x 160 | 15,360 | 13 | 425 ms | 81 ms |
| 192 x 320 | 61,440 | 17 | 2.1 s | 100 ms |
| 384 x 640 | 245,760 | 18 | 9.2 s | 110 ms |

The pressure solve alone sits AT the plan's whole-step bar (100 ms per 1e5 cell-modes) on the CPU
and a step has twelve such block solves plus the explicit terms, so the CPU is ~15x short of G4's
speed criterion at V3 size, as expected -- the criterion was written for the GPU. The same code
runs on CuPy (`device="gpu"`); measured on the Spark below.
**On the GB10** (`upict25` container from the `lssem-cupy` image, pyamg built in place with g++ and
python3-dev; `bench_modesolve.py gpu 32`, hierarchy built on the host, levels moved to the device):

| plane | N | iterations | solve GPU | per 1e5 cell-modes GPU | CPU |
|---|---|---|---|---|---|
| 24 x 80 | 1,920 | 8 | 24 ms | 37 ms | 36 ms |
| 96 x 160 | 15,360 | 13 | 73 ms | 14 ms | 81 ms |
| 192 x 320 | 61,440 | 17 | 431 ms | 21 ms | 100 ms |
| 384 x 640 | 245,760 | 18 | 2.2 s | 26 ms | 110 ms |
| 768 x 1280 | 983,040 | 19 | 9.2 s | 27 ms | -- |

Identical iteration counts and residuals (same arithmetic, same hierarchy). Below ~1e4 cells the
GPU is launch-latency bound (the Python V-cycle issues ~40 small kernels per iteration) and no
faster than the CPU; above it the pressure block solve settles at **26-27 ms per 1e5 cell-modes**,
4x the CPU, and flat to 1e6 cells x 34 columns (33M unknowns in 9.2 s). Per iteration that is ~60
GB/s of effective bandwidth against the GB10's ~270: cupyx's sparse-times-dense is not at the
roofline, so a 2-4x remains in a fused kernel. Against the G4 bar of 100 ms per step per 1e5
cell-modes the pressure solve is a quarter of the budget; the momentum families (5 iterations,
diagonally dominant) are cheaper; the rest of the step -- convection on the padded planes, FFTs,
gradients, Rhie-Chow -- is still numpy on the host and has to move as well before the criterion
can be measured on a whole step. That port (an `xp` backend through `PISO25` and the operators it
calls) is the next item.

**First PPO run, result (40k steps, 25 rollouts, 3.7 h on the Spark):** the mean episode return went
-172 -> -165 and stayed there; the last rollouts' episodes score -161 to -174 against the do-nothing
-64. The policy did not learn: from the first rollout the Gaussian's unit initial std saturated the
jets (mean |a| 0.60-0.65 through the run, i.e. jets at 0.3 U_inf on average, adding the loads the
reward penalises) and PPO's clipped updates cannot walk the std down fast enough in 25 updates.
That is the same start-up that cost the cylinder training its first runs. Second run: `log_std_init
-1.5` (initial std 0.22, jets at ~0.1 U_inf) so the exploration starts inside the range where doing
less is better than doing more; same everything else. Logs of the first run kept:
`rl_logs/naca40_gust_probe/`.

**L4 step 2: the whole step on the device (2026-09-24).** `PISO25(device="gpu")`: the solver was
rewritten against an array-module handle (`self.xp`, numpy or cupy) and a device copy of the mesh
(`self.dm`), operators (the gradient matrices, Laplacians and their deferred-correction matrices --
now exposed by `uops.laplacian` -- and the face scatter) and boundary-condition masks; the block AMG
families live on the device, the pyamg hierarchy is built on the host. Same code on the CPU:
Taylor-Green energies to all printed digits against the previous version, `test_uchannel_laminar`
and `test_usgs` pass; on the GB10 the energies and divergence equal the CPU path's.

| 3D TGV, AMG path | CPU (one core) | GB10 | per 1e5 cell-modes, GB10 |
|---|---|---|---|
| 32^2 x 16 | 68 ms/step | 143 | 1555 ms |
| 64^2 x 32 | 572 | 231 | 331 |
| 128^2 x 64 | | 1,166 | 216 |
| 256^2 x 64 | | 6,168 | 285 |
| 384^2 x 64 | | 14,749 | 303 |

**2-3x over the G4 bar** (100 ms per step per 1e5 cell-modes) and only 2.5x faster than one CPU
core at 64^2 x 32 -- the block pressure solve alone measured 26 ms per 1e5 cell-modes, so the rest
of the step is where the time goes now. Phase profile on the GB10, per RK stage (each phase timed with device synchronisation):

| 256^2 x 64 | nonlinear (padded planes) | diffusion x3 | momentum solves 3 comps x 2 inner | Rhie-Chow | pressure solve | rest |
|---|---|---|---|---|---|---|
| ms | 137 | 35 | 6 x 354 (6 it, AMG V-cycle) | 34 | 495 (9 it) | ~35 |

The momentum solves were the bulk: a diagonally dominant family (V/dt on the diagonal) does not
need the multigrid cycle, whose ~10 matvecs per iteration cost more than the 5-6 iterations save.
`ModeFamily(precond="jacobi")` for momentum: the momentum solve
drops 354 -> 169 ms per stage-component (10 Jacobi-PCG iterations against 6 with AMG, at a sixth
of the cost each); CPU answers unchanged (E to 8 digits). Whole step: 64^2 x 32 116 ms (166 per 1e5
cell-modes), 128^2 x 64 673 ms (**125**), 256^2 x 64 4.06 s (188). At 128^2 the bar is within 25%;
at 256^2 the profile is momentum 6 x 169, pressure 487, nonlinear 137 per stage -- and every one of
those is a CSR matrix times a row-major block that cupyx runs at ~25 GB/s (17 ms for a 5-point
65k-row matrix times 66 columns). `src/ucuda.py` `DevCSR`: a CuPy raw kernel
for CSR x row-major block, one thread per (row, column) so the gathers of X[j, :] coalesce -- 0.81 ms
against cupyx's 2.02 for the 65k x 66 test (257 GB/s of block reads, at the GB10's bandwidth),
identical to scipy. Wired into every operator (gradients, Laplacians and their corrections, the face
scatter) and every multigrid level. Whole step: 64^2 x 32 81 ms (116 per 1e5 cell-modes), **128^2 x 64
480 ms (89 -- inside the bar)**, 256^2 x 64 3.0 s (139); per stage at 256^2: momentum 6 x 142 ms (10
Jacobi-PCG iterations: now the un-fused vector updates, not the matvec), pressure 306, nonlinear 112.
Fusing the two PCG vector updates (`cupy.fuse`) and a separate momentum tolerance
(`mom_rtol` 1e-7: energy identical to 9 digits against 1e-9 and LU, 4 iterations either way at 64^2):

| 3D TGV, GB10, AMG path, DevCSR, fused PCG | ms/step | per 1e5 cell-modes | G4 bar 100 |
|---|---|---|---|
| 64^2 x 32 | 77 | 110 | (launch-latency bound) |
| 128^2 x 64 | 497 | **92** | met |
| 256^2 x 64 | 2,720 | 126 | +26% |
| 384^2 x 64 | 6,762 | 139 | +39% |

Per stage at 256^2 x 64: momentum 6 x 133 ms (10 Jacobi-PCG iterations, ~13 ms each of which the
matvec is 0.8: the remainder is the per-column reductions and updates on the (N, 66) block),
pressure 296 (9 iterations), nonlinear 114, everything else 75. Against the CPU: 397 -> 77 ms at
64^2 x 32 (5x), and the CPU cannot run the larger cases in the same session. **G4 speed criterion:
met at 128^2 x 64, 26-39% over at 256^2-384^2.** The next factor is inside the PCG iteration
(transposed block layout so the column reductions are contiguous, or one fused CG kernel), not in
the operators; recorded as the open item. The three G4 criteria: pressure iterations 8-19 to 1e-8
(< 30, met); speed as above; a dt change costs nothing by construction (RK3 has no history) --
not separately measured.

**Second PPO run (initial std 0.22; 40k steps, 3.4 h):** mean return -87.5 -> -82.0, monotone over
the 25 updates (-87.5, -88.4, -87.7, ..., -83.1, -82.8, -82.4, -82.0); the best final episodes -77.
Learning, and 2x better than run 1 at every point, but still below the do-nothing -64: the policy's
own noise (std held at 0.17-0.18 throughout, the entropy did not fall) costs about 20 per episode
against a controller that would sit still, and 25 clipped updates are not enough to walk it down.
Continued from its final checkpoint for 80k more steps (`rl_logs/naca40_gust_probe_std022_cont/`).
What would close the gap faster is an action penalty in the reward or an entropy schedule; HydroGym's
own m-AIA environment has neither, and the point of this run is to reproduce its task as defined.

**Continuation (80k more steps, 7.0 h, 120k total):** mean return -82 -> -64.2, still falling at
about 0.4 per update at the end, the last rollout's episodes -57.6 to -63.4 -- **at the do-nothing
level and crossing it**. The policy that emerged is steady suction on the upper-surface jet
(mean action -0.20 on jet 1, -0.05 on the nose, 0 on the lower jet; std still 0.20). Continued
again for 80k (`rl_logs/naca40_gust_probe_std022_cont2/`).

## 58. V1: Taylor-Green Re 1600 on the GB10 (2026-09-25)

The resolution matrix the V1 preview (section 52) asked for, now that the step runs on the device:
64^2 x 64, 96^2 x 96 and 128^2 x 128 (plane cells x spanwise planes; 33/49/65 modes), no model and
WALE, dt 0.02, T = 20, RK3 with the block AMG solves (`mom_rtol` 1e-7). Wall time 1.5 / 5 / 14 min
per run. Reference: the in-house SEM DNS of `les_findings.md` (2 nu Z peak 0.012299 at t = 8.93;
the published pseudo-spectral value is 0.0128 at t ~ 9.0). `figures/utgv1600_v1.png`.

| run | -dE/dt peak | at t | value vs 0.012299 | time vs 8.93 | resolved eps_d peak | model + numerical share | <nu_t>/nu at the peak |
|---|---|---|---|---|---|---|---|
| 64^2 x 64, no model | 0.01303 | 8.4 | +5.9% | -5.9% | 0.01159 | 11% | 0 |
| 64^2 x 64, WALE | 0.01188 | 8.5 | -3.4% | -4.8% | 0.00435 | 63% | 1.32 |
| 96^2 x 96, no model | 0.01355 | 8.3 | +10.2% | -7.1% | 0.01258 | 7% | 0 |
| 96^2 x 96, WALE | 0.01218 | 8.3 | **-1.0%** | -7.1% | 0.00598 | 51% | 0.70 |
| 128^2 x 128, no model | 0.01197 | 8.4 | -2.7% | -5.9% | 0.01131 | 5.5% | 0 |
| 128^2 x 128, WALE | 0.01192 | 8.4 | -3.1% | -5.9% | 0.00704 | 41% | 0.42 |

**Value: met** at every resolution with WALE (-1.0 to -3.4%, criterion 5%), and the model's share of
the peak falls as it should with resolution (63 -> 51 -> 41%) while the no-model run converges onto
the reference from above (+5.9, +10.2, -2.7%: the 96^2 no-model overshoot is the pile-up at the
cut-off before it is resolved). **Time: not met.** Every run peaks at t = 8.3-8.5, 5-7% before the
reference's 8.93, and the offset does not move with resolution (64 -> 128) or with the model. That
rules out the small scales and the closure; what is left is the time integration (dt 0.02 at
CFL 0.4 is far inside the RK3 range, but the O(dt) Rhie-Chow stage damping is not zero), the
reference's own time axis, or the in-plane FV truncation of the large scales, which second-order
central differencing does not remove by 128^2. The dt test (64^2 x 64 at dt 0.01 and 0.04) is
running; the WALE curves also show the section-52 shoulder before the peak (t 4.5-7), shrinking
with resolution.

The dt test settled it and then the reference did: at 64^2 x 64 the peak sits at t = 8.47 / 8.40 /
8.38 for dt 0.01 / 0.02 / 0.04 -- dt-independent. And the reference is not what the plan said. The
"0.012299 at t = 8.93" of `les_findings.md` is `results/tgv_diag_re800_88.npz`, the in-house
spectral-element run at **Re = 800** (nu = 0.00125 in the file; its 2 nu Omega at t = 0 is 0.000938,
which is Re 800 with Omega = (1/2) int |omega|^2 -- at Re 1600 the initial dissipation is
0.000469, which is what every run above starts from). So the Re 1600 matrix was scored against a
Re 800 curve: a lower Reynolds number peaks EARLIER and LOWER (Brachet's series), which is exactly
the pattern above read backwards -- our Re 1600 peaks were "late" by the DNS's own 6% and "high".
The published Re 1600 peak (van Rees et al. 2011, the high-order workshop) is about 0.0128 at t of
about 9, from memory and not on disk, so it is not used as a criterion here. Instead the matrix is
rerun at Re 800 against the reference we actually have, whose full history is on disk:

**Re 800 matrix against the in-house SEM DNS (88^3-equivalent), full history, T = 15, dt 0.02
(`figures/utgv800_v1.png`):** DNS 2 nu Omega peak 0.01230 at t = 8.93.

| run | -dE/dt peak | at t (parabolic) | value | time | model + numerical share | <nu_t>/nu | rms error of the -dE/dt curve, t < 12 |
|---|---|---|---|---|---|---|---|
| 64^2 x 64, no model | 0.01238 | 8.39 | +0.6% | -6.1% | 4.6% | 0 | 13.5% |
| 64^2 x 64, WALE | 0.01132 | 8.73 | -7.9% | -2.3% | 45% | 0.60 | 23.4% |
| 96^2 x 96, no model | 0.01195 | 8.25 | -2.9% | -7.6% | 2.6% | 0 | 8.2% |
| 96^2 x 96, WALE | 0.01171 | 8.32 | -4.8% | -6.8% | 31% | 0.30 | 13.9% |
| 128^2 x 128, no model | 0.01144 | 8.42 | -7.0% | -5.7% | 1.2% | 0 | **5.9%** |
| 128^2 x 128, WALE | 0.01166 | 8.39 | -5.2% | -6.0% | 21% | 0.17 | 10.0% |

Same picture at the right Reynolds number, so the early peak is the solver's, not the reference's.
The curves say where: the 128^2 no-model run follows the DNS to within 1-2% up to t = 7.5, then the
DNS keeps rising to 8.93 while ours turns over at 8.4 and sits 10-20% below the DNS from t = 9 on.
That is the small-scale end of the cascade: the dissipation that the DNS still gains between 8.4 and
8.9 comes from scales the second-order plane discretisation has already truncated, and the
post-peak deficit is energy parked at the cut-off. Doubling 64 -> 128 moved the peak value 7% and the
curve error 13.5 -> 5.9% but not the peak time -- second-order central converges slowly in exactly
this quantity. WALE at every resolution over-dissipates the LAMINAR phase (a shoulder at t = 4.5-6,
ratio to the DNS 1.1-1.3 there; 63 -> 21% model share at the peak as the grid refines) and then
under-dissipates after it; the implicit run is the better LES at 96^2 and 128^2 on this case.
Variants to locate the timing:

| 64^2 x 64 unless stated, Re 800, no model | -dE/dt peak | at t | E(6)/V (DNS 0.10606) |
|---|---|---|---|
| baseline | 0.01238 | 8.39 | 0.10442 |
| **rotated initial condition** (x -> y -> z -> x: the span carries the w-like component) | 0.01081 | **7.50** | 0.10590 |
| no spanwise dealiasing | 0.00853 | 5.30, then negative dissipation | 0.11127 |
| n_inner 1 | 0.01238 | 8.39 | 0.10442 |
| 32^2 x 32 | 0.01118 | 8.90 | 0.10556 |
| 48^2 x 96 (span finer than plane) | 0.01129 | 8.12 | 0.10359 |
| 96^2 x 48 (plane finer than span) | 0.01251 | 8.34 | 0.10560 |

Three things. (1) **The numerics are anisotropic in a way that moves the transition.** The
Taylor-Green problem is invariant under the cyclic rotation of the axes; the solver is not, and the
rotated run peaks 0.9 time units earlier and 13% lower than the baseline. In the standard
orientation w = 0 and every z-dependence is the single exact Fourier mode cos z, so the spectral
direction is nearly free of error; rotated, the in-plane second-order central scheme carries more
of the dynamics and its dispersion error is the symmetry-breaking perturbation that brings the
breakdown forward. (2) The peak time does not converge monotonically with resolution (8.90, 8.39,
8.25, 8.42 for 32, 64, 96, 128): the coarse grid's own dissipation delays the breakdown, the finer
grids' dispersion advances it; the 3% time window is not reachable with this plane discretisation
at these resolutions. (3) The z-dealiasing is essential and correct: without it the run is
garbage by t = 5 (negative dissipation). n_inner is irrelevant on an orthogonal mesh, as it should
be. The energy balance of the rotated field is as clean as the standard one (Re 100 check below),
so this is discretisation error, not a bug in the w-equation.

**V1 verdict.** Peak value within 5% at 96^2 and 128^2 with WALE (-4.8, -5.2%) and the 128^2 implicit
run within 6% rms over the whole curve to t = 12; peak time 6% early at every resolution, model and
orientation-dependent -- **not met as posed**, and the reason is the second-order in-plane
convection (its dispersion error at the breakdown scales), which the model cannot fix and which
resolution does not remove at a useful rate. Two ways to close it: a fourth-order in-plane
reconstruction (a substantial change to the FV core), or re-pose the criterion on the curve rms and
the peak value, which is what the LES literature on this case usually reports. Recorded as open;
the implicit-LES result (5.9% rms at 128^2 x 128) is the number to carry forward. WALE's laminar
over-dissipation (the t = 4.5-6 shoulder) is a second, separate finding: the model needs the
laminar-phase guard (sigma-model behaviour, or a dynamic procedure) before V2-style cases where
transition matters.




## 59. The A100 package and the run-time estimate (2026-09-25)

`tools/a100/`: Dockerfile (nvidia/cuda 12.4 runtime + numpy, scipy, pyamg, cupy-cuda12x),
requirements, `make_bundle.sh` (solver, drivers, meshes, reference data -> one 9.5 MB tarball),
`profile_step.py` (whole-step and per-phase timing on whatever GPU it finds, plus the run-time table
for the target cases), README with the install, the correctness check, the runs and the memory
budget. Nothing in the code is architecture-specific: the one raw CUDA kernel compiles at first use.
`run_uchannel25.py` and `run_ucylinder25.py` gained `--device gpu` (and `--sgs` for the cylinder);
both smoke-tested on the GB10.

Measured on the GB10: box 128^2 x 64 77 ms per 1e5 cell-modes, 256^2-384^2 x 64 108-112; the
**fine butterfly (27968 quads) x 64 modes with WALE and three non-orthogonal passes 256** -- a real
mesh costs 2.3x the box (16 pressure iterations instead of 9, the WALE term). Launch floor 56 ms.
Estimates for V3 (cylinder Re 3900, dt 0.002, 200 D/U = 1e5 steps): 66 h on the GB10 on the
butterfly-fine plane; on an A100-80GB between 10 h (bandwidth ratio 7.5 against the GB10's real
273 GB/s) and 20 h (the 3.7 the device properties imply, which double-count the GB10's bus). A
1e5-cell plane, which Re 3900 wants (wall cell ~0.002 D), is 3.5x that: 35-70 h on the A100. The
package is `upict25_a100.tar.gz` at the repo root.

Re_tau 395 case prepared for it (`run_uchannel25.py --re-tau 395`, MKM 1999 reference in
`reference/mkm_chan395/`, initial field = the 180 DNS field with the mean shifted to MKM's, minimal
box 96 x 160 x 128, dt 0.001; notebook `tools/a100/A100_channel_re395.ipynb`). Measured on the GB10 at
that size: 2.1-2.8 s/step, i.e. 18-23 h for the 30-time-unit run -- the real-mesh rate (~230 ms per
1e5 cell-modes), not the box rate the first estimate used; 3-6 h expected on an A100. Committed and
pushed (`8f20e9c`, branch `unstructured`).

**L4 step 3, launch count and reductions (2026-09-25).** Prompted by the first A100 profile
(A100-SXM4-40GB, cloud host): 47.6 ms per 1e5 cell-modes at 384^2 x 64 -- only 2.6x the GB10 for 5.7x
the bandwidth -- behind a 237 ms per-step launch floor (four times the GB10's; ~700 kernel
launches per step from Python at ~300 us each on that host). Three changes, no change of answers:
(1) `src/ucuda.py` `spmm_shift` and `jacobi_shift`: the shifted family matvec A X + s D X and the
whole damped-Jacobi sweep as one raw kernel each (were ~6 launches each); (2) the momentum
equations for u, v, w solved as one block of 6 n_k columns when their operators coincide (channel,
box) -- one solve, one convergence test, a third of the launches; (3) the column reductions of the
PCG (three per iteration along the slow axis of the row-major block) as cuBLAS matrix-vector
products, and the convergence test every second iteration. CPU path unchanged (Taylor-Green E to 9
digits, channel+WALE to 6, laminar-channel tests pass); GB10 energies and divergence bitwise equal
to before.

| GB10, 3D TGV | before | after | per 1e5 cell-modes |
|---|---|---|---|
| 64^2 x 32 | 77 ms | 47 | 67 |
| 128^2 x 64 | 480 | 329-340 | 61-63 |
| 256^2 x 64 | 2,720 | 1,428-1,587 | 66-73 |
| 384^2 x 64 | 6,762 | 3,275 | 67 |
| fine butterfly 27968 x 64, WALE, n_nonorth 3 | 2,366 | 1,290 | 140 |

Launch floor 56 -> 46 ms. **The G4 speed criterion (100 ms per step per 1e5 cell-modes) is now met
at every size on the GB10**, and the real mesh at 140 is within 1.4x. The step's bulk is now the
nonlinear term (248 of ~640 ms per stage at 384^2: the face gathers on the 3/2-padded planes and
the pad/truncate FFTs), then the pressure solve (276, 10 iterations); the momentum solves are 55 x 6.
On the A100 the same changes cut the floor from 237 ms to a third and the large-case rate toward
25 ms per 1e5 cell-modes (to be measured there); the run-time table of `profile_step.py` remains a
box-rate estimate -- use `--mesh` for a real mesh.

## 60. An external review of the LES capability, assessed against the record (2026-09-25)

The review (five points: convective scheme and TVD, pressure-gradient consistency on skewed meshes,
WALE over Smagorinsky, BDF2 as the time scheme, anisotropic filter width) was written against the
structured code and the plan as first drafted. Against the measurements: (1) the 2.5D convection is
explicit central + skewness on the divergence-form face sum, skew-symmetric to round-off on quads
(section 48), 0.04%/turnover for the full RK3 step (50); there is no TVD path in it -- already the
state. (2) The dual Green-Gauss gradient's inconsistency is confined to the momentum pressure term
and measured harmless (16-24); the triangle failure in turbulence is the non-bipartite pressure mode
and a momentum imbalance (55), which iterating the gradient does not touch -- quads for LES is the
answer, and is in the plan. (3) WALE is the model in use and its y^3 exponent is verified (52); the
review misses its measured laminar-phase over-dissipation (58) -- the sigma-model port is the
actionable item, added to the plan. (4) BDF2 is not the LES scheme: RK3 with per-stage projection,
for the measured 100x difference in energy loss (50). (5) max(dx, dy, dz) as the filter width would
INCREASE nu_t and damp more, the opposite of the review's aim; V^(1/3) reproduced the channel within
1% (54). The review's one useful recommendation is recorded; the design rules the runs established
are now in `piso_unstructured_formulation.md` (section "Design rules the LES runs established").

## 61. The Re_tau 395 channel on the A100: a CFL divergence, and the adaptive step (2026-09-25)

First run of `run_uchannel25.py --re-tau 395` on an A100-SXM4-40GB (96 x 160 x 128, WALE, dt 0.001,
645 ms/step): healthy for ten time units -- u_tau 0.93-1.01, U_b 17.65 -> 17.90 settling toward
MKM's 17.54 from above, <nu_t>/nu 0.10-0.12, pressure two-colour 0.00% -- then **diverged at step
10178 (t = 10.18)** with the per-component CFL reading 0.68-0.70 throughout. In the face-flux Courant
measure (sum over faces of the outgoing flux over the cell volume, plus |w|/dz, times dt) that is
about 1.2: the Re 180 channel ran at 0.5-0.9 in the same measure and never failed, so the explicit
central RK3 stage with the explicit WALE term has its practical limit near 1 here, below the 1.73
of the linear analysis, and a rare local excursion crossed it. The driver's post-processing also
crashed on the NaN statistics (fixed: a divergence now ends the run cleanly, the last valid
checkpoint stays).

Adaptive step (`PISO25.cfl_max`, `--cfl-max`): before each step the face-flux Courant number per
unit dt is measured (one sparse product), and dt is set to the largest of dt0 / 2^m (m <= 3) that
keeps it below `cfl_max`, shrinking at once and growing one level per step; RK3 has no history so a
change costs nothing, and the per-dt momentum and pressure families are cached (a level costs one
extra AMG setup, 0.2 s). On the Re 180 channel the measure reads 0.52 at dt 0.002 (the 0.31 of the
old per-component reading), and `--cfl-max 0.8` leaves that run alone; on the 395 case it will
halve dt to 0.0005 (C ~ 0.6) for most of the run: 60k steps, ~11 h on the A100. The notebook
launches with `--cfl-max 0.8` and resumes the diverged run from its t = 10 checkpoint.

**Third continuation (200k steps total, 2026-09-25; interrupted once by a Spark reboot and resumed
from the 48k checkpoint):** mean return -53.2 -> -52.1 over the last 32k steps, the final rollout's
episodes -43.8 to -51.2 -- **20-30% better than doing nothing (-64) with the exploration noise still
on** (std 0.22). The policy is steady suction on the upper-surface jet at -0.27 (0.14 U_inf), the
nose jet off, a trace of blowing on the lower jet. The learning has flattened (0.03 per update);
what remains to measure is the deterministic policy without its noise, which is the controlled
return proper (`eval_naca_policy.py`, running). Against the estimate of section 56 (-35 to -45 for
a good policy) this sits at the weak end; an action penalty or entropy schedule would take it
further, both outside HydroGym's task definition.

**Deterministic evaluation of the 200k-step policy** (`eval_naca_policy.py`, three snapshot phases,
each with and without the jets; `figures/naca40_gust_control_eval.png`, `results/naca/eval_cont3.npz`):

| | return | mean |C_L - C_L0| during the gust | after | peak C_L | action (jet 1, 2, 3) |
|---|---|---|---|---|---|
| jets off | -64.24 +- 0.05 | 0.71 | 0.10 | 2.97 | 0 |
| learned policy, no exploration noise | **-30.33 +- 0.15** | **0.21** | 0.09 | 1.87 | -0.275, -0.006, +0.036 |

**A 53% better return and a 70% smaller lift excursion during the gust**, and -- the field
episodes (`eval_naca_fields.py`, `figures/naca40_gust_control_fields.png`) corrected an earlier
reading of the mean action -- **the policy is gust-reactive, not a constant**: from the probe's
velocity it ramps the upper-surface jet from -0.3 before the gust to -0.75/-0.80 (suction at
0.4 U_inf) at the gust peak and the lower jet from +0.07 to +0.6 (blowing), holding C_L at 0.95-1.05
against the 1.9-2.2 of the uncontrolled airfoil while the free stream doubles, then relaxes to
(-0.17, +0.03, -0.09) after it; C_D at the peak 1.79 against 2.50. The upper-surface suction thins
the separated shear layer over the suction side and the lower blowing weakens the trailing-edge
vortex (t = 21.6 panels). The noise in training cost the 22 points between -52 and -30. This
lands at the good end of the -35 to -45 estimated above. The after-gust shedding penalty is
untouched (0.09 vs 0.10): with no actuation cost in the task the policy keeps a mild suction on
afterwards but does not attempt to suppress the shedding.

## 62. The fluidic pinball on HydroGym's mesh, against HydroGym's Firedrake (2026-09-25/26)

Three cylinders of diameter 1 at the corners of an equilateral triangle of side 1.5 D, apex upstream
(centres (0, 0), (1.30, +-0.75)); domain -6..20 x +-6; inlet u = 1, lateral symmetry, outlet p = 0.
HydroGym ships two all-triangle meshes as LFS pointers; the real files were fetched from GitHub's LFS
store and converted to msh2 (`meshes/pinball_medium.msh` 50864 cells, `pinball_fine.msh` 109258;
96 wall faces per cylinder, wall cell h/D 0.030 on both -- "fine" refines only the far wake;
`figures/pinball_meshes.png`). Drivers: `run_upinball.py` (ours, per-cylinder wall-flux forces),
`tools/hydrogym_cmp/hg_pinball.py` (their Firedrake P2-P1, Newton steady solve then a transient;
`--impulsive` starts from our initial condition instead). Same mesh in both.

**Re 30.** Firedrake's Newton solve from a symmetric guess gives the symmetric steady state:
C_D 1.488 / 1.571 / 1.571 (total 4.630), C_L 0 / +0.526 / -0.526, and its transient from that
state plus a 1e-3 random perturbation stays there for 150 time units. Our run from an impulsive
start (uniform flow plus a y-even blob) agrees with that near field but develops a growing
oscillation of the far wake from t ~ 90 (`figures/pinball_re30_history.png`): frequency 0.083,
rear-pair lift amplitude 0.08 by t = 150, total drag creeping 4.68 -> 4.81; the far-field
vorticity meanders with a 10 D wavelength where Firedrake's is straight
(`figures/pinball_re30_fields.png`). The pressure is smooth (face high-pass share 0.011, ten times
below the quad channel's): not the triangle pressure mode. Three runs then located it:

| Re 30 discriminators | result |
|---|---|
| ours, RK3, dt 0.005 | same oscillation: f 0.081, amplitudes 0.037 / 0.062 / 0.063, drag 4.79 |
| ours, Re 15 | steady to 1e-7 in dC_D/dt; C_L amplitudes 0.001: no numerical instability |
| **Firedrake, from OUR impulsive start** | **the same oscillation**: f 0.081, amplitudes 0.036 / 0.060 / 0.060, drag 4.69 |

`figures/pinball_re30_forces_compare.png`: from the same start the two solvers' rear-pair lift
histories lie on top of each other -- same phase, amplitude and period from t = 60 to 120 -- and
in the window t 90-120 the drag is 1.500 / 1.602 / 1.606 (Firedrake) against 1.517 / 1.620 / 1.624
(ours), +1.1% in total, a uniform 0.017 per cylinder (the one-sided wall-flux stress on 96 faces,
the same offset direction as the single cylinder, section 38). Firedrake's rerun to t = 150 with the field
saved (window 90-150: C_D 1.503 / 1.613 / 1.612, total 4.728; amplitudes 0.045 / 0.078 / 0.078; f
0.082 -- against ours 4.780, 0.046 / 0.079 / 0.079, 0.083) gives the matched field figure
`figures/pinball_re30_fields_matched.png`: at the same instant the two far wakes meander in phase,
crest for crest, with peak vorticity 11.07 against 11.28 and identical pressure fields
(-0.526..+1.037 against -0.526..+1.035). The symmetric steady state is one
solution; a far-wake oscillation that a finite disturbance excites in this confined domain is
another, and both solvers find the second when started the same way. **The reference number for
the pinball at Re 30 depends on how the reference was started**, which matters before a controller
is trained against the steady value; the published bifurcation sequence for this geometry (pitchfork
near Re 18, Hopf near 68) is for a different confinement and was not used as a criterion.

**Re 100.** Firedrake from the Newton state + 1e-3 noise held the (unstable) symmetric branch for the
whole 200 time units (total drag 3.53, C_L +-0.090, amplitudes 0.005 only at the end); from the
impulsive start it leaves it within 40 time units. Ours from the impulsive start: total drag 3.72
with the rear pair asymmetric (1.41 / 1.32) at t 150. Matched-start comparison:

| Re 100, both from the impulsive start | Firedrake P2-P1 (t 150-250) | ours (t 120-200) | difference |
|---|---|---|---|
| C_D front / top / bottom | 0.9809 / 1.3974 / 1.3068 | 0.9925 / 1.4106 / 1.3197 | +1.2 / +0.9 / +1.0% |
| total C_D | 3.6852 | 3.7228 | +1.0% |
| C_L mean front / top / bottom | +0.0064 / +0.1121 / -0.0658 | +0.0065 / +0.1078 / -0.0639 | -4% / -3% on the rear pair |
| C_L amplitude front / top / bottom | 0.0032 / 0.0170 / 0.0282 | 0.0035 / 0.0172 / 0.0290 | +1 / +3% |
| St (total lift) | 0.1112 | 0.1110 | -0.2% |

`figures/pinball_re100_fields_matched.png`: the same asymmetric shedding state -- top rear cylinder
carrying the higher drag, the staggered street of the far wake, the pressure around the bodies --
with the near-wake vortex pattern the same and the street's phase differing (the two snapshots are
at t = 250 and t = 200). Drag is again a uniform 1% above Firedrake's (the wall-flux stress), lift
amplitudes within 3%, Strouhal within 0.2%: the same margins as the single cylinder on their mesh
(section 45). The pinball on HydroGym's own mesh is therefore validated against their solver at both
Reynolds numbers, once both are started the same way -- and at Re 30 that condition is the finding.
The asymmetric mean lift at Re 100 (+0.11 top, -0.065 bottom, net +0.05) is the pitchfork-broken
state of the literature, reached by both solvers from the y-even impulsive start through numerical
asymmetry; the mirror image is the other attractor.


## 63. The Re_τ 395 channel on the A100, against MKM 1999 (2026-09-26)

The run §61 prepared finished: `run_uchannel25.py --re-tau 395 --nx 96 --ny 160 --nz 128 --cfl-max 0.8
--device gpu`, WALE, constant pressure gradient, minimal box L_x = π, L_z = 0.34π (L_x⁺ 1241, L_z⁺ 422),
Δx⁺ 12.9, Δy⁺ 1.0 at the wall and 9.4 at the centre, Δz⁺ 3.3, to t = 30 with statistics over t = 10–30
(46,582 samples, about ten flow-through times). Initial field: the Re_τ 180 FOSLS DNS field with its mean
shifted to the MKM 395 mean. Outputs: `results/uchan395/uchan395_96x160x128_wale_cpg/` (statistics,
final field, checkpoint, history, log); Google Drive `PICT-Python_runs/uchan395_96x160x128_wale_cpg`.
Figures: `figures/uchannel_re395_profiles.png` (`plot_utility/plot_uchannel_re395.py`),
`figures/uchannel_re395_nearwall_yp12.png`, `_planes.png`, `_spectra.png`
(`plot_utility/plot_uchannel_re395_nearwall.py`).

**Cost and stepping.** 68,656 steps at 380–440 ms on the A100 (about 7.8 h of GPU time; the §59
estimate was 10.7 h at fixed dt 0.001, which C ≈ 1.2 made impossible — §61). The adaptive step sat at
dt 0.0005 for 77% of the reports and 0.00025 for 23%, face-flux Courant number never above 0.80; no
divergence, no two-colour pressure mode (0.00% of p_rms on the quads throughout). Three Colab session
losses were absorbed by the Drive checkpoints (restarts at t = 4.64 and twice at t = 6.90); the
statistics window is untouched.

**Statistics against MKM 1999 (Re_τ 392.24):**

| | LES | MKM DNS | |
|---|---|---|---|
| u_τ (window mean), Re_τ | 0.9958, 393.4 | 1, 392.2 | +0.3% |
| U⁺ in 30 < y⁺ < 120 | | | mean +0.10, max 0.17 u_τ (+0.6%) |
| u'⁺ peak (at y⁺) | 2.687 (14.4) | 2.739 (14.2) | −1.9% |
| v'⁺ max | 0.988 | 0.997 | −1.0% |
| w'⁺ max | 1.243 | 1.289 | −3.6% |
| −⟨u'v'⟩⁺ max | 0.830 | 0.837 | −0.9% |
| U_c⁺ | 20.99 | 20.13 | +4.3% |
| ⟨ν_t⟩/ν over the window | 0.118 | | |

The V2 criteria (U⁺ within 3% in the log region, u_rms peak within 5%, Re_τ within 2%, two-colour
mode < 1%) are all met at more than twice the validation Reynolds number, on the same quad topology
and the same solver settings as the Re_τ 180 run, with 20% of the cells per wall unit in x
(Δx⁺ 12.9 against 23.6) and the same y⁺ 1 wall cell.

**Where it departs, and why.** Above y⁺ ≈ 150 the LES core runs high (U_c⁺ +4.3%) and u' runs low
(−15% at y⁺ 200, −25% at y⁺ 300); v' and w' stay within 5% to the centreline and the shear stress
follows the DNS line to the centre. This is the box, not the model: Jiménez & Moin (1991) and Flores &
Jiménez (2010, *Phys. Fluids* 22, 071704) show the minimal unit reproduces the full-channel statistics
only below y ≈ 0.3 L_z — here 0.3 × 1.07h = 0.32h, i.e. y⁺ ≈ 125, exactly where the departure begins —
and above it the flow is a single box-filling structure with a higher core velocity and starved
streamwise fluctuations. The spanwise spectrum at y⁺ 153 confirms it: all of k_z E_uu sits in the
first box mode (λ_z⁺ 422). The Re_τ 180 run had L_z⁺ 192, 0.3 L_z = y⁺ 58, and showed the same
signature in a milder form (§54: U_c⁺ high, u' low in the core). A full MKM box (2π × π, four times the
cells and eight times the cost) is the check, if the outer layer ever becomes a target; the near-wall
comparison, which is what the LES is for, does not need it.

**The field.** MKM publish statistics only, so the instantaneous field is checked the way the DNS
papers characterise it. On the wall-parallel plane y⁺ 11.4 (`_nearwall_yp12.png`): four low-speed
streaks across L_z⁺ 422 — spacing ≈ 100 wall units — meandering over the whole 1241⁺ box length,
u' rms 2.62 (MKM 2.68 at this y⁺), p' rms 2.49 u_τ², streamwise vorticity ν ω_x/u_τ² rms 0.141 in
streak-flanking pairs. The premultiplied spanwise spectrum of u at y⁺ 11 (one field, x-averaged) puts
its two largest bins at λ_z⁺ 105 and 211 — the box quantises λ_z⁺ to 422/n, so the DNS spacing of
100 falls in the 105 bin and the 105/211 pair brackets it; at y⁺ 50 the peak moves to 141, the
outward growth of the spacing that Kim, Moin & Moser (1987) report. The single field's rms profile
(walls folded) lies on the 20-turnover statistics to within the sampling noise, so t = 30 is a
representative state and not a transient. Cross planes (`_planes.png`) show the wall-normal ejections
reaching y⁺ 300 in the y–z cut and the inclined shear layers over the low-speed regions in the x–y cut.

**Verdict.** T19 done: first LES beyond the validation Reynolds number, all V2 criteria met on the
near-wall statistics, the outer-layer departure identified as the minimal-box effect with the
literature's y ≈ 0.3 L_z boundary reproduced. The A100 notebook, Drive checkpointing and adaptive step
(§59, §61) carried the 7.8 h run across three session losses without intervention.

### 63a. Near-wall u, p and ω_x against DNS at Re_τ 395 (2026-09-26)

`plot_utility/plot_uchannel_re395_nearwall_dns.py`. There is no DNS *field* at Re_τ 395 — the Drive DNS
folder holds only the Re_τ 180 FOSLS run and MKM 1999 publish statistics — so the comparison is made
twice. `figures/uchannel_re395_nearwall_vs_dns.png` puts the LES plane at y⁺ 10 beside the FOSLS Re_τ 180
plane at the same y⁺ and the same wall-unit scale (structure at this height scales in wall units): the same
streak width and spacing, the same size of pressure patch and of the streamwise-vorticity pairs that flank
the streaks; the LES box is 2.2 × 2.2 the DNS box in wall units, so it holds four streaks to the DNS's two.
`figures/uchannel_re395_nearwall_stats.png` compares the same three quantities with the MKM data at the
same Re, using the parts of the MKM tarball (`chandata.tar.gz`, UT Austin) now kept in
`reference/mkm_chan395/`: `chan395.velp` (p variance), `chan395.vortvar` (vorticity variances),
`chan395.zspec.10/.20` (spanwise spectra of u, v, w, p at y⁺ 9.5 and 19.7) and `chan395.zcorr.10/.20`
(two-point correlations). MKM's E(k_z) is one-sided with Σ_k E = variance at Δk = 2 (L_z = π); ours is
the one-sided per-mode power at Δk = 2π/L_z; both are plotted as k_z E/Δk.

| at y⁺ ≈ 10 | LES (one field) | LES (t = 10–30 statistics) | MKM 395 DNS | FOSLS 180 field |
|---|---|---|---|---|
| u'⁺ rms | 2.54 (plane), 2.45 (folded) | 2.56 | 2.59 | 2.65 |
| p'⁺ rms | 2.47 (plane), 2.13 (folded) | 2.19 | 2.20 | 1.52 |
| ω_x ν/u_τ² rms | 0.135 | — | 0.150 | 0.139 |
| R_uu minimum | Δz⁺ 49, −0.24 | | Δz⁺ 58, −0.14 | |
| k_z E_uu peak | 105/141 bins | | ≈ 100 | 96 (box quantised) |

**u.** The premultiplied spanwise spectra at y⁺ 10 and 20 lie on the MKM curves point for point from
λ_z⁺ 20 to 200, including the peak at λ_z⁺ ≈ 100 and the y⁺ 20 curve sitting above the y⁺ 10 curve at
long wavelengths; only the two box modes (211, 422) carry less than the DNS's continuum there. The
correlation minimum, the classic streak-spacing measure, is at Δz⁺ 49 against 58 (spacing 98 vs 116), and
deeper (−0.24 vs −0.14): the minimal box with four streaks across 422⁺ makes them slightly closer and
more regular than the wide-box DNS. u' rms from the statistics is within 1.5% at y⁺ 10 and within 2% at
the peak. The single snapshot is 4% low there — a normal one-field fluctuation, not a bias.

**p.** The statistics reproduce the MKM p_rms profile to 1% at the wall and at y⁺ 10, 3% at the y⁺ 30
peak and 4% at y⁺ 50; the near-wall pressure spectrum matches from λ_z⁺ 20 to 150. Above λ_z⁺ 200 the
LES has only the two box modes and they hold about 60% of the DNS's premultiplied energy, and R_pp turns
negative at Δz⁺ 175 where the DNS's stays positive to 600: the near-wall pressure carries the footprint of
the outer flow, and the minimal box truncates that footprint — the same box effect as the core velocity
in §63, seen in the one near-wall quantity that is not local. The single-plane p' rms (2.47 on one wall,
1.78 on the other, 2.13 folded) shows how much of p' at this height is a few box-filling patches.

**ω_x.** The single-field rms profile follows the MKM shape — minimum at y⁺ 5, maximum at y⁺ 18 — at
8–10% below it through the buffer layer (0.137 vs 0.150 at y⁺ 10; 0.152 vs 0.164 at the maximum) and
24% below at the wall (0.19 vs 0.25). The wall value is ∂w/∂y in the first cell and ∂w/∂x at Δx⁺ 12.9,
the finest streamwise vortices are the scales the WALE term acts on; 10% low on ω_x rms with u', p' and
their spectra on the DNS is what a wall-resolved LES at this Δx⁺ should show.

**Verdict.** At the same Reynolds number, the near-wall u and p statistics and spectra of the LES are on
the MKM DNS to a few percent from the wall to y⁺ 100 and from λ_z⁺ 20 to 200; streak spacing 98 against
116; ω_x rms 10% low in the buffer layer. What departs is the box, not the discretisation: the two
longest spanwise modes and the pressure's outer footprint.
