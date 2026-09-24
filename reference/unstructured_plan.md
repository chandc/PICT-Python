# Unstructured collocated PISO — implementation plan

Branch `unstructured`. Follows *Migrating Curvilinear PISO Codes to Unstructured Collocated
Layouts*, with four decisions taken where the guide conflicted with HydroGym's source or was
internally inconsistent. Those conflicts and their resolutions are recorded here because each
one changes what "correct" means for the verification suite.

## Why we pivoted here

The structured multiblock path cannot express a Cartesian tiling wrapped around an O-ring. For
the north row to share one y-distribution, the east and west transition faces are forced onto
identical node counts, and likewise north/south; the square perimeter then receives `2p+2q`
nodes for `2p+2q-4` distinct positions. Four corners are over-supplied **by construction**, and
shifting the ownership converts the duplicates into gaps. Measured, this put two cells at one
physical point holding velocities 0.20 apart and flooded the O-ring with spurious vorticity:
`|w|` median **1.80** where the butterfly gives **0.00065**.

Unstructured removes the question rather than working around it — no blocks, no face
orientations, no ghost padding, no seam ownership. And it lets us read HydroGym's own
`medium.msh`, so conformance stops being approximated with stretching parameters.

## Decisions

| question | decision | consequence |
|---|---|---|
| jet actuator | **Firedrake spec**: cosine, `omega = pi/18` (10 deg), centres `+-pi/2`, **not** ZNMF | guide's section 7.3 ZNMF mass-conservation test does not apply and is dropped |
| Rhie–Chow | **time-step independent** (Choi / Yu–Kim): carry the previous face flux | guide's printed formula is spatial-only despite its heading, and would fail its own four-decade dt test |
| Re=100 validation | **HydroGym's own numbers on `medium.msh`** | guide's St 0.165 / C_D 1.34–1.35 are *unconfined* values; this mesh is beta = 0.10, where C_D is legitimately higher |
| discrete adjoint | **design for it now** | operators stay explicit sparse matrices, state updates functional; no matrix-free shortcuts in the operator layer |

## Phases

**P0 — mesh and connectivity.** DONE (`830ab94`), `src/umesh.py`. Gmsh 2.2 reader, cell/face
topology, physical tags. On `medium.msh`: 17,258 cells, 26,002 faces, 230 boundary, audit clean;
cell area 199.2150 against an analytic 199.2146.

**P1 — face geometry and gradients.** Over-relaxed decomposition `S_f = E_f + T_f` with
`E_f = d (d.S)/(d.d)`. Least-squares gradient assembled ONCE as two sparse matrices (needed for
the adjoint anyway).
*Tests:* `grad(3x - 2y) == (3, -2)` to 1e-10. NOT the guide's `E_f + T_f == S_f`, which is an
algebraic identity and tests nothing; instead Laplacian of a linear field on a deliberately
skewed mesh.

**P2 — operators.** Divergence; Laplacian with implicit `E_f` and deferred `T_f`; convection.
*Tests:* `div(uniform) == 0`, `lap(linear) == 0`. These are the unstructured analogues of the
checks that caught the handedness and duplicated-seam defects, and they go in before any solve.

**P3 — pressure Poisson.** Assemble once; solve through the existing `src/linsolve.py`
unchanged (scipy / PETSc / AmgX all take a sparse matrix and an RHS).
*Tests:* manufactured solution, then MMS with an O(h^2) slope under refinement.

**P4 — Rhie–Chow and the PISO loop.** Time-step-independent form.
*Test:* the guide's checkerboard suppression across `dt = 1.0 ... 1e-4`.

**P5 — boundary conditions**, all four verified from HydroGym source:
inlet `DirichletBC(V, U_inf, INLET)`; outlet `DirichletBC(Q, 0, OUTLET)`; Freestream
`DirichletBC(V.sub(1), 0, FREESTREAM)` — y-component only, i.e. SYMMETRY, with the full-vector
version commented out in their source; cylinder no-slip.

**P6 — canonical validation, BEFORE the cylinder.** The cylinder has no reference we can
obtain: Firedrake is not installable here, the 1.0.0 sdist ships no tests or reference values,
and the paper's baseline tables are in Supplementary 3/5. Both of these DO have authoritative
references, so they validate the solver rather than merely exercising it.

  **P6a — plane Poiseuille channel.** Exact solution, so the error is known pointwise and the
  spatial order can be measured directly. Also the cheapest check that the pressure-velocity
  coupling is right: a wrong Rhie-Chow still produces a plausible parabola but the wrong
  flow rate for a given pressure gradient.

  **P6b — Ghia, Ghia & Shin (1982) lid-driven cavity at Re = 1000.** Published centreline
  profiles (u along x=0.5, v along y=0.5). The standard benchmark, and a hard one: the
  corner singularities and the secondary vortices punish a sloppy convection scheme.

Run BOTH at `perturb=0` and `perturb>0`. On an unperturbed right triangulation the mesh is
exactly orthogonal (measured cos(d,S) = 1.0000), so T_f vanishes and a broken deferred
correction passes. The skewed variant is what makes the test mean anything.

**P7 — Re = 100 cylinder**, on HydroGym's `medium.msh` with the Firedrake specs. By this point
the discretisation is validated against known answers, so a disagreement here is attributable to
the case setup or to blockage rather than to the solver.

**P8 — jets.** Profile already written and verified analytically (peak 36.0000, support 5.55%
of the circle, flux 2.000000 per unit control); port to face-based application.

**P9 — discrete adjoint.**

## Tollgates

Ordered so that each one can only fail for a reason the previous ones have already excluded.
Every gate runs at `perturb=0` AND `perturb>0`.

| # | gate | what it catches | reference |
|---|---|---|---|
| T1 | operator exactness | assembly, signs, skewness | grad/lap/conv of a linear field, to 1e-11. **PASSING** |
| T2 | **spatial** order, Poisson | discretisation order | MMS, slope 2. **PASSING** (Dirichlet 2.02, Neumann 2.09) |
| T3 | **spatial** order, Navier–Stokes | coupled discretisation | MMS with a manufactured `u, v, p`. With LSQ: 1.42 → 0.72, plateau. **With `grad_p=GGDualGradient`: 1.38 / 1.20** — 4–5× lower error, order holding but NOT second. Convective path is the open item. See `skew_unstructured_literature.md` §19 |
| T4 | **temporal** order | the time scheme alone | fix the mesh, refine `dt`, slope 2 for BDF2. **Found first order (0.85–1.00)**: the deferred non-orthogonal correction was lagged one time level. **PASSES with `n_inner=3`: 2.31 / 2.33 / 2.29** (n=128, three points before the spatial floor). Steady T5 bit-identical with and without the inner loop. `n_inner=2` measured sufficient (2.70 / 2.01) and is now the default. See `skew_unstructured_literature.md` §18 |
| T5 | Stokes | pressure–velocity coupling WITHOUT convection | manufactured Stokes solution. **VELOCITY PASSES on the uniform mesh: 1.98 / 1.99** over three levels, with the divergence-dual Green–Gauss pressure gradient (`grad_p = GGDualGradient`, opt-in; the LSQ default is order 0.02). **On QUAD meshes, uniform or clustered to β=2.0: velocity 1.94–2.00 AND pressure 1.92–2.05, checkerboard decaying at order 3.5–4** (§27) — the complete pass. On triangles: velocity passes uniform only (pressure order 1.00; irregular triangles sit on a diagnosed floor, §22–24). Hybrid quad-skin-on-triangle-core FAILS at the interface (§27). Mesh code now supports polygons (§26). See `skew_unstructured_literature.md` §16–27 |
| T6 | Poiseuille | flow rate for a given pressure gradient | exact solution; a wrong Rhie–Chow gives a plausible parabola at the wrong flow rate. **PASS (§47)**: `test_upoiseuille.py`, periodic x via `Mesh.make_periodic`, rates 2.01/2.00/2.00 on sin(πy); parabola second order with the half-cell offset |
| T7 | Ghia cavity Re=1000 | convection, corner singularities, secondary vortices | `src/ghia.py`, `U_RE1000` / `V_RE1000`. **Converging** (32→64 order 0.7–1.3 on triangles). Wall clustering β=1.5 gives core vorticity within 0.1% of Ghia. **On 8281 quads: u rms 0.0100, all extrema within 1.6%** — 2.6× better than the best triangle mesh at equal cell count (§31) |
| T8 | Orr–Sommerfeld Re=7500 | numerical DISSIPATION, temporal accuracy | growth `alpha*Im(c) = 0.00223497`, phase `0.249892`; `orr_sommerfeld.py` reproduces both to 5+ digits. **PASS (§47)**: `test_uorr_sommerfeld.py`, 48×100/200/400 quads at dt 0.05: growth −16.6% / −0.9% / +1.4%, phase −1.6% (temporal-limited, as the structured code); grows on every grid; 4 min for what took the structured code 80. Orders measured one variable at a time: **spatial 2nd order both directions; temporal was FIRST order (phase-speed error halving with dt) because the convection matrix used the lagged flux F^n — fixed by 2F^n−F^(n−1), now the default (`PISO.conv_flux_extrap`), after which the results are dt-independent.** Best: 96×400 growth +0.3%, phase −0.16% (§47). T9 numbers recorded earlier used the lagged flux |
| T9 | cylinder Re=100 | the target case | no obtainable reference — see above; T1–T8 are what make its output trustworthy. **Meshes ready**: HydroGym `medium.msh` regenerated exactly from `medium.geo` (17258 tri); structured butterfly O-grid `cylinder_butterfly.msh` (6992 quads, all-quad, same 112 wall edges) — matches the triangle mesh on the Stokes MMS with 2.5× fewer cells (§30). Wake refinement band not yet reproduced in the butterfly. Mesh code supports polygons; Gmsh reader accepts quads. **Both meshes done (T=150, St by zero-crossing): butterfly 6992 quads St 0.1750 / Cd 1.453 / Cl amp 0.346; HydroGym 17258 tris St 0.1775 / Cd 1.476 / Cl amp 0.376; Cd_p agrees to 0.2%** — 5–8% above unconfined literature, consistent with β=0.10 blockage; tri/quad differences are in the shear terms (§35–36). **Steady branch vs HydroGym's unit test (C_D 1.2840, same mesh/BCs): ours 1.2702 on their mesh; butterfly 1.2442/1.2627/1.2744 coarse/medium/fine, approaching 1.284 from below, extrapolations 1.278–1.295 bracket it** (§37). **HydroGym run ourselves (Spark, §38): P2-P1 medium St 0.1791 / Cd 1.4862 / Cl amp 0.3582; steady converged 1.2861. Ours on their mesh: St −0.9%, Cd −0.7%, Cl amp +4.8%; butterfly −2 to −3%.** **With wall-flux force (§39): butterfly Cd 1.4866 (+0.03%), Cl amp −1.8%, St −2.3%; wake-refined butterfly 13952 cells Cd +0.2%, Cl amp +1.8%, St −2.5%; tris Cd +0.9%, Cl amp +6%, St −0.9%.** Open: butterfly St deficit (near-body O-grid or time scheme), tri Cl amplitude. Tri vorticity speckle = two-colour mode, 3x the common 8e-3 floor at cell level, gone under vertex averaging; RC damping >1 unstable, Neumann-extrapolation −18% only (§40). **Time-converged (2F^n−F^(n−1) flux, §47): butterfly St 0.1755 (−2.0%), Cd 1.479 (−0.45%), Cl amp 0.333 (−7.1%) — the earlier lift agreement was the first-order flux lag cancelling a spatial deficit; fine butterfly and tri reruns done (§47): **fine butterfly 27968 St 0.1782 (−0.5%) / Cd 1.4878 (+0.1%) / Cl amp 0.3520 (−1.7%); HydroGym's tris with our solver St 0.1780 (−0.6%) / Cd 1.4927 (+0.4%) / Cl amp 0.3606 (+0.7%) — T9 closed at the 1% level; the coarse butterfly's deficits were near-body resolution (wall cell 0.024 vs 0.012). Fine tris 25376 (wall 0.009, matched to the fine butterfly): St 0.1785 (−0.3%) / Cd 1.4851 (−0.07%) / Cl amp 0.3611 (+0.8%).** **RL on HydroGym's own env (§43–44): shipped jets → constant suction, −30.6% in 2 episodes (reward loophole); ZNMF jets → nothing in 100k steps.** |
| T10 | NACA0012, Re=100, α=20° (HydroGym's airfoil case, their solver is m-AIA-only) | **done (§45)**: steady; C_D 0.5703 / C_L 0.7729 vs HydroGym config 0.577 / 0.783 (−1.2/−1.3%), L/D 1.355 vs 1.357; self-made elliptic C-grid 33k quads (`meshes/gen_naca_cgrid.py`, `run_uairfoil.py`). **α=40° (§46): shedding, St_c 0.2331 (St_h 0.150), C_D 1.068 / C_L 1.010 vs HydroGym 1.081 / 1.027 (−1.2/−1.6%)**. Open: far-field R=16 and one refinement; **both T10 runs used the lagged convecting flux (pre-§47 default) and need rerunning with `conv_flux_extrap` before their 1.3% agreement is claimed time-converged** |

**T3 and T4 must be separated.** A scheme that is second order in space and first in time still
shows slope 2 under joint refinement if the spatial error dominates. Refine one at a time.

**T8 is the sharpest gate and the reason `central` convection is not a preference.** The growth
rate is 2.2e-3 per unit time; numerical damping of comparable size does not merely add error, it
flips the sign and reports a stable flow. The structured solver's own study records SOU removing
~10% of kinetic energy per turnover, which would swamp it by orders of magnitude, and needed
48x401 to get within 1.2%. A second-order scheme has to reach that resolution the hard way.

**T5 before T6/T7.** Stokes has no convection, so if it passes and Poiseuille fails, the fault is
convection; if Stokes fails, nothing downstream is worth debugging.

## T5 status -- divergence fixed, order still failing

**Fixed: the correction coefficient was the bare diagonal.** PISO diverged ~32x/step for any
nu >= 0.1. Root cause: `u' = -(V/a_P) grad p'` approximates `A^-1` by `1/a_P`, but a viscous
operator has zero row sum, so its contribution cancels out of the true response while inflating
`a_P`. Measured at nu=1, dt=0.05, h=1/16 (diffusion number 12.8): `a_P` = 6.06 against a row sum
of 0.058594 = `a_t*V` exactly, so `D` was **92x too small** and `p += pp` amplified geometrically.
SIMPLEC (`a_P - sum a_N`, which IS the matrix row sum) fixes it; Stokes is now stable from
nu = 0.01 to nu = 10, the last being a diffusion number of 128.

Corrector count is NOT the cure, and measuring it is what localised this: n_corr 1, 2, 5, 20 and
50 gave a **bit-identical** divergent trajectory, ruling out the PISO splitting.

Rhie-Chow keeps `V/a_P`. That coefficient is not a free parameter -- it comes from the momentum
equation's own algebraic form, which is what makes the `H/a_P` terms cancel. Substituting the
SIMPLEC value there over-damps and blows up (L2(u) 4.0e+27, then NaN).

**Still failing: the converged solution is mesh-independent.** Run to a genuine fixed point
(d|u|/step < 1e-12) rather than a fixed step count -- the 400-step runs in `test_ustokes.py`
were NOT converged and their apparent order 1.1 was an artefact of that:

| n | steps | L2(u) | order | L2(p) | p-error correlation across faces |
|---|---|---|---|---|---|
| 16 | 1261 | 1.2802e-02 | -- | 1.1160e-01 | -0.331 |
| 32 | 4795 | 1.2227e-02 | 0.07 | 7.1764e-02 | -0.106 |

What has been EXCLUDED by measurement:
* not the momentum path -- steady momentum with the exact `p`, deferred loop converged, gives
  clean second order (1.52e-03, 3.84e-04, 9.67e-05, 2.43e-05 at n=16..128);
* not the LSQ gradient, despite it being only FIRST order pointwise (interior included) -- its
  error is oscillatory and cancels in the solve: analytic vs LSQ `grad p` in the momentum source
  give 1.47e-03 vs 1.52e-03, both order 1.99;
* not the boundary Laplacian. `ef_over_d` is 2.4 where the exact normal-distance coefficient is
  3.0, but the deferred cross term recovers it exactly -- order 1.99. Forcing the normal-distance
  form instead makes it WORSE (order 1.54);
* not the velocity correction -- the converged `u` satisfies its own momentum equation to 8e-11;
* not the corrector count, the time step (fixed point varies only 1.278e-02 to 1.321e-02 across
  dt 0.4 ... 0.003125), or the cell-shape concern (all 256 wall cells are the identical right
  triangle, area ratio 1.000).

What the evidence DOES point at: the pressure carries a checkerboard (error anti-correlated
across interior faces, -0.33), and the velocity error scales inversely with the Rhie-Chow damping
-- the signature of a spurious mass source the pressure has to fight:

| rc_scale | 0.1 | 1 | 3 | 10 |
|---|---|---|---|---|
| L2(u), n=16 | 3.79e-02 | 1.28e-02 | 5.98e-03 | diverges |

More damping helps but never restores order, and above ~3 it is unstable. `PISO.rc_scale` is left
in place as the diagnostic knob that produced this table.

Two further things this turned up, both worth fixing independently:
* **`dp_compact` and `dp_wide` are different directional derivatives.** `(p_N - p_P)/|d|` is
  `grad p . dhat`; `gpf . nhat` is `grad p . nhat`. They agree only where `d` is parallel to `S`.
  Invisible at perturb=0 (interior faces are exactly orthogonal) but wrong on any skewed mesh.
* **Interior faces are SKEWED even at perturb=0.** The centroid line crosses the face at 2h/3
  while the face centre is at h/2 -- an offset of exactly h/6, measured. `convection` already
  carries a skewness correction for this; Rhie-Chow's `ubar` does not. Adding the same correction
  there makes `div(Fbar)` worse (1.35e-01 against 9.68e-02), because the gradient it leans on is
  itself only first order -- so this needs a better gradient, not just the correction term.

## Gradient experiment -- 8x on magnitude, but NOT the fix for T5

Question asked: would a wider stencil fix the T5 plateau. Answer: no. `src/ugrad.py` implements a
quadratic k-exact gradient on the vertex-neighbour stencil (`KGradient`); `PISO.grad`,
`PISO.grad_rc` and `PISO.grad_p` are separable hooks so it can be placed selectively. Defaults are
UNCHANGED -- the k-exact path is opt-in.

**Width is not the variable; polynomial degree is.** A linear fit recovers the gradient of a curved
field to O(h) however many points it is fitted through, and fitting a plane over a larger curved
region is worse. Measured gradient error on cos(pi x)cos(pi y):

| stencil / degree | n=16 | n=32 | n=64 | order |
|---|---|---|---|---|
| face-neighbour, linear (current) | 4.90e-02 | 2.51e-02 | 1.27e-02 | 0.97 |
| vertex-neighbour, linear | 1.17e-01 | 4.71e-02 | 1.98e-02 | 1.32 |
| vertex-neighbour, quadratic k-exact | 3.23e-02 | 9.64e-03 | 3.08e-03 | **1.75** |

Stencil sizes on `rect_mesh`: face-neighbour mean 3.00 and **100%** of cells below the 5
coefficients a 2D quadratic needs; vertex-neighbour mean 13.88 and **0%** below. Operator density
goes 3.97 -> ~14 nnz/row, but ONLY on the explicit path -- the implicit matrices stay two-point,
so solve cost and conditioning are untouched. If MPI ever returns, this commits a two-deep halo.

**T5 outcome:**

| configuration | n=16 | n=32 | n=64 | order |
|---|---|---|---|---|
| baseline | 1.2802e-02 | 1.2227e-02 | 1.2080e-02 | 0.07 / 0.02 |
| k-exact everywhere | 2.7403e-03 | 1.5739e-03 | **diverges** | 0.80 |
| k-exact in the momentum source only | 2.9646e-03 | 1.7944e-03 | 1.5184e-03 | 0.72 / **0.24** |

So the gradient accounted for roughly **8x of the error magnitude and none of the plateau**. The
baseline plateau is now confirmed at three refinement levels, so it is not an artefact.

**The k-exact gradient is unstable inside the pressure-correction loop.** At n=64 `max|p|` grew
~2.7x per step while `div F` stayed at 1e-12 -- the projection was sound, the feedback loop was
not. Keeping it out of that loop (momentum source only) is stable to n=64. The standalone momentum
deferred iteration is stable with it at n=64 too (converges to 2e-16, error 9.29e-05 against the
narrow gradient's 9.67e-05), which is what localised the instability to the pressure loop.

**Three hypotheses this killed, recorded so they are not re-tried:**
* *Conditioning was the cause of the n=64 divergence.* There IS a real conditioning flaw -- the
  quadratic basis mixes O(h) and O(h^2) columns, so the condition number doubled per refinement
  (43.5 / 87.1 / 174.1). Column equilibration fixes it (1.766 at every level) and is kept. But the
  results came back BIT-IDENTICAL, so it was never the cause.
* *The Rhie-Chow damping needs a smoothing gradient.* The reasoning was that
  `(dp_compact - dp_wide)` is dissipative only if the wide gradient smooths. Keeping the narrow
  gradient in `dp_wide` still diverged at n=64.
* *The wide stencil slows the deferred iteration.* There was no convergence problem at all -- the
  1e-12 threshold was below the linear solver's ~1e-10 noise floor, so converged runs looked
  capped. Error was frozen to 7 digits from step 5000 to 30000. Use a stagnation criterion on the
  ERROR, not `d|u|/step`, for every future order measurement in this suite.

**Where to look next.** Not the gradient, and not the momentum path -- both are now excluded by
measurement. The pressure still carries a checkerboard (correlation -0.39, and the k-exact gradient
made it slightly WORSE, not better), and the velocity error still scales inversely with the
Rhie-Chow damping. That points at the discrete continuity/momentum-interpolation relation itself.

## Mesh audit gaps

* `audit()` never looks at orthogonality. Every BOUNDARY face of `rect_mesh` at perturb=0 is
  **26.6 degrees** non-orthogonal (cos 0.8944, |Tf|/|S| = 0.447) while interior faces are exactly
  1.0000, and the orthogonality figures quoted earlier in this work were interior-only. Add
  boundary faces to the audit so this is never again reported as an orthogonal mesh.
* **`medium.msh` is not on this machine.** Every copy found is a 130-byte git-lfs pointer. The P0
  numbers (17,258 cells, area 199.2150) came from a real file since cleaned out of the scratchpad.
  T9 needs `git lfs pull` before it can run.

## Work items not yet started

* **Periodic faces in `umesh.py`** — PREREQUISITE FOR T8. Orr–Sommerfeld needs the domain
  periodic in x over `[0, 2pi]`. Periodicity means pairing two boundary faces into an interior
  one: matching them by their tangential coordinate modulo the period, setting `neigh` to the
  partner's owner, and carrying the period shift in `dcc` so the cell-to-cell vector does not
  jump backwards across the seam. Far cheaper than the structured seam machinery -- no
  orientations, no ghost padding, no ownership -- but it is a real change to the connectivity
  builder and to `audit()`, not a boundary-condition flag.
  The structured code's own scar is worth borrowing: its periodic path had to shift coordinates
  by one period or "the ghost coordinates jump backwards across the seam, collapsing the
  Jacobian there".
* **Face-based jet application** — the profile is written and verified analytically
  (`cylinder_rect_bc.jet_amplitude`); it needs porting onto cylinder boundary faces.
* **Bounded second-order convection for Re > ~500.** The convection term is deferred-correction
  central (`scheme="central"`), kept because T8's growth rate punishes any first-order dissipation.
  At Re = 1000 on the airfoil the mid-chord cell Péclet number is ~30 (h ~ 0.03, U ~ 1) and central
  differencing will oscillate. On unstructured meshes that means a limited linear-upwind or TVD
  scheme in the Darwish–Moukalled r-factor formulation (the upwind-side "far" value reconstructed
  as phi_C - grad(phi)_C . d, so the Sweby limiter needs no upstream cell), applied as a deferred
  correction on top of the implicit upwind part so the matrix stays diagonally dominant. Goal is
  bounded second order, not "no artificial dissipation": limiters add dissipation wherever they act.
  Gate: T3 order must not fall below the central scheme's on smooth flow, and the first Re-1000
  test is the NACA0012 at 20° against Kurtulus (2015) St and force amplitudes. Not started.
* **MPI** — deferred. On the structured code MPI measured NEGATIVE at this problem size
  (1 rank 4.58 s/step, 8 ranks 9.72 s/step) because the implicit solves gather to rank 0.
  Revisit only if the cell count grows by an order of magnitude.

## Standing notes

* The solver is cell-centred FV; HydroGym is Taylor–Hood P2–P1. Same mesh, same BCs, different
  discretisation — agreement on St and C_D validates us, but exact agreement is not expected,
  and their P2 velocity carries more accuracy per cell than our one value per cell.
* Every seam/geometry defect in the structured work passed `validate()` and produced a
  plausible-looking field. The invariant tests in P1/P2 exist for that reason and should be run
  on every mesh, not only on the one being developed against.
