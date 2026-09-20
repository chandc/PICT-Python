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

## Standing notes

* The solver is cell-centred FV; HydroGym is Taylor–Hood P2–P1. Same mesh, same BCs, different
  discretisation — agreement on St and C_D validates us, but exact agreement is not expected,
  and their P2 velocity carries more accuracy per cell than our one value per cell.
* Every seam/geometry defect in the structured work passed `validate()` and produced a
  plausible-looking field. The invariant tests in P1/P2 exist for that reason and should be run
  on every mesh, not only on the one being developed against.
