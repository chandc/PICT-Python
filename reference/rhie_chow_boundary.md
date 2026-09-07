# The Rhie–Chow gap at physical boundaries, and how PICT and OpenFOAM avoid it

**Status: mechanism identified and the upstream remedy is confirmed at source level. Not yet
implemented.**

## The gap

`Domain.pressure_face_fluxes(..., rhie_chow=True)` returns `compact - wide`: the compact face
difference of the pressure minus the face-interpolated cell-centred (wide) gradient. That
difference is the checkerboard damping. On any face whose two cells are not both interior it is
**dropped entirely**:

    else:
        val = 0.0 * val   # no valid wide stencil -> no correction here

The reason is real. `pad_field` gets its ghosts from `_ghost_field`, which returns `None` at a
wall or an inflow -- there is no pressure ghost outside a physical boundary -- so `np.gradient`
falls back to a one-sided `edge_order=2` stencil at those cells. That stencil extrapolates the
INTERIOR field and knows nothing about the pressure boundary condition, so `compact - wide` stops
being the O(h^3) difference of two centred second-order approximations and becomes an O(1) term.
It was tried, and it diverged: at an inflow the correction grew from 5e+01 to 4e+02 over 30 steps.

Measured consequence of dropping it instead: **4,096 of 533,568 faces carry exactly zero
damping**, split 50/50 between the body surface and the outer boundary, none in the interior --
i.e. precisely one cell layer adjacent to every physical boundary. This is consistent with the
periodic TGV being clean (0.000-0.029% Nyquist energy) while the channel and the cylinder are not,
and with the cylinder's far-field Dong-arc odd-even oscillation (edge/middle ratio 8.1x).

## What upstream PICT does

`tum-pbs/PICT` contains no occurrence of "Rhie" or "Chow". The damping is implicit, from the
mismatch between a compact face-based Laplacian and a wide cell-centred gradient in the velocity
correction -- the same construction as OpenFOAM. All references are to
`extensions/PISO_multiblock_cuda_kernel.cu`.

**1. The boundary face flux is prescribed, never interpolated** (`computeFluxesNDLoop`, l.1553):

    case BoundaryType::FIXED:
        // enforce flux
        fluxes[bound] = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &block.boundaries[bound].fb, domain);

**2. The pressure matrix contributes nothing on that face** (`PISO_build_pressure_matrix`, l.4798):

    const bool atPrescribedBound = atBound && isEmptyBound(bound, s_block.boundaries);
    if(!atPrescribedBound){ ...assemble... } else { indices[bound+1] = -1; }

Homogeneous Neumann for pressure, consistent with (1): the flux through that face is already
known, so the pressure correction must not change it.

**3. The cell-centred gradient stays a two-point central difference EVERYWHERE**, because a ghost
is always supplied (`getPressureGradient` l.802 calling `getPressureAtWithBounds` l.2161):

    case BoundaryType::VALUE: case BoundaryType::DIRICHLET_VARYING:
    case BoundaryType::FIXED: case BoundaryType::GRADIENT:
        // enforce 0 pressure gradient to avoid changing the prescribed value
        tempPos.a[dim] = 0;
        return zeroBound ? 0 : block.pressure[flattenIndex(tempPos, block)];

`getPressureGradient` switches its factor from 0.5 to 1 -- a genuinely one-sided difference --
ONLY for `isEmptyBound`, which marks a degenerate/absent axis, never a wall.

So the ghost is `p_ghost = p_boundary_cell`, and the wide gradient at a boundary cell is
`(p[1] - p[0]) * 0.5`.

## What OpenFOAM does

`applications/solvers/incompressible/pisoFoam/pEqn.H`:

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA) + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)));
    adjustPhi(phiHbyA, U, p);
    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, U, phiHbyA, rAU, MRF);
    fvScalarMatrix pEqn(fvm::laplacian(rAU, p) == fvc::div(phiHbyA));
    ...
    phi = phiHbyA - pEqn.flux();
    U = HbyA - rAU*fvc::grad(p);

Same three pieces:

1. `constrainHbyA` / `constrainPhiHbyA` overwrite the patch values on every patch whose U is not
   assignable with the PRESCRIBED value (`HbyAbf[patchi] = U.boundaryField()[patchi]`,
   `phiHbyAbf[patchi] = Sf & U_b`). No interpolation from the interior.
2. `constrainPressure` sets the `fixedFluxPressure` patch gradient to
   `(phiHbyA_b - rho*(Sf & U_b)) / (magSf * rhorAU_b)` -- the gradient that makes the boundary
   flux equal the prescribed flux exactly. `fixedFluxPressure` derives from `fixedGradient`.
3. `fvc::grad(p)` is a Gauss sum over faces INCLUDING boundary faces, using that patch value. It
   is therefore defined at every cell, with no one-sided interior extrapolation anywhere.

## The comparison

|  | wide gradient at a boundary cell |
|---|---|
| PICT | `(p[1] - p[0]) * 0.5` -- zero-gradient ghost |
| OpenFOAM | Gauss sum using the `fixedFluxPressure` patch value |
| this repo | one-sided `np.gradient(edge_order=2)`, then DISCARDED |

Our boundary FACES are already correct: `pressure_face_fluxes` does `continue` at the domain
boundary (zero flux, no matrix entry), which is PICT's `atPrescribedBound` and OpenFOAM's
`fixedFluxPressure`. Only the FIRST INTERIOR FACE is wrong.

## Why the ghost is safe where the one-sided stencil was not

They are not the same fix, and the earlier divergence does not argue against the ghost.

A one-sided `edge_order=2` gradient extrapolates the interior field. Nothing bounds it relative to
the compact difference, so `compact - wide` can take either sign and any magnitude -- hence 5e+01
to 4e+02 in 30 steps.

A zero-gradient ghost gives `wide = compact / 2` at a boundary cell, exactly. So
`compact - wide = compact / 2`: same sign as the compact term, magnitude between half and all of
it. It is a bounded, correctly-signed damping by construction and cannot run away. At a wall it
makes the damping STRONGER than the interior value, not wilder.

## Before implementing

`getPressureAtWithBounds`'s ghost is homogeneous Neumann, which is the right BC for a
prescribed-velocity boundary. Confirm that is what our pressure solve actually assumes at walls
and at the inflow before adopting it -- the outflow (Dong) is the case to check, since a
convective outflow is not a prescribed-flux boundary and OpenFOAM would not put
`fixedFluxPressure` there.

See also `pressure_checkerboard.md`, `rhie_chow_ddt_instability.md`, `les_findings.md`.

---

# The fix, and how it is tested

## The repo already contains the correct implementation

`mb_adjoint.cell_gradient_matrix` builds the wide gradient as `P @ F`, where `P` averages the
faces either side of a cell with weight 0.5 each. At a wall the boundary cell has only ONE face,
so the half-weight survives and the matrix produces

    (p[1] - p[0]) / (2h)

which is EXACTLY PICT's zero-gradient ghost. Measured on a uniform channel (h_y = 0.4,
1/(2h) = 1.25):

    cell j= 0  ->  j=0: -1.250000, j=1: +1.250000     <- wall: half-weight one-sided
    cell j= 1  ->  j=0: -1.250000, j=2: +1.250000     <- interior: central
    cell j= 5  ->  j=4: -1.250000, j=5: +1.250000     <- wall

So the two implementations of the same operator DISAGREE, and `verify_rc_divergence` -- which
predates this investigation -- already measures it:

| domain | max abs err | scale | relative |
|---|---|---|---|
| periodic box 8^3 | 2.274e-13 | 1.002e+03 | 2.3e-16 |
| periodic box 8^3, 2 blocks | 2.274e-13 | 1.002e+03 | 2.3e-16 |
| **channel 8x12x8 (walls in y)** | **8.155e+05** | 1.436e+04 | **5.7e+01** |

and the discrepancy is confined to the two wall-adjacent layers, machine-zero everywhere else:

    j =  0   8.154740e+05   <- wall
    j =  1   7.105278e+04   <- wall
    j =  2   2.728484e-12
    ...                     (1e-13 through j = 9)
    j = 10   5.021088e+04   <- wall
    j = 11   5.762712e+05   <- wall

This is a DEMONSTRATION, not the earlier inference from face counting.

## Recommended fix

Make the flux routine match the matrix. Nothing in `mb_adjoint.py` changes -- it is already right.

1. **`src/multiblock.py`, `pressure_face_fluxes`.** Where `pad_field` supplied no ghost (physical
   boundary, `lo2[axis] == 0` or `hi2[axis] == 0`), overwrite the edge entry of `g2` with the
   zero-gradient-ghost value `0.5 * (pp2[1] - pp2[0]) / h` (and the mirror at the upper edge)
   instead of `np.gradient`'s one-sided `edge_order=2` result. Then DELETE `dpw_ok` and the
   `val = 0.0 * val` branch -- with a ghost everywhere there is no invalid face left.

2. **`src/phase5_fluxes.py`.** The single-block twin has the same defect through
   `deriv(p, h[a], a, per[a])` on a non-periodic axis. Same treatment.

3. **`src/mb_adjoint.py`.** No change. That is the point.

The boundary FACES stay as they are -- `continue`, zero flux, no matrix entry -- matching PICT's
`atPrescribedBound` and OpenFOAM's `fixedFluxPressure`.

### On the outflow

`build_diffusion_matrix` enumerates only interior faces, one periodic wrap per axis, and one face
per connection. Physical boundary faces contribute nothing, so the pressure operator is ALREADY
homogeneous Neumann at wall, inflow AND outflow. The zero-gradient ghost therefore matches the
operator we already invert in all three cases, which is what the damping term needs. Whether
Neumann is the physically right OUTFLOW pressure BC is a separate, pre-existing question that
this fix neither creates nor resolves.

## Test plan

Cheap and decisive first; the expensive confirmations last.

| # | test | criterion | why it can fail |
|---|---|---|---|
| T1 | `verify_rc_divergence` on a wall-bounded channel | rel err ~1e-16, matching the periodic rows | **currently 5.7e+01.** Predates the fix; a half-done fix (flux but not matrix, or either alone) leaves it failing |
| T2 | Gate 0 bitwise, 640 arrays, periodic/TGV | 640/640 identical | proves the change is confined to boundaries. Run on the machine owning the reference (it is numpy/scipy-version specific) and pin `momentum_tol` |
| T3 | zero-damping face count on the channel and cylinder meshes | 0 interior faces; only domain-boundary faces, which carry none by design | currently 4,096 of 533,568, split 50/50 body/outer |
| T4 | `test_channel_laminar.py` | 4/4, parabola stationary to 0.009%, u_tau 0.9986 | the fix ADDS damping at the wall, where the profile matters. Laminar Poiseuille has dp/dy ~ 0 at the wall so the term should be ~0 -- if u_tau moves, it is doing something it should not |
| T5 | Stokes refinement study (`stokes_verification.md`) | observed order stays ~2 | the term is O(h) at the wall cell. This is the legitimate objection to the fix and the test that could honestly kill it |
| T6 | square cylinder (inflow + Dong outflow), instrumented `|RC|/|F|` as in `rhie_chow_ddt_instability.md` | ratio stays O(0.1), does not grow | **the falsifiable prediction.** The one-sided attempt went 5e+01 -> 4e+02 in 30 steps. The ghost is bounded (`wide = compact/2` at a boundary cell, so `compact - wide = compact/2`), so it must not |
| T7 | scheme-axis channel probe, same seed/dt/mesh | rotational amplification collapses from 15.292x toward chorin's 0.001x; Nyquist energy fraction 99.3% -> TGV level (<3%) | the payoff test. Expensive; run last |

T1-T5 are minutes. T6-T7 are hours.

`verify_rc_divergence` needs no mangle to prove it bites -- it is failing right now.

---

# Results

## The fix as implemented

`src/multiblock.py` `pressure_face_fluxes`: where `pad_field` returned no ghost (`lo2[axis] == 0`
or `hi2[axis] == 0`), the edge entry of `g2` is replaced by `0.5 * (p_next - p_edge) / h`, the
zero-gradient ghost. `dpw_ok` and the `val = 0.0 * val` branch are gone.

`src/phase5_fluxes.py`: the same, into a SEPARATE `dpw` array rather than patching `dp`. `deriv`
is shared with the grid metrics and `dp` also feeds the non-orthogonal cross term; neither wants
this ghost.

`src/mb_adjoint.py`: unchanged.

## T1 -- the two implementations agree. PASS

| domain | before | after |
|---|---|---|
| periodic box 8^3 | 2.269e-16 | 2.269e-16 |
| periodic box 8^3, 2 blocks | 2.269e-16 | 2.269e-16 |
| channel 8x12x8 (walls) | **5.679e+01** | **4.282e-16** |

## T2a -- periodic is bitwise unchanged. PASS

60/60 digests identical (u, v, w, p; 5 steps; 1 block and 2 blocks; periodic box, rotational,
rhie_chow=True, persistent_flux=True) against `git show HEAD`. The change is confined to physical
boundaries, as it must be.

**T2b, and it is a consequence, not a failure:** the Gate 0 bitwise reference is the CYLINDER
(`reference/gate0/cylinder_re100.npz`), which has a wall, an inflow and an outflow. The fix
changes it BY DESIGN. That reference has to be regenerated, and the DD gates re-based on it.

## T3 -- no interior face is left undamped. PASS

Driven with a random pressure so a zero can only come from the operator. Structurally-absent
faces are excluded PER SIDE -- deciding per axis instead counted `k = n` on a block with a
connection below and a wall above, which is what produced a spurious 896 survivors on the first
cut.

| mesh | before | after |
|---|---|---|
| channel 24x40x24, 1 block | 1152 / 70464 | **0** |
| channel 24x40x24, 2 blocks | 1152 / 71424 | **0** |
| square cylinder, 8 blocks | 2240 / 169360 | **0** |

**And a finding that matters more than the count.** Every one of the channel's 1152 was on
AXIS 1, the wall-normal axis -- 576 per wall, exactly nx*nz. x and z are periodic, so they always
had a full ghost and full damping. The reported checkerboard is STREAMWISE (99.3% of the energy
at lambda_x+ = 47.1 = 2 dx+). So the fix repairs a demonstrated defect on wall-normal faces, and
"missing wall-normal damping produces a streamwise 2dx mode" remains UNDEMONSTRATED. T7 is what
decides it; the scheme axis (rotational 15.292x vs chorin 0.001x) may be a separate cause.

## T4 -- the wall solution is not polluted. PASS

`test_channel_laminar.py` 4/4, numbers unchanged: parabola drift 0.0090% of u_c, u_tau 0.9986
against the exact 1 at dy+ = 1.00.

## T5 -- the duct version was VACUOUS, and this is worth recording

The first attempt refined the square duct (four walls, corners, exact Fourier series). It gave
rates 2.059, 2.037 both before and after -- because fully developed parallel duct flow has a
UNIFORM cross-sectional pressure. Measured spread: **1.591e-08**. `rhie_chow=True` was
bit-identical to `rhie_chow=False`, so the test could not see the fix at all, and "the order did
not change" meant nothing.

Redone on the wall-bounded Stokes mode (`test_stokes_channel`), which has genuine wall-normal
pressure structure (spread 1.2e-04) and an exact eigenvalue sigma = -9.313739. Non-vacuity is
checked and PRINTED before the order study: rc=True vs rc=False differ by 2.8e-02 in sigma.

## Other regressions

`test_rhie_chow.py` 28/28, `test_checkpoint.py` 25/25, `test_implicit_cross.py` 10/10,
`test_energy_conservation.py` 6/6, `test_numerical_dissipation.py` 4/4,
`test_mb_adjoint_seam.py` 16/16, `test_mb_adjoint_state.py` 9/9, `test_mb_adjoint_bc.py` 10/10,
`test_forces.py` 6/6, `test_duct.py` rates 2.06/2.04 -- all matching the Gate 0 suite baseline.


## The single-block path had the SAME defect by a different mechanism

`phase5_fluxes` has no drop -- and needs none, because the one-sided stencil cancels the damping
identically. At the wall-adjacent face, with `deriv`'s `edge_order=2`:

    dp[0] = (-3 p0 + 4 p1 - p2) / (2h)          one-sided at the wall cell
    dp[1] = (p2 - p0) / (2h)                    central
    wide  = 0.5 c [dp0 + dp1] = 0.5 c (4 p1 - 4 p0)/(2h) = c (p1 - p0)/h
    compact = c (p1 - p0)/h

so `compact - wide = 0` EXACTLY. Measured, wall-bounded 8x10x8, random p:

| axis-1 face | before | after |
|---|---|---|
| k = 1 (first interior, wall side) | **0.000000** | **27.281994** |
| k = 5 (mid-channel) | 24.679488 | 24.679488 |

Two independent implementations, two different mechanisms -- an explicit `val = 0.0 * val` in
`multiblock` and pure algebra in `phase5_fluxes` -- arriving at exactly zero damping on the same
face. That is independent corroboration of the diagnosis, and it is why the O(1) one-sided term
that diverged at an INFLOW was invisible at a wall: at a wall it does not merely misbehave, it
cancels.

## A trap in the TEST HARNESS, recorded because it nearly published a false result

`t5_stokes.py`, `t6_stability.py`, `t7_gridmode.py` and `check_phase5.py` each opened with

    sys.path.insert(0, "/Users/danielchan/Dropbox/PICT-Python")

added so the scratchpad scripts could import the repo's top-level drivers. It puts the PATCHED
tree ahead of `PYTHONPATH=<baseline tree>`, so **both arms of every before/after comparison ran
patched code.** T5 came back identical to the last printed digit in both arms, which is what
exposed it; had the fix been slightly lossy rather than exactly inert on that case, the
comparison would have looked like a clean pass.

The fix is to rely on `PYTHONPATH=<base>:.` alone -- the baseline tree carries only `src/`, so
the drivers still come from the repo -- and to `diff` the baseline tree against `git show HEAD`
before spending hours on it. T1-T4 never used that harness.

## T5 -- order of accuracy, wall-bounded Stokes, RC ON

Exact sigma = -9.313739. Non-vacuity printed first: rc=True vs rc=False differ by 2.4e-02.

| ny | baseline err | patched err |
|---|---|---|
| 17 | -4.522e-02 | -4.091e-02 |
| 33 | -8.106e-03 | -7.546e-03 |
| 65 | -1.680e-03 | -1.632e-03 |
| **observed order** | **2.480, 2.271** | **2.439, 2.209** |

PASS. Second order is preserved -- the O(h) damping at one wall-adjacent cell layer does not
degrade the global rate, which is what upstream's second-order accuracy already implied but had
not been shown here. The patched errors are marginally SMALLER at every resolution.

## T6 -- the stability test the one-sided attempt failed. PASS

Square cylinder (inflow + Dong outflow + body), 42,864 cells, 600 steps to t = 6.

| step | baseline | patched |
|---|---|---|
| 1 | 0.0438 | 0.0438 |
| 25 | 0.0150 | 0.0157 |
| 400 | 0.0108 | 0.0112 |
| 600 | **0.0104** | **0.0108** |

Both DECREASE monotonically. The patched ratio sits ~4% above baseline -- slightly more damping,
as the ghost predicts -- and stays two orders of magnitude below the 0.184 that
`rhie_chow_ddt_instability.md` records for a healthy run, against the 5e+01 -> 4e+02 in 30 steps
that the one-sided stencil produced. The falsifiable prediction held.

Instrumented by INTERCEPTING `Domain.pressure_face_fluxes` and `Domain.face_fluxes` rather than
recomputing: the first cut recomputed with coef = 1, because Gamma = J/rowsum(A) is a local in
`step()`, and reported 3.4 for a quantity the solver never forms.

## What this fix DOES and DOES NOT invalidate

Checked rather than assumed:

* **`reference/gate0/` (cylinder, 10 steps, rhie_chow=True): INVALIDATED.** Regenerate on the
  machine that owns it -- the bitwise reference is numpy/scipy-version specific.
* **Gate 3 and Gate 4 references: INVALIDATED.** Both run `m.step()` (NSTEPS = 10).
* **`reference/gate5_serial.npz`: NOT invalidated.** Gate 5 calls `measure()` on the state
  straight out of `checkpoint.load` and takes NO steps, so it is a rank-reduction consistency
  check on a fixed field. Re-captured under the baseline tree and confirmed byte-identical to the
  post-fix capture -- "differs in NOTHING".

  (It was overwritten in the course of finding this out, because `test_gate5.py` CAPTURES at
  1 rank and only COMPARES at more than one. Restored, and the restore verified reproducible:
  two baseline captures are byte-identical.)

The circular cylinder itself is affected -- 2048 of 531,520 interior faces went from zero damping
to none, split evenly between the body wall (axis0 side0, 16 blocks) and the far field (axis0
side1, 16 blocks), which is the 50/50 body/outer split noted earlier at a different nz.

---

# Why the cylinders showed nothing near the wall, and what that implies

The zero-damping layer is one cell deep at EVERY physical boundary in all three cases -- body
surface and far field alike (the measured 50/50 split). So "the cylinders had no near-wall
checkerboard" is not explained by where the gap was. It is explained by the CELL REYNOLDS
NUMBER in that layer: where viscosity already damps a 2-cell mode, losing the Rhie-Chow damping
costs nothing.

Measured (`U * delta / nu` in the first cell in from each physical boundary):

| zero-damping layer | first cell | Re_cell | observed |
|---|---|---|---|
| circular cylinder, body surface | 0.00600 | **0.60** | nothing |
| square cylinder, body surface | 0.03180 | **3.18** | nothing |
| channel LES, wall-normal (dy+ 1.00) | 0.00556 | **1.00** | nothing |
| circular cylinder, far field (r = 20D) | 0.28526 | **28.5** | Dong-arc odd-even |
| square cylinder, outer box | 0.89-0.99 | **89-99** | same far-field family |

A wall is refined precisely because the gradients there are steep, which drives Re_cell to O(1);
the far field is 50-150x coarser and nothing damps it. Every one of the five boundary
observations follows from that single number.

## The consequence for the channel

The channel's checkerboard is STREAMWISE -- Re_cell = 363, 99.3% of the energy at 2 dx. But x is
PERIODIC: it always had a full ghost and full damping. **The boundary gap was never present in
the direction where the channel actually fails.**

So the four boundary observations are explained by this defect, and the channel's streamwise mode
is a DIFFERENT phenomenon. This fix should not be expected to cure it. The remaining candidate is
the one the scheme axis already isolated: rotational amplification 15.292x against chorin's
0.001x, running at Re_cell 363 where nothing damps it and where van Driest deliberately drives
nu_t -> 0 at the wall.

T7 is therefore not a pass/fail on the fix. If its arms stay together to t = 12, that RETIRES the
boundary hypothesis for the channel by measurement instead of by argument, and points the next
investigation at the rotational pressure scheme. `channel_les_status.md` items (3) and (4) --
"at walls and outflows there is NO damping, so the mode grows there" and "once seeded at the
boundary it spreads inward" -- should be re-scoped accordingly: (3) is now demonstrated and
FIXED, and (4) is what T7 tests.

---

# T7 -- STOPPED at t = 4.75 of 12, and the partial result is decisive enough

Both arms ran 4750 steps (dt = 1e-3) on the Re_tau = 180 channel, identical except for the fix,
seeded with a 2dx streamwise mode. Logs: `results/logs/t7_base.log`, `t7_patch.log`.

| t | amp_x base | amp_x patched | E_2dy base | E_2dy patched |
|---|---|---|---|---|
| 2.00 | 0.052 | 0.052 | 5.9533e-02 | 5.9734e-02 |
| 3.25 | 0.081 | 0.081 | 1.5482e-01 | 1.5544e-01 |
| 4.50 | 0.087 | 0.087 | 2.6757e-01 | 2.6287e-01 |
| 4.75 | 0.100 | 0.100 | 2.9380e-01 | 2.8699e-01 |

**The streamwise mode is identical to three digits at every one of 19 sample points.** The
wall-normal mode differs by 0.6-2.5%, so the fix IS active and is acting exactly where T3 said it
would -- on wall-normal faces -- and not where the checkerboard is.

## Why it was stopped rather than run to t = 12

At t ~ 4.75 both runs slowed by roughly 18x -- no output for 2h43m at 100% CPU, against 0.86
s/step earlier. Not a hang: RSS was identical to the byte over that period and the main thread
sampled into numpy rather than `_sparsetools`, which points at the pressure solve's ITERATION
COUNT exploding rather than a deadlock. Reaching t = 12 would have taken ~30 hours.

That both arms slowed identically, at the same time, is itself consistent with the conclusion:
whatever degrades the pressure system at t ~ 4.75 is unaffected by this fix.

## Conclusion

T7 does not confirm the boundary hypothesis for the channel -- it retires it, which is what it
was for. Combined with the two independent arguments already recorded (the zero-damping faces
were wall-normal only while the mode is streamwise; the cell-Reynolds table explains all five
boundary observations without needing the channel), the channel's streamwise checkerboard is a
DIFFERENT phenomenon from the one fixed here.

`channel_les_status.md` items (3) and (4) should now read: (3) demonstrated and FIXED, and it
explains the CYLINDERS' far-field oscillation; (4) refuted for the channel. The next
investigation is the rotational pressure scheme -- amplification 15.292x against chorin's 0.001x
-- running at Re_cell 363 in the streamwise direction where nothing damps it and van Driest
drives nu_t -> 0.

An 18x slowdown of the pressure solve at t ~ 4.75 is worth its own look: it is a measurable
symptom, on a case that reaches it in about an hour, and nobody has instrumented the iteration
count through it.
