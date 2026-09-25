# Unstructured LES: status, measured rules, open items

*2026-09-25. One page on where the unstructured 2.5D LES solver stands. The evidence is in
`skew_unstructured_literature.md` §§48–60 (cited as [§n]); the plan and its gates are
`unstructured_les_plan.md`; the discretisation is `piso_unstructured_formulation.md`; the GPU
package is `tools/a100/README.md`.*

## What the solver is

`src/upiso25.py` (`PISO25`): the unstructured cell-centred collocated finite-volume scheme in the
x–y plane (quads or triangles, Rhie–Chow, dual Green–Gauss pressure gradient, deferred central
convection with skewness correction), Fourier in a periodic span with 3/2-rule dealiasing, RK3 with
a pressure projection per stage (Le & Moin), WALE or Smagorinsky (`src/usgs.py`), constant-pressure-
gradient or constant-mass-flow forcing, binned statistics, bitwise restart (`src/ustats.py`). Per-mode
implicit systems solved as one block by PCG with a shared Ruge–Stuben AMG hierarchy
(`src/umodesolve.py`); the whole step runs on CuPy with raw CSR-block kernels (`src/ucuda.py`).
Mode 0 reproduces the 2D solver to 1e-15 [§51], so the 2D validation ladder (T1–T11) carries over.

## Gates

| gate | criterion | status |
|---|---|---|
| G0 attribution of energy loss | measure, decide L1a/L1b | passed [§49] |
| G1 RK3 per-stage projection | T4 order, T8, T11 ≤ 0.2%/turnover, T9 within 0.3% | passed on T4/T8/T11; T9 Cl amplitude −1.2% is the dt-independent Rhie–Chow damping BDF2 keeps, criterion mis-posed [§50] |
| G2 2.5D solver | TGV energy balance 0.5%, spectral in z, order 2 in plane, cylinder = 2D | passed on all four [§51] |
| G3 SGS | variable-ν order 2, rotation checks, WALE ∝ y³, ν_t reported | passed [§52] |
| G4 solvers and speed | < 30 pressure iterations; ≤ 100 ms per step per 10⁵ cell-modes on the GPU; dt change free | 8–19 iterations; **61–73 ms on the GB10 at every size**, 140 on a real mesh; dt change holds by construction [§57] |
| G5 infrastructure | lossless restart; statistics reproduce an analytic mean | passed [§53] |
| V1 Taylor–Green (Re 800, in-house SEM DNS) | peak time 3%, value 5%, WALE | **value met** (−4.8/−5.2% at 96²/128²), curve rms 5.9% (128² implicit); **time not met**: 6% early at every resolution, orientation-dependent [§58] |
| V2 channel Re_τ 180 (FOSLS DNS) | U⁺ 3%, u_rms 5%, Re_τ 2%, pressure two-colour mode < 1% | **passed on quads**: Re_τ 179.1, U⁺ +1.0%, u_rms −0.3%, −u'v' −1.2%, two-colour 0.00% [§54]. **Fails on every triangle mesh** [§55] |
| V3 cylinder Re 3900, periodic span | St 3%, C_D 5%, recirculation 10% | not started: needs a Re 3900 mesh; 10–20 h on an A100 for the fine butterfly [§59] |

## Design rules, measured

1. **RK3 with a projection per stage is the LES integrator.** Energy loss 0.04%/turnover against
   3.9% for BDF2-PISO at dt 0.005 [§50]. BDF2-PISO remains the default for laminar 2D cases only.
2. **No dt-independent Rhie–Chow transient term inside RK3.** It left a 1.2–1.7%/turnover floor;
   the plain O(dt) stage damping vanishes with the step [§50].
3. **Quads for wall-bounded LES.** Split quads and hybrid triangle cores diverge or carry 14% of the
   pressure rms in the two-colour mode with a zigzag Reynolds stress; isotropic graded triangles
   run but the bulk velocity drifts +14% while τ_w ≈ f (the discrete momentum balance does not
   close), U⁺ +1.2 u_τ, v_rms −27% [§55]. Triangles stay in laminar external flow with
   vertex-averaged post-processing.
4. **WALE with Δ = (V δz)^{1/3}** gives the channel within 1% [§54] but over-dissipates the laminar
   phase of a transition (ν_t/ν ≈ 0.8 on the laminar TGV field, a shoulder at t 4.5–6) [§58]. The
   σ-model, which vanishes for laminar and 2D states, is the port to make before transitional cases.
5. **Second-order in-plane convection advances a transition**: the TGV breakdown comes 6% early at
   every resolution and moves 0.9 time units when the initial condition is rotated [§58]. Only a
   fourth-order in-plane reconstruction changes that.
6. **Spanwise dealiasing is mandatory**: without it the TGV is unusable by t = 5 [§58].
7. **The in-house "Re 1600" TGV reference is the Re 800 SEM file** (`results/tgv_diag_re800_88.npz`,
   ν = 1/800); V1 is judged at Re 800 against its full history [§58].

## Cost (GB10; A100 measured 2.3× before the last optimisation, remeasure after pulling)

| case | ms/step | per 10⁵ cell-modes |
|---|---|---|
| TGV 128²×64 | 340 | 63 |
| TGV 384²×64 | 3,275 | 67 |
| fine butterfly 27,968 quads × 64 modes, WALE | 1,290 | 140 |
| channel Re_τ 395, 96×160 × 128 modes | ~1,400 | ~140 |

Launch floor ~46 ms per step (~250 kernel launches after the fusions); the nonlinear term is now
40% of the step. Run-time table and profiler: `tools/a100/profile_step.py`.

## Open items (in order)

1. Port the σ-model; gate: TGV Re 800 curve rms at 96²×96 below the implicit run's.
2. Run the Re_τ 395 channel on the A100 (`tools/a100/A100_channel_re395.ipynb`; MKM 1999
   reference in `reference/mkm_chan395/`; initial field = the 180 DNS field with the mean shifted).
3. Fuse the nonlinear term's padded-plane gathers on the GPU; CFL-adaptive stepping.
4. A Re 3900 cylinder mesh (wall cell ~0.002 D, 10⁵ cells) for V3.
5. Fourth-order in-plane reconstruction, only if a transition's peak timing becomes a criterion.

## Not doing

TVD or limited convection in the LES momentum equations; triangles or tetrahedra for wall-bounded
LES; iterating the dual Green–Gauss gradient (its inconsistency is confined to the momentum
pressure term and measured harmless [§§16–24]; the triangle failure is a different mechanism);
max(Δx, Δy, Δz) as the filter width (it increases ν_t; the channel result does not ask for it) [§60].
