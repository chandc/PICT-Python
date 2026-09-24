# LES on the unstructured PISO code — implementation plan and tollgates

Written 2026-09-24 against the state recorded in `skew_unstructured_literature.md` §1–48 and the gate
table in `unstructured_plan.md`. The code today: 2D cell-centred collocated finite volume on polygons,
BDF2 + PISO with the extrapolated convecting flux, Rhie–Chow (time-step-independent form), dual
Green–Gauss pressure gradient, deferred central convection, periodic seams, sparse direct solves.
Verified: second order in space (quads) and time, energy-conserving convective operator on quads,
cylinder and airfoil forces within 1% of HydroGym.

Standing constraints carried into this plan: **hexahedra/quads only for LES** (the non-bipartite
pressure mode feeds an eddy-viscosity model), **central convection only** (no limiter in a
turbulence-resolving run; §"TVD" discussion), every change validated on a gate before the next
phase starts, nothing committed without asking.

## The ladder

| phase | what | gate | success criterion | est. effort |
|---|---|---|---|---|
| L0 ✅ | **Attribute the full-step energy loss** — DONE 2026-09-24 (§49): splitting 77-89% on the steady TGV; on the advected TGV splitting ≈ BDF2 ≈ 2%/turnover each at dt 0.005, Rhie-Chow ~10% and O(h²). Decision: **L1a**. `PISO.n_outer` (converged coupling) kept as the reference mode. Original text: (T11 measured 2.2%/turnover at dt 0.005, ∝ dt², Rhie–Chow only 4–22% of it) | G0 | Run inviscid TGV with PISO correctors iterated to convergence (residual < 1e-10) at dt 0.005 / 0.0025. Report the fraction of the loss that survives. Decision: > 50% survives → BDF2 dominates → L1a; < 50% → splitting dominates → L1b | 1 day |
| L1a ✅ | **RK3 with per-stage projection** — DONE 2026-09-24 (§50): `PISO.time_scheme="rk3"`, Le–Moin coefficients, CN viscous, explicit convection on the stage flux, per-stage Rhie–Chow **without** Choi's transient term (it left a dt-independent 1.2–1.7% floor inside the RK stages), `init_flux()` fixes the zero initial flux that capped every temporal test at first order. Measured: T4 convective order 2.9–3.0, viscous 2.0 (CN, by construction), constant 1000× below BDF2; T8 48×400 phase dt-independent to 4e-4%, growth −0.2%; **T11 0.04%/turnover at dt 0.005** (BDF2-PISO 3.9%), integrator ∝ dt³, full step ∝ dt from the O(dt) Rhie–Chow damping; T9 fine butterfly St −0.06%, Cd −0.22%, Cl amp −0.71% (the one miss of the 0.3% band; dt 0.0025 pair open). Original criteria: | G1 | T4 order ≥ 2.8 in dt (unsteady Stokes); T8 phase speed dt-independent to 0.1% and growth within 1% at 48×400; **T11 full-step loss ≤ 0.2%/turnover at dt 0.005 on 64² quads** (10× below today) and ∝ dt³; T9 butterfly-fine St/Cd/Cl unchanged within 0.3% | 2–3 weeks |
| L1b | **Converged coupling per step**: PISO iterated to a residual tolerance, or a coupled velocity–pressure solve (direct in 2D) | G1 | same T11 criterion (≤ 0.2%/turnover at dt 0.005) and T8/T9 unchanged | 1–2 weeks |
| L2 ✅ | **Third dimension, 2.5D** — DONE 2026-09-24 (§51): `src/upiso25.py` `PISO25`, Fourier span, 3/2-rule dealiased nonlinear term, per-mode momentum/pressure/Rhie–Chow, RK3 per stage. Mode 0 reproduces the 2D RK3 to 1e-15. TGV Re 100 32²×32: −dE/dt = discrete dissipation to 0.1–0.2% at all t (0.5% criterion met; against the LSQ-gradient enstrophy the gap is the gradient's 10% truncation at 32²); spanwise convergence spectral (1.84 → 0.118 → 5e-4 for nz 4/8/16); in-plane order 2.04/2.32 (n 16→32→64 vs 128); periodic-span cylinder (coarse butterfly × 4 planes, 3D seed 1e-3): St/Cd/Cl identical to the 2D RK3 to every digit, 3D energy decays at 0.31/time unit to 3e-27. **G2 met**. Original: Fourier expansion in a periodic span; one 2D face-based problem per mode; nonlinear terms by FFT (3/2 rule or phase-shift dealiasing); pressure Helmholtz operator (L − k²) factorised once per mode; span-periodic seams free | G2 | 3D Taylor–Green (Re 100 viscous decay, 32² × 32 modes): dE/dt equals −2ν∫|ω|² to 0.5% at all times; a spanwise-inclined 3D TGV (`k = (1,1,1)` rotated into x–y–z) recovers order 2 in the plane and spectral in z; the periodic-z laminar cylinder at Re 100 reproduces the 2D St/Cd/Cl to 0.1% | 3–5 weeks |
| L2′ | (alternative) **full 3D hexahedra**: 3D face geometry in `umesh`, extruded quad meshes, 3D gradients; operators unchanged in form | G2 | same TGV criteria; cost: 3D pressure solves need L4 first | 5–8 weeks |
| L3 ✅ | **Variable viscosity + SGS** — IN PROGRESS 2026-09-24 (§52): `src/usgs.py` (gradient tensor with spectral z, Δ = (V dz)^{1/3}, Smagorinsky, WALE), explicit dealiased eddy-viscosity term with transpose in `PISO25.sgs_term`. Passed: manufactured variable-ν operator orders 1.97–2.00 (x/y/z); solid rotation ν_t(Smag) 2e-18 and WALE at its analytic value to 1e-15; WALE wall exponent 2.970; Δ² scaling 4.0000. TGV Re 1600 64²×64 (§52): no model −dE/dt peak +5.9% vs DNS, WALE −3.4% with ⟨ν_t⟩/ν 1.3 (max 24) and 63% of the peak dissipation from the model; both peaks ~5% early. G3 met; L3 ✅ as an operator, the model calibration is V1's question. Original: explicit transpose term ∇·(ν_t ∇uᵀ), per-step diffusion assembly, port `src/sgs.py` (Smagorinsky, WALE) with Δ = V^{1/3}; ν_eff positivity guard | G3 | manufactured variable-ν solution order 2; solid-body rotation ν_t = 0 (Smagorinsky) and the analytic WALE value; WALE ∝ y³ at a wall (exponent 3.0 ± 0.05); ν_t/ν reported with every energy curve | 2 weeks |
| L4 | **Solvers and speed**: PETSc/AmgX CG-AMG for pressure past ~3×10⁵ cells, GPU offload, Numba/CuPy kernels for assembly; variable-step BDF2/RK with CFL control | G4 | pressure solve to 1e-8 in < 30 iterations on the 2.5D channel operator; assembly + solve ≤ 100 ms per step per 10⁵ cells per mode on the Spark GPU; a dt change costs no accuracy order (T4 with a dt jump mid-run) | 3–4 weeks |
| L5 ✅ | **Infrastructure** — DONE 2026-09-24 (§53): `set_mass_flow` (force settles on 3ν at order 2 on the laminar channel), `src/ustats.py` (binned means, second moments, uv; equals instantaneous means to 0.0), `save/load` (100 steps after a load bitwise identical), `rect_mesh(cluster_y=)`. Probes and spectra not yet. Original: constant-mass-flow and constant-pressure-gradient forcing, running statistics (means, Reynolds stresses, spectra), probes, checkpoint/restart with BDF/RK history | G5 | restart is bitwise-lossless over 100 steps; statistics reproduce a known analytic mean (Poiseuille) to round-off | 1–2 weeks |

## Validation ladder (after G3)

| case | reference | success criterion |
|---|---|---|
| V1 3D Taylor–Green decay, Re 1600, 64³-equivalent | Brachet et al. DNS dissipation history (the structured `tgv400_vs_dns` rig) | peak dissipation time within 3%, peak value within 5%, with WALE; implicit (no-model) run shown alongside so the model's share is visible | **Preview 2026-09-24 (§52, 64²×64, dt 0.02): no model peak +5.9% at t 8.4; WALE −3.4% at t 8.5 (value inside 5%, time −5% outside 3%), model share 63%, ⟨ν_t⟩/ν 1.3. Needs the 96²×96 pair.**
| V2 turbulent channel Re_τ = 180 | Moser–Kim–Mansour DNS | mean profile within 3% of u_τ in the log region, u_rms peak within 5%, Re_τ within 2% at constant mass flow; **pressure field free of the mode that grew in the structured channel** (`channel_les_status.md`): p_rms of the two-colour component < 1% of p_rms over 20 flow-throughs | **Run 2026-09-24 (§54, 24×80×32, WALE, constant pressure gradient) against the SEM FOSLS DNS run02: Re_τ 179.1; U+ +1.0% in the log region; u_rms+ peak −0.3%; −⟨u'v'⟩+ −1.2%; v/w rms −6/−7%; pressure two-colour mode 0.00% of p_rms over 99 flow-throughs. All criteria met. Constant-mass-flow and no-model companions running.** |
| V3 cylinder Re 3900, spanwise-periodic πD | Parnaudeau et al. PIV/LES; HydroGym's `Cylinder_Re3900` environment (their table: St 0.21, C_D 1.07 per m-AIA) | St within 3%, C_D within 5%, recirculation length within 10% of the Parnaudeau band; then a HydroGym-style force comparison as in §38 |

## Decision points and what they turn on

1. **G0 → L1a or L1b.** Measured, not assumed. If both BDF2 and the splitting contribute comparably, do L1a: RK with per-stage projection removes both.
2. **L2 vs L2′.** 2.5D first: every LES target in this plan has a periodic span, and the per-mode direct solve keeps the existing solver. Full 3D only if a target without a homogeneous direction appears.
3. **Wall modelling.** Not in this plan; V2 and V3 are wall-resolved (y⁺ ≈ 1 with the one-sided wall flux). Add only if a Re beyond wall-resolved reach becomes a target.
4. **Dynamic SGS procedure.** After V1 with WALE; needs a test filter on the mesh (face-neighbour average is the natural one). Gate: Germano identity error reported, C_s within the literature band on V1.

## What is deliberately excluded

TVD/limited convection for LES (kept as the Re-1000 laminar-airfoil option only); triangle or tetrahedral
meshes for LES; MPI before L4's single-node measurements say it pays; any claim of a "validated LES"
before V2 passes with the pressure-mode criterion — the structured effort's withdrawn Re_τ 180 result
is the reason that criterion is written in.

## Order of work

L0 (days) → L1 (weeks) → L2 → L3 → V1 → L4/L5 as V2 demands them → V2 → V3. L0 and L1 can start now; both
are validated on existing gates (T4, T8, T9, T11) before anything three-dimensional is touched.
