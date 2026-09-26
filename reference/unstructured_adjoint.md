# Unstructured adjoint: the running record

*The measured record of `reference/unstructured_adjoint_plan.md`, in the style of
`production_adjoint.md`. Every number below was produced by the command next to it on
2026-09-26 (CPU, float64, torch 2.14, numpy 2.0.2, scipy 1.13.1).*

## Status

| stage | state | what landed |
|---|---|---|
| U0 decisions | ✅ | superset pattern; detached, recorded and replayable branch masks; Poisson sweep counts recorded and replayed; float64 CPU |
| U1 constant operators | ✅ | `src/uadj_ops.py`: `TMesh`, `SparseConst`, `TGradient` (LSQ and dual GG from the production matrices) |
| U2 solution-dependent operators | ✅ | convection (matrix and deferred rhs), Laplacian (matrix and deferred rhs, linear in γ), SIMPLEC, Rhie–Chow with the dt-independent history, pressure-correction flux, `BC.effective` with differentiable boundary values |
| U3 solves | ✅ | `src/uadj_solve.py`: SuperLU factor reused forward, **transpose solve** backward, pinned-row singular systems adjointed exactly, a **forward-mode rule** (needed by A12 now and P7 later), residual certificate that raises |
| U4 BDF2 step | ✅ | `src/uadj_step.py` `TorchUPISO`: equal to `PISO.step()` to round-off on six meshes including two production cylinder meshes |
| U5 gradient gates | 🔧 partly | A10 (state, body force, boundary values, ν), A11 (seven path cuts), A12 done on every mesh; A13–A15 open |
| U6–U11 | not started | |

## Gate results

`python test_uadj_ops.py` (0.5 s) and `python test_uadj_step.py` (30 s) are the fast suite: four
meshes, every gate.

| gate | cavity_quads (M1) | cavity_tris (M3) | channel_open (M3′) | channel_periodic (M7) | tol |
|---|---|---|---|---|---|
| A1 gradients vs production | 0 | 0 | 0 | 0 | 1e-14 |
| A2 ⟨y,Ax⟩ = ⟨Aᵀy,x⟩ | 6e-17 | 1e-16 | 2e-16 | 3e-17 | 1e-13 |
| A3 dual-GG duality | 0 | 7e-16 | 4e-16 | 4e-16 | 1e-13 |
| A4 linearity (worst of three) | 1e-15 | 1e-15 | 2e-15 | ≤ 1e-15 | 1e-13 |
| A5 operators vs production (worst of four) | 2e-16 | 2e-16 | 1e-16 | ≤ 2e-16 | 1e-14 |
| A6 LU solve dL/dA, dL/db vs FD (worst) | 1e-10 | 8e-11 | 1e-11 | 2e-11 | 1e-8 |
| A7 residual certificate raises | ✓ | ✓ | ✓ | ✓ | — |
| A8 step vs production, 1 step | 5e-16 | 4e-16 | 2e-14 | 1e-13 | 1e-10 |
| A8 over 10 steps | 4e-15 | 6e-15 | 7e-14 | 8e-13 | 1e-9 |
| A10 FD vs adjoint, 1/2/3 steps, worst of 14 inputs | ≤ 3e-9 | ≤ 3e-9 | ≤ 2e-9 | ≤ 4e-9 | 1e-6 |
| A11 every path live | 6/7 + N/A | 7/7 | 7/7 | 6/7 + N/A | ≥ 1e-6 change |
| A12 ⟨w,Jv⟩ = ⟨Jᵀw,v⟩ | 0 | 2e-16 | 2e-16 | 2e-15 | 1e-10 |

Production meshes, `python test_uadj_step.py --production cylinder_butterfly cylinder_tris`
(the T9 set-up, `run_ucylinder.py`'s boundary conditions):

| mesh | cells | A8 after 1 step | A8 over 10 steps | torch / production per step |
|---|---|---|---|---|
| `cylinder_butterfly_coarse.msh` (quads) | 1,776 | 3.1e-14 | 7.7e-13 | 32 / 24 ms |
| `cylinder_medium.msh` (HydroGym's triangles) | 17,258 | 4.6e-14 | 4.1e-13 | 749 / 266 ms |

(A10–A12 on these two meshes: see the next section once the run completes.)

## Findings

1. **The torch step is the production step on the first attempt.** No discrepancy had to be
   hunted. That is the dividend of the unstructured solver being written as explicit matrices
   and short face-array functions: the transcription is mechanical and every operator could be
   checked against its production twin on its own (A1, A5) before the step was assembled.

2. **A10's FD step must be scaled to the field, not to the input.** The first run failed on
   8 of 12 rows. Every failing input was identically zero (the body force in the cavity, the
   wall values), so a step of 1e-6 × max|input| fell back to 1e-9 and round-off dominated. With
   the step scaled to the velocity (the body force to ν·U) and swept over four decades, the error
   follows h² at large h and ε/h at small h on every input, the signature of a correct adjoint.
   A wrong adjoint would show at every h. The gate reports the best of the sweep and prints all
   four with `-v`.

3. **Two of the operator gates were posed wrongly at first, and both corrections are physics.**
   * A3: the dual-GG duality Σ V u·Gp + Σ V p Du = 0 holds only with the boundary pairing the
     projection uses: p_b = p_owner in the gradient and zero flux in the divergence. Each interior
     face then gives (u_O p_O − u_N p_N)·S_f, which telescopes by cell closure. With p_b = 0 the
     boundary faces break the closure and the identity is off by O(1).
   * A2: normalising by |⟨y, Ax⟩| divided by a near-cancelling sum. It is now normalised by
     Σ|y_i A_ij x_j|, the scale of the terms.

4. **A6 on the Dirichlet pressure operator needed a fourth-order FD stencil.** The second-order
   stencil's h² truncation sat at 3e-8. The fourth-order stencil gives 1e-11.

5. **The non-orthogonal Poisson path is absent, not dead, on orthogonal meshes.** Cutting it
   changes the gradient by 1e-15 on uniform quads, because T_f ≡ 0 there. A11 reports it as N/A
   when max|T_f| = 0, and exercises it on the skewed meshes, where it moves the gradient by 4–8 %.

6. **What the cuts show about where the gradient lives** (A11, 2 steps, change in the gradient
   norm when the path is cut):

   | path | quads | skewed tris | open channel |
   |---|---|---|---|
   | Rhie–Chow dt-independent history | 100 % | 98 % | 94 % |
   | BDF2 history | 6 % | 6 % | 8 % |
   | deferred momentum terms | 2 % | 7 % | 9 % |
   | convecting flux → momentum matrix | 1 % | 2 % | 16 % |
   | aP → Rhie–Chow D | 0.1 % | 0.2 % | 0.3 % |
   | aC → pressure operator | 5e-6 | 2e-4 | 1e-3 |

   The history row is close to 100 % because it contains the only path to two of the state
   inputs (Ff_old, Fbar_old). The aC row is small because the SIMPLEC coefficient sits on its
   floor a_t V in most cells. That is the kink the recorded masks exist for.

## Open (U5 remainder, then U6)

* A13: gradients in a recirculation bubble (faces with F ≈ 0) with switch counts reported.
* A14: a seam-crossing source/loss pair on M7 (the seam is already inside A8/A10/A12, but not
  isolated).
* A15: per-boundary-type rows, including the jets.
* Performance: the torch step is 1.3× production on the butterfly and 2.8× on the 17k-cell
  triangle mesh. Most of it is per-call Python overhead in `index_add`/`index_copy` and the
  per-solve residual check. Not optimised yet: correctness first.
* U6: the wall-traction force routine in torch and actuation through boundary values.
