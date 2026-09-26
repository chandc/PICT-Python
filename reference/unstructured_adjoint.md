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
| U5 gradient gates | ✅ | A10–A15 on every listed mesh (A13 bubble, A14 seam invariance, A15 every boundary type) |
| U6 forces and actuation | ✅ | `src/uadj_control.py`: wall traction, cylinder jets, HydroGym's NACA jets from the production `JetSet`, rotation, inlet modulation |
| U7 replay | ✅ | `src/uadj_replay.py`: replay == tape to 9e-14; memory flat in the horizon |
| physical tests | 🔧 P1, P2, P5, P6 pass; P8 (a) passes; P4 and P8 (b) not yet run | `test_uadj_physics.py` |
| U8–U11 | not started | |

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

| gate | butterfly quads | HydroGym triangles | tol |
|---|---|---|---|
| A10 FD vs adjoint, 1/2/3 steps, worst of 14 inputs | 3e-9 / 4e-8 / 5e-8 | 5e-8 / 9e-9 / 9e-8 | 1e-6 |
| A11 paths live | 7/7 (the butterfly is not orthogonal: non-orthogonal Poisson path 2.6 %) | 7/7 | ≥ 1e-6 |
| A12 ⟨w,Jv⟩ = ⟨Jᵀw,v⟩ | 1e-14 | 6e-15 | 1e-10 |

The whole production run takes 666 s.

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

## U5 remainder, U6, U7: `python test_uadj_control.py` (about 4 minutes)

Butterfly cylinder at Re 100 after 20 steps, unless stated.

| gate | measured | tol |
|---|---|---|
| A16 forces, torch vs the production formula | 2e-16 | 1e-14 |
| A16 dC_D/d(u, v, p) vs FD | 3e-13 | 1e-8 |
| A15 inflow u (Dirichlet) | 5e-12 | 1e-6 |
| A15 wall u / wall v (no-slip) | 1e-10 / 4e-11 | 1e-6 |
| A15 freestream v (symmetry) | 1e-12 | 1e-6 |
| A15 outlet p (Dirichlet) | 9e-10 | 1e-6 |
| A15 dL/d(u value) on the Neumann faces (freestream, outlet) | **exactly 0** | 0 |
| U6 cylinder jets ±90°, opposing and symmetric: dC_D/da, dC_L/da, d(wake v)/da | ≤ 6e-9 | 1e-6 |
| U6 rotation dC_L/dω | 2e-10 | 1e-6 |
| A17 replay == tape, 5 control steps: loss / worst dL/da_k | 0 / 9e-14 | 1e-12 / 1e-9 |
| A13 Re 40 bubble (112 cells; 159 faces with \|F\| < 1e-6 max\|F\|), recorded branch | 3e-11 | 1e-6 |
| A14 seam moved by half a period: loss / gradient field cell for cell | 0 / 3e-12 | 1e-12 / 1e-10 |
| U6 NACA α 40°: torch slot values vs production `JetSet.ramp` | 0 | 1e-15 |
| U6 NACA: forces vs `env.forces()` | 0 | 1e-13 |
| U6 NACA dC_L/da_j, j = 1, 2, 3 | 7e-11, 2e-10, 7e-11 | 1e-6 |

**Memory and cost (A18, A19).** Saved-tensor bytes plus live LU-factor bytes, butterfly cylinder:

| horizon H | tape MB | replay peak MB |
|---|---|---|
| 3 | 65 | 28 |
| 10 | 230 | 28 |
| 30 | 703 | 28 |

The tape grows linearly (10.8× from H = 3 to 30). Replay is flat to 0.2 %. On an uncontended
run the tape backward cost 0.2–0.4× its forward and a whole replay 1.0–2.0× a taped forward,
inside the A19 bars (2.5× and 3.5×). The timing columns of the second run are not quoted: the
Taylor–Green job was sharing the CPU.

## Physical tests: `python test_uadj_physics.py` (about 4 minutes)

**P1, plane Poiseuille, 200 steps from rest** (quads, ν = 0.1, f = 0.2):

| ny | Q_h | A: dQ/dν, dQ/df vs FD | P: dQ/dν = −Q/ν | P: dQ/df = Q/f |
|---|---|---|---|---|
| 8 | 1.3750000 | 5e-12, 1e-12 | 6e-10 | 1e-15 |
| 16 | 1.3437500 | 7e-12, 8e-13 | 5e-10 | 8e-16 |
| 32 | 1.3359375 | 6e-12, 5e-13 | 4e-10 | 1e-16 |

Continuum: dQ/df against 2/(3ν) is off by 3.1e-2, 7.8e-3, 2.0e-3, which is order 2.00 and
identical (to 2e-16) to the forward Q's own error. The 4–6e-10 in the ν row is the residual time
transient after 200 steps, not the adjoint: the FD agrees with the adjoint to 1e-11.

**P2, 2D Taylor–Green, T = 1, ν = 0.05, dt ∝ h:**

| n | steps | E(T) (exact 0.2046827) | A: vs FD | P: \|dE/dν + 4T E\| / \|4T E\| |
|---|---|---|---|---|
| 16 | 25 | 0.2036172 | 8e-13 | 2.0e-2 |
| 32 | 50 | 0.2046238 | 2e-12 | 4.6e-3 |
| 64 | 100 | 0.2046960 | 1e-12 | 1.1e-3 |

The adjoint through 100 time steps equals the discrete derivative to 1e-12 and converges to the
exact law dE/dν = −4T E at order 2.13.

**A first physical reading of the NACA gradients (P10, preliminary).** At a developed α 40° state
and over two steps, dC_L/da is +1.10 for the upper-surface jet, +1.24 for the nose jet and −1.07
for the lower-surface jet. Blowing on the suction side raises lift and blowing on the pressure
side lowers it. That is the direction the trained PPO policy used to hold C_L down during the gust
(suction on the upper jet, blowing on the lower, record §56). P10 itself needs the gradient at the
gust peak and over a control interval, so this is a sign check, not the test.

## Findings, continued

7. **A18 needs two counters.** Saved-tensor hooks see only tensors. The LU factors sit in the
   solve's ctx and are invisible to them, and on the tape they are most of the memory. `LUFactor`
   now keeps live and peak byte counts, and the gate adds both.

8. **Live FD probes flip upwind decisions in the bubble.** A 1e-3 probe flips 12 upwind decisions
   over two steps at Re 40. Unreplayed, those probes would compare two different discrete
   operators. With the branch recorded and replayed, the bubble gradient is FD-exact to 3e-11.
   The recorded masks are what make gradients through recirculation testable at all.

9. **A14 was first posed as a ratio and failed at 2.0 against 10.** It asked that the gradient
   across the seam be ten times the mid-domain gradient. That was mis-posed: the pressure solve
   couples every cell in one step, so the ratio says nothing about the seam. It is replaced by an
   exact invariance: the same box with the seam moved by half a period must give the same loss
   and the same gradient field cell for cell. Measured 0 and 3e-12.

10. **P1's continuum criterion was first posed as ≤ 1e-3 at ny = 32 and failed at 1.95e-3.** That
    is exactly the forward solver's discretisation error of Q: Q_h = Q(1 + 2/ny²), from the
    one-sided wall face. The plan's claim that the FV parabola is exact on uniform quads was
    wrong. The gradient inherits the forward error and adds nothing (equal to 2e-16), so the
    criterion is now that equality plus order ≥ 1.8.

11. **P2 was first run at fixed dt and gave order 1.61 from 32² to 64², failing 1.8.** The plan
    poses the refinement at fixed dt/h. At fixed dt, the O(dt) Rhie–Chow damping puts a floor
    under the error. At fixed dt/h the order is 2.13.

## P5, P6, P8: `python test_uadj_physics.py --only run_p5_p6` / `--only run_p8`

**P5, exact symmetry zeros** at the steady symmetric Re 40 wake. The coarse butterfly's nodes are
snapped to exact mirror pairs (`uadj_cases.mirror_nodes`; they sat ≤ 4e-4 off), and the state is
mirror-symmetric to 2e-14, with C_D 1.824 and C_L 6e-15.

| N steps | dC_L/da_sym ÷ dC_L/da_anti | dC_D/da_anti ÷ dC_D/da_sym | dC_D/dω ÷ dC_L/dω |
|---|---|---|---|
| 2 | 3e-14 | 8e-15 | 5e-15 |
| 20 | 1e-14 | 9e-16 | 1e-14 |

The non-zero ones equal FD to 2e-8 (jets) and 1e-10 (rotation).

**P6, steady symmetric blowing over 10 convective times** (500 steps, through the replay):
dC_D/da_sym = +0.27626, FD +0.27626 (5e-7). Blowing raises the drag and suction lowers it.

**P8, the Re 100 limit cycle** (coarse butterfly spun up to t = 200, C_L amplitude 0.278 constant
to 4 digits since t = 75, state in `results/uadj_shed_cylinder_butterfly_coarse.npz`; period 5.939,
St 0.168, coarse mesh, reported only):
* (a) time shift, N = 100: ⟨∇_{x0} C_L(t_N), x₁ − x₀⟩ = 3.4963e-4 against C_L(t_{N+1}) − C_L(t_N) =
  3.5007e-4, 1.3e-3 apart (tol 1e-2). With half the shift the error falls by 2.06 (a linearisation
  error, as it must be). **Pass.**
* (b) gradient norm over 1–8 periods: 1 period gives ‖∇‖ = 0.911. The 2-, 4- and 8-period runs
  were stopped: with three jobs on four cores one period took 49 min. **Not yet run.**

**P4, Orr–Sommerfeld: not yet run.** Its first attempt hit its own 7,000 s timeout under CPU
contention. The re-run was queued behind the regression, and the queue never started it: its
wait loop's `pgrep -f` matched its own command line.

## Findings, continued (2)

12. **A symmetric state sits on the upwind kink, and the adjoint must take the midpoint there.**
    On faces crossing the symmetry axis v = 0, so F = 0 exactly, and F·φ_upwind has one-sided
    derivatives φ_owner and φ_neighbour, which differ at O(1). The branch round-off picked made
    dC_D/dω 5.7e-4 of dC_L/dω, while the forward keeps the symmetry to 2e-11. The SIMPLEC floor
    has the same kind of tie in every cell whose viscous row sum cancels. At ties (|F| ≤ 1e-12
    max|F|, row sum within 1e-12 of its floor) the derivative is now the average of the two
    one-sided derivatives. The forward value is untouched (`uadj_ops.st_mask`), and FD probes
    re-decide ties live so the central difference straddles the kink (`Masks.straddle`; off for
    the linearity gate A4). Before this change P6's 500-step gradient differed from FD by 5e-3,
    and after it by 5e-7. The full regression over the change passes (ops, step, control,
    physics).

13. **`cylinder_bf2_coarse` is not a flow case.** Symmetric as written, but it is the MMS mesh on
    a different domain (record §S13), and production gives C_D < 0 within 100 steps at Re 40.

## Running the rest locally

    python test_uadj_ops.py                        # 1 s
    python test_uadj_step.py                       # 35 s  (--production: + 11 min)
    python test_uadj_control.py                    # 5-10 min (needs gymnasium)
    python test_uadj_physics.py                    # P1, P2, P5, P6 (about 10 min)
    python test_uadj_physics.py --only run_p4      # Orr-Sommerfeld, est. 30-60 min alone
    python test_uadj_physics.py --only run_p8      # limit cycle; (b) is 15 periods of replay

Dependencies: numpy 2.0, scipy 1.13, torch, pyamg, gymnasium.

## Open

* P4 and P8 (b), commands above. Then P3, P7, P9–P12.
* Training: port `replay_policy_grad` (the action computed inside each replayed step from that
  step's observation) with a replay == tape gate, then a DPC smoke run on the coarse cylinder's
  jets before the NACA gust task.
* U8: DPC through the replay on the NACA gust task.
* Performance: the torch step is 1.3× production on the butterfly and 2.8× on the 17k-cell
  triangle mesh, mostly per-call Python overhead and the per-solve residual check. Not optimised
  yet: correctness first.
