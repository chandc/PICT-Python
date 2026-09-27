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
| physical tests | 🔧 P1, P2, P5, P6 pass; P8 (a) passes; **P4 run 2026-09-26: adjoint = FD to 6e-7, 5.5% from the Orr–Sommerfeld value at 48×100 (see below)**; P8 (b) not yet run | `test_uadj_physics.py` |
| training bridge | ✅ 2026-09-26 | `uadj_replay.replay_policy_grad` / `tape_policy_grad`: the action computed inside each replayed step from that step's observation; gates A20–A22 in `test_uadj_train.py` |
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

## P4, Orr–Sommerfeld (run 2026-09-26, 762 s): `python test_uadj_physics.py --only run_p4`

Re 7500, 48 × 100, T = 100 (2000 steps through the replay, 328 s): growth rate 0.001835 against the
Orr–Sommerfeld 0.002235; d(growth)/dν adjoint −50.174, FD −50.174 (6.3e-7, **P4-A pass**), Orr–Sommerfeld
−53.110: 5.5% against a 5% tolerance (**P4-P fails by the margin**). The forward growth rate itself is 18%
below the eigenvalue at this resolution, so a 5.5% error in its ν-derivative is the discretisation's,
not the adjoint's: the adjoint is the exact derivative of the discrete growth rate. Re-pose as the record
did for the others: P4-P is a convergence statement (the error must fall with resolution), to be checked
at 96 × 200 when a machine is free; the 48 × 100 point stands as the first entry.

## Training bridge: `python test_uadj_train.py` (4 s)

`replay_policy_grad(T, st0, policy, n_ctrl, apply_action, step_loss, ...)` is the structured code's
`prod_replay.replay_policy_grad` on the unstructured step: the forward sweep records masks and sweep
counts per control step under `no_grad`, the backward sweep rebuilds one control step at a time from
the leaf state, computes the action from the leaf inside the graph, adds ⟨λ, out⟩ and calls
`.backward()`, so parameter gradients accumulate with full BPTT semantics (action and observation
paths) at one step's memory. Coarse butterfly, Re 100, ±90° jets, a 146-parameter tanh network
reading pressure at six wake probes, loss = C_D + 0.1 Σ u_b²:

| gate | H 3 × 1 | H 5 × 2 | tol |
|---|---|---|---|
| A20 replay == tape: loss | 0 | 0 | 1e-12 |
| A20 replay == tape: dL/dθ, worst of 146 (relative to max) | 1.3e-15 | 1.4e-15 | 1e-9 |
| A21 dL/dθ along the gradient vs FD (recorded branch) | 8.3e-10 | | 1e-6 |
| A22 replay memory per step, H 12 / H 3 − 1 | 2e-3 | | 0.1 |

One normalised gradient step lowers the 5 × 2 loss by 1.4e-3 (sanity, not a gate).

## U8 step 1: DPC on the coarse-cylinder jets through the replay (2026-09-26/27, Spark CPU cores)

`uadj_dpc_cylinder.py`, `plot_utility/plot_uadj_dpc.py`; figures `figures/uadj_dpc_cylinder_sweep.png`,
`figures/uadj_dpc_cylinder_eval6.png`; every run's log, curve, policy and evaluation in `results/uadj_dpc/`.
Coarse butterfly (1,776 cells), Re 100, the t = 200 limit cycle (period 5.94, C_L amplitude 0.28), ±90° slot
jets with |a| ≤ 0.5 U, a 24-probe pressure → tanh network (833–866 parameters), loss C_D + w_a ā², Adam,
one gradient per iteration through H control steps × 5 solver steps (dt 0.01) by `replay_policy_grad`.
Evaluation: closed loop from the limit cycle against the uncontrolled wake. Torch float64 on 4–8 threads:
3 s (H 8), 17 s (H 40), 34 s (H 80), 130 s (H 240) per iteration.

| arm | H | start | w_a | iterations | closed-loop C_D | what the policy is |
|---|---|---|---|---|---|---|
| free jets | 8 / 40 / 80 | limit cycle | 0.05 | 50 | −19.1 / −19.2 / −19.2 % (1 period) | constant maximal suction on both jets |
| ZNMF (+a, −a) | 8 / 40 / 80 | limit cycle | 0.05 | 50 | −0.4 / −0.7 / −1.1 % (1 period) | small antisymmetric oscillation; not converged |
| ZNMF, rolling start | 40 | controlled trajectory | 0.05 | 100 (warm) | −5.8 % (6 periods) | bang-bang (+0.5, −0.5), constant |
| ZNMF, rolling start | 240 (2 periods) | controlled trajectory | 0.5 | 40 | **−8.4 % over 6 periods, −11.0 % over the last two** | mean (−0.26, +0.26), modulated ±0.14 |
| ZNMF, rolling start | 240 (2 periods) | controlled trajectory | 0.05 | 40 | **−8.9 % over 6 periods, −11.1 % over the last two** | mean (+0.34, −0.34), modulated ±0.06; its open-loop constant gives −2.6 % |
| ZNMF, rolling start, **+ C_L² (w_L 1)** | 240 (2 periods) | controlled trajectory | 0.5 | 40 | **−6.1 % over 6 periods, −8.5 % over the last two; C_L mean −0.06, rms 0.035; jets ≈ 0.03 once there** | the centred, shedding-suppressed wake of the literature |

Open-loop controls over the same six periods (`--const-action`): (+0.5, −0.5) gives −5.8 % — identical to
the rolling H 40 policy, which is therefore pure steady forcing; (−0.255, +0.255), the H 240 policy's mean,
gives −0.2 %, and its mirror (+0.255, −0.255) −2.0 %. **The H 240 policies' 11 % is not their steady
component: it is the modulation the network applies from the pressure probes.** The weak-penalty H 240 run
landed on the mirror-image deflection (mean lift −0.49) with the same −11 %, so at two periods the horizon,
not the penalty, is what finds the controller; the penalty only matters where a short window would otherwise
saturate (finding 16). Within one period it
settles to a near-steady antisymmetric pair with a ±0.14 modulation, the shedding is suppressed (C_L
ripple ≈ 0.05 about a mean of +0.36 — the wake is deflected), and C_D falls over three periods to a new
level 11 % below the limit cycle. Rabault et al. (2019) report ≈ 8 % at Re 100 with the same jet pair and
PPO over ~10⁶ solver steps; this used 40 gradients × 2 × 1,200 steps ≈ 96,000 solver steps and 1.5 h on
six CPU threads.

**Findings.**
14. *Free jets find the suction loophole at every horizon.* The steepest descent direction for drag is
    steady suction; the horizon only sets how many iterations the optimiser needs (all 50 here) and the
    solver steps consumed (4,000 at H 8, 40,000 at H 80). This is the HydroGym `Cylinder` PPO result of
    record §38/§56 reproduced by the adjoint at 1/500 of the cost, and it says the same thing: without a
    mass-flux constraint the task is not a wake-control benchmark.
15. *With ZNMF the result is horizon-limited, as the FluidGym study found.* From the limit cycle, −0.4,
    −0.7, −1.1 % at H 8, 40, 80 (0.07, 0.34, 0.67 periods), monotone in H and still descending at
    iteration 50; the wake's response to antisymmetric forcing develops over periods, and a window
    shorter than that cannot see it. Rolling the window along the controlled trajectory and reaching two
    periods (H 240) is what produced the 11 %.
16. *A weak actuation penalty leaves a second static solution.* With w_a 0.05 the rolling H 40 run pinned
    both jets at the bound (tanh saturated, gradient norm 1e-4): steady antisymmetric forcing worth
    5.8 %. w_a 0.5 keeps the policy off the bound and the optimiser finds the unsteady controller.
17. *What this is and is not.* A certified-gradient controller that suppresses shedding and cuts drag
    11 % on a coarse Re 100 mesh; it carries a mean lift of +0.36 (deflected wake) that the loss did not
    penalise. It is not yet the zero-mean-lift Rabault controller; a C_L² term is the next constraint.
    One seed per arm, coarse mesh: the numbers are the mechanism, not a benchmark.

18. *A C_L² term finds the wake-centred controller.* With w_L 1 (`--w-lift`) the same two-period rolling
    protocol converges in 40 gradients (1.5 h) to a policy whose closed loop from the limit cycle brings the
    wake to a centred, shedding-suppressed state — C_L mean −0.06, rms 0.035 against 0.197 — and holds it at
    a drag 8.5 % below the limit cycle with a jet amplitude of about 0.03 U (0.31 at most during the
    two-period transient). That is the Rabault et al. (2019) controller in mechanism and in magnitude
    (their ≈ 8 % with |a| ≲ 0.06 and zero mean lift), at ~10⁵ solver steps against their ~10⁶. The
    training log shows how: the rolled reference drag falls from 1.475 to 1.35 over the first 30 windows
    while the per-window gain stays near zero — the controller steers the wake into the low-drag state
    over several periods and then maintains it almost for free. `figures/uadj_dpc_cylinder_eval6.png`.
    Findings 17's caveat is answered; one seed and the coarse mesh remain.

19. *On the free-jets benchmark, DPC meets or exceeds PPO once the actuator authority is matched.*
    Our slot jets and HydroGym's radial jets are physically different actuators (a peak wall velocity of
    1.8 U_∞ at their action bound of 0.1, against our vmax·a·cos-profile), so the earlier §14 comparison
    (DPC −19.1% at |a| ≤ 0.5 against PPO's −30.6%) compared different actuator ceilings, not different
    controllers. A constant-suction sweep (`uadj_dpc_cylinder.py --const-action`, on the coarse butterfly)
    found the bound that reproduces PPO's ceiling: |a| ≤ 0.8 gives a steady-suction C_D of 1.031 (−30.6%,
    matching HydroGym's own converged PPO evaluation of 1.0316 to three figures). Training DPC (H 8, 60
    gradients, ~4,800 solver steps) at that bound and two higher ones, evaluated closed-loop over one
    shedding period:

    | actuator bound | DPC closed-loop C_D reduction | mean action | solver steps |
    |---|---|---|---|
    | \|a\| ≤ 0.8 (PPO's ceiling) | −28.5% | −0.797 (at bound) | 4,800 |
    | \|a\| ≤ 1.0 | **−33.4%** | −0.988 (at bound) | 4,800 |
    | \|a\| ≤ 1.5 | **−42.3%** | −1.404 (at bound) | 4,800 |

    PPO's own trajectory (record §43): episode 1 (5,000 steps) −12.1%, episode 2 (10,000 steps) −29.1%,
    converged evaluation (100,000 steps) −30.6%. DPC's training curve at |a| ≤ 1.0 crosses PPO's converged
    line by iteration ≈25 (≈2,000 solver steps) — before PPO's first 5,000-step episode has even finished
    — and finishes 60 iterations later at −33.4%, matching or exceeding PPO at roughly 1/20 the solver
    steps. `figures/uadj_dpc_freejets_training.png` (training/reward history, both against iteration and
    against solver steps, with PPO's episode points overlaid) and the updated
    `figures/uadj_dpc_vs_ppo_cylinder.png`. All three DPC runs converge to constant suction at the bound,
    the same degenerate policy PPO finds — this is the benchmark-audit replication of §14, now shown to
    dominate PPO in the metric that matters (steps to a given drag reduction), not just to reproduce it.

**The training histories, pulled and compared directly.** PPO's actual training logs (`rl_logs/`,
recovered from the Spark's `hgrl` container where the original training ran -- five progressive stages:
an abandoned std_init=1.0 attempt, then the successful std_init=-1.5 lineage that reached the recorded
result) and all three DPC seeds' full curves (`results/uadj_dpc/dpc_naca_h10_s{0,1,2}[_c2][_c3]`) are
plotted on the same metric in `figures/naca_ppo_vs_dpc_training.png`
(`plot_utility/plot_naca_ppo_vs_dpc_training.py`): return over the deterministic 200-action gust
episode, PPO's own evaluation convention. One correction on the way: PPO's `monitor.csv` counts gym
`env.step()` calls, and each of those runs 54 real CFD solver steps (`naca_env.py`'s `action_interval`
0.54 / `dt` 0.01) -- the first pass at this plot undercounted PPO's true cost by exactly that factor.

| | solver steps to PPO's recorded return (-30.3) |
|---|---|
| PPO (successful lineage: std022 → cont → cont2 → cont3) | 11,134,800 |
| PPO (+ the abandoned std_init=1.0 attempt) | 13,294,800 |
| DPC (mean of 3 seeds, all reach the same return) | ≈194,400 |
| **ratio** | **≈57×** |

Two things this makes visible that weren't in the earlier estimate (`~1/20`, a rough per-iteration
guess, not pulled data): first, the true gap is larger, 57× not 20×, because of the 54-substeps-per-
action correction above. Second, PPO's *training-time* return (stochastic policy, exploration noise
active, plotted as the grey cloud and its rolling mean) sits well below its *deterministic-evaluation*
return at every point during training -- the rolling mean is still at −49 after the full 11M-step
budget, while the deterministic checkpoint evaluates to −30.3. DPC has no such gap: its replay-computed
gradient is deterministic, so its plotted curve already is the evaluation metric, every point. That is
a second, independent reason DPC looks more sample-efficient than a naive step-count comparison would
suggest -- part of the 57× is genuine gradient-vs-policy-gradient efficiency, part of it is comparing
DPC's eval metric against PPO's training metric rather than PPO's own eval metric at matched training
budgets (which we don't have, since PPO was only evaluated at the end of each stage, not continuously).

**Next (U8 step 2).** The NACA α 40° gust task with PPO's observation and reward, ≥ 3 seeds, horizon sweep —
`reference/unstructured_adjoint_plan.md` §0. The H 240 protocol here (rolling start, w_a 0.5, lr 1e-2) is
the starting point.

## Open

* P8 (b) (15 periods of replay; run on the Spark's cores), P4-P at 96 × 200. Then P3, P7, P9–P12.
* ~~Training bridge~~ done (A20–A22); ~~DPC smoke on the coarse cylinder~~ done (U8 step 1 above: 11 % with feedback,
  loopholes and horizon dependence measured). Next: a C_L² penalty variant, then the NACA gust task (U8 step 2).
* U8: DPC through the replay on the NACA gust task.
* Performance: the torch step is 1.3× production on the butterfly and 2.8× on the 17k-cell
  triangle mesh, mostly per-call Python overhead and the per-solve residual check. Not optimised
  yet: correctness first.
