# Paper 1 — outline (draft)

*Draft 2026-09-26. Every number below is quoted from the record on the `origin/unstructured` branch
(document and section named in brackets); none was re-run for this outline. Paths such as
`reference/fluidgym_parity.md` and `figures/...` refer to that branch, not to `main`.*

---

## Working title (pick one)

1. *Horizon, not method: differentiable-physics control closes the gap to reinforcement learning
   when its gradients reach far enough — and verified on a certified solver*
2. *Verified adjoint gradients for flow control, and what current benchmarks actually reward*
3. *Long-horizon differentiable control of cylinder wakes: a replication, a benchmark audit, and a
   certified adjoint*

## Thesis in one paragraph

On FluidGym's `CylinderJet2D-easy`, differentiable predictive control (DPC) is published as
weaker than SAC (≈5.9% vs ≈6.9% drag reduction). Using the same environment, architecture and
gradient machinery, we show the ordering is set by the training horizon: H = 8 makes drag 12.6%
*worse*, the published H = 40 gives 5.8%, and full-episode H = 80 gives 6.55%, a statistical tie
with SAC. We also find that the published stack's backward linear solves silently fail to converge
and its gradients degrade with horizon, and that a popular benchmark (HydroGym's jet cylinder as
shipped) is solved by constant suction rather than feedback control. To remove the gradient
uncertainty, we build a discrete adjoint of a *validated* production PISO solver whose gradients
are finite-difference-certified end to end and whose memory stays flat in the horizon.

## Claims, and how far the evidence already goes

| # | claim | status | source |
|---|---|---|---|
| C1 | DPC's deficit to SAC on CylinderJet2D is a horizon effect: H 8 / 40 / 80 → −12.6% / +5.76% / +6.55%; SAC +6.66% (our eval of their checkpoint) | **landed, single training seed per arm** | `fluidgym_parity.md` "M0/M3 closing table" |
| C2 | Exact replication: their config in our clean-room trainer lands 0.17% in drag from their published policy; uncontrolled C_D matches their reference to five digits | **landed** | same |
| C3 | The published stack's adjoint solves hit iteration caps and return unconverged iterates; FD audit error grows 0.25% → 0.03% → 1.45% at H = 1 / 4 / 16 | **landed, one action value** | `fluidgym_parity.md` "M0 first light", "M0 results" |
| C4 | Published D-MPC artifacts give 3.76% mean reduction, not the ≈8% the paper's figure family implies | **landed, from their uploaded CSVs** | `fluidgym_parity.md` "M0 mid-flight" |
| C5 | The learned H = 80 policy is a mode-selective, phase-leading stabiliser: antisymmetric gain maps, action leads lift by ≈⅓ period, corr 0.92 | **landed** | `fluidgym_parity.md` "What the trained policy learned" |
| C6 | HydroGym's jet cylinder as shipped: PPO converges to constant maximal suction, C_D −30.6%, with no feedback; the zero-net-mass-flux variant learns nothing in the same budget | **landed** | `skew_unstructured_literature.md` §43–44 |
| C7 | A production-solver adjoint, field-for-field equal to the forward step (u 3e-9, p 5e-9), FD-exact through assembly, jets and forces (3.9e-7), memory 180 MB vs 581 MB tape at 3 steps (≈200 MB vs ≈15 GB projected at H = 80) | **landed as machinery** | `production_adjoint.md` 9.1–9.5 |
| C8 | Training *on our solver* reproduces C1 on a shedding wake | **NOT landed** — 15 noisy DPC iterations (≈2% drag dips bought with lift, `figures/dpc_shed_regimes.png`); SAC flat below baseline on the shedding mesh (`figures/sac_mid_progress.png`) | commits `ed6714d`, `ac00343` |
| C9 | The solver underneath is validated: Re 100 cylinder St 0.1673 / C_D 1.321; pinball and NACA0012 within 1–3% of HydroGym's Firedrake | **landed** | `hydrogym_backend.md`, `cylrect_r11_adjustments.md`, §45–46, pinball commits |

**The framing question to settle first.** C1 was obtained in *FluidGym's own* stack with its
ordinary tape; memory did not bind on the GB10's unified memory. So the horizon result does not
depend on our adjoint. That leaves two honest framings:

* **Framing A (recommended now):** the paper is C1–C6 — replication, horizon, gradient audit,
  benchmark audit — with C7 as "the remedy, certified", demonstrated on gradient gates and a short
  training run. Publishable today after the seed work below.
* **Framing B (stronger, later):** additionally land C8, so the headline number is reproduced on a
  certified solver. Needs the shedding-mesh training to converge, which it has not.

---

## Section outline

### 1. Introduction (≈1 page)
* Two families of learned flow control: model-free RL (Rabault et al. 2019 lineage) and
  differentiable physics (DPC / D-MPC). Benchmarks (FluidGym, HydroGym) now compare them head to head.
* The published ordering (RL ahead) is being read as a property of the methods.
* Contributions, as bullets C1, C3, C6, C7 (+ C8 under framing B).

### 2. Problem and environments (≈1 page)
* CylinderJet2D-easy: Re 100, opposing ±90° jets, one scalar action held 25 solver steps,
  reward r = C_D,ref − ⟨C_D⟩ − |⟨C_L⟩|; why the lift term dominates the uncontrolled reward.
* Table: env spec (from `fluidgym_parity.md` top table).
* HydroGym jet cylinder as shipped, and our ZNMF variant (§43–44).
* **Table 1** — environment parity: uncontrolled C_D 3.3281 ± 0.0004 vs shipped 3.3281555.

### 3. Methods (≈2 pages)
* 3.1 DPC trainer: MLP 453→64→64→1 (note: ours sees pressure, theirs velocity only — must be
  controlled for, see §7), truncated vs full-episode BPTT, γ = 0.999.
* 3.2 The gradient audit: FD on the scalar action through H steps.
* 3.3 The certified adjoint (C7): affine momentum assembly probed once per mesh (A = A₀ + T x);
  every linear map probed from the production code; residual-gated solves; replay for flat memory.
  **Figure 1** — architecture diagram (the mermaid in `fluidgym_parity.md`, redrawn).
* 3.4 Solver validation summary (C9), one paragraph plus **Table 2** (St, C_D, pinball, NACA).

### 4. Result 1 — the horizon decides the ordering (≈1.5 pages)
* **Table 3** — the closing table (uncontrolled, D-MPC artifacts, H 8, their DPC, our exact
  config, H 80, SAC ours, SAC artifacts).
* **Figure 2** — training reward vs iteration against their published log
  (`figures/fluidgym_dpc_reward_progress.png`).
* **Figure 3** — vorticity, uncontrolled vs H = 80, base-flow subtracted
  (`figures/fluidgym_jet_vorticity_h80_basesub.png`).

### 5. Result 2 — what the policy learned (≈1 page)
* **Figure 4** — linearised gain maps at sensor positions (`figures/fluidgym_h80_policy_gains.png`):
  antisymmetry, v at separation, p in the far wake.
* Phase lead of ≈2 t.u. over lift; decay of the action once shedding is suppressed.

### 6. Result 3 — gradient quality (≈1 page)
* Unconverged backward solves in the published stack (log excerpts).
* **Figure 5** — FD relative error vs horizon, theirs (0.25/0.03/1.45%) against ours (3.9e-7 for
  dC_D/d(jet), 7e-7 for dL/d(state), 8e-5 for dL/d(source) through the production step,
  `production_adjoint.md` 9.3, 9.5). Note ours are at 1–3 steps: extend to long horizons first.
* Memory: tape vs replay, measured and projected (`figures/prod_dpc_capability.png`).

### 7. Result 4 — the benchmark audit (≈1 page)
* **Table 4** — constant-actuation references (§43): suction −0.1 gives C_D 1.04, C_L rms 0.02.
* **Figure 6** — PPO training and evaluation (`figures/hydrogym_rl_eval.png`), and the fields
  (`figures/hydrogym_control_fields_shipped.png`): no vortex anywhere, mass removal.
* ZNMF: chattering policy, 0.0% change (`figures/hydrogym_rl_training_znmf.png`).
* Recommendation to benchmark maintainers: ZNMF jets plus an actuation cost.

### 8. Discussion (≈1 page)
* Horizon as the controlling hyperparameter for DPC; why truncated BPTT cannot see the delayed
  drag benefit of opposing a forming vortex.
* When memory-flat adjoints matter (larger meshes, 3D, smaller GPUs) and when they do not
  (C1 itself did not need them).
* Limits: 2D, Re 100, single scalar actuator, one geometry.

### 9. Conclusions (≈½ page)

### Appendices
* A. Exact hyperparameters of every arm; FluidGym artefact provenance (HF dataset paths).
* B. Gate tables for the adjoint (9.1–9.5).
* C. Solver validation detail (R11 table, pinball, NACA).
* D. Engineering notes needed to reproduce (container recipe, `sm_120` arch flag, weak-pointer
  keepalive for `set_state`).

---

## Work required before submission, in priority order

1. **Seeds.** ≥ 5 training seeds for H = 8, 40, 80 and our SAC re-training; report mean ± CI.
   C1 currently rests on one training seed per arm, and SAC vs H = 80 (6.66% vs 6.55%) is inside
   any plausible seed spread — so the claim must be worded "ties", never "beats".
2. **Horizon sweep.** Add H = 20, 60 (and 120 if affordable) so the paper shows a curve, not three
   points.
3. **Input-channel control.** Re-run H = 80 with velocity-only inputs (their 302). Our net also
   sees pressure; without this a referee can attribute the gain to the extra sensor channel.
4. **Rule out the cheap competitor** (`fluidgym_parity.md` "two honest gaps", item 2): gradient
   checkpointing in their stack. If H = 80 already runs there, say plainly that the horizon result
   does not need our adjoint (framing A), and give our adjoint's value as certification and memory.
5. **FD audit breadth.** Repeat the gradient audit at several action values and H up to 80, on
   several states, with both stacks.
6. **Harder tiers.** At least `CylinderJet2D-medium` (Re 250) to show the horizon effect is not a
   Re 100 accident.
7. **(Framing B only)** a converged DPC run on our shedding mesh with preconditioned solves.
8. **Contact the benchmark authors** about C3, C4 and C6 before submission. They are claims
   about others' published work and should be checked with them, not only against their artifacts.
9. **Independent re-derivation** of every number in Tables 1–4 from the raw run files.

## Threats to validity (to address in the text)

* Single-seed training (item 1).
* Different observation channels (item 3).
* The D-MPC exact-config rerun hung after 11 of 80 steps; the D-MPC number is theirs, not ours.
* The HydroGym result is for PPO defaults, one CFD step per action, (C_L, C_D) observations; the
  ZNMF "learns nothing" finding is budget- and observation-limited, not a statement about ZNMF.
* Everything is 2D, Re 100.

## Authorship, lineage, licensing

* This work builds on PICT (Franz, Wei, Guastoni & Thuerey, 2025) and runs FluidGym, which itself
  uses PICT. Check `LICENSE`/`NOTICE` obligations; consider inviting the PICT authors.
* Cite FluidGym and HydroGym for environments, checkpoints and artifacts, with dataset versions.
* Release: tag a commit, publish the trainer, the audit scripts and the figure scripts, and an
  archive of run logs sufficient to regenerate every table.
