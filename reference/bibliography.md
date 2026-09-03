# Bibliography

Every external result this repo relies on, in one place, so a claim in the code can point at a
source rather than at a surname.

**Provenance is marked, because it matters here.** Two entries were read directly from the
source during this work and their numbers verified against the tables. The rest are standard
references given from knowledge: author, year, title and journal are given, and **exact volume
and page numbers are omitted where they were not checked**, because an invented page number is
worse than a missing one. Anyone using these for a publication should verify the unmarked
entries.

Legend: **[V]** verified against the source in this repo's work · **[U]** unverified details

---

## Benchmarks this repo is validated against

**[V] Sohankar, A., Norberg, C. & Davidson, L. (1998).** *Low-Reynolds-number flow around a
square cylinder at incidence: study of blockage, onset of vortex shedding and outlet boundary
condition.* International Journal for Numerical Methods in Fluids **26**(1), 39–56.
[PDF](https://www.cfd-sweden.se/lada/postscript_files/Sohankar_num-fluids.pdf) ·
[publisher](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0363%2819980115%2926%3A1%3C39%3A%3AAID-FLD623%3E3.0.CO%3B2-P)

Tables II–V read directly. Used in `reference_data.py` and `square_cylinder_vortex_street.md`
§6.1 for the matched 5%-blockage comparison: St 0.146, C_D 1.460, C_L′ 0.139, −C_pb 0.661,
C_ps 1.052, L_r 2.20, and Re_cr = 51.2 ± 1.0. **The zero-blockage experimental estimate of
47 ± 2 quoted therein is NOT the right comparison for a 5%-blockage grid** — see
`measurement_traps.md` §1.

**[U] Ghia, U., Ghia, K.N. & Shin, C.T. (1982).** *High-Re solutions for incompressible flow
using the Navier–Stokes equations and a multigrid method.* Journal of Computational Physics.
Used by `test_*cavity*` / `plot_ghia*.py`.

**[U] Armaly, B.F., Durst, F., Pereira, J.C.F. & Schönung, B. (1983).** *Experimental and
theoretical investigation of backward-facing step flow.* Journal of Fluid Mechanics. Used by
`plot_armaly_bfs.py` and the BFS cases.

**[U] Kim, J., Moin, P. & Moser, R. (1987).** *Turbulence statistics in fully developed channel
flow at low Reynolds number.* Journal of Fluid Mechanics. The Re_τ = 180 channel DNS that the
van Driest profile work is aimed at.

**[U] Hoyas, S. & Jiménez, J. (2006).** *Scaling of the velocity fluctuations in turbulent
channels up to Re_τ = 2003.* Physics of Fluids. The channel reference PICT compares against.

---

## Numerical method

**[U] Issa, R.I. (1986).** *Solution of the implicitly discretised fluid flow equations by
operator-splitting.* Journal of Computational Physics. The PISO algorithm.

**[U] Rhie, C.M. & Chow, W.L. (1983).** *Numerical study of the turbulent flow past an airfoil
with trailing edge separation.* AIAA Journal. The collocated-grid momentum interpolation in
`src/piso_multiblock.py` and `reference/rhie_chow_ddt_instability.md`.

**[V, in-repo] Dong, S., Karniadakis, G.E. & Chryssostomidis, C. (2014).** *A robust and
accurate outflow boundary condition for incompressible flow simulations on severely-truncated
unbounded domains.* Journal of Computational Physics **261**, 83–105. The outflow condition in
`_dong_nodes`; already cited in `reference/outflow_bcs.md` §"dong", including this repo's
measurement that its regularised switch leaves a spurious near-wall traction of order
`U₀² δ²`.

**[U] Maliska, C.R.** *Heat Transfer and Fluid Mechanics Computation.* The curvilinear
non-orthogonal treatment PICT's appendix follows (their eqs. 12.184, 12.193–12.195).

---

## Subgrid-scale modelling

**[U] Smagorinsky, J. (1963).** *General circulation experiments with the primitive equations:
I. The basic experiment.* Monthly Weather Review. The eddy-viscosity closure
`ν_t = (C_s Δ)² |S|` in `src/sgs.py`.

**[U] Lilly, D.K. (1967).** *The representation of small-scale turbulence in numerical
simulation experiments.* IBM Scientific Computing Symposium on Environmental Sciences. Source of
`C_s ≈ 0.17` for isotropic turbulence — the value `CS_SMAGORINSKY` uses.

**[U] Nicoud, F. & Ducros, F. (1999).** *Subgrid-scale stress modelling based on the square of
the velocity gradient tensor.* Flow, Turbulence and Combustion. The WALE model and `C_w = 0.55`
in `src/sgs.py`. Its `y³` near-wall behaviour is measured here at **2.997**
(`test_sgs_models.py`).

**[U] van Driest, E.R. (1956).** *On turbulent flow near a wall.* Journal of the Aeronautical
Sciences. The damping function `1 − exp(−y⁺/A⁺)` with `A⁺ = 26`. **Note the correction recorded
in this repo:** applied to the mixing length it gives `ν_t ~ y⁴`, not `y³` — measured at 3.993 —
so "van Driest fixes the near-wall scaling" is false as usually stated.

**[U] Germano, M., Piomelli, U., Moin, P. & Cabot, W.H. (1991).** *A dynamic subgrid-scale eddy
viscosity model.* Physics of Fluids A. The dynamic procedure named as future work in
`les_implementation.md` §6; needs a test filter, i.e. the multi-block halo.

**[U] Comte-Bellot, G. & Corrsin, S. (1971).** *Simple Eulerian time correlation of full- and
narrow-band velocity signals in grid-generated, "isotropic" turbulence.* Journal of Fluid
Mechanics. The decaying-isotropic-turbulence spectra proposed as the first real LES validation.

---

## The solver this port descends from

**[V] Franz, A., Wei, H., Guastoni, L. & Thuerey, N. (2025).** *PICT – A Differentiable,
GPU-Accelerated Multi-Block PISO Solver for Simulation-Coupled Learning Tasks in Fluid
Dynamics.* arXiv:2505.16992v2. [PDF](https://arxiv.org/pdf/2505.16992) ·
[journal](https://www.sciencedirect.com/science/article/pii/S0021999125007156) ·
[code](https://github.com/tum-pbs/PICT)

Read directly for `reference/les_learning_plan.md`. Quotations used there, with their location:

| statement | where |
|---|---|
| "we train an SGS model in form of a corrector `G_θ` … a correcting force `S_θ` at a low spatial resolution of 64 × 32 × 32" | §5.3, p. 16 |
| "`G_θ` receives the instantaneous velocity and the normalized wall distance `1 − \|y/δ\|` as inputs" | §5.3, p. 16 |
| "a simple CNN with layers using 8, 64, 64, 32, 16, 8, 4, and 3 filters … 198931 trainable parameters" | §5.3, p. 17 |
| `L = L_stats + λ_S (1/N) Σ ‖S_θⁿ‖²₂` (their eq. 16); forcing constrained to [−2, 2] | §5.3, p. 17 |
| "To ensure divergence free flow motions, we include the gradient modification from eq. (11) for `S_θ`" | §5.3, p. 17 |
| "we exclude the gradients of the linear solves from the optimization … include the terms at a later" | §5.3, p. 17 |
| "The differentiable quantities are `u`, `ν`, `ρ`, and `S` … as well as any derived intermediate quantities like matrices and RHS of the linear systems" | App. A.5, p. 27 |
| "training the SGS model while only supervising in terms of velocity moments" | §1, p. 2 |
| Re_τ at target 550: no SGS 390, Smagorinsky 452, learned CNN 548 | Fig. 11, p. 18 |

**A caution worth recording.** An automated summary of this same PDF reported that the learned
model outputs an *eddy viscosity*. It does not — it outputs a correcting force, as the first row
above shows. The summary's "quotes" were paraphrase. Read the source.
