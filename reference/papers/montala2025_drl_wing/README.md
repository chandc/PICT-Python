# Montalà et al. (2025) — DRL active flow control, 3D NACA0012 wing at Re = 1,000

Archived 2026-10-03 from arXiv. **License: CC BY 4.0** (confirmed on the arXiv abstract page), which
permits redistribution with attribution — this is why a copy can live in this public repo at all.

## Citation

R. Montalà, B. Font, P. Suárez, J. Rabault, O. Lehmkuhl, R. Vinuesa and I. Rodriguez (2025).
*Deep Reinforcement Learning for Active Flow Control around a Three-Dimensional Flow-Separated Wing
at Re = 1,000.* arXiv:2509.10195v1 [cs.CE], submitted 12 September 2025.
DOI: https://doi.org/10.48550/arXiv.2509.10195 ·
[abs](https://arxiv.org/abs/2509.10195) · [html](https://arxiv.org/html/2509.10195v1)

Affiliations: UPC (Montalà, Rodriguez), TU Delft (Font), KTH FLOW (Suárez, Vinuesa), independent
(Rabault), Barcelona Supercomputing Center (Lehmkuhl). Presented as *Proceedings of the 1st
International Symposium on AI and Fluid Mechanics*; 12 pages, preprint, no journal reference listed.

## Files

* `2509.10195v1.pdf` — the paper as served by arXiv (12 pages, 11 MB; figures are the bulk).
* `2509.10195v1.html` — arXiv's HTML rendering, for text search without opening the PDF.

## What it does

PPO agent driving active flow control on a **three-dimensional NACA0012 at Re = 1,000, AoA = 20°**
(fully separated, vortex-shedding regime). Domain 32c × 30c × 3c. Actuation is two sets of three
jets, at the leading edge (x/c = 0.01) and downstream (x/c = 0.4); the rear jets are constrained to
the opposite sign of the front ones. Reward is drag reduction penalised by lift oscillation.

Framework: the **GPU-accelerated spectral-element CFD solver SOD2D** coupled to **TF-Agents** through
a **Redis in-memory database** (SmartSim), i.e. a Fortran-solver/Python-DRL bridge rather than an
in-process differentiable stack. Policy net: two hidden layers of 512. Training ran **10 parallel CFD
simulations × 3 pseudo-environments = 30 trajectories per action**, stopped at **67 episodes** when
the reward plateaued.

## Results, as the paper states them — and one number that does not check out

Baseline validation (their Fig. 3) is against Gupta et al. and Kouser et al. (2D and 3D), for C_l,
C_d and St versus AoA; their uncontrolled point sits on those curves at AoA = 20°.

| quantity | paper's claim | checked against their own figures |
|---|---|---|
| mean drag | **ΔC_d = −21%** | **consistent** — Fig. 5b shows baseline C_d ≈ 0.40 vs controlled ≈ 0.315, i.e. −21% |
| lift RMS | **ΔC_l,rms = −124%** | **NOT reproducible** — see below |
| shedding Strouhal | St 0.56 → 0.64 | consistent with Figs. 3c/3d; the learned actuation's dominant frequency locks to the controlled St = 0.64 (Fig. 6b) |

**The −124% lift-RMS figure appears to be an error in the paper, and should not be cited without
checking with the authors.** It is printed twice — in the body (p. 6: "the root-mean-square of the
signal has been noticeably reduced (ΔC_l,rms = −124%)") and in the Conclusions (p. 9: "the
root-mean-square of the lift coefficient is reduced by 124%"). A >100% reduction in a
root-mean-square — a strictly non-negative quantity — is not possible under any normalisation where
"reduction" means what it usually means. Their own Fig. 4a shows C_l,rms falling from ≈0.20 to ≈0.115
across training (≈43%), and the signals in Fig. 5a imply a fluctuation-RMS reduction of roughly
70–75%. None of those routes yields 124%. Most likely a normalisation or typesetting slip; flagged
here rather than propagated, per `reference/bibliography.md`'s standing rule that a wrong number is
worse than a missing one.

## Why this is archived here

Directly adjacent to this repo's own control work, and useful as an external reference point for it:

* **Same geometry family, harder case.** This repo's DPC-vs-PPO sample-efficiency result (claim C10
  in `reference/paper1_outline.md`) is on a 2D NACA0012 gust-rejection task; this is a 3D separated
  wing with jet actuation. It is the natural "next tier" configuration if that line is extended.
* **A PPO baseline from the RL-for-flow-control community itself**, with authors (Rabault, Vinuesa)
  whose earlier work is already the lineage this repo's benchmark audits sit in (C3/C4/C6).
* **Sample cost is reported only in episodes (67), never in solver steps**, which is exactly the
  reporting gap C10's methodology note is about — their 67 episodes × 30 trajectories hides however
  many CFD steps each trajectory actually ran, so no cross-method efficiency comparison is possible
  from the paper as published. Worth citing as evidence that the gap is general, not specific to the
  HydroGym logs this repo had to reconstruct by hand.
