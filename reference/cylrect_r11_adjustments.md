# Everything R11 needed to work: the adjustment ledger

The butterfly-grid shedding validation (`cylrect_r11_spark`, St 0.1673 / C_D 1.321,
see `oring_rect_design.md`) looked from the outside like one run. It was eleven
launches. This file records every adjustment between "grid validates, smoke test
passes" and "18 h production run, zero incidents", in causal order, so the next
campaign on a new grid does not rediscover them. Grid-construction fixes (weight
rescaling, two-pass radial counts, seam dx matching) are already in
`oring_rect_design.md`; measurement discipline is in `measurement_traps.md`
(traps 21-24 were earned here).

## The one PHYSICS adjustment (the actual cure)

**Orthogonal-only pressure projection is unstable on this grid.**
`MultiBlockPISO` defaults to `implicit_cross=False`: the pressure operator and the
flux correction drop the non-orthogonal cross terms. On near-orthogonal grids
(channel, O-grid, square-cylinder) that approximation is benign. On the butterfly
grid's sheared trapezoid corner cells (min J 3e-5) it leaves unprojected divergence
in exactly those cells every step; the field doubles per step from t~0.1 (east
trapezoid first, N/S symmetric pair next) and blows up by step ~17. Measured with
per-step per-block max|vel| tracing after every linear solve was verified exact --
and independently fatal on pure scipy, so no solver was ever the root cause.

Fix: `PICT_IMPLICIT_CROSS=1` -> `_solve_cross_dc` in `src/piso_multiblock.py`,
**deferred correction**: each sweep = one orthogonal solve on the fast backend
(AmgX) + one cross-flux evaluation on the lagged pressure; iterate to relative
change `PICT_CROSS_DC_TOL` (1e-3) or `PICT_CROSS_DC_ITERS` (6). The pre-existing
matrix-free full-operator path (`_solve_cross`) is exact but pays one cross
evaluation PER KRYLOV MATVEC -- observed 70+ minutes inside a single solve -- and
is kept only behind `PICT_CROSS_DC=0`. Production cost of the cure: 2.17 s/step
vs ~1.4 s/step orthogonal-only (which does not survive to step 20).

## The AmgX adjustments (all real bugs, none the root cause)

These were found because the blow-up's early phase makes the momentum operator
Krylov-hostile before the field visibly explodes, and every latent solver defect
fired at once. Each is worth having regardless of grid.

1. **Check the solve status** (`src/amgx/binding.py`). `AMGX_solver_solve`'s
   return code reports API health only; a DIVERGED solve returns rc 0 and a
   garbage vector. R10 ran 2 h at 90% GPU on NaN fields because nobody asked.
   Now `AMGX_solver_get_status` is read after every solve and an unhealthy solve
   raises with the true config, rtol, iterations, and NaN census.

2. **Never let AmgX write into the caller's x0** (`binding.solve`).
   `np.ascontiguousarray` does NOT copy an already-contiguous array, so
   `AMGX_vector_download` wrote the (possibly failed) solution INTO the warm-start
   buffer in place. One rejected solve then poisoned every retry, every fresh
   solver, and the full context reset -- a single-solve failure masqueraded as
   permanent process corruption for hours. `x` is now an explicit copy.

3. **Enforce convergence in the caller as ||b - Ax|| <= rtol ||b||**
   (`src/linsolve.py::_amgx_solve`). AmgX's `RELATIVE_INI_CORE` measures against
   the INITIAL residual, and it monitors the PRECONDITIONED residual -- both lie.
   A good warm start makes "reduce by 1e-9" an unattainable sub-machine-precision
   target: the solver grinds to max_iters on an already-exact solution, and
   BiCGStab breakdown on the floor residual manufactures NaN. Now: one SpMV skips
   the solve when the warm start already meets tolerance; a "not converged" solve
   whose TRUE residual meets tolerance is accepted; anything else is rejected no
   matter what AmgX's status claims. `pbicgstab_jac_1e9.json` max_iters 20000->500
   so a doomed attempt costs ~1 s, not 40.

4. **Escalation chain on a rejected solve** (`_amgx_solve`):
   reconstruct the solver -> full AmgX context reset (`binding.reset()`: close
   every solver via the `linsolve._all_caches` registry, destroy the per-tolerance
   configs and shared resources, `AMGX_finalize`, re-initialize) -> **cached-splu
   direct solve** with a cooldown (`PICT_AMGX_COOLDOWN`, production 600) so a hard
   stretch is not re-litigated through the whole chain on every solve. splu is the
   refuge because the pre-blow-up momentum matrices defeat every AmgX
   Krylov/preconditioner combination while direct factorisation solves them to
   1e-16 (value-bisection between the step-1 and step-20 matrices shows a smooth
   conditioning cliff: 2 -> 9 -> 38 -> 132 iters -> divergence). The factorisation
   is reused while A's values are bit-identical (correctors share one matrix).
   In the final production run this machinery was armed and NEVER fired.

5. **NaN fail-fast at solve entry** (`_solve_timed`): a NaN system never
   converges, it grinds to max_iters at full GPU -- ~0.5 ms of `isnan` per solve
   converts hours of silent grinding into a stack trace naming the step.

6. **Report the config actually loaded** (`binding.config_used`). Errors used to
   print the pre-routing template path (`AMGX_CONFIG`), which sent the
   investigation down a config-routing rabbit hole while the tight
   (`AMGX_CONFIG_TIGHT`) substitution was working fine all along.

7. **Momentum solver family**: PCG collapses on the nonsymmetric rect-grid
   momentum operator (280k iterations); PBICGSTAB + BLOCK_JACOBI is the tight
   config (`pbicgstab_jac_1e9.json`); a MULTICOLOR_DILU variant
   (`pbicgstab_dilu_1e9.json`) is kept for harder operators.

## Operational adjustments

- **`python -u` for every containerised run**, log to a file, never through a
  pipe. R10's first relaunch showed a 2 h-stale log because Python block-buffers
  redirected stdout while the C library's prints (unbuffered) got through --
  worst possible combination: the log looked alive and said nothing.
- **Verify the remote copy actually has the fix before attributing its failure**
  (trap 21): the "fixed" R10 relaunch ran with a stale rsync; md5 of every
  touched file on both ends is now part of the launch ritual.
- **Diagnostics that stay in the tree**: `PICT_SOLVE_STATS=N` per-cache solve
  stats; `AMGX SLOW` warning above 2000 iterations (iteration growth is the
  leading indicator of field degradation); `PICT_DUMP_SOLVE=<dir>` dumps each
  distinct system once AND every finally-rejected system (`failing_*.npz`) for
  offline replay. The Spark working copy keeps the probe harnesses
  (`growth_probe.py` per-step per-block max|vel|; `solve_trace.py` per-solve
  warm-start/result/residual trace) that localised the blow-up to block 1.
- **Seeing the fields truthfully**: `plot_utility/plot_rect_vorticity.py` --
  metric-aware omega_z (per-node Jacobian inversion; per-axis gradients are
  garbage on curvilinear blocks and painted a fake instability over a healthy
  startup field), seam stitching (single-owner seam nodes leave one-cell unpainted
  strips that look like cracks), `--grid` overlay.

## The launch that worked

```
docker run -d --rm --gpus all \
  -v $HOME/amgx:/tmp/AMGX/build -e AMGX_LIB=/tmp/AMGX/build/libamgxsh.so \
  -e AMGX_CONFIG=/w/src/amgx/pcg_agg_1e6.json \
  -e AMGX_CONFIG_TIGHT=/w/src/amgx/pbicgstab_jac_1e9.json \
  -e PICT_IMPLICIT_CROSS=1 -e PICT_AMGX_COOLDOWN=600 -e PICT_SOLVE_STATS=2000 \
  -v $HOME/pict_sqcyl:/w -w /w pict-amgx:1.0 \
  sh -c "python -u run_cylinder_rect.py --backend amgx --tol 1e-6 \
         --tag cylrect_r11_spark > results/logs/cylrect_r11_spark.log 2>&1"
```

8000 settle + 30000 shed, 2.17 s/step, 18 h, restarts every 500 steps, zero
incidents. Commits: `c518e44` (fixes), `c266cb9` (verdict), `92bc7d1`
(literature table).
