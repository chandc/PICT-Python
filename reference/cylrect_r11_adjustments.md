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

## Post-verdict finding: unclaimed seam-endpoint corner nodes (fixed)

The R11 vorticity figures showed small blemishes on the lateral boundaries near
x = 7. Verified against the raw fields: the trapezoid diagonal seam rays end
exactly at (X_HAND, +-Y_HALF) and (-X_IN, +-Y_HALF), and the E/W trapezoids that
OWN those ray endpoints have all four in-plane faces CONNECTED -- so wall_mask
(which skips connected faces) never marks those nodes and no BC ever claims them,
even though they sit geometrically ON the domain boundary. Each such column
settles at a fixed point of the edge-replicated corner discretisation: u = 1.41
at (7, +-10), frozen from t = 80 to t = 380. Consequences: (a) the run's
far-field metric floor of 0.4157 was this node all along, not wake physics --
the metric still worked as a GROWTH detector, but its resting value was the
defect; (b) about half the visible blemish size was plot-side (the seam
stitching's edge-replicated pad corner inflates local |omega| 2.9 -> 12.3).
The W-trap endpoints coincide with the inlet corners and stayed ~1.0 naturally.

Fix (`cylinder_rect_bc.py`): `seam_endpoint_columns()` finds every such node by
geometry; `apply()` pins them to freestream AND enrolls them in the solver's
Dirichlet set (m.wall/m.bnd/m.interior rebuilt) -- writing the bc arrays alone
does nothing for a node wall_mask never marked. Verified: 60-step probe, corner
u = 1.0000 exactly (was 1.4077 at the same t), inboard halo 1.19 -> 1.03,
trajectory elsewhere unchanged. R11's published St/C_D/C_L are unaffected (the
defect was static, 10 D from the body); future runs also get a physical far-field
metric floor.

## Post-campaign work (2026-09-11): speed, confinement check, smoothing verdict

**1. Deferred-correction pressure solve made 34% faster (2.197 -> 1.453 s/step at the
step-500 benchmark), physics bit-comparable** (v_probe matches R11 to 6 digits; 20-step
A/B field diff at the solver-tolerance floor: velocity 6e-6, pressure 4e-4). R11 spent
65% of its runtime in 24 full-tolerance inner pressure solves per step. Three changes in
`_solve_cross_dc`:
- **Exact-math block skip**: cross-flux evaluation omitted for blocks whose cross
  metrics are identically zero (threshold 1e-13 relative) -- the tensor wake block,
  ~45% of the cells.
- **Slot-keyed seeding**: the lagged cross term starts from the previous STEP's
  converged pressure FOR THE SAME (picard, corrector) slot. Consecutive calls solve
  different systems (|p| 6.3e3 vs 1.1e3 across the two correctors -- a shared seed is
  useless, measured), but the same slot one step apart is nearly identical.
- **Loose sweeps + reused-RHS polish**: intermediate sweeps solve to
  PICT_CROSS_DC_INNER (1e-4) in a second SolveCache; ONE final full-tolerance solve
  REUSES the last sweep's RHS (a one-iterate-older cross lag, the same order as the
  sweep truncation itself), so it warm-starts as a ~10 ms polish with zero extra cross
  evaluations. First attempt recomputed the RHS for the final solve and was NET SLOWER
  than R11 (2.633 s/step): three extra cross evaluations ate the entire loose-solve
  saving. Measure, don't assume.
Also measured: the DC fixed point converges slowly here (delta ratio ~0.4/sweep; the
sheared-corner coupling is strong), so R11 was ALWAYS running 6 truncated sweeps --
the new scheme keeps that accuracy contract and cuts its price.

**2. Y_HALF=20 confinement grid** (`PICT_Y_HALF`, env; 221,212 cells, validate 0 FAIL,
worst seam jump 1.63x = the Y10 warn level; Y10 stays BIT-IDENTICAL via explicit
guards). Two grid-generation changes were forced by the 2:1 wall-length asymmetry:
- `_ramp_hold_sym`: the west wall (40 D) meets the north/south walls (17 D); a uniform
  parameterisation put a 2.35x cell-width jump at the shared corners (validate FAIL).
  End-graded symmetric distribution matches the corner spacings; the frame end stays
  uniform (a quarter's two ends may use different params -- the east quarter always
  has).
- `_two_sided_n`: east-trap columns span 6-20 D and pure geometric distributions end
  anywhere between 0.1 and 0.8 against the wake block's single dx (3.8x seam FAIL).
  Two-sided (Vinokur-style) distributions pin BOTH end spacings and absorb the length
  in a mid-column bulge; every trap column now ends at wake_dx0 and the wake seam
  matches by construction.
Run: `cylrect_r12_y20_spark`, same protocol as R11. Expected verdict: St drops from
0.1673 toward 0.164-0.165 if the confinement attribution is right.

**3. Trapezoid corner smoothing: evaluated and REJECTED** (negative result, recorded in
`cylinder_ring_grid.py`). Measurements that killed it: (a) min J = 6.1e-5 lives in the
RING blocks' first wall layer on the diagonals -- that is wall RESOLUTION, not
distortion, and the earlier "sheared corner cells" framing misattributed it; (b) the
worst skew (0.83 / grid lines at 34 deg) sits at PINNED corner nodes an interior
smoother cannot move, and the pervasive 0.69 (46 deg) along every diagonal seam is
TOPOLOGICAL -- ray directions are fixed by the block decomposition; (c) 30 Jacobi
sweeps on the trap outer halves changed the worst skew by 0.001; 100 sweeps degraded
seam spacing to 2 validate FAILs with the skew unchanged. Non-orthogonality on this
topology is handled where it can be: the implicit_cross projection.

### Confinement-check pivot: Y20 is topologically out of reach; Y7 runs instead

The Y_HALF=20 launch blew up from ~step 25 (east trapezoid first -- the R11
instability signature) EVEN WITH implicit_cross. Root cause is geometric, not
numerical: at Y=20 the east fan's corner rays to (7, +-20) run 18 degrees off the
outer wall and the max skew reaches 0.97 (grid lines at 14 degrees), regardless of
column distribution -- the cross coupling then dominates the orthogonal operator and
no truncated deferred correction can hold it. The skew law is
cos(90deg - arctan(Y_HALF/6)): the butterfly fan cannot healthily span a wall much
taller than its handoff width. Y_HALF beyond ~12 needs a slab-extension topology
(butterfly core at +-10 plus tensor slabs above/below), noted as future work.

The confinement attribution is instead tested in the direction where geometry
IMPROVES: **Y_HALF=7** (corner rays at 45 deg, max skew 0.82 < Y10's 0.83; grid 0
FAILs after scaling the E<->N/S seam ramp start by sqrt(Y/10); 80-step probe
monotone-stable). Literature blockage laws are monotone in D/(2Y), so two points --
Y10 (St 0.1673) and Y7 (expected HIGHER) -- extrapolate to the open-domain St and
test the same hypothesis as Y20 would have. Run: `cylrect_r12_y7_spark`.
