# AmgX configs live here, not in /tmp

`pcg_amg_1e6.json` is the config for the square-cylinder production runs: PCG preconditioned by
one AMG V-cycle, converging to **relative residual 1e-6**.

## The caller's rtol now wins — FIXED, and confirmed on the GPU

`_amgx_solve()` used to discard the `rtol` it was handed, so on the GPU path the JSON *was* the
tolerance and tightening `tol` did nothing. It now passes `rtol` down to `AmgXSolver`.

**How, and why not the obvious way.** The first version called
`AMGX_config_add_parameters(&cfg, "main:tolerance=...")` on the already-created handle. AmgX
answered `Caught amgx exception: Invalid/null C wrapper` and the run died at step 1: that call
does not augment a live handle the way its name suggests. `_config_with_tolerance()` now reads
the template, substitutes `solver.tolerance`, and writes
`_generated_tol_<rtol>.json` beside it, so `AMGX_config_create_from_file` does the only thing it
is known to do correctly — and a failed run leaves the exact config on disk to inspect. The
generated files are gitignored.

Because the config is a **process-wide singleton**, a second solver asking for a different
tolerance cannot quietly get one: that raises, rather than returning an answer converged to
somebody else's tolerance. One tolerance per process.

Exercised on hardware: the `sqcyl_v3` run on spark-b85b (82,096 cells, `--tol 1e-6`,
`backend=amgx`) initialises from `_generated_tol_1.000e-06.json` and runs clean.

## Why the file is ours anyway

Pointing `AMGX_CONFIG` at a file in
`/tmp/AMGX/build/configs/` on one machine means the tolerance of a production run is set by a
file nobody versions and anybody can overwrite. That is the same class of mistake that produced
the 56x speedup claim: SciPy grinding to 1e-11 against AmgX at 1e-10, compared as though matched.

`AMGX/build/configs/PCG_CLASSICAL_V_JACOBI.json` happens to carry `tolerance: 1e-06` today. That
is luck, not configuration.

## Two settings that are not cosmetic

* **`monitor_residual` must stay 1 on the outer solver.** It reads like a printing flag and is
  not -- it drives convergence checking. Setting it to 0 previously made every resolution
  diverge. The inner smoother and preconditioner are a different matter and are 0 here.
* **`max_iters: 300`**, raised from the stock 100. Measured 53 iterations at 1e-10 on the BFS
  operator, so 100 is usually ample -- but a solver that silently stops at its cap returns an
  unconverged answer with no error, and the cost of the headroom is nothing when it is not used.

Printing is off (`print_solve_stats`, `print_grid_stats`) because a 30,000-step run otherwise
emits tens of thousands of lines; the binding also registers a no-op print callback, and
`AMGX_solver_get_iterations_number` still reports what we need.

## 2026-09-07: the rebuilt library does not work; R7 ran on scipy

The Sep 6 Spark reboot erased /tmp/AMGX (as SESSION_STATE.md warned). Four rebuild attempts of
the same v2.5.0 tag failed four ways, and the differences from the working Aug 30 build are
recorded here so the next attempt starts from evidence:

| build | env | outcome |
|---|---|---|
| arch=native, no --gpus at build | pytorch:25.12 | wrong-arch SASS; abort at resources_create when run without --gpus |
| arch=121 (two racing makes) | pytorch:25.12 | linked; hangs at 100% CPU, 0% GPU before first solve |
| arch=121, clean | pytorch:25.12 | same hang |
| arch=native with --gpus, clean | pict-amgx:1.0 itself (25.11 base) | clean RuntimeError: solver_setup rc 6, even on a 1e4 1-D Poisson |

Facts that constrain the cause: the GPU is healthy post-reboot (31 h of torch DNS on it);
the run container sees the device (nvidia-smi -L inside); pict-amgx:1.0 is based on the
25.11 NGC image while the first three builds used 25.12 (runtime mismatch explains at most
the first three, not the matched fourth); the Aug 30 banner shows the SAME runtime/driver
pair (13.1 on 13.0) that now fails. Remaining suspects: whatever container actually built
the Aug 30 library (unrecorded -- exactly the reproducibility hole the Dockerfile header
complains about), and an AmgX-vs-driver interaction specific to solver setup. Next probe:
build in a CUDA 13.0 devel image (matching the DRIVER), and bisect the JSON config on rc 6.

Silent-fallback trap, worth its own line: `_amgx_solve` returns None on any init failure and
the caller falls back to scipy without a visible line at default verbosity -- a "hung" AmgX
run and a healthy scipy run look identical for the first ~50 minutes at 1e-6 on 82k cells.
Two R7 attempts were killed on that ambiguity. `fell_back` exists; it should PRINT.

## RESOLVED (2026-09-07 late): AmgX working again — the recipe, and the real saboteur

Working stack (verified at 0.666 s/step on the 164k-cell R7, GPU active):
* source: AMGX main @ 91a8413 ("Fix CUDA 13 tuple compatibility", 2026-07-09). The v2.5.0
  RELEASE TAG does not work on CUDA 13 (classical-AMG setup: THRUST_FAILURE rc 6 / hangs) --
  the Aug 30 "2.5.0" that worked was main, whose banner also says 2.5.0.
* builder: cpn-spark:latest (NGC pytorch 25.11, CUDA 13.0.88 -- matches the driver), --gpus all,
  cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_NO_MPI=1.
* config: aggregation AMG (pcg_agg_1e6.json). The classical D2 path fails on every 2026-09 build.
* the library now lives at ~/amgx/libamgxsh.so on Spark (REBOOT-PROOF) with RECIPE.md beside it;
  the launcher bind-mounts it over /tmp/AMGX/build.

And the hidden saboteur that outlived every library fix: the binding held ONE process-wide
config, keyed to the first rtol requested. This week's momentum-tolerance decoupling made the
momentum solver (1e-9) claim it first, the pressure solver (1e-6) was refused, and the silent
fallback ran the run on scipy while every library looked "hung". Fixed: one config per
tolerance against the shared resources object (which genuinely must be a singleton), and the
fallback now prints. The teardown wart remains: AMGX_solver_destroy throws "Mode not found"
at interpreter exit, after all output is written.

## 2026-09-08: the cylinder shootout — one config per SYSTEM, and a benchmark trap

The round-cylinder R8 arms exposed that "the" AmgX config is really two. Measured on the REAL
operators (dumped from a live run via PICT_DUMP_SOLVE, solved cold, one process per config):

| system | Jacobi-PCG | aggregation AMG (SIZE_2+BJ) | scipy CG+Jacobi (CPU) |
|---|---|---|---|
| pressure, n=160,640, rtol 1e-6 | 0.599 s / 65 it | **0.062 s / 61 it** | 2.52 s / 888 it |
| momentum, n=158,720, rtol 1e-9 | **0.014 s / 1 it** | 55.9 s / 20,000 it, DIVERGED | -- |

Also: SIZE_4/SIZE_8 + MULTICOLOR_DILU diverge on the pressure system; classical D2 still rc 6.
The momentum operator is dt-scaled and diagonally dominant -- Jacobi converges in ONE
iteration and an AMG hierarchy is actively wrong for it. That divergence, not the pressure
solve, was the whole of R8 take 1's >22 s/step. The binding now takes AMGX_CONFIG_TIGHT, a
second template used for rtol <= 1e-8; production sets pcg_agg_1e6.json (pressure) +
pcg_jac_1e6.json (momentum).

Two caveats for the record. (1) A cold-solve shootout RANKS preconditioners but does not
decompose a step budget: in-run solves warm-start from the previous step, so the cylinder ran
at the same 2.71 s/step under all-Jacobi and under the split -- the floor is CPU-side
per-block assembly (16 blocks, GPU at 14%), not the solver. The split config is kept because
it removes the divergence failure mode, not because it bought speed. (2) The pressure system
is mildly NON-symmetric (Dong outflow rows) and is being solved by PCG; it converges cleanly
(61 its) but an FGMRES outer solver is the principled alternative if that ever degrades.
