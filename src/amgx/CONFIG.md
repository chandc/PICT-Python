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
