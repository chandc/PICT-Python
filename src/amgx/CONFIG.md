# AmgX configs live here, not in /tmp

`pcg_amg_1e6.json` is the config for the square-cylinder production runs: PCG preconditioned by
one AMG V-cycle, converging to **relative residual 1e-6**.

## The caller's rtol now wins — FIXED

`_amgx_solve()` used to discard the `rtol` it was handed, so on the GPU path the JSON *was* the
tolerance and tightening `tol` did nothing. It now passes `rtol` down to `AmgXSolver`, which
applies `main:tolerance=<rtol>` through `AMGX_config_add_parameters` after loading the file.

Because the config is a **process-wide singleton**, a second solver asking for a different
tolerance cannot quietly get one: that raises, rather than returning an answer converged to
somebody else's tolerance. One tolerance per process.

**Not yet exercised on hardware** — written while the GPU machine was unreachable. The SciPy path is
untouched and regressions pass (multiblock 55/55), but the AmgX branch needs a GPU run to
confirm `AMGX_config_add_parameters` is accepted with this config's scoping.

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
