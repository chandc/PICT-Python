# PICT-Python

An **educational, pure-Python/NumPy reimplementation** of the numerical methods behind
[PICT](https://github.com/tum-pbs/PICT) — the differentiable GPU multi-block PISO solver from the
Physics-based Simulation group at TU Munich.

**Not a fork.** This repository contains no upstream source. Where it follows PICT's design —
conservative Thomas & Lombard metrics, the multi-block connection model, the PISO corrector
structure — it was rebuilt from the described method so that each piece could be tested in
isolation. Apache-2.0, matching upstream; see `NOTICE`.

The point is legibility: every non-obvious decision is written down next to the code that depends
on it, usually with the measurement that forced it.

## What works

| case | result | reference |
|---|---|---|
| Square cylinder, Re = 100 | **St = 0.1467** ± 0.0067 | 0.145–0.150 (CFD, 5% blockage) |
| Backward-facing step, 5 domains | x_r/S = 2.80 / 4.85 / 6.72 / 8.02 | Armaly ≈ 3 / 5 / 6–7 |
| Circular cylinder, Re = 100 | grid + case built, not yet run | St ≈ 0.164, C_D ≈ 1.33 |

## Start here

```bash
uv run python square_cylinder_grid.py     # build and validate a grid
uv run python run_square_cylinder.py --tol 1e-6 --steps 20000 --nz 4
uv run python test_multiblock.py          # 55 checks
```

`uv` reads `pyproject.toml` and builds the environment. Versions are pinned because two of them
have bitten this project — see the comments there.

## The one setting that matters most

**`--tol` is the linear solver tolerance, and at `1e-4` the square cylinder converges bitwise
steady and never sheds** — at Re 100, 200 *and* 300, at every grid resolution tried. The pressure
correction is under-resolved there, so every velocity update is damped: harmless for a steady
problem, fatal for a marginally unstable one. Use `1e-6` for anything that oscillates.

That failure is silent and total. It does not degrade the answer; it deletes the physics and
returns a beautifully converged steady solution. `reference/linear_solver_tolerance.md` has the
full story.

## Layout

```
src/            solver core -- multiblock topology, PISO, metrics, linear solvers, AmgX binding
reference/      why things are the way they are; read these before changing the solver
plot_utility/   figure generation
test_*.py       invariants that produce silent, plausible corruption when violated
docker/         pinned image for GPU runs with NVIDIA AmgX
results/        measurements (histories, logs). Checkpoints are gitignored -- see results/fields/
```

## Reference documents worth reading first

* `reference/linear_solver_tolerance.md` — the tolerance trap above
* `reference/square_cylinder_vortex_street.md` — what it took to make a vortex street appear,
  including four hypotheses that were wrong and how each was killed
* `reference/reentrant_corner_gcl.md` — seam consistency vs freestream preservation, and why they
  could not both hold until the grid builder supplied background coordinates
* `reference/rhie_chow_ddt_instability.md` — a term with unit gain by construction
* `reference/pressure_checkerboard.md` — detection needs an (amplitude, flip-fraction) pair;
  either number alone is wrong in a different direction

## Testing

Tests here gate **invariants**, not outputs — the failures that look plausible are the dangerous
ones. `test_reentrant_corner_stretched.py` also fails if the *without-fix* case stops reproducing
the original defect, so it cannot quietly stop testing anything.

```bash
uv run python test_multiblock.py                    # 55 checks
uv run python test_rhie_chow.py                     # 28 checks
uv run python test_obstacle_topology.py
uv run python test_reentrant_corner_stretched.py
```
