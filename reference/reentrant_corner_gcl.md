# The reentrant corner: seam consistency and GCL

**Status: fixed.** Grids with an obstacle now supply a `background` node distribution; both
invariants hold together. The history is kept because the defect was invisible to the test
written to catch it, and the two obvious fixes both failed.

## Two invariants, both required

**Seam consistency.** The two blocks sharing a face must compute the same flux through it, or
mass is injected at the seam.

**GCL / freestream preservation.** A uniform flow has identically zero divergence on any valid
grid. `div(u = const)` therefore contains no flow at all — it is a pure test of the geometry.

## They used to be traded against each other

`face_fluxes` has two paths, and before the fix each was exact in one invariant and O(1) wrong
in the other:

| grid | path | seam mismatch | GCL |
|---|---|---|---|
| square cylinder | own-metrics *(shipped)* | 0.00e+00 | **1.81e+00** |
| square cylinder | padded-geometry | 5.43e+01 | 6.15e-15 |
| BFS 5-domain | own-metrics *(shipped)* | 0.00e+00 | **1.93e+00** |
| BFS 5-domain | padded-geometry | 2.63e+00 | 5.49e-15 |

The `own-metrics` path was introduced to fix a 4.2e+00 seam mismatch at the BFS step corner. It
did, and silently traded it for a freestream-preservation failure of the same magnitude.

**Cause.** Each block derives its metrics from its own padded coordinates. A block abutting the
obstacle has a **wall** there, so it was not padded and fell back to **one-sided** differences
along that axis — while the block across the seam used **central** differences with real
neighbour data. The GCL sum at a corner node mixes metric terms from both and stops telescoping.

## Why the existing test could not see it

`test_obstacle_topology.py` gates this exact topology at 3.8e-11. It builds its coordinates as
one `np.linspace` sliced into contiguous chunks — a perfectly **uniform** grid, the one case in
which one-sided and central differences of a linear coordinate map agree exactly. Its own
docstring says bug 3 was the pressure flux using extrapolated corner coordinates; the grid it
uses cannot detect a recurrence. Stretching the same topology, changing nothing else:

| stretch | GCL, before |
|---|---|
| 0.00 | 4.9e-15 ← what the existing test measures |
| 0.05 | 2.1e-03 |
| 0.50 | 4.0e-02 |

Nine blocks **without** the hole stay at 1.3e-15 on the same stretched coordinates, so the seam
machinery was sound and the reentrant corner was not. It did not converge away either —
3.99e-02 → 2.90e-02 across a 64× cell increase — so it was an inconsistency, not truncation.

## What it cost

Prescribing freestream on every boundary makes `u = (1,0,0)` an exact solution, so any departure
is purely discretisation. After 20 steps:

| stretch | max abs(u−1) before | after |
|---|---|---|
| 0.05 | 5.1e-04 | **7.8e-15** |
| 0.50 | **5.1e-03** | **2.7e-15** |

About **0.5% spurious velocity manufactured out of an undisturbed freestream**, now at machine
precision.

## The fix

`Block(..., background=(gx, gy, gz, offset))` — the node distributions of the **whole background
rectangle, solid region included**, and where this block starts in them. `pad_coords` then cuts
ghosts straight out of those arrays instead of walking neighbours.

Every block slices the **same** arrays, so ghost coordinates are single-valued everywhere —
including inside the body, where no block stores anything — and every block uses the identical
central stencil. That is precisely what the metric identity needs. A face with no data beyond it
in the background is a true domain boundary and stays one-sided, as the single-block code does.

Only valid for a separable background, which is what an H-grid around a box is; blocks without
`background` keep the neighbour-walking path unchanged.

One consequence worth knowing: obstacle walls now carry ghosts that no *field* can supply, so
`padded_geometry` trims its result back to the extent `pad_field` produces, or the two arrays
stop broadcasting.

### After

| grid | seam mismatch | GCL | was |
|---|---|---|---|
| square cylinder | 0.00e+00 | **6.15e-15** | 1.81e+00 |
| BFS 5-domain | 0.00e+00 | **4.81e-15** | 1.93e+00 |

## What it changed in a real answer

Armaly BFS at Re = 100, coarse five-domain grid (nx 14/28/28, ny 12/13, nz 8), Dong outflow,
both runs identical except for the background coordinates and both converged at step 872:

| metrics | x_r/S | GCL |
|---|---|---|
| GCL-violating (old) | 2.7428 | 1.96e+00 |
| background (fixed) | 2.7223 | 2.32e-15 |

**Reattachment moved 0.75%.** So the published BFS numbers were not badly wrong — the defect
perturbs the answer rather than wrecking it — but they do shift, and the shift is not something
the old code could have told you about. Measured on a coarse grid; the production grid clusters
harder and has not been re-run.

## Fixes that were tried and rejected

Both measured, neither kept:

* **Linear extrapolation instead of edge replication in `_match_extent`.** Changes the corner
  ghost coordinates as intended and moves the GCL residual by *nothing* — bit-for-bit identical,
  because the own-metrics path never reads padded coordinates there. It does improve the padded
  path's seam mismatch, 5.43e+01 → 2.41e+01, which is still useless.
* **Padding wall faces with extrapolated coordinates.** Helps the BFS (1.93 → 1.21e-01) and
  makes the square cylinder **worse** (1.81 → 2.88): linear extrapolation is not the
  continuation the neighbouring block assumes on a clustered grid. This is the near miss — the
  right idea (pad the wall) with the wrong data, and the reason the fix has to come from the
  builder rather than from anything a block can infer locally.

Gate: `test_reentrant_corner_stretched.py`, which fails if either invariant breaks **or** if the
without-background case stops reproducing the original defect. Logs in `results/logs/`.
