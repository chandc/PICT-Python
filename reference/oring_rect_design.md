# O-ring-in-rectangle grid for the round cylinder — design

**Why.** The round cylinder's O-grid put a Dong arc and freestream Dirichlet on ONE circular
boundary; the junction corners between the two BC types are a striping source no intervention
at the boundary could stabilise (rhie_chow_boundary.md R8/R9 + junction study). The
literature never builds this: upstream PICT's vortex street is a rectangular channel, wake
stability work uses C-grids with a flat outflow plane, and Dong's OBC is designed for whole
planes. This grid moves the round cylinder into the square-cylinder topology that R7
validated end-to-end: rectangular outer boundary, inflow left, slip laterals, **Dong on the
single flat right face**, BC junctions only at far rectangle corners in quiet freestream.

## Topology (v2 -- after studying the seam machinery)

The connection machinery is strictly whole-face-to-whole-face with equal node counts, one
owner per seam line, NO split faces and NO corner logic (mixed wall/connected corners are the
repo's documented scar, patched only by `background`, which is tensor-only). That rules out a
ring inside square_domain's H-grid: the hole faces fragment at corner vertices no ring
partition can match. The topology that satisfies every constraint is the classic BUTTERFLY
plus a tensor wake extension -- 9 blocks:

* 4 ring quarters (E, N, W, S): circle R=0.5D to the square frame (half-width L1), radial x
  tangential, rays through uniform frame-edge nodes; quarter q spans corner-ray to corner-ray
  and OWNS its start diagonal (cyclic, like the O-grid's azimuthal seams).
* 4 trapezoids: frame side to the corresponding rectangle side, cut by the diagonals from
  frame corners to rectangle corners; trapezoid q's inner face has exactly the quarter's
  tangential count -- equal-count whole-face connections by construction; trapezoid-trapezoid
  seams along the extended diagonals with the same start-diagonal ownership. No reentrant
  corners anywhere, so curvilinear blocks need no `background`.
* 1 tensor wake block: the EAST trapezoid stops at x_hand (a few D); its outer face connects
  whole-face to a rectangular wake block that carries the validated wake plateau
  (dx = 0.15 D to 15 D, ratio 1.06) out to X_OUT. **Dong lives on this block's flat far
  face** -- a single tensor plane, the literature-standard arrangement.
* Outer BCs: each rectangle side owned entirely by one block (trapezoid or wake block):
  left = inflow, top/bottom = slip, right = Dong. BC junctions only at rectangle corners
  between wall-type faces, the same benign situation square_domain has.

One computational-spacing convention shared by all blocks (the diffusion assembly refuses
seams whose h differ); z periodic throughout.

## Sizing (first cut)

n_frame = 33 nodes across the hole side -> dx_frame = 0.0625 D; circle quarter arc 0.785 D
across 32 cells -> body spacing 0.019-0.038 D (cos^2 variation), comparable to the O-grid's
azimuthal 0.012 at the body and the square case's 0.032. Radial: first = 0.008 D, ratio 1.10
-> ~17 cells to the side centres. Outer strips identical to square_domain with X_IN/X_OUT/
Y_HALF as in the square case (or the wider round-cylinder values; blockage check applies).
Wake plateau dx = 0.15 D to 15 D as validated.

## BCs and runner

`cylinder_rect_bc.py`: classify by geometry -- ring inner faces (r = R_CYL) no-slip;
rectangle left inflow, right Dong (single plane), top/bottom slip. Reuse
square_cylinder_bc's role logic for the rectangle. Runner `run_cylinder_rect.py` adapts
run_cylinder (same probe/force/report/sponge machinery; forces integrate over the ring inner
faces).

## Validation ladder

1. `__main__` grid check: validate(), min(J) > 0, spacing ratios <= 1.10, ownership audit.
2. Laminar Re = 20 steady: symmetric twin vortices, C_D vs literature (~2.0).
3. Split-equals-whole style check against the O-grid solution at Re = 100 base flow.
4. Full shedding run: St vs 0.1643/0.164, C_D vs 1.27-1.33, far field clean at the flat
   outflow, NO junction striping anywhere (the point of the exercise).

## R11 VERDICT (2026-09-10): VALIDATED

Full shedding run (`cylrect_r11_spark`, 142,624 cells nz=4, dt=0.01, tol 1e-6, AmgX +
implicit_cross deferred correction, 8000 settle + 30000 shed, 18 h at 2.17 s/step, ZERO
solver incidents):

- **St = 0.1673** over 21 saturated cycles t=[250,380] (zero-crossing; cycle scatter
  <1e-4). Legacy O-grid measured 0.1643; canonical open-domain 0.164. The +1.8% is the
  documented confined-domain shift for lateral freestream walls at +-10 D -- consistent
  with the runner's own confined reference (C_D 1.33).
- **C_D = 1.321** (window mean; spurious normal stress +0.0047 accounted). Literature
  band 1.27-1.33.
- **C_L rms = 0.23** (band 0.23-0.33).
- **No junction striping anywhere** -- the point of the exercise. Vorticity contours
  (figures/rect_vorticity_cylrect_r11_mid.png): the Karman street crosses the trapezoid->
  wake handoff at x=7 invisibly and exits the flat Dong plane at x=30 without reflection.
  Far-field metric pinned at 0.416 for the entire run (abort was 1.5); on the O-grid the
  same metric rode the Dong-arc corner stripes.

Two prerequisites found on the way (details: reference/measurement_traps.md and the
c518e44 commit message): the orthogonal-only pressure projection is UNSTABLE on this
grid's sheared trapezoid corners (field doubles per step from t~0.1; cured by
PICT_IMPLICIT_CROSS=1 deferred correction at ~2x step cost), and the AmgX binding needed
status/aliasing/true-residual hardening before any of the above could even be measured.
