"""
At a reentrant corner the two flux paths are each exact in ONE invariant and O(1) wrong in the
other. There is currently no setting in which both hold.

TWO INVARIANTS, BOTH REQUIRED.

  SEAM CONSISTENCY -- the two blocks sharing a face must compute the same flux through it.
  Violating it injects or destroys mass at the seam. This is what the `own-metrics` path in
  `face_fluxes` was introduced to fix (measured 4.2e+00 at the five-domain BFS step corner).

  GCL / FREESTREAM PRESERVATION -- a uniform flow has identically zero divergence on any valid
  grid, so `div(u = const)` is a pure test of the discretisation with no flow in it. Violating
  it means the solver manufactures pressure to "correct" an undisturbed freestream.

MEASURED, on the two production grids (uniform flow for GCL, a smooth analytic field for the
seam check):

    grid              path              seam mismatch      GCL
    square cylinder   own-metrics           0.00e+00     1.81e+00
    square cylinder   padded-geometry       5.43e+01     6.15e-15
    BFS 5-domain      own-metrics           0.00e+00     1.93e+00
    BFS 5-domain      padded-geometry       2.63e+00     5.49e-15

`own-metrics` is what ships, so THE FIVE-DOMAIN BFS RESULTS IN THIS REPO CARRY A GCL VIOLATION
OF 1.9 AT THE STEP CORNER.

WHY, AND WHY IT HID FOR SO LONG. Each block computes its metrics from its own padded
coordinates. A block abutting the obstacle has a WALL there, so it is not padded and falls back
to ONE-SIDED differences along that axis -- while the block across the seam uses CENTRAL
differences with real neighbour data. The GCL sum at a corner node mixes metric terms from both,
so it no longer telescopes. On a UNIFORM grid one-sided and central differences of a linear
coordinate map agree exactly, which is why `test_obstacle_topology.py` -- whose coordinates are
a single `np.linspace` sliced into chunks -- reads 3.8e-11 and cannot see any of this.

    stretch   GCL, own-metrics path
      0.00      4.9e-15     <- what the existing obstacle test measures
      0.01      3.9e-04
      0.50      4.0e-02

Not convergent under refinement: 3.99e-02 -> 2.90e-02 across a 64x cell increase. The same nine
blocks WITHOUT the hole stay at 1.3e-15 on the same stretched coordinates, so the seam machinery
is sound and it is specifically the reentrant corner that fails.

THINGS THAT LOOK LIKE FIXES AND ARE NOT (each measured, each rejected):

  * Replacing `_match_extent`'s edge-replication fill with linear extrapolation. Changes the
    corner ghost coordinates as intended and moves the GCL residual by NOTHING at all --
    bit-for-bit identical -- because the own-metrics path never reads padded coordinates there.
    It does improve the padded path's seam mismatch (5.43e+01 -> 2.41e+01), nowhere near zero.
  * Padding wall faces with extrapolated coordinates so every block uses a central stencil.
    Helps the BFS (1.93 -> 1.21e-01) and makes the square cylinder WORSE (1.81 -> 2.88),
    because linear extrapolation is not the continuation the neighbouring block assumes on a
    strongly clustered grid.

WHAT WOULD ACTUALLY FIX IT. The ghost inside the solid has to be the continuation of the GLOBAL
grid, which no block knows locally -- in an H-grid it is the obstacle's own interior tensor
grid, stored by nobody. Either the grid builder supplies explicit ghost coordinates for obstacle
faces, or the metrics are corrected so GCL holds by construction (a freestream-preservation
correction, localised to the ~32 affected nodes). Both are design changes, not patches.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id, face_axis_side

CLEAN = 1e-10


def build(stretch, m=1, hole=True, background=False):
    """The topology of test_obstacle_topology.py; stretch=0, m=1 reproduces its grid exactly."""
    NXc, NYr, NZ = (6*m, 4*m, 8*m), (5*m, 4*m, 5*m), 4
    L, H = 3.0, 2.0
    Nx, Ny = sum(NXc), sum(NYr)
    t = np.linspace(0, 1, Nx); xs = L * (t + stretch * np.sin(2*np.pi*t) / (2*np.pi))
    t = np.linspace(0, 1, Ny); ys = H * (t + stretch * np.sin(2*np.pi*t) / (2*np.pi))
    zs = np.arange(NZ) / NZ
    cx = [slice(0, NXc[0]), slice(NXc[0], NXc[0]+NXc[1]), slice(NXc[0]+NXc[1], Nx)]
    cy = [slice(0, NYr[0]), slice(NYr[0], NYr[0]+NYr[1]), slice(NYr[0]+NYr[1], Ny)]
    h = (L/(Nx-1), H/(Ny-1), 1.0/NZ)
    names, blocks = {}, []
    for j, ry in enumerate("BMT"):
        for i, rx in enumerate("LMR"):
            if hole and i == 1 and j == 1:
                continue
            X, Y, Z = np.meshgrid(xs[cx[i]], ys[cy[j]], zs, indexing="ij")
            bgd = (xs, ys, zs, (cx[i].start, cy[j].start, 0)) if background else None
            b = Block((NXc[i], NYr[j], NZ), X, Y, Z, h, background=bgd)
            b.faces[face_id(2, 0)] = b.faces[face_id(2, 1)] = "periodic"
            names[rx+ry] = len(blocks); blocks.append(b)
    C = []
    for j, ry in enumerate("BMT"):
        for i in range(2):
            a, b = "LMR"[i]+ry, "LMR"[i+1]+ry
            if a in names and b in names:
                C.append(Connection(names[a], face_id(0, 1), names[b], face_id(0, 0)))
    for i, rx in enumerate("LMR"):
        for j in range(2):
            a, b = rx+"BMT"[j], rx+"BMT"[j+1]
            if a in names and b in names:
                C.append(Connection(names[a], face_id(1, 1), names[b], face_id(1, 0)))
    return Domain(blocks, C)


def gcl_residual(d):
    """max |div| over FREE nodes for u = (1,0,0). Exactly zero on a correct discretisation."""
    wall = d.wall_mask()
    u = {b: np.ones(bl.shape) for b, bl in enumerate(d.blocks)}
    z = {b: np.zeros(bl.shape) for b, bl in enumerate(d.blocks)}
    return max(np.abs(np.where(wall[d.global_ids(b)], 0.0,
                               d.divergence(b, d.face_fluxes(b, u, z, z),
                                            d.block_metrics_cached(b)[0]))).max()
               for b in range(len(d.blocks)))


def seam_mismatch(d):
    """max disagreement between the two blocks' flux through a face they share."""
    f = lambda bl, p: np.sin(1.3*bl.x + p) * np.cos(0.9*bl.y - p) + 0.3*np.cos(2.1*bl.z)
    u = {b: f(bl, 0.0) for b, bl in enumerate(d.blocks)}
    v = {b: f(bl, 1.1) for b, bl in enumerate(d.blocks)}
    w = {b: f(bl, 2.2) for b, bl in enumerate(d.blocks)}
    F = {b: d.face_fluxes(b, u, v, w) for b in range(len(d.blocks))}
    worst = 0.0
    for c in d.connections:
        aa, sa = face_axis_side(c.fa); ab, sb = face_axis_side(c.fb)
        A = np.moveaxis(F[c.ba][aa], aa, 0)[F[c.ba][aa].shape[aa]-1 if sa == 1 else 0]
        B = np.moveaxis(F[c.bb][ab], ab, 0)[F[c.bb][ab].shape[ab]-1 if sb == 1 else 0]
        worst = max(worst, np.abs(A - c.align(B)).max())
    return worst


if __name__ == "__main__":
    print("  GCL residual (uniform flow) on the 8-block obstacle topology.\n"
          "  The flow is constant, so this is pure geometry: it must be ~1e-15.\n")
    print(f"  {'stretch':>9}{'no background':>16}{'with background':>18}{'seam':>12}")
    bad_before, ok_after, seams_ok = [], [], []
    for st in (0.0, 0.01, 0.05, 0.2, 0.5):
        before = gcl_residual(build(st))
        after = gcl_residual(build(st, background=True))
        sm = seam_mismatch(build(st, background=True))
        bad_before.append(before); ok_after.append(after); seams_ok.append(sm)
        print(f"  {st:>9.2f}{before:>16.3e}{after:>18.3e}{sm:>12.3e}")

    print("\n  CONTROLS at stretch = 0.5 -- isolating the hole from the seam machinery:")
    print(f"    9 blocks, NO hole, no background   {gcl_residual(build(0.5, hole=False)):.3e}")
    print(f"    8 blocks, hole,    no background   {gcl_residual(build(0.5)):.3e}")
    print(f"    8 blocks, hole,    background      {gcl_residual(build(0.5, background=True)):.3e}")

    print("\n  THE PRODUCTION GRIDS (both now supply a background):")
    from armaly_bfs5_grid import bfs5_domain
    from square_cylinder_grid import square_domain
    prod = []
    for nm, mk, was in (("BFS 5-domain", bfs5_domain, 1.93e0),
                        ("square cylinder", lambda: square_domain()[0], 1.81e0)):
        d = mk(); g, sm = gcl_residual(d), seam_mismatch(d)
        prod.append((g, sm))
        print(f"    {nm:<18} GCL {g:.2e}  seam {sm:.2e}   (GCL was {was:.2f})")

    fails = []
    if not all(v < CLEAN for v in ok_after):
        fails.append("GCL still violated with a background supplied")
    if not all(v < CLEAN for v in seams_ok):
        fails.append("seam consistency broken by the background path")
    if not all(g < CLEAN and s < CLEAN for g, s in prod):
        fails.append("a production grid still violates GCL or seam consistency")
    if max(bad_before[1:]) < CLEAN:
        fails.append("the without-background case no longer reproduces the defect, so this "
                     "test has stopped testing anything")

    if fails:
        print("\n  [FAIL] " + "; ".join(fails))
        sys.exit(1)
    print(f"\n  [PASS] GCL and seam consistency now hold together. Worst GCL with a background:")
    print(f"         {max(ok_after + [g for g, _ in prod]):.2e}, against {max(bad_before):.2e} without.")
