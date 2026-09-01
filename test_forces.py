"""Surface force integration, checked against closed-form integrals rather than against a plot.

Every one of these has an exact answer, which is the point: a force routine that is wrong by a
constant factor -- the easiest mistake to make, since the area vector carries an h_eta h_zeta
that is easy to drop -- produces perfectly plausible C_D. Only an analytic target catches it.

  1. Uniform pressure on a closed surface  ->  zero net force.
  2. Linear pressure p = a.x                ->  F = +a V on a box that CONTAINS the fluid, and
                                                F = -a V on a body the fluid SURROUNDS. Same
                                                divergence theorem, opposite enclosed volume:
                                                closed integral of x_j n_i dA = V delta_ij with
                                                n outward from the enclosed region, and the
                                                traction normal points out of the SOLID, which
                                                is into the box and out of the cylinder.
  3. Couette u = gamma y on a flat wall     ->  F_x = nu gamma A, the textbook wall shear.
  4. Same as 2 on the CURVILINEAR cylinder O-grid, where the area vectors are not axis-aligned
     and a wrong metric convention no longer cancels.
"""
import numpy as np

from src.multiblock import Block, Domain, face_id
from src.forces import surface_force

PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def box(n=(9, 9, 5), L=(2.0, 1.5, 1.0)):
    """One Cartesian block, all six faces solid, spanning [0, L] in each direction."""
    xs = [np.linspace(0.0, L[a], n[a]) for a in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    h = tuple(1.0 / (n[a] - 1) for a in range(3))
    return Domain([Block(n, X, Y, Z, h)]), L


def zeros_like_block(d):
    return {0: np.zeros(d.blocks[0].shape)}


def all_faces():
    return [(0, face_id(a, s)) for a in range(3) for s in (0, 1)]


print("\n" + "=" * 70 + "\n  Surface forces\n" + "=" * 70)

# ---------------------------------------------------------------- 1. uniform pressure
d, L = box()
u = zeros_like_block(d); v = zeros_like_block(d); w = zeros_like_block(d)
p = {0: np.full(d.blocks[0].shape, 3.7)}
R = surface_force(d, all_faces(), u, v, w, p, nu=0.0); F, Fp, Fv = R['total'], R['pressure'], R['viscous']
scale = 3.7 * 2 * (L[0]*L[1] + L[1]*L[2] + L[0]*L[2])
check(np.abs(F).max() / scale < 1e-12,
      f"uniform pressure on a closed box gives zero net force: |F| = {np.abs(F).max():.3e} "
      f"against {scale:.2f} of one-sided load")

# ---------------------------------------------------------------- 2. linear pressure
a = np.array([2.0, -1.0, 0.5])
blk = d.blocks[0]
p = {0: a[0]*blk.x + a[1]*blk.y + a[2]*blk.z}
R = surface_force(d, all_faces(), u, v, w, p, nu=0.0); F, Fp, Fv = R['total'], R['pressure'], R['viscous']
V = L[0] * L[1] * L[2]
# The fluid is INSIDE this box, so the outward-from-solid normal points into the enclosed
# volume and the sign flips relative to the cylinder below. Physically: p grows with x, the
# x = L wall carries the higher load, the container is pushed in +x.
want = +a * V
err = np.abs(F - want).max() / np.abs(want).max()
check(err < 1e-12, f"linear pressure on a box containing the fluid gives F = +a V: "
                   f"{np.round(F, 9)} against {np.round(want, 9)}, rel err {err:.2e}")

# ---------------------------------------------------------------- 3. Couette wall shear
gamma, nu = 1.7, 0.023
u = {0: gamma * blk.y}
v = zeros_like_block(d); w = zeros_like_block(d)
p = zeros_like_block(d)
bottom = [(0, face_id(1, 0))]
R = surface_force(d, bottom, u, v, w, p, nu=nu); F, Fp, Fv = R['total'], R['pressure'], R['viscous']
want_x = nu * gamma * L[0] * L[2]
check(abs(F[0] - want_x) / want_x < 1e-12,
      f"Couette shear on the bottom wall: F_x = {F[0]:.9f} against nu*gamma*A = {want_x:.9f}")
check(np.abs(Fp).max() < 1e-14, "zero pressure field contributes no pressure force")

# ---------------------------------------------------------------- 4. curvilinear O-grid
from cylinder_grid import cylinder_domain, R_CYL

dc, r, arc = cylinder_domain(nblk=16, nz=4)
span = float(dc.blocks[0].period[2])
uc = {b: np.zeros(bl.shape) for b, bl in enumerate(dc.blocks)}
vc = {b: np.zeros(bl.shape) for b, bl in enumerate(dc.blocks)}
wc = {b: np.zeros(bl.shape) for b, bl in enumerate(dc.blocks)}

ac = np.array([2.0, 0.0, 0.0])
pc = {b: ac[0]*bl.x + ac[1]*bl.y for b, bl in enumerate(dc.blocks)}
body = [(b, face_id(0, 0)) for b in range(len(dc.blocks))]     # the cylinder surface
R = surface_force(dc, body, uc, vc, wc, pc, nu=0.0); F, Fp, Fv = R['total'], R['pressure'], R['viscous']
V = np.pi * R_CYL**2 * span
want = -ac[0] * V
err = abs(F[0] - want) / abs(want)
# the fluid SURROUNDS this body, so the sign is the opposite of the box above
# the surface is a POLYGON of 256 nodes, not a circle, so the enclosed area is the inscribed
# polygon's: pi R^2 is approached as O(1/N^2), and 256 nodes puts that at ~2.5e-4
check(err < 2e-3, f"linear pressure on the curvilinear cylinder: F_x = {F[0]:.6f} against "
                  f"-a pi R^2 L = {want:.6f}, rel err {err:.2e} (polygon vs circle is O(N^-2))")

pc = {b: np.full(bl.shape, -0.9) for b, bl in enumerate(dc.blocks)}
F = surface_force(dc, body, uc, vc, wc, pc, nu=0.0)['total']
ref = 0.9 * 2 * np.pi * R_CYL * span
check(np.abs(F).max() / ref < 1e-12,
      f"uniform pressure on the closed cylinder ring: |F| = {np.abs(F).max():.3e} "
      f"against {ref:.3f} of one-sided load")

print("=" * 70)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 70)
raise SystemExit(1 if FAIL else 0)
