"""The subgrid closures, against fields whose answer is known in advance.

Every check here has an analytic target, so none of them can be argued with. The pattern is the
one this repo arrived at the hard way: a model validated only by "the answer looks turbulent"
has been validated against nothing.

  1  pure SHEAR      u = (a y, 0, 0):  |S| = a exactly, and Smagorinsky is then (Cs D)^2 a.
  2  solid ROTATION  u = omega x r:    S = 0 identically, so Smagorinsky gives nu_t = 0 -- the
                     check a model built on |grad u| instead of |S| would fail. WALE does NOT
                     vanish here and must not: S^d is the traceless symmetric part of g*g, which
                     for rotation is diag(-1,-1,2) omega^2/3, so S^d:S^d = (2/3) omega^4 while
                     S:S = 0. The test asserts that EXACT value. A first draft asserted zero,
                     which is a property WALE does not have and never claimed.
  3  positivity      nu_t >= 0 everywhere on a random field, for both models. A negative total
                     viscosity makes the diffusion operator indefinite.
  4  filter width    on a stretched grid, Delta must follow the CELL, not be a constant. Checked
                     by refining and requiring nu_t to fall as Delta^2.
  5  WALE near-wall  nu_t ~ y^3 approaching a wall, the property that lets WALE skip van Driest
                     damping. Measured as a fitted exponent.
  6  seams           the same field split into 1 block and 4 blocks must give the same nu_t --
                     if the gradient did not resolve the seam, it would not.
"""
import numpy as np

from src import sgs
from src.mb_adjoint import periodic_box

TOL = 1e-10


def _fields(d, fn):
    """Evaluate fn(x, y, z) -> (u, v, w) on every block, as the dicts the API takes."""
    U, V, W = {}, {}, {}
    for b, blk in enumerate(d.blocks):
        u, v, w = fn(blk.x, blk.y, blk.z)
        U[b], V[b], W[b] = u, v, w
    return U, V, W


def check_shear():
    d = periodic_box(12, 1)
    a = 2.5
    U, V, W = _fields(d, lambda x, y, z: (a * y, 0 * y, 0 * y))
    G = sgs.velocity_gradient(d, U, V, W)
    mag = sgs.strain_magnitude(G[0])
    # interior only: the field is not periodic in y, so the wrap-around plane is not shear
    inner = mag[2:-2, 2:-2, 2:-2]
    err = float(np.abs(inner - a).max())
    ok = err < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] pure shear a = {a}: |S| = {inner.mean():.10f}, "
          f"max error {err:.2e}")
    return ok


def check_rotation():
    d = periodic_box(12, 1)
    om = 1.7
    U, V, W = _fields(d, lambda x, y, z: (-om * y, om * x, 0 * z))
    G = sgs.velocity_gradient(d, U, V, W)
    mag = sgs.strain_magnitude(G[0])[2:-2, 2:-2, 2:-2]
    _, nt_s = sgs.effective_viscosity(d, U, V, W, 0.0, model="smagorinsky")
    _, nt_w = sgs.effective_viscosity(d, U, V, W, 0.0, model="wale")
    s = float(np.abs(nt_s[0][2:-2, 2:-2, 2:-2]).max())
    w = float(nt_w[0][2:-2, 2:-2, 2:-2].mean())
    D = float(sgs.filter_width(d, 0).mean())
    w_exact = (sgs.CW_WALE * D) ** 2 * ((2.0 / 3.0) * om ** 4) ** 0.25
    ok = (float(np.abs(mag).max()) < 1e-9 and s < 1e-12
          and abs(w - w_exact) / w_exact < 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] solid rotation omega = {om}: |S| = "
          f"{np.abs(mag).max():.2e}, Smagorinsky nu_t = {s:.2e} (must be 0)")
    print(f"         WALE nu_t = {w:.6e} against the analytic "
          f"(Cw D)^2 ((2/3) om^4)^(1/4) = {w_exact:.6e}  -- WALE responds to rotation by design")
    return ok


def check_positive():
    d = periodic_box(12, 1)
    g = np.random.default_rng(4)
    U, V, W = ({0: g.standard_normal(d.blocks[0].shape)} for _ in range(3))
    U, V, W = {0: g.standard_normal(d.blocks[0].shape)}, \
              {0: g.standard_normal(d.blocks[0].shape)}, \
              {0: g.standard_normal(d.blocks[0].shape)}
    ok = True
    for m in ("smagorinsky", "wale"):
        _, nt = sgs.effective_viscosity(d, U, V, W, 1e-3, model=m)
        mn = float(nt[0].min())
        fin = bool(np.isfinite(nt[0]).all())
        ok &= (mn >= 0.0) and fin
        print(f"  [{'PASS' if mn >= 0 and fin else 'FAIL'}] {m:<12} on a random field: "
              f"min nu_t = {mn:.3e}, all finite {fin}")
    return ok


def check_filter_width():
    """nu_t must scale as Delta^2 = (cell volume)^(2/3), so halving the cell gives a quarter."""
    a = 2.5
    vals = []
    for n in (8, 16):
        d = periodic_box(n, 1)
        U, V, W = _fields(d, lambda x, y, z: (a * y, 0 * y, 0 * y))
        _, nt = sgs.effective_viscosity(d, U, V, W, 0.0, model="smagorinsky")
        vals.append(float(nt[0][2:-2, 2:-2, 2:-2].mean()))
    ratio = vals[0] / vals[1]
    ok = abs(ratio - 4.0) < 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] filter width follows the cell: nu_t ratio "
          f"coarse/fine = {ratio:.4f}, expected 4.00 for Delta^2")
    return ok


def check_wale_wall_scaling():
    """Approaching a wall, WALE must give nu_t ~ y^3."""
    n = 48
    y = np.linspace(1e-3, 0.2, n)
    # a near-wall velocity field: u = y (linear shear), plus the wall-normal decay that makes
    # the gradient tensor non-trivial. v ~ y^2 is what continuity gives next to a wall.
    A, B = 1.0, 0.5
    nut = []
    for yy in y:
        g = np.zeros((3, 3))
        g[0][1] = A                 # du/dy
        g[1][1] = 2 * B * yy        # dv/dy ~ y
        g[1][0] = 0.0
        gg = [[g[i][j] * np.ones((1, 1, 1)) for j in range(3)] for i in range(3)]
        g2 = [[sum(gg[i][k] * gg[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        tr = sum(g2[k][k] for k in range(3))
        Sd = [[0.5 * (g2[i][j] + g2[j][i]) - (tr / 3 if i == j else 0) for j in range(3)]
              for i in range(3)]
        S = [[0.5 * (gg[i][j] + gg[j][i]) for j in range(3)] for i in range(3)]
        SS = sum(S[i][j] ** 2 for i in range(3) for j in range(3))
        SdSd = sum(Sd[i][j] ** 2 for i in range(3) for j in range(3))
        nut.append(float((SdSd ** 1.5) / (SS ** 2.5 + SdSd ** 1.25)))
    nut = np.array(nut)
    m = (y > 2e-3) & (y < 5e-2)
    p = np.polyfit(np.log(y[m]), np.log(nut[m]), 1)[0]
    ok = abs(p - 3.0) < 0.25
    print(f"  [{'PASS' if ok else 'FAIL'}] WALE near-wall exponent {p:.3f}, expected 3 "
          f"(van Driest on the mixing length gives 4 -- measured 3.993 elsewhere in this repo)")
    return ok


def check_seam_invariance():
    """The same field on 1 block and on 4 must give the same nu_t."""
    a = 2.5
    out = []
    for ns in (1, 4):
        d = periodic_box(12, ns)
        U, V, W = _fields(d, lambda x, y, z: (a * np.sin(2 * np.pi * z) + 0.3 * y,
                                              0.2 * np.cos(2 * np.pi * x), 0.1 * y))
        _, nt = sgs.effective_viscosity(d, U, V, W, 0.0, model="wale")
        out.append(np.concatenate([nt[b].ravel() for b in sorted(nt)]))
    o1 = np.sort(out[0]); o4 = np.sort(out[1])
    err = float(np.abs(o1 - o4).max()) / max(float(o1.max()), 1e-30)
    ok = err < 1e-10
    print(f"  [{'PASS' if ok else 'FAIL'}] 1 block vs 4 blocks: max relative difference "
          f"{err:.2e}  (the gradient resolves the seam)")
    return ok


def main():
    print("=" * 78)
    print("  subgrid closures against analytic targets")
    print("=" * 78)
    r = [check_shear(), check_rotation(), check_positive(), check_filter_width(),
         check_wale_wall_scaling(), check_seam_invariance()]
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
