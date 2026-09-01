"""Stage 5c, Layer 1 — the variable-viscosity operator, before any network exists.

Plan: `reference/nn_piso_plan.md` Stage 5c and `reference/implementation_plan.md` 5.2.

`build_momentum_matrix` now takes `nu` as a scalar OR a per-block array of
nu_eff = nu + nu_t(x). Folding it in at `Jg_of` rather than at the face is what keeps the
operator symmetric: every face takes 0.5*(Jg_lo + Jg_hi), so a per-cell nu is arithmetically
face-averaged along with J and the metric, and the two rows a face writes stay equal.

THE ORDER OF THESE TESTS IS THE POINT. 5c.1 is the obvious one and it is BLIND to the thing most
likely to be wrong: interpolating a CONSTANT to a face is exact however you do it, so a
first-order or one-sided face coefficient passes it. Only a nu that varies can catch that, which
is what 5c.7 is for.
"""
import numpy as np
from scipy import sparse

from src.mb_adjoint import MultiBlockMiniPISO, periodic_box

PASS = FAIL = 0
DT = 0.02


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def build(d, p, nu, dt=DT):
    """A(u=0): the diagonal J/dt plus the diffusion operator, convection switched off."""
    z = {b: np.zeros(bl.shape) for b, bl in enumerate(d.blocks)}
    return sparse.csr_matrix(d.build_momentum_matrix(p.Js, p.ms, z, z, z, nu, dt))


def flat(d, per_block):
    out = np.zeros(d.n_cells)
    for b in range(len(d.blocks)):
        out[d.global_ids(b).ravel()] = np.asarray(per_block[b]).ravel()
    return out


print("\n" + "=" * 78 + "\n  Stage 5c Layer 1 — variable effective viscosity\n" + "=" * 78)

d = periodic_box(10, 2)
p = MultiBlockMiniPISO(d, 0.05, DT)
NU0, C = 0.05, 0.017

# ---------------------------------------------------------------- 5c.1 constant nu_t
A_scalar = build(d, p, NU0 + C)
A_const = build(d, p, {b: np.full(bl.shape, NU0 + C) for b, bl in enumerate(d.blocks)})
dif = (A_const - A_scalar).tocoo()
e1 = float(np.abs(dif.data).max()) if dif.nnz else 0.0
check(e1 < 1e-14,
      f"5c.1  a CONSTANT nu_t array reproduces the scalar path: max|dA| = {e1:.2e}. "
      f"Necessary and blind -- interpolating a constant to a face is exact however it is done")

# ---------------------------------------------------------------- 5c.2 symmetry
rng = np.random.default_rng(3)
nuf = {b: rng.uniform(0.02, 0.4, bl.shape) for b, bl in enumerate(d.blocks)}
Av = build(d, p, nuf)
asym = (Av - Av.T).tocoo()
e2 = float(np.abs(asym.data).max()) if asym.nnz else 0.0
check(e2 == 0.0,
      f"5c.2  with a VARYING nu the operator stays exactly symmetric: max|A - A^T| = {e2:.1e} "
      f"(a one-sided face coefficient would break this)")

# ---------------------------------------------------------------- 5c.9 row sums
Jd = sparse.diags(flat(d, {b: p.Js[b] / DT for b in range(len(d.blocks))}))
D = (Av - Jd).tocsr()
rs = np.abs(np.asarray(D.sum(axis=1)).ravel()).max()
scale = np.abs(D).max()
check(rs / scale < 1e-12,
      f"5c.9  row sums of the diffusion part vanish for any nu(x) > 0: max|sum| = {rs:.2e} "
      f"against entries of {scale:.2e} -- a constant field is in the null space, which is "
      f"momentum conservation")

# ---------------------------------------------------------------- 5c.10 definiteness
# RANDOM PROBING DOES NOT WORK HERE, and finding that out is most of the value. A nu with a
# negative PATCH still gave min v.Dv = +4.9e+04 over 200 random vectors: the bad direction is a
# localised mode and a random vector has almost no overlap with it. The smallest EIGENVALUE
# finds it immediately -- and the control below proves the test can fail.
from scipy.sparse.linalg import eigsh

lo_ok = float(eigsh(D, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
nu_bad = {b: np.full(bl.shape, 0.1) for b, bl in enumerate(d.blocks)}
nu_bad[0][3, 3, 3] = -0.5                       # ONE cell with a negative eddy viscosity
D_bad = (build(d, p, nu_bad) - Jd).tocsr()
lo_bad = float(eigsh(D_bad, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
check(lo_ok > -1e-10,
      f"5c.10 the diffusion part is positive semi-definite for nu > 0: smallest eigenvalue "
      f"{lo_ok:+.3e} (zero is the constant null vector, which 5c.9 is about)")
check(lo_bad < -1.0,
      f"      control: ONE cell of negative nu_t makes it indefinite, smallest eigenvalue "
      f"{lo_bad:+.3e}. Diffusion must remove energy at every wavenumber, and this is where an "
      f"unclipped network output surfaces -- not in a run that diverges three hours later")

# ---------------------------------------------------------------- 5c.7 MMS, varying nu
# The operator is the VOLUME-INTEGRATED -div(nu grad .), so dividing by J recovers the
# differential operator and the error must fall as h^2.
print(f"\n  {'ntot':>6}{'max err':>14}{'order':>9}")
prev = None
rates = []
for ntot in (8, 16, 32):
    dm = periodic_box(ntot, 2)
    pm = MultiBlockMiniPISO(dm, 0.05, DT)
    nu_f, pe_f, rhs_f = {}, {}, {}
    for b, bl in enumerate(dm.blocks):
        X, Y, Z = 2 * np.pi * bl.x, 2 * np.pi * bl.y, 2 * np.pi * bl.z
        nu_f[b] = 1.0 + 0.5 * np.sin(X)
        pe_f[b] = np.sin(X) * np.sin(Y) * np.sin(Z)
        lap = -3.0 * (2 * np.pi) ** 2 * pe_f[b]
        dnu_dx = 0.5 * 2 * np.pi * np.cos(X)
        dp_dx = 2 * np.pi * np.cos(X) * np.sin(Y) * np.sin(Z)
        rhs_f[b] = -(nu_f[b] * lap + dnu_dx * dp_dx)      # -div(nu grad p)
    Am = build(dm, pm, nu_f)
    Jm = sparse.diags(flat(dm, {b: pm.Js[b] / DT for b in range(len(dm.blocks))}))
    Dm = (Am - Jm).tocsr()
    Jflat = flat(dm, {b: pm.Js[b] for b in range(len(dm.blocks))})
    got = (Dm @ flat(dm, pe_f)) / Jflat
    err = float(np.abs(got - flat(dm, rhs_f)).max() / np.abs(flat(dm, rhs_f)).max())
    order = np.log2(prev / err) if prev else float("nan")
    if prev:
        rates.append(order)
    print(f"  {ntot:>6}{err:>14.3e}{order:>9.2f}")
    prev = err
check(min(rates) > 1.8,
      f"5c.7  MMS with a VARYING nu converges at order {min(rates):.2f}-{max(rates):.2f}. "
      f"This is the test 5c.1 cannot do: a first-order or cell-valued face coefficient shows "
      f"up here as a rate near 1")

print("=" * 78)
print(f"  {PASS}/{PASS + FAIL} checks passed  (Layer 2 and the transpose term still to come)")
print("=" * 78)
raise SystemExit(1 if FAIL else 0)
