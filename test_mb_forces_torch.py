"""Stage 8 — the force integral in torch: same numbers, and now differentiable.

Plan: `reference/nn_multiblock_plan.md`.

  8.0  the torch port reproduces the NumPy `surface_force` on RANDOM fields, every part
       separately -- pressure, viscous tangential, viscous normal. Random rather than physical,
       because a physical field makes several parts small and a wrong one can hide in the noise.
  8.1  the four analytic checks from `test_forces.py`, re-run through the torch path.
  8.2  dC_D/d(fields) against central finite differences.
  8.3  the tangential-only and full-traction gradients must DIFFER, or the contamination
       described in src/forces.py cannot be trained away.
  8.4  dC_L must be antisymmetric under y -> -y on a mirror-symmetric configuration.
  8.5  FD vs adjoint through a WALL-BOUNDED chain, Dirichlet velocity rows eliminated.
  8.6  dC_D/d(actuation) -- drag as the loss, through the solver and the wall integral.
  8.7  a Dong outlet makes the pressure operator NON-SINGULAR.
  8.8  FD vs adjoint with that non-singular solve.
  8.9  Dong's boundary value depends on the interior solution, and that dependence is in the
       graph -- a nonlinear BC, not a constant.
"""
import numpy as np
import torch

from src.forces import surface_force as np_surface_force
from src.forces_torch import coefficients, face_geometry, surface_force
from src.multiblock import Block, Domain, face_id

torch.set_default_dtype(torch.float64)
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
    xs = [np.linspace(0.0, L[a], n[a]) for a in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    return Domain([Block(n, X, Y, Z, tuple(1.0 / (n[a] - 1) for a in range(3)))]), L


def as_torch(per_block, requires_grad=False):
    return {b: torch.tensor(np.asarray(a), requires_grad=requires_grad)
            for b, a in per_block.items()}


print("\n" + "=" * 76 + "\n  Stage 8 — surface forces in torch\n" + "=" * 76)

# ---------------------------------------------------------------- 8.0 port fidelity
from cylinder_grid import cylinder_domain, R_CYL

dc, r, _ = cylinder_domain(nblk=16, nz=4)
body = [(b, face_id(0, 0)) for b in range(len(dc.blocks))]
geom = face_geometry(dc, body)
rng = np.random.default_rng(0)
fields = {nm: {b: rng.standard_normal(blk.shape) for b, blk in enumerate(dc.blocks)}
          for nm in "uvwp"}
NU = 0.01

ref = np_surface_force(dc, body, fields["u"], fields["v"], fields["w"], fields["p"], NU,
                       check_wall=False)
got = surface_force(geom, *(as_torch(fields[nm]) for nm in "uvwp"), NU)
worst, scale = 0.0, 0.0
for key in ("pressure", "viscous_tangential", "viscous_normal", "total"):
    a = got[key].detach().numpy()
    worst = max(worst, np.abs(a - ref[key]).max())
    scale = max(scale, np.abs(ref[key]).max())
check(worst / scale < 1e-13,
      f"8.0  torch reproduces NumPy on random fields, part by part: worst {worst:.2e} "
      f"of {scale:.3e} (rel {worst/scale:.2e})")

# ---------------------------------------------------------------- 8.1 analytic checks
d, L = box()
blk = d.blocks[0]
allf = [(0, face_id(a, s)) for a in range(3) for s in (0, 1)]
gbox = face_geometry(d, allf)
z3 = {0: np.zeros(blk.shape)}

F = surface_force(gbox, as_torch(z3), as_torch(z3), as_torch(z3),
                  as_torch({0: np.full(blk.shape, 3.7)}), 0.0)["total"].detach().numpy()
check(np.abs(F).max() < 1e-12, f"8.1  uniform pressure on a closed box: |F| = {np.abs(F).max():.2e}")

a = np.array([2.0, -1.0, 0.5])
F = surface_force(gbox, as_torch(z3), as_torch(z3), as_torch(z3),
                  as_torch({0: a[0]*blk.x + a[1]*blk.y + a[2]*blk.z}),
                  0.0)["total"].detach().numpy()
want = a * L[0] * L[1] * L[2]
check(np.abs(F - want).max() / np.abs(want).max() < 1e-12,
      f"     linear pressure, fluid INSIDE the box: F = +aV, rel "
      f"{np.abs(F-want).max()/np.abs(want).max():.2e}")

gam, nu = 1.7, 0.023
F = surface_force(face_geometry(d, [(0, face_id(1, 0))]), as_torch({0: gam * blk.y}),
                  as_torch(z3), as_torch(z3), as_torch(z3), nu)["total"].detach().numpy()
want_x = nu * gam * L[0] * L[2]
check(abs(F[0] - want_x) / want_x < 1e-12,
      f"     Couette wall shear: F_x = {F[0]:.9f} vs nu*gamma*A = {want_x:.9f}")

span = float(dc.blocks[0].period[2])
pc = {b: 2.0 * bl.x for b, bl in enumerate(dc.blocks)}
zc = {b: np.zeros(bl.shape) for b, bl in enumerate(dc.blocks)}
F = surface_force(geom, as_torch(zc), as_torch(zc), as_torch(zc), as_torch(pc),
                  0.0)["total"].detach().numpy()
want = -2.0 * np.pi * R_CYL ** 2 * span
check(abs(F[0] - want) / abs(want) < 2e-3,
      f"     linear pressure, fluid AROUND the cylinder: F_x = {F[0]:.6f} vs -a*pi*R^2*L = "
      f"{want:.6f} (polygon vs circle is O(N^-2))")

# ---------------------------------------------------------------- 8.2 gradient vs FD
ut, vt, wt, pt = (as_torch(fields[nm], requires_grad=True) for nm in "uvwp")
cd, cl = coefficients(geom, ut, vt, wt, pt, NU, span)
cd.backward()
gp = {b: pt[b].grad.detach().numpy().copy() for b in pt}
gu = {b: ut[b].grad.detach().numpy().copy() for b in ut}
scale82 = max(max(np.abs(g).max() for g in gp.values()),
              max(np.abs(g).max() for g in gu.values()))


def cd_of(pert_field, b, idx, delta):
    mod = {nm: {bb: fields[nm][bb].copy() for bb in fields[nm]} for nm in "uvwp"}
    mod[pert_field][b][idx] += delta
    c, _ = coefficients(geom, *(as_torch(mod[nm]) for nm in "uvwp"), NU, span)
    return float(c)


eps, worst82 = 1e-4, 0.0
for nm, gdict in (("p", gp), ("u", gu)):
    flat = np.concatenate([g.ravel() for g in gdict.values()])
    for pick in np.argsort(-np.abs(flat))[:3]:
        b = int(pick // gdict[0].size)
        idx = np.unravel_index(int(pick % gdict[0].size), gdict[0].shape)
        fd = (cd_of(nm, b, idx, eps) - cd_of(nm, b, idx, -eps)) / (2 * eps)
        worst82 = max(worst82, abs(fd - gdict[b][idx]) / scale82)
check(worst82 < 1e-8,
      f"8.2  dC_D/d(fields) vs central FD on the largest-|g| entries: worst {worst82:.2e} of "
      f"max|g| = {scale82:.3e}")

# ---------------------------------------------------------------- 8.3 the two objectives
ut2, vt2, wt2, pt2 = (as_torch(fields[nm], requires_grad=True) for nm in "uvwp")
cd_t, _ = coefficients(geom, ut2, vt2, wt2, pt2, NU, span, tangential_only=True)
cd_t.backward()
gu_t = {b: ut2[b].grad.detach().numpy().copy() for b in ut2}
diff = max(np.abs(gu_t[b] - gu[b]).max() for b in gu) / scale82
check(diff > 1e-3,
      f"8.3  tangential-only and full-traction gradients DIFFER by {diff:.2e} of max|g|. If "
      f"they did not, a network told to reduce C_D could not be prevented from reducing the "
      f"spurious normal stress instead, and the objective would need rethinking")

# ---------------------------------------------------------------- 8.4 lift antisymmetry
# A mirror-symmetric field must give zero lift, and the SENSITIVITY of lift to a perturbation
# must flip sign under y -> -y. Both follow from the geometry alone, so a failure is an
# asymmetry in the wall integration -- the same class of defect as the far-field BC offset.
X = np.concatenate([b.x[:, :, 0].ravel() for b in dc.blocks])
Y = np.concatenate([b.y[:, :, 0].ravel() for b in dc.blocks])
sym = {b: (np.cos(2 * np.pi * bl.x) * np.cos(2 * np.pi * bl.y)) for b, bl in enumerate(dc.blocks)}
pts, uts = as_torch(sym, requires_grad=True), as_torch(zc)
_, cl_s = coefficients(geom, uts, as_torch(zc), as_torch(zc), pts, NU, span)
cl_s.backward()
gsym = np.concatenate([pts[b].grad.detach().numpy()[:, :, 0].ravel() for b in pts])
key = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(zip(X, Y))}
mir = np.array([key.get((round(x, 9), round(-y, 9)), -1) for x, y in zip(X, Y)])
ok = mir >= 0
anti = np.abs(gsym[ok] + gsym[mir[ok]]).max() / max(np.abs(gsym).max(), 1e-300)
check(ok.all() and anti < 1e-12,
      f"8.4  dC_L/dp is ANTISYMMETRIC under y -> -y: worst {anti:.2e} of max|g|, over "
      f"{int(ok.sum()):,} mirror pairs (|C_L| itself = {abs(float(cl_s)):.2e})")

# ------------------------------------------------- 8.5 / 8.6 drag AS the loss, on a wall
# Everything above differentiates the force with respect to the FIELDS. The objective Stage 10
# needs is the force with respect to an ACTUATION, which means chaining it to a solver on a
# domain that has a wall at all.
from src.forces_torch import coefficients as t_coefficients
from src.mb_adjoint import MultiBlockWallChain, channel_box

dch = channel_box(10, 2)
wall_chain = MultiBlockWallChain(dch, 0.05, 0.02)
walls = [(b, face_id(1, s)) for b in range(len(dch.blocks)) for s in (0, 1)]
wgeom = face_geometry(dch, walls)
span_ch = 1.0
rng5 = np.random.default_rng(41)
S5 = [rng5.standard_normal(wall_chain.N) * 0.05 for _ in range(3)]


def drag_of(arrays):
    src = [torch.tensor(a) if not torch.is_tensor(a) else a for a in arrays]
    u, pf = wall_chain.rollout(src, return_fields=True)
    ub = wall_chain.to_blocks(u)
    pb = wall_chain.to_blocks(pf)
    zb = {b: torch.zeros_like(ub[b]) for b in ub}
    cd, _ = t_coefficients(wgeom, ub, zb, zb, pb, 0.05, span_ch)
    return cd


src5 = [torch.tensor(a, requires_grad=True) for a in S5]
L5 = wall_chain.rollout(src5)
L5.backward()
g5 = [s.grad.detach().numpy().copy() for s in src5]
scale5 = max(np.abs(g).max() for g in g5)
worst85 = 0.0
for k_step in (0, 2):
    gg = g5[k_step]
    for k in np.argsort(-np.abs(gg))[:3]:
        pert = [a.copy() for a in S5]
        pert[k_step][k] += 1e-2
        Lp = float(wall_chain.rollout([torch.tensor(a) for a in pert]))
        pert[k_step][k] -= 2e-2
        Lm = float(wall_chain.rollout([torch.tensor(a) for a in pert]))
        worst85 = max(worst85, abs((Lp - Lm) / 2e-2 - gg[k]) / scale5)
check(worst85 < 1e-6,
      f"8.5  FD vs adjoint through a WALL-BOUNDED chain ({int(wall_chain.wall.sum())} Dirichlet "
      f"nodes eliminated, {len(wall_chain.interior)} unknowns): worst {worst85:.2e} of max|g|")

srcd = [torch.tensor(a, requires_grad=True) for a in S5]
cd_val = drag_of(srcd)
cd_val.backward()
gcd = [s.grad.detach().numpy().copy() for s in srcd]
scale86 = max(np.abs(g).max() for g in gcd)
worst86 = 0.0
for k_step in (0, 2):
    gg = gcd[k_step]
    for k in np.argsort(-np.abs(gg))[:3]:
        pert = [a.copy() for a in S5]
        pert[k_step][k] += 1e-2
        cp = float(drag_of(pert))
        pert[k_step][k] -= 2e-2
        cm = float(drag_of(pert))
        worst86 = max(worst86, abs((cp - cm) / 2e-2 - gg[k]) / scale86)
check(worst86 < 1e-6 and scale86 > 0,
      f"8.6  dC_D/d(actuation) through the solver AND the wall integral: worst {worst86:.2e} "
      f"of max|g| = {scale86:.3e}, C_D = {float(cd_val):+.5f}. This is the objective Stage 10 "
      f"would train against -- a body force in the interior, a drag on the wall, and one "
      f"gradient joining them")

# ------------------------------------------------- 8.7 / 8.8 / 8.9 the Dong outflow
# Everything above ran with a SINGULAR pressure system -- periodic or walled, pure Neumann -- so
# LinearSolve always took the `singular=True` branch with its projection in both passes. A Dong
# outlet prescribes a pressure and the null space disappears.
from src.mb_adjoint import MultiBlockDongChain, outflow_channel

ddg = outflow_channel(10, 2)
dong = MultiBlockDongChain(ddg, 0.05, 0.02)
one = np.ones(dong.M_ff.shape[0])
resid = np.abs(dong.M_ff @ one).max()
check(resid > 1e-6,
      f"8.7  the reduced pressure operator is NON-SINGULAR: |M_ff @ 1| = {resid:.3e} over "
      f"{dong.M_ff.shape[0]} unknowns with {len(dong.pD)} Dong nodes eliminated (0 would mean "
      f"the constant is still in the null space and the projection is still needed)")

rng7 = np.random.default_rng(51)
S7 = [rng7.standard_normal(dong.N) * 0.05 for _ in range(3)]
src7 = [torch.tensor(a, requires_grad=True) for a in S7]
L7 = dong.rollout(src7)
L7.backward()
g7 = [s.grad.detach().numpy().copy() for s in src7]
scale7 = max(np.abs(g).max() for g in g7)
worst88 = 0.0
for k_step in (0, 2):
    gg = g7[k_step]
    for k in np.argsort(-np.abs(gg))[:3]:
        pert = [a.copy() for a in S7]
        pert[k_step][k] += 1e-2
        Lp = float(dong.rollout([torch.tensor(a) for a in pert]))
        pert[k_step][k] -= 2e-2
        Lm = float(dong.rollout([torch.tensor(a) for a in pert]))
        worst88 = max(worst88, abs((Lp - Lm) / 2e-2 - gg[k]) / scale7)
check(worst88 < 1e-6,
      f"8.8  FD vs adjoint through the chain with a non-singular pressure solve: worst "
      f"{worst88:.2e} of max|g| = {scale7:.3e}")

src7d = [torch.tensor(a, requires_grad=True) for a in S7]
dong.rollout(src7d, detach_dong=True).backward()
g7d = [s.grad.detach().numpy().copy() for s in src7d]
rel89 = max(np.abs(a - b).max() for a, b in zip(g7, g7d)) / scale7
check(rel89 > 1e-12,
      f"8.9  Dong's pressure DEPENDS on the interior solution and that dependence carries a "
      f"gradient: detaching it changes dL/dS by {rel89:.2e} of max|g|. The boundary value is "
      f"nu (u_n - u_n,i)/dn - 0.5|u|^2 theta, so u_n,i at the first interior node feeds back "
      f"through the pressure -- a nonlinear BC in the graph, not a constant")

print("=" * 76)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 76)
raise SystemExit(1 if FAIL else 0)
