"""Stage 5c Layer 2 — van Driest as a MANUFACTURED SOLUTION for the variable-viscosity path.

The correlation defines both the eddy viscosity and, through the total-stress balance, the mean
profile that must result:

    nu_t+ = (kappa y+ D)^2 |dU+/dy+|,  D = 1 - exp(-y+/A+),  kappa = 0.41, A+ = 26
    (1 + nu_t+) dU+/dy+ = 1 - y/delta

So prescribing nu_t(y) and driving the channel with G = u_tau^2/delta makes the van Driest
profile the EXACT steady solution. No DNS data is needed and nothing is fitted: this is a known
answer for the whole variable-nu path -- the array viscosity in the momentum matrix, its face
average, the rotational pressure correction that multiplies by nu, and the deferred cross term.

NORMALISATION. u_tau = 1, delta = 1, so nu = 1/Re_tau, G = 1, and y+ = y*Re_tau. Everything
comes out in wall units with no rescaling.

GRID. 97 points wall to wall, geometric from each wall at ratio 1.05 with the first cell at
y+ = 1. Measured on the 1-D problem beforehand: that combination gives 0.25% error in the bulk
velocity, against 4.07% for 65 uniform points and 1.06% for 129 uniform. The clustering is worth
a factor of about five in point count, and all of it is bought in the sublayer -- Delta y+ = 4
at the wall costs 2.7% however many points sit in the core.
"""
import numpy as np

from src.multiblock import Block, Connection, Domain, face_id
from src.piso_multiblock import MultiBlockPISO

KAPPA, APLUS, RE_TAU = 0.41, 26.0, 180.0
NY, DY0_PLUS, RATIO = 97, 1.0, 1.05
NU = 1.0 / RE_TAU
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def van_driest_reference(n=2000001):
    """(y+, U+, nu_t+) on the half channel, integrated finely."""
    yp = np.linspace(0.0, RE_TAU, n)
    D = 1.0 - np.exp(-yp / APLUS)
    lm = KAPPA * yp * D
    tau = 1.0 - yp / RE_TAU
    dU = 2.0 * tau / (1.0 + np.sqrt(1.0 + 4.0 * lm ** 2 * tau))
    U = np.concatenate([[0.0], np.cumsum(0.5 * (dU[1:] + dU[:-1]) * np.diff(yp))])
    return yp, U, lm ** 2 * dU


def clustered_y():
    """97 points over y in [0, 2], geometric from BOTH walls at `RATIO`."""
    dy0 = DY0_PLUS / RE_TAU
    ys, dy = [0.0], dy0
    while ys[-1] < 1.0:
        ys.append(ys[-1] + dy)
        dy *= RATIO
    ys = np.array(ys) * 1.0 / ys[-1]
    full = np.concatenate([ys[:-1], 2.0 - ys[::-1]])
    return full


def channel(y):
    """One block: walls at y = 0 and 2, periodic in x and z. The solution is 1-D in y."""
    nx, nz = 4, 4
    x = np.arange(nx) / nx * 0.5
    z = np.arange(nz) / nz * 0.5
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    blk = Block((nx, len(y), nz), X, Y, Z, (1.0 / nx, 1.0 / (len(y) - 1), 1.0 / nz),
                period=(0.5, 1.0, 0.5))
    blk.faces[face_id(0, 0)] = blk.faces[face_id(0, 1)] = "periodic"
    blk.faces[face_id(2, 0)] = blk.faces[face_id(2, 1)] = "periodic"
    blk.faces[face_id(1, 0)] = blk.faces[face_id(1, 1)] = "wall"
    return Domain([blk])


print("\n" + "=" * 78 + "\n  Stage 5c Layer 2 — van Driest channel, Re_tau = 180\n" + "=" * 78)

yp_ref, U_ref, nut_ref = van_driest_reference()
Ub_ref = float(np.trapezoid(U_ref, yp_ref) / RE_TAU)

y = clustered_y()
d = channel(y)
blk = d.blocks[0]
yplus = np.minimum(blk.y, 2.0 - blk.y) * RE_TAU              # distance to the NEAREST wall
nu_eff = NU * (1.0 + np.interp(yplus, yp_ref, nut_ref))

print(f"  {len(y)} points in y, first cell y+ = {(y[1]-y[0])*RE_TAU:.2f}, "
      f"max spacing y+ = {np.diff(y).max()*RE_TAU:.2f}, ratio {RATIO}")
print(f"  nu_eff/nu spans {(nu_eff/NU).min():.2f} to {(nu_eff/NU).max():.2f}")

m = MultiBlockPISO(d, {0: nu_eff}, 0.05, 2, 1e-12, time_scheme="bdf2", scheme="rotational",
                   picard_iters=1, rhie_chow=False)
m.velocity_source = [1.0, 0.0, 0.0]                          # G = u_tau^2 / delta = 1
for _ in range(4000):
    m.step()

U = m.u[0][0, :, 0]
np.savez("results/van_driest_profile.npz", y=y, U=U, nu_eff=nu_eff[0, :, 0], NU=NU,
         RE_TAU=RE_TAU, yp=np.minimum(y, 2 - y) * RE_TAU)
Ub = float(np.trapezoid(U, y) / 2.0)
U_at = np.interp(np.minimum(y, 2.0 - y) * RE_TAU, yp_ref, U_ref)
err_prof = float(np.abs(U - U_at).max() / U_ref.max())

print(f"\n  {'y+':>8}{'U+ solver':>12}{'U+ van Driest':>15}{'diff':>10}")
for t in (1, 5, 30, 100, 179):
    i = int(np.argmin(np.abs(np.minimum(y, 2 - y) * RE_TAU - t)))
    print(f"  {np.minimum(y,2-y)[i]*RE_TAU:>8.1f}{U[i]:>12.3f}{U_at[i]:>15.3f}"
          f"{U[i]-U_at[i]:>10.4f}")

check(err_prof < 0.01,
      f"profile matches the manufactured solution: max|U+ - U+_vD| = {err_prof:.3%} of U_max")
check(abs(Ub - Ub_ref) / Ub_ref < 0.01,
      f"bulk velocity U_b+ = {Ub:.4f} against the exact {Ub_ref:.4f} "
      f"({abs(Ub-Ub_ref)/Ub_ref:.2%}; the 1-D study predicted 0.25% for this grid)")

# THE LOG-LAW CONSTANT MUST BE COMPARED LIKE FOR LIKE. At Re_tau = 180 there is barely a log
# layer -- y+ = 150 is y/delta = 0.83, deep in the wake -- so the accepted B = 5.0-5.2 does not
# apply. The van Driest REFERENCE itself gives B = 4.23 over [50,150] here, rising to 4.99 at
# Re_tau = 590 and 5.19 at 5200. Testing the solver against 5.2 measured the Reynolds number,
# not the solver: it read B = 4.27 and "failed" while agreeing with its own target to 0.04.
mask = (np.minimum(y, 2 - y) * RE_TAU > 50) & (np.minimum(y, 2 - y) * RE_TAU < 150)
ypm = np.minimum(y, 2 - y)[mask] * RE_TAU
B_solver = float(np.mean(U[mask] - np.log(ypm) / KAPPA))
mref = (yp_ref > 50) & (yp_ref < 150)
B_ref = float(np.mean(U_ref[mref] - np.log(yp_ref[mref]) / KAPPA))
check(abs(B_solver - B_ref) < 0.1,
      f"log-law constant matches the reference at the SAME Re_tau and window: solver "
      f"{B_solver:.3f} vs van Driest {B_ref:.3f}. Neither is the textbook 5.0-5.2, because "
      f"Re_tau = 180 has no log layer to speak of -- the same integration gives 4.99 at "
      f"Re_tau = 590 and 5.19 at 5200")

# a control: the same run with molecular viscosity only must be far off
m2 = MultiBlockPISO(d, NU, 0.05, 2, 1e-12, time_scheme="bdf2", scheme="rotational",
                    picard_iters=1, rhie_chow=False)
m2.velocity_source = [1.0, 0.0, 0.0]
for _ in range(4000):
    m2.step()
Ub_lam = float(np.trapezoid(m2.u[0][0, :, 0], y) / 2.0)
check(Ub_lam > 3 * Ub_ref,
      f"control: the SAME force with molecular nu alone gives U_b+ = {Ub_lam:.1f}, "
      f"{Ub_lam/Ub_ref:.1f}x the turbulent value -- so the eddy viscosity is genuinely "
      f"carrying the momentum transport, not decorating it")

print("=" * 78)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 78)
raise SystemExit(1 if FAIL else 0)
