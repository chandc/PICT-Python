"""Two controlled experiments on the cylinder's far-field oscillation.

The disturbance grows monotonically with cell size -- 0.13 at Pe = 28 to 4.08 at Pe = 194 --
which points at central differencing on a coarsening grid. But its ANGULAR peak sits in
22-45 deg, immediately outside the Dong outflow arc at 21.8 deg, so a boundary-condition
junction cannot be ruled out by that evidence alone.

These are the two independent knobs:

  arc     widen the outflow arc from 21.8 to 44.3 deg (sector 45 -> 90). If the disturbance
          pattern is unchanged, the BC junction is not driving it.
  sponge  raise nu_eff in the outer region so the cell PECLET number falls, with the grid
          untouched. Pe = |u| dr / nu_eff, so raising nu is equivalent to shrinking dr for
          this mechanism. If the disturbance collapses, Peclet is confirmed as the generator.

Both restart from the SAME snapshot so the comparison is matched, and the production run
continues untouched as the control.
"""
import argparse
import os
import time

import numpy as np

import cylinder_bc
import cylinder_grid
from cylinder_grid import cylinder_domain, D, R_CYL, outer_role
from cylinder_bc import apply as apply_bc, U_INF
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO

RE, NU = 100.0, 0.01


def disturbance(d, m, nb):
    """max |u - U_inf| by radial band, away from the wake -- the metric under test."""
    X = np.concatenate([b.x[:, :, 0].ravel() for b in d.blocks])
    Y = np.concatenate([b.y[:, :, 0].ravel() for b in d.blocks])
    U = np.concatenate([m.u[b][:, :, 0].ravel() for b in range(nb)])
    V = np.concatenate([m.v[b][:, :, 0].ravel() for b in range(nb)])
    R = np.hypot(X, Y)
    TH = np.degrees(np.arctan2(Y, X))
    dist = np.hypot(U - U_INF, V)
    away = np.abs(Y) > 3.0
    bands = {}
    for lo, hi in ((5, 10), (10, 14), (14, 17), (17, 19), (19, 20.1)):
        s = away & (R >= lo) & (R < hi)
        bands[f"r {lo}-{hi:g}"] = float(dist[s].max()) if s.sum() else float("nan")
    # THE PEAK ANGLE IS THE WHOLE EXPERIMENT. The disturbance sits at theta = -24.6, just
    # outside the outflow arc edge at 21.8. If it is the junction that generates it, moving the
    # edge to 44.3 moves the peak with it; if it is anything else -- the grid, the wake, the
    # outer boundary as such -- the peak stays where it is. That is a far sharper test than
    # asking whether the pattern "looks different".
    m = away & (R > 0.85 * R.max())
    i = int(np.argmax(dist[m]))
    peak = (float(np.abs(TH[m][i])), float(R[m][i]), float(dist[m].max()))
    ang = {}
    for lo, hi in ((0, 22), (22, 45), (45, 90), (90, 180.1)):
        s = m & (np.abs(TH) >= lo) & (np.abs(TH) < hi)
        ang[f"|th| {lo}-{hi}"] = float(dist[s].max()) if s.sum() else float("nan")
    return bands, ang, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("control", "arc", "sponge"), required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--restart", default="results/fields/cyl_expt_base.npz")
    p.add_argument("--sponge-from", type=float, default=12.0)
    p.add_argument("--sponge-mult", type=float, default=5.0)
    p.add_argument("--sponge-outflow", action="store_true",
                   help="also raise nu in the outflow blocks (the first attempt did, and "
                        "diverged: the Dong condition uses nu, so a 50x sponge over the arc "
                        "rewrites the outflow BC instead of damping the far field)")
    a = p.parse_args()

    if a.mode == "arc":
        # force a 90-degree sector, i.e. |theta| <= 44.3 with 16 blocks. cylinder_bc holds its
        # OWN reference to outer_role, so patching the module constant is not enough.
        cylinder_bc.outer_role = lambda dom, nblk, flow_x=True, sector=90.0: \
            outer_role(dom, nblk, flow_x, 90.0)

    d, r, _ = cylinder_domain(nblk=16, nz=4)
    nb = len(d.blocks)
    roles = outer_role(d, nb, sector=90.0 if a.mode == "arc" else 45.0)
    outs = sorted(b for b in roles if roles[b] == "outflow")

    nu = NU
    if a.mode == "sponge":
        # THE FIRST SPONGE DIVERGED AND THE HYPOTHESIS WAS NOT WHY. It multiplied nu by 50
        # everywhere past r = 14, WHICH INCLUDES THE OUTFLOW ARC, and the Dong condition is
        # written in terms of nu -- so it did not damp the far field against an unchanged
        # boundary condition, it rewrote the boundary condition. |u - U_inf| went 0.126 ->
        # 1.011 in six time units and then to NaN, with a divide-by-zero in the Rhie-Chow row
        # sums. That says nothing about whether the far field is convectively unstable.
        #
        # So: leave the outflow blocks alone by default, and default the multiplier to 5, which
        # still takes the outer cell Peclet number from 29 to 6 -- well below the 29 that the
        # grid rebuild reached, and the grid rebuild is what showed 218 -> 29 was not enough.
        nu = {}
        for b, blk in enumerate(d.blocks):
            rad = np.hypot(blk.x, blk.y)
            ramp = np.clip((rad - a.sponge_from) / (r[-1] - a.sponge_from), 0.0, 1.0) ** 2
            if roles.get(b) == "outflow" and not a.sponge_outflow:
                ramp = np.zeros_like(ramp)
            nu[b] = NU * (1.0 + (a.sponge_mult - 1.0) * ramp)

    m = MultiBlockPISO(d, nu, 0.01, 2, 1e-6, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b in range(nb):
        m.u[b][:] = U_INF; m.v[b][:] = 0.0; m.w[b][:] = 0.0
    apply_bc(m, d, nb)
    # the sponge changes nu, which the checkpoint records as configuration -- a deliberate
    # change, which is exactly what strict=False documents
    checkpoint.load(m, a.restart, strict=(a.mode != "sponge"))

    print(f"  mode = {a.mode};  outflow blocks {outs} "
          f"({'|theta| <= 44.3' if a.mode == 'arc' else '|theta| <= 21.8'})")
    if a.mode == "sponge":
        mx = max(float(v.max()) for v in nu.values())
        print(f"  sponge from r = {a.sponge_from}, nu_eff up to {mx/NU:.0f}x -> "
              f"max Pe {0.285/mx:.0f};  outflow blocks {outs} "
              f"{'INCLUDED' if a.sponge_outflow else 'left at molecular nu'}")
    b0, a0, p0 = disturbance(d, m, nb)
    print(f"  start t = {m.time:.1f}:  " + "  ".join(f"{k} {v:.3f}" for k, v in b0.items()))
    print(f"  start peak |theta| = {p0[0]:.1f} deg at r = {p0[1]:.1f}, |u-U| = {p0[2]:.4f}")

    t0 = time.time()
    for i in range(1, a.steps + 1):
        m.step()
        if i % 500 == 0:
            # SAVE AS WE GO. Both of these experiments were once stopped mid-flight to free the
            # GPU and lost everything, because the only save was after the loop. A restartable
            # experiment costs one file write per 500 steps.
            checkpoint.save(m, f"results/fields/cyl_expt_{a.mode}.npz")
        if i % 200 == 0:
            bb, aa, pp = disturbance(d, m, nb)
            print(f"  t = {m.time:7.1f}  peak |th| {pp[0]:5.1f} deg  r {pp[1]:4.1f}  "
                  f"|u-U| {pp[2]:.4f}   " + "  ".join(f"{k} {v:.3f}" for k, v in bb.items())
                  + f"   {(time.time()-t0)/i:.2f} s/step", flush=True)
    bb, aa, pp = disturbance(d, m, nb)
    print(f"\n  FINAL t = {m.time:.1f}   peak |theta| = {pp[0]:.1f} deg at r = {pp[1]:.1f}")
    print("    radial:  " + "  ".join(f"{k} {v:.4f}" for k, v in bb.items()))
    print("    angular (r>25):  " + "  ".join(f"{k} {v:.4f}" for k, v in aa.items()))
    os.makedirs("results/fields", exist_ok=True)
    checkpoint.save(m, f"results/fields/cyl_expt_{a.mode}.npz")


if __name__ == "__main__":
    main()
