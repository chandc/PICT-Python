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
    for lo, hi in ((5, 13), (13, 20), (20, 28), (28, 30.1)):
        s = away & (R >= lo) & (R < hi)
        bands[f"r {lo}-{hi:g}"] = float(dist[s].max()) if s.sum() else float("nan")
    ang = {}
    m25 = away & (R > 25)
    for lo, hi in ((0, 22), (22, 45), (45, 90)):
        s = m25 & (np.abs(TH) >= lo) & (np.abs(TH) < hi)
        ang[f"|th| {lo}-{hi}"] = float(dist[s].max()) if s.sum() else float("nan")
    return bands, ang


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("control", "arc", "sponge"), required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--restart", default="results/fields/cyl_expt_base.npz")
    p.add_argument("--sponge-from", type=float, default=12.0)
    p.add_argument("--sponge-mult", type=float, default=50.0)
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
        nu = {}
        for b, blk in enumerate(d.blocks):
            rad = np.hypot(blk.x, blk.y)
            ramp = np.clip((rad - a.sponge_from) / (r[-1] - a.sponge_from), 0.0, 1.0) ** 2
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
              f"max Pe {2.176/mx:.0f}")
    b0, a0 = disturbance(d, m, nb)
    print(f"  start t = {m.time:.1f}:  " + "  ".join(f"{k} {v:.3f}" for k, v in b0.items()))

    t0 = time.time()
    for i in range(1, a.steps + 1):
        m.step()
        if i % 250 == 0:
            bb, aa = disturbance(d, m, nb)
            print(f"  t = {m.time:7.1f}  " + "  ".join(f"{k} {v:.3f}" for k, v in bb.items())
                  + f"   {(time.time()-t0)/i:.2f} s/step", flush=True)
    bb, aa = disturbance(d, m, nb)
    print(f"\n  FINAL t = {m.time:.1f}")
    print("    radial:  " + "  ".join(f"{k} {v:.4f}" for k, v in bb.items()))
    print("    angular (r>25):  " + "  ".join(f"{k} {v:.4f}" for k, v in aa.items()))
    os.makedirs("results/fields", exist_ok=True)
    checkpoint.save(m, f"results/fields/cyl_expt_{a.mode}.npz")


if __name__ == "__main__":
    main()
