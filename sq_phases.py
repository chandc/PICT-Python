"""One shedding period of the square cylinder, saved at eight equally spaced phases.

The existing sqcyl_phase01-05 files are on the OLD grid -- 63,280 cells, written before the grid
fingerprint existed -- and are spaced 3 time units apart, which is not a fraction of the period.
They cannot illustrate this solution.

The period is measured, not assumed: zero crossings of C_L over the last 35 time units give
T = 6.7199 +/- 0.0218, so St = 0.1488 +/- 0.0005. An FFT of the same record returns 0.1429 with
a bin width of 0.0286 -- the crossings are 60x sharper, because the information is in the timing
of the crossings and not in the spectral resolution.
"""
import argparse
import os
import time

import numpy as np

from square_cylinder_bc import apply as apply_bc
from square_cylinder_grid import square_domain
from src import checkpoint
from src.piso_multiblock import MultiBlockPISO

RE, NU, U_INF = 100.0, 0.01, 1.0
T_PERIOD = 6.7199


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restart", default="results/fields/sqcyl_v3_forces.npz")
    p.add_argument("--phases", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nz", type=int, default=4)
    a = p.parse_args()

    d, _ = square_domain(nz=a.nz)
    nb = len(d.blocks)
    m = MultiBlockPISO(d, NU, a.dt, 2, 1e-6, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b in range(nb):
        m.u[b][:] = U_INF; m.v[b][:] = 0.0; m.w[b][:] = 0.0
    apply_bc(m, d)
    checkpoint.load(m, a.restart)
    print(f"  restarted at t = {m.time:.1f}; period {T_PERIOD:.4f} = "
          f"{T_PERIOD/a.dt:.0f} steps, {a.phases} phases", flush=True)

    n_tot = int(round(T_PERIOD / a.dt))
    marks = [int(round(k * n_tot / a.phases)) for k in range(1, a.phases + 1)]
    os.makedirs("results/fields", exist_ok=True)
    checkpoint.save(m, "results/fields/sqph_00.npz")
    print(f"  phase 0 saved at t = {m.time:.3f}", flush=True)
    t0 = time.time()
    k = 0
    for i in range(1, n_tot + 1):
        m.step()
        if i in marks:
            k += 1
            checkpoint.save(m, f"results/fields/sqph_{k:02d}.npz")
            print(f"  phase {k} saved at t = {m.time:.3f}  "
                  f"({i}/{n_tot} steps, {(time.time()-t0)/i:.2f} s/step)", flush=True)


if __name__ == "__main__":
    main()
