"""Re-stamp checkpoints written before the fingerprint was made machine-independent.

WHY THIS IS SAFE HERE AND WOULD NOT BE IN GENERAL. The stored digest is the only record of the
grid a file was written on, so nothing inside the file can prove two machines built the same
grid. This migration is justified by a DIRECT MEASUREMENT instead: cylinder_grid.py was run on
both machines and the coordinates compared element by element --

    max |spark - mac| = 3.553e-15 over x and y, exactly one ULP at r = 20
    r identical to the bit
    numpy 2.4.6 on the GB10, 2.0.2 on the Mac

-- so the grids are the same to round-off and the digests differed only because the old
fingerprint hashed raw float64 bytes. Files are re-stamped ONLY if their block count and field
shapes match the rebuilt domain, and only the named ones, never a blanket sweep.
"""
import sys

import numpy as np

from src import checkpoint
from cylinder_grid import cylinder_domain

FILES = ["cyl_v2.npz", "cyl_v2_base.npz", "cyl_clean_t60.npz", "cyl_v2_t60.npz",
         "cyl_expt_control.npz", "cyl_expt_arc.npz"]


class _Holder:
    pass


def main():
    d, _, _ = cylinder_domain(nblk=16, nz=4)
    h = _Holder(); h.d = d
    want = checkpoint.grid_fingerprint(h)
    print(f"  quantised fingerprint for the current cylinder grid: {want}\n")
    for name in FILES:
        p = f"results/fields/{name}"
        try:
            z = dict(np.load(p, allow_pickle=False))
        except FileNotFoundError:
            print(f"  {name:<26} absent, skipped")
            continue
        if int(z["nblocks"]) != len(d.blocks):
            print(f"  {name:<26} REFUSED: {int(z['nblocks'])} blocks, grid has {len(d.blocks)}")
            continue
        bad = [b for b in range(len(d.blocks))
               if z[f"u_{b}"].shape != d.blocks[b].x.shape]
        if bad:
            print(f"  {name:<26} REFUSED: field shape mismatch in blocks {bad}")
            continue
        was = str(z["grid"]) if "grid" in z else "(none)"
        if was == want:
            print(f"  {name:<26} already current")
            continue
        z["grid"] = np.array(want)
        np.savez_compressed(p, **z)
        print(f"  {name:<26} {was[:12]} -> {want[:12]}   t = {float(z['time']):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
