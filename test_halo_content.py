"""Gate 2 preparation: every halo cell carries the RIGHT value, not merely some value.

WHY THIS EXISTS, AND WHY IT EXISTS NOW. The Gate 0/1 equivalence test compares a ten-step
trajectory against digests. It is sensitive -- both mangles fire at step 1 -- but it is a
DETECTOR, not a diagnostic: it says the run diverged, not which seam, which axis, or which
corner. That is adequate for Gate 1, where the change was two call sites. It is not adequate for
Gate 2, where a halo exchange can be wrong on one face of one block in one corner and still
produce a smooth, plausible field.

So this is built BEFORE any exchange code, while everything is still serial and therefore known
good. A test written after the code it judges tends to encode the same misunderstanding.

THE IDEA. Fill a field with an analytic function of position, f(x, y, z). Pad it. Then every
halo cell must equal f evaluated AT THAT HALO CELL'S OWN COORDINATE -- and the coordinates come
from `pad_coords`, the other half of the seam. A halo that moved the right bytes to the wrong
place fails. A halo that moved the wrong neighbour's data fails. A halo that forgot the periodic
displacement fails, because f at the wrapped coordinate differs from f at the unwrapped one.

f MUST BE GENUINELY PERIODIC on the periodic axes, or the test would demand of the code
something the physics does not: a velocity field is periodic, so its halo across a periodic seam
is the far-side value UNSHIFTED. Coordinates are the opposite -- they ramp and jump back. Using
sin/cos at the box wavenumber makes f periodic by construction, so a correct halo and a correct
analytic value agree without either being special-cased.

ATTRIBUTION IS THE POINT. Failures are reported per (axis, side) and separately for CORNER cells
-- those padded on two or three axes at once, which is where the recursive padding earns its
complexity and where an exchange is most likely to be wrong. "17 cells differ" is a detection;
"every cell on the -x face of block 3, corners only" is a diagnosis.
"""
import numpy as np

from src.domains import channel_box, clustered_y, periodic_box

WIDTH = 2


def f(x, y, z, L=(1.0, 1.0, 1.0)):
    """Analytic field: periodic on every axis at the box wavenumber, and not separable."""
    kx, ky, kz = (2 * np.pi / l for l in L)
    return (np.sin(kx * x) * np.cos(ky * y)
            + 0.5 * np.sin(kz * z) * np.sin(kx * x)
            + 0.25 * np.cos(ky * y + kz * z))


def _check_domain(d, L, name, periodic_axes=(0, 1, 2)):
    """Pad an analytic field on every block; every halo cell must match f at its coordinate."""
    nb = len(d.blocks)
    fields = {b: f(d.blocks[b].x, d.blocks[b].y, d.blocks[b].z, L) for b in range(nb)}
    worst = 0.0
    per_face = {}
    corner_bad = 0
    corner_tot = 0
    total = 0
    for b in range(nb):
        got, lo, hi = d.pad_field(b, fields, width=WIDTH)
        X, Y, Z, clo, chi = d.pad_coords(b, width=WIDTH)
        if (list(lo), list(hi)) != (list(clo), list(chi)):
            print(f"  [FAIL] block {b}: field padding {lo}/{hi} disagrees with coordinate "
                  f"padding {clo}/{chi} -- the two halves of the seam do not match")
            return False, 0.0
        want = f(X, Y, Z, L)
        err = np.abs(got - want)
        # only judge cells that are actually HALO: outside the interior box
        n = got.shape
        interior = tuple(slice(lo[a], n[a] - hi[a]) for a in range(3))
        mask = np.ones(n, bool)
        mask[interior] = False
        if not mask.any():
            continue
        total += int(mask.sum())
        worst = max(worst, float(err[mask].max()))
        # per (axis, side) attribution
        for a in range(3):
            for side, sl in ((0, slice(0, lo[a])), (1, slice(n[a] - hi[a], n[a]))):
                if sl.stop <= sl.start:
                    continue
                idx = [slice(None)] * 3
                idx[a] = sl
                e = err[tuple(idx)]
                k = (a, side)
                per_face[k] = max(per_face.get(k, 0.0), float(e.max()))
        # corners: padded on two or more axes at once
        depth = np.zeros(n, int)
        for a in range(3):
            d_a = np.zeros(n, int)
            idx = [slice(None)] * 3
            idx[a] = slice(0, lo[a])
            d_a[tuple(idx)] = 1
            idx[a] = slice(n[a] - hi[a], n[a])
            d_a[tuple(idx)] = 1
            depth += d_a
        corner = depth >= 2
        corner_tot += int(corner.sum())
        corner_bad += int((err[corner] > 1e-12).sum()) if corner.any() else 0

    ok = worst < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {total:,} halo cells, max error {worst:.2e} "
          f"({corner_tot:,} of them corners, {corner_bad} wrong)")
    if not ok:
        for (a, side), e in sorted(per_face.items()):
            flag = "  <-- " if e > 1e-12 else ""
            print(f"          axis {a} side {side}: max error {e:.2e}{flag}")
    return ok, worst


def check_periodic_split(n_split):
    d = periodic_box(12, n_split, L=2 * np.pi)
    ok, _ = _check_domain(d, (2 * np.pi,) * 3, f"periodic box, {n_split} block(s)")
    return ok


def check_channel():
    """Walls on axis 1: those faces are NOT padded, and the test must not demand that they are."""
    ny = 24
    y = clustered_y(ny, re_tau=180.0)
    Lx, Lz = np.pi, 0.34 * np.pi
    d = channel_box(12, ny, 12, 2, Lx=Lx, Lz=Lz, y_nodes=y)
    # f must be periodic in x and z; in y it is merely smooth, which is fine because the wall
    # faces are not padded and so are never compared.
    ok, _ = _check_domain(d, (Lx, 2.0, Lz), "channel, 2 blocks, walls on axis 1")
    return ok


def check_mangle_is_detected():
    """Corrupt one seam's data and require a failure, or this test proves nothing."""
    from src.comm import Comm

    class Bad(Comm):
        def fetch_field_slab(self, b, k, oaxis, oside, width, local):
            lay, olo, ohi = super().fetch_field_slab(b, k, oaxis, oside, width, local)
            lay = lay.copy()
            lay.flat[0] += 1e-9                   # ONE cell, ONE seam, tiny
            return lay, olo, ohi

    d = periodic_box(12, 2, L=2 * np.pi)
    d.comm = Bad(len(d.blocks))
    nb = len(d.blocks)
    fields = {b: f(d.blocks[b].x, d.blocks[b].y, d.blocks[b].z, (2 * np.pi,) * 3)
              for b in range(nb)}
    worst = 0.0
    for b in range(nb):
        got, lo, hi = d.pad_field(b, fields, width=WIDTH)
        X, Y, Z, _, _ = d.pad_coords(b, width=WIDTH)
        worst = max(worst, float(np.abs(got - f(X, Y, Z, (2 * np.pi,) * 3)).max()))
    ok = worst > 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] a single corrupted halo cell (1e-9) is DETECTED: "
          f"max error {worst:.2e}")
    return ok


def main():
    print("=" * 78)
    print("  halo content: every ghost cell carries the right value, attributable by face")
    print("=" * 78)
    r = [check_periodic_split(1), check_periodic_split(2), check_periodic_split(3),
         check_channel(), check_mangle_is_detected()]
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
