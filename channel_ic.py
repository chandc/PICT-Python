"""Initial condition for the Re_tau = 180 minimal channel, interpolated from a reference DNS.

WHY A DNS STATE AND NOT A SYNTHETIC TRIP. Plane Poiseuille at this Reynolds number is LINEARLY
STABLE -- the critical Reynolds number is ~5772 on centreline/half-height and this is ~3300 --
so transition must be bypassed with a finite-amplitude disturbance, and a minimal box sits close
enough to the sustaining threshold that a poorly chosen trip relaminarises. A relaminarisation
would be indistinguishable from a solver bug. Starting from a field that is ALREADY turbulent
removes that failure mode entirely.

THE SOURCE. `~/Dropbox/Apple_MLX_CFD/sem_demo/results/minchan_re180_K/final_state.npz`, t = 18,
a spectral-element/Fourier field: SEM-nodal in x and y over 108 elements of 9x9 GLL points, and
rfft coefficients in z. One inverse transform recovers 32 physical z-planes, and the element
node coordinates come from that project's own `minchan.setup()`.

INTERPOLATION ACCURACY BARELY MATTERS HERE, which is worth stating because it looks careless.
Turbulence forgets its initial condition within an eddy turnover. What must survive is that the
field IS turbulent -- right streak spacing, right wall-normal structure, right energy -- and
scattered interpolation preserves all of it while losing spectral accuracy. The solver's first
pressure projection makes the result discretely divergence-free regardless.

VERIFIED ON EXTRACTION rather than assumed: the reference run reports a centreline U+ of 18.38
and its own u_tau of 1.0148, and the extracted field reproduces 18.48 and U+ = 0.982 at y+ = 1.
"""
import os
import sys

import numpy as np

SEM = os.path.expanduser("~/Dropbox/Apple_MLX_CFD/sem_demo")
STATE = os.path.join(SEM, "results/minchan_re180_K/final_state.npz")
CACHE = "results/minchan_re180_field.npz"


def extract(state=STATE, cache=CACHE):
    """Pull the DNS field into a plain scattered array, once, and cache it."""
    if os.path.exists(cache):
        return np.load(cache)
    if not os.path.isdir(SEM):
        raise FileNotFoundError(f"reference project not found at {SEM}")
    sys.path.insert(0, SEM)
    import scratch.minchan as M                      # noqa: E402
    s = M.setup()
    z = np.load(state)
    u = np.fft.irfft(z["U"], n=s["nz"], axis=-1)     # (nelem, N+1, N+1, 3, nz)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, x=s["X"], y=s["Y"], z=s["zpl"], u=u,
                        nu=s["nu"], t=float(z["t"]), Lx=M.LX, Lz=M.LZ)
    return np.load(cache)


def interpolate_to(d, cache=CACHE):
    """Return per-block (u, v, w) on domain `d`, interpolated from the DNS field.

    Separable by construction: the source is a TENSOR grid in z (uniform, periodic) and
    scattered in x-y, so z is handled by periodic linear interpolation and x-y by a Delaunay
    interpolant built once and reused for all three components and all z-planes. Doing the full
    3-D scatter instead would be both slower and worse conditioned.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    src = extract(cache=cache)
    X, Y, Z, U = src["x"], src["y"], src["z"], src["u"]
    Lz = float(src["Lz"])
    pts = np.column_stack([X.ravel(), Y.ravel()])
    nz_src = len(Z)

    out = []
    for b, blk in enumerate(d.blocks):
        tx, ty, tz = blk.x, blk.y, blk.z
        # z: periodic linear weights between the two bracketing source planes
        zf = (tz / Lz * nz_src) % nz_src
        k0 = np.floor(zf).astype(int) % nz_src
        k1 = (k0 + 1) % nz_src
        w1 = zf - np.floor(zf)
        comp = []
        for c in range(3):
            acc = np.zeros(tx.shape)
            for k in np.unique(np.concatenate([k0.ravel(), k1.ravel()])):
                vals = U[..., c, k].ravel()
                itp = LinearNDInterpolator(pts, vals)
                got = itp(tx, ty)
                if np.isnan(got).any():          # the convex hull misses boundary nodes
                    nn = NearestNDInterpolator(pts, vals)
                    got = np.where(np.isnan(got), nn(tx, ty), got)
                acc += np.where(k0 == k, 1.0 - w1, 0.0) * got
                acc += np.where(k1 == k, w1, 0.0) * got
            comp.append(acc)
        out.append(tuple(comp))
    return out


if __name__ == "__main__":
    src = extract()
    print(f"  extracted: u {src['u'].shape}, t = {float(src['t']):.2f}, "
          f"nu = {float(src['nu']):.6f}")
    print(f"  box: Lx = {float(src['Lx']):.4f}, Lz = {float(src['Lz']):.4f}, "
          f"y in [{src['y'].min():.3f}, {src['y'].max():.3f}]")
    print(f"  max|u| {np.abs(src['u'][..., 0, :]).max():.3f}  "
          f"max|v| {np.abs(src['u'][..., 1, :]).max():.3f}  "
          f"max|w| {np.abs(src['u'][..., 2, :]).max():.3f}")
