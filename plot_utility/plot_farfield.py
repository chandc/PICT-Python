"""The WHOLE domain: streamlines over speed, and vorticity, out to the boundaries.

The near-wake plots crop to a few diameters, which is where the physics is but not where the
boundary conditions are. This one shows everything, because the questions it answers are about
the edges:

  Does the wake reach the outflow still carrying structure, and does it leave cleanly or pile up
  against the boundary? The square domain runs to 30 D downstream and the vortex street is still
  coherent well past 20 D.

  Is the lateral boundary far enough away that the free stream is undisturbed there? Blockage is
  5% for the square and 1.7% for the cylinder, and this is the picture that shows whether that
  number is doing its job.

  Is the far field clean? A boundary condition that leaks -- as the cylinder's did, exiting 77%
  of its mass through the flanks -- shows up here as streamlines bending toward the sides long
  before they reach the body.

VORTICITY NEEDS ITS OWN SCALE OUT HERE. The near wake reaches |omega| ~ 4 and the far wake is
two orders below that, so the near-field colour limits render everything past x ~ 10 as blank
white. The scale is set from a high PERCENTILE of the field rather than its maximum, so the
weak structure that survives to the outflow is actually visible.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from src import checkpoint

NX, NY = 1100, 620


def wake_resolution_limit(d, tag, St=0.149):
    """Last x on the wake centreline where the cell is finer than lambda/20.

    The striping that appears in the far wake is not a solver artefact and not turbulence: it is
    the grid. The wake plateau holds a fine spacing only so far, and past that the cells stretch
    until a shedding wavelength spans too few of them to be represented. Marking the crossing
    turns an unexplained texture into a stated limit.
    """
    lam = 1.0 / St                       # D / St, in units of D
    best = float("nan")
    for blk in d.blocks:
        y0 = np.abs(blk.y[:, :, 0]).min()
        if y0 > 0.6:                      # not a block on the wake centreline
            continue
        xs = blk.x[:, 0, 0]
        if xs.max() <= 0.5:
            continue
        dx = np.diff(xs)
        ok = np.where(dx <= lam / 20.0)[0]
        if len(ok):
            cand = float(xs[ok[-1]])
            best = cand if not np.isfinite(best) else max(best, cand)
    return best


def build(tag, nz=4):
    """(domain, body patch factory, extent) for whichever case the tag names."""
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    if tag.startswith("sqcyl"):
        from square_cylinder_grid import square_domain, D, X_IN, X_OUT, Y_HALF
        d, _ = square_domain(nz=nz)
        patch = lambda: plt.Rectangle((-.5*D, -.5*D), D, D, color="k", zorder=9)
        solid = lambda X, Y: (np.abs(X) <= 0.5*D) & (np.abs(Y) <= 0.5*D)
        return d, f, meta, patch, solid, (X_IN, X_OUT, -Y_HALF, Y_HALF), "square cylinder"
    from cylinder_grid import cylinder_domain, D, R_CYL
    d, r, _ = cylinder_domain(nblk=meta["nblocks"], nz=nz)
    R = float(r[-1])
    patch = lambda: plt.Circle((0, 0), R_CYL, color="k", zorder=9)
    solid = lambda X, Y: (X**2 + Y**2) <= R_CYL**2
    return d, f, meta, patch, solid, (-R, R, -R, R), "circular cylinder"


def main(tag="sqcyl_v3", nz=4):
    d, f, meta, patch, solid, (XL, XR, YB, YT), name = build(tag, nz)
    P, U, V = [], [], []
    for b in range(len(d.blocks)):
        blk = d.blocks[b]
        P.append(np.column_stack([blk.x[:, :, 0].ravel(), blk.y[:, :, 0].ravel()]))
        U.append(f["u"][b][:, :, 0].ravel())
        V.append(f["v"][b][:, :, 0].ravel())
    P = np.vstack(P); U = np.concatenate(U); V = np.concatenate(V)
    gx = np.linspace(XL, XR, NX); gy = np.linspace(YB, YT, NY)
    GX, GY = np.meshgrid(gx, gy)
    gu = LinearNDInterpolator(P, U)(GX, GY)
    gv = LinearNDInterpolator(P, V)(GX, GY)
    m = solid(GX, GY)
    gu[m] = np.nan; gv[m] = np.nan

    spd = np.sqrt(gu**2 + gv**2)
    wz = np.gradient(gv, gx, axis=1) - np.gradient(gu, gy, axis=0)
    lim = float(np.nanpercentile(np.abs(wz), 99.0))

    # THE FIGURE MUST FOLLOW THE DOMAIN. Both panels are aspect-equal, so a fixed figure size
    # fits the square cylinder's 2:1 domain and leaves the cylinder's 1:1 domain floating in
    # half a page of white. Size the canvas from the extent instead.
    pw = 13.0
    ph = pw * (YT - YB) / (XR - XL)
    fig, axes = plt.subplots(2, 1, figsize=(pw + 1.6, 2 * ph + 1.4))

    ax = axes[0]
    im = ax.pcolormesh(GX, GY, spd, cmap="viridis", shading="auto", vmin=0, vmax=1.4)
    ax.streamplot(gx, gy, gu, gv, density=(3.0, 1.6), linewidth=0.5, color="w", arrowsize=0.6)
    ax.add_patch(patch())
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("streamlines over speed — the whole domain, boundaries included", fontsize=11)
    fig.colorbar(im, ax=ax, label="|u| / U", fraction=0.025, pad=0.01)

    ax = axes[1]
    im = ax.pcolormesh(GX, GY, wz, cmap="RdBu_r", shading="auto", vmin=-lim, vmax=lim)
    ax.add_patch(patch())
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"vorticity, scale set by the 99th percentile |$\\omega_z$| = {lim:.2f} "
                 f"(peak {np.nanmax(np.abs(wz)):.1f}) so the far wake is visible", fontsize=11)
    fig.colorbar(im, ax=ax, label=r"$\omega_z D / U$", fraction=0.025, pad=0.01)

    # How far the wake still carries vorticity, and how far the GRID can still represent it.
    # Peak |omega| lives on the body and is ~26 here, so a threshold relative to it measures
    # near-body dominance and nothing about the wake; 0.1 U/D is a fixed, interpretable level.
    cols = np.where(np.nanmax(np.abs(wz), axis=0) > 0.1)[0]
    x_end = gx[cols[-1]] if len(cols) else float("nan")
    x_res = wake_resolution_limit(d, tag)
    if np.isfinite(x_res):
        for ax_ in axes:
            ax_.axvline(x_res, color="crimson", ls="--", lw=1.2, zorder=10)
        axes[1].text(x_res + 0.4, 0.72 * YT, "grid coarser than\n$\\lambda$/20 past here",
                     color="crimson", fontsize=9)
    fig.suptitle(f"{name}, {tag} — t = {meta['time']:.1f};  wake carries "
                 f"$|\\omega_z| > 0.1$ to x = {x_end:.1f} D, grid resolves the shedding "
                 f"wavelength to x = {x_res:.1f} D", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"figures/{tag}_farfield.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  domain x [{XL:.0f}, {XR:.0f}], y [{YB:.0f}, {YT:.0f}];  t = {meta['time']:.1f}")
    print(f"  |omega| peak {np.nanmax(np.abs(wz)):.2f}, 99th percentile {lim:.3f}")
    print(f"  wake carries |omega| > 0.1 to x = {x_end:.1f} D")
    print(f"  grid resolves the shedding wavelength (lambda/20) to x = {x_res:.1f} D")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_v3")
