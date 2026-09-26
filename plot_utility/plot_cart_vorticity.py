"""Vorticity on the Cartesian-background cylinder grid.

VORTICITY VIA THE METRICS, not np.gradient on 1-D coordinate arrays. The ring and transition
blocks are curvilinear -- a block is NOT a tensor product of an x array and a y array -- so the
square-cylinder plotter's shortcut is wrong there. The chain rule through each block's own
metrics is exact on every block type:

    dv/dx = xi_x dv/dxi + eta_x dv/deta + zeta_x dv/dzeta      (and likewise du/dy)

NO INTERPOLATION ONTO A UNIFORM RASTER. Each block is drawn on its own mesh so a node-to-node
oscillation survives to the eye instead of being averaged away.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cylinder_cart_grid import cart_ring_domain, D, R_CYL
from src import checkpoint

PATH = _sys.argv[1] if len(_sys.argv) > 1 else "results/fields/cart_re100_base.npz"
d, idx = cart_ring_domain(nz=2)
d.prepare_geometry()
fl, meta = checkpoint.load_fields(PATH)      # (fields, meta); fields is {name: {block: arr}}
t, nstep = meta["time"], meta["nstep"]
u, v = fl["u"], fl["v"]

KEYS = (("xi_x", "xi_y", "xi_z"), ("eta_x", "eta_y", "eta_z"), ("zeta_x", "zeta_y", "zeta_z"))

def vort(b):
    """w_z on block b, differenced across SEAMS rather than one-sided at them.

    np.gradient on the bare block is one-sided at every block edge, and each side of a seam
    then differences different data -- which draws a discontinuity at every seam whether or not
    the solution has one. Padding the velocity as a field and the geometry alongside it (the
    same pair the solver's own operators use) makes the stencil central at the seam, so what is
    left in the picture is the solution's, not the plotter's.
    """
    blk = d.blocks[b]
    up, lo, hi = d.pad_field(b, u, 1)
    vp = d.pad_field(b, v, 1)[0]
    _Jp, mp, glo, ghi = d.padded_geometry(b, 1)
    # IN-PLANE AXES ONLY: nz = 2 is too thin for a second-order z difference, and the grid is a
    # pure z-extrusion so zeta_x = zeta_y = 0 (measured 1e-13) -- that term is not part of w_z.
    du = [np.gradient(up, blk.h[a], axis=a, edge_order=2) for a in range(2)]
    dv = [np.gradient(vp, blk.h[a], axis=a, edge_order=2) for a in range(2)]
    dvdx = sum(mp[KEYS[a][0]] * dv[a] for a in range(2))
    dudy = sum(mp[KEYS[a][1]] * du[a] for a in range(2))
    core = tuple(slice(lo[a], lo[a] + blk.shape[a]) for a in range(3))
    return (dvdx - dudy)[core][:, :, 0]

W = {b: vort(b) for b in range(len(d.blocks))}

# DRAW EACH SEAM ONCE. Coincident seam nodes are SEPARATE degrees of freedom in this code --
# the two blocks' boundary cells are coupled through the matrix face, and it is the FLUX that
# is continuous there (1e-14), not the nodal value, which differs by O(0.5) on the validated
# butterfly grid too. Plotting both blocks paints two different values at one physical point,
# which drew a spurious vorticity sheet around the body at the annulus seam. Trim the upper
# side of every connected face so each location is coloured once.
from src.multiblock import face_axis_side as _fas
trim = {b: [slice(None)] * 3 for b in range(len(d.blocks))}
for c in d.connections:
    for blk_i, fid in ((c.ba, c.fa), (c.bb, c.fb)):
        ax, side = _fas(fid)
        if side == 1:                      # keep the lower side's copy, drop the upper's
            trim[blk_i][ax] = slice(0, -1)
            break
lim = 5.0
fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                         gridspec_kw={"height_ratios": [1.0, 1.25]})
views = [((-5, 15), (-5, 5), "full domain"), ((-1.5, 6.0), (-2.2, 2.2), "near wake")]
for ax, (xl, yl, ttl) in zip(axes, views):
    for b, blk in enumerate(d.blocks):
        tb = tuple(trim[b][:2])
        im = ax.pcolormesh(blk.x[:, :, 0][tb], blk.y[:, :, 0][tb], W[b][tb],
                           cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_CYL * np.cos(th), R_CYL * np.sin(th), color="0.55",
            edgecolor="k", lw=1.0, zorder=5)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=r"$\omega_z$")
wmax = max(float(np.abs(W[b]).max()) for b in range(len(d.blocks)))
fig.suptitle(f"Re=100 cylinder, Cartesian-background grid  --  t = {t:.1f}, step {nstep}, "
             f"28,265 cells 2D   (|w|max {wmax:.1f}, clipped at {lim})", fontsize=12)
fig.tight_layout()
fig.savefig("figures/cart_re100_vorticity.png", dpi=115)
print(f"  wrote figures/cart_re100_vorticity.png   t={t:.1f} step={nstep}  |w|max {wmax:.2f}")
