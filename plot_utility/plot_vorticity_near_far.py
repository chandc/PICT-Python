"""Near- and far-field vorticity at NODE resolution, with a per-region oscillation audit.

Two lessons from the pressure check are built in:

  NO INTERPOLATION. Rasterising the eight blocks onto a uniform grid averages neighbouring nodes,
  which is exactly what annihilates a node-to-node mode. Each block is drawn on its own mesh with
  `shading="nearest"` -- one coloured cell per node -- and the cuts plot raw node values.

  REGIONS, NOT BLOCKS. The trailing-edge pressure wiggle (flip fraction 0.950) was invisible in the
  whole-block metric (0.297) because each block also holds thousands of quiet freestream nodes.
  The audit here is restricted to boxes: near body, near wake, mid wake, far wake.

Vorticity is taken per block with np.gradient against the PHYSICAL 1-D coordinates. That is exact
for this grid because every block is a tensor product of an x array and a y array; it is
second-order in the interior and one-sided at block edges, so the audit uses block interiors.
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

from square_cylinder_grid import square_domain, D
from diag_checkerboard import checkerboard
from src import checkpoint

REGIONS = (("near body",  (-1.5, 1.5), (-1.5, 1.5)),
           ("near wake",  (1.5, 6.0),  (-2.5, 2.5)),
           ("mid wake",   (6.0, 15.0), (-3.0, 3.0)),
           ("far wake",   (15.0, 28.0), (-4.0, 4.0)))
VLIM = 3.0


def vorticity(d, f):
    """w_z per block on the z=0 plane, via physical-coordinate differences."""
    out = []
    for b, blk in enumerate(d.blocks):
        x, y = blk.x[:, 0, 0], blk.y[0, :, 0]
        u, v = f["u"][b][:, :, 0], f["v"][b][:, :, 0]
        dvdx = np.gradient(v, x, axis=0) if len(x) > 1 else np.zeros_like(v)
        dudy = np.gradient(u, y, axis=1) if len(y) > 1 else np.zeros_like(u)
        out.append(dvdx - dudy)
    return out


def audit(d, fields, region):
    """Worst (flip fraction, amplitude, scale) over the region, block interiors only."""
    (x0, x1), (y0, y1) = region
    worst_f, worst_a, scale, n = 0.0, 0.0, 0.0, 0
    for b, blk in enumerate(d.blocks):
        x, y = blk.x[:, 0, 0], blk.y[0, :, 0]
        xi = np.where((x > x0) & (x < x1))[0]
        yj = np.where((y > y0) & (y < y1))[0]
        # drop each block's own edge row: np.gradient is one-sided there
        xi = xi[(xi > 0) & (xi < len(x) - 1)]
        yj = yj[(yj > 0) & (yj < len(y) - 1)]
        if len(xi) < 4 or len(yj) < 4:
            continue
        sub = fields[b][xi[0]:xi[-1] + 1, yj[0]:yj[-1] + 1]
        scale = max(scale, float(np.abs(sub).max()))
        n += sub.size
        for ax in (0, 1):
            a, fl = checkerboard(sub, axis=ax)
            if fl > worst_f:
                worst_f, worst_a = fl, a
    return worst_f, worst_a, scale, n


def draw(ax, d, w, edges=False):
    for b, blk in enumerate(d.blocks):
        ax.pcolormesh(blk.x[:, :, 0], blk.y[:, :, 0], w[b], cmap="RdBu_r",
                      vmin=-VLIM, vmax=VLIM, shading="nearest",
                      edgecolors="0.55" if edges else "face", linewidth=0.1 if edges else 0)
    ax.add_patch(plt.Rectangle((-.5 * D, -.5 * D), D, D, color="k", zorder=9))


def main(tag="sqcyl_v3_forces"):
    d, idx = square_domain(nz=4)
    f, meta = checkpoint.load_fields(f"results/fields/{tag}.npz")
    w = vorticity(d, f)
    p = [f["p"][b][:, :, 0] for b in range(len(d.blocks))]

    rows = []
    for name, rx, ry in REGIONS:
        fw, aw, sw, n = audit(d, w, (rx, ry))
        fp, ap, sp, _ = audit(d, p, (rx, ry))
        rows.append((name, n, fw, aw, sw, fp, ap, sp))

    fig = plt.figure(figsize=(15, 13))

    ax = fig.add_axes([0.05, 0.735, 0.92, 0.205])
    draw(ax, d, w)
    ax.axvline(15.0, color="k", ls="--", lw=1.0)
    ax.text(15.2, 4.2, "grid stretching begins (x = 15): pressure column mode\n"
            "switches on here, in phase at every y", fontsize=8.5, va="top")
    ax.set_xlim(-3, 28); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title(f"FAR FIELD — vorticity at node resolution, t = {meta['time']:.0f}", fontsize=11)

    ax = fig.add_axes([0.05, 0.415, 0.44, 0.265])
    draw(ax, d, w, edges=True)
    ax.set_xlim(-1.2, 3.2); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xlabel("x / D"); ax.set_ylabel("y / D")
    ax.set_title("NEAR FIELD — one cell per node, cell edges drawn", fontsize=11)

    # raw-node cuts: a sawtooth between adjacent markers is a node-to-node mode
    ax = fig.add_axes([0.55, 0.585, 0.42, 0.115])
    b = idx["RT"]; blk = d.blocks[b]
    j = int(np.argmin(np.abs(blk.y[0, :, 0] - 0.75)))
    m = blk.x[:, 0, 0] < 8
    ax.plot(blk.x[m, 0, 0], w[b][m, j], "o-", ms=2.5, lw=0.9, color="#5b6c8f")
    ax.set_title(f"raw nodes along y = {blk.y[0, j, 0]:.2f} (upper shear layer)", fontsize=9.5)
    ax.set_xlabel("x / D", fontsize=9); ax.grid(alpha=.3); ax.tick_params(labelsize=8)

    ax = fig.add_axes([0.55, 0.43, 0.42, 0.115])
    for nm, col in (("RB", "#e4572e"), ("RM", "#5b6c8f"), ("RT", "#76b041")):
        bb = idx[nm]; bk = d.blocks[bb]
        i = int(np.argmin(np.abs(bk.x[:, 0, 0] - 2.0)))
        yy = bk.y[i, :, 0]; mm = np.abs(yy) < 3
        ax.plot(yy[mm], w[bb][i, mm], "o-", ms=2.5, lw=0.9, color=col, label=nm)
    ax.set_title("raw nodes across the wake at x = 2", fontsize=9.5)
    ax.set_xlabel("y / D", fontsize=9); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_axes([0.05, 0.22, 0.92, 0.15])
    h = np.load("results/sqcyl_v3_history.npy")
    hf = np.load("results/sqcyl_v3_forces_history.npy")
    ax.plot(h[:, 0], h[:, 1], lw=0.5, color="#5b6c8f", label="probe v at (2D, 0.5D)")
    ax.plot(hf[:, 0], hf[:, 1], lw=0.5, color="#e4572e", label="restart for forces")
    ax.axvspan(0, 80, color="0.88", zorder=0)
    ax.text(40, ax.get_ylim()[1] * 0.75, "settle", ha="center", fontsize=8, color="0.35")
    ax.axvline(80, color="crimson", ls="--", lw=1)
    ax.text(82, ax.get_ylim()[1] * 0.75, "sinuous kick", fontsize=8, color="crimson")
    ax.set_xlabel("t U / D"); ax.set_ylabel("v"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("probe history — settle, kick, growth, saturated limit cycle", fontsize=11)

    ax = fig.add_axes([0.05, 0.02, 0.92, 0.15]); ax.axis("off")
    hdr = f"{'region':<12}{'nodes':>7}   {'vorticity flips':>15}{'amp / max|w|':>14}   " \
          f"{'pressure flips':>14}{'amp / max|p|':>14}"
    ax.text(0.0, 0.95, hdr, family="monospace", fontsize=9.5, weight="bold", va="top")
    for k, (name, n, fw, aw, sw, fp, ap, sp) in enumerate(rows):
        flag = "  <-- OSCILLATION" if max(fw, fp) > 0.35 else ""
        line = f"{name:<12}{n:>7}   {fw:>15.3f}{100*aw/max(sw,1e-30):>13.2f}%   " \
               f"{fp:>14.3f}{100*ap/max(sp,1e-30):>13.2f}%{flag}"
        ax.text(0.0, 0.76 - 0.17 * k, line, family="monospace", fontsize=9.5, va="top",
                color="crimson" if flag else "0.1")
    ax.text(0.0, 0.02, "flip fraction ~1 = node-to-node mode; < 0.35 = smooth. "
            "Amplitude is only a wiggle size when flips are high.", fontsize=8.5, color="0.35")

    fig.suptitle("Square cylinder Re = 100, rebuilt grid (82,096 cells) — "
                 "vorticity, near and far field", fontsize=13, y=0.995)
    out = f"figures/{tag}_vorticity_near_far.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")

    print(f"  t = {meta['time']:.1f}")
    print(f"  {'region':<12}{'nodes':>7}{'w flips':>9}{'w amp%':>8}{'p flips':>9}{'p amp%':>8}")
    for name, n, fw, aw, sw, fp, ap, sp in rows:
        print(f"  {name:<12}{n:>7}{fw:>9.3f}{100*aw/max(sw,1e-30):>8.2f}"
              f"{fp:>9.3f}{100*ap/max(sp,1e-30):>8.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(_sys.argv[1] if len(_sys.argv) > 1 else "sqcyl_v3_forces")
