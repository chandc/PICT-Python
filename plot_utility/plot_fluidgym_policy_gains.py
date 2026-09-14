"""Sensor-gain map of a trained FluidGym DPC policy: what the MLP learned.

Loads the policy state dict (453 -> 64 -> 64 -> 1: 151 probes x (pressure,
u, v), sorted-key flattening from dpc_train.py), computes the zero-point
linearized gain g = W3 W2 W1 per input, and scatters the three channels at
the PHYSICAL sensor positions -- reconstructed exactly from FluidGym's
_get_sensor_locations_2d (wake grid 8x7, two rings of 36 at r = 1.0 and
0.625, 23 near-wake extras; concatenation order [wake, ring1, ring2,
extras]). Findings for the H=80 policy documented in
reference/fluidgym_parity.md ("What the trained policy learned").

  .venv/bin/python plot_utility/plot_fluidgym_policy_gains.py <policy.pt> <out.png>
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sensor_positions():
    xg, yg = np.meshgrid(np.arange(1.0, 5.0, 0.5), np.arange(-1.5, 1.75, 0.5),
                         indexing="ij")
    wake = np.stack([xg.ravel(), yg.ravel()])
    ang = np.linspace(0, 2 * np.pi, 36)
    c1 = np.stack([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    c2 = np.stack([0.625 * np.cos(ang), 0.625 * np.sin(ang)])
    x1 = np.arange(-0.25, 1, 0.25)
    x2 = np.concatenate([[-0.25], np.arange(0.25, 1.25, 0.25)])
    x3 = np.array([0.75] * 3)
    add = np.stack([np.concatenate([x1, x1, x2, x2, x3]),
                    np.concatenate([np.full_like(x1, -1.5), np.full_like(x1, 1.5),
                                    np.full_like(x2, 1.0), np.full_like(x2, -1.0),
                                    np.array([-0.5, 0, 0.5])])])
    P = np.concatenate([wake, c1, c2, add], axis=1)
    assert P.shape[1] == 151
    return P


def main(policy_path, out_png):
    sd = torch.load(policy_path, map_location="cpu")
    W1, W2, W3 = (sd["0.weight"].numpy(), sd["2.weight"].numpy(),
                  sd["4.weight"].numpy())
    g0 = (W3 @ W2 @ W1).ravel()
    gp, gu, gv = g0[:151], g0[151::2], g0[152::2]
    xs, ys = sensor_positions()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    vm = max(np.abs(g).max() for g in (gp, gu, gv))
    for ax, (name, g) in zip(axes, (("pressure sensors", gp),
                                    ("u sensors", gu), ("v sensors", gv))):
        s = ax.scatter(xs, ys, c=g, cmap="RdBu_r", vmin=-vm, vmax=vm, s=48,
                       edgecolors="0.6", linewidths=0.3)
        th = np.linspace(0, 2 * np.pi, 100)
        ax.fill(0.5 * np.cos(th), 0.5 * np.sin(th), color="0.3", zorder=5)
        for sgn in (1, -1):
            phi = np.radians(np.linspace(80, 100, 20)) * sgn
            ax.plot(0.52 * np.cos(phi), 0.52 * np.sin(phi), color="lime",
                    lw=3, zorder=6)
        ax.set_title(f"{name}  (sum|gain| {np.abs(g).sum():.2f})", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("x/D")
        ax.set_xlim(-1.6, 5.0)
        ax.set_ylim(-1.9, 1.9)
    axes[0].set_ylabel("y/D")
    fig.colorbar(s, ax=axes, label="linearized gain  da/d(obs)", shrink=0.85)
    fig.suptitle("Per-sensor gains of the trained DPC policy", fontsize=13)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m0_H80_policy.pt",
         sys.argv[2] if len(sys.argv) > 2 else "policy_sensor_gains.png")
