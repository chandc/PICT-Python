"""Usage: python plot_utility/plot_uadj_dpc_vs_ppo.py
DPC (this solver's replay adjoint) against PPO on the cylinder wake-control benchmark: drag reduction vs.
solver steps consumed, grouped by actuator constraint (free jets vs. zero-net-mass-flux). PPO points are
(a) Rabault et al. 2019, the literature ZNMF benchmark this solver has not trained PPO on, and (b) our own
PPO training run on HydroGym's Firedrake Cylinder env, free jets (record: hydrogym-jet-env-suction-loophole)."""
import os as _os, sys as _sys; _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path.insert(0, _ROOT); _os.chdir(_ROOT)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# (label, method, constraint, steps, drag_reduction_pct, mean_Cl, marker, color)
rows = [
    ("Rabault et al. 2019\n(literature, PPO)",        "PPO",  "ZNMF",  1.0e6, 8.0,  0.0,   "s", "0.35"),
    ("HydroGym Firedrake\n(our PPO run, free jets)",   "PPO",  "free",  1.0e5, 30.6, None,  "s", "C3"),
    ("DPC, free jets\n(this work, H 8-80)",            "DPC",  "free",  4.0e3, 19.1, None,  "o", "C3"),
    ("DPC, ZNMF\n(this work, H 8-80, no penalty)",     "DPC",  "ZNMF",  4.0e4, 1.1,  None,  "o", "0.6"),
    ("DPC, ZNMF + $C_L^2$\n(this work, H 240 rolling)", "DPC", "ZNMF",  1.0e5, 8.5,  -0.04, "o", "C0"),
]
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
offsets = {"Rabault et al. 2019\n(literature, PPO)": (0, -42, "top"),
           "HydroGym Firedrake\n(our PPO run, free jets)": (0, -42, "top"),
           "DPC, free jets\n(this work, H 8-80)": (10, -30, "top"),
           "DPC, ZNMF\n(this work, H 8-80, no penalty)": (0, -38, "top"),
           "DPC, ZNMF + $C_L^2$\n(this work, H 240 rolling)": (0, 20, "bottom")}
for lab, method, con, steps, red, cl, mk, c in rows:
    ax.scatter(steps, red, s=260, marker=mk, facecolor=c, edgecolor="k", lw=1.3, zorder=5)
    dx, dy, va = offsets[lab]
    ax.annotate(lab, (steps, red), xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=9, va=va)
ax.plot([4.0e3, 4.0e4, 1.0e5], [19.1, 1.1, 8.5], "C3--", lw=0.8, alpha=0.4, zorder=1)
ax.plot([1.0e6, 1.0e5], [8.0, 8.5], "C0--", lw=1.2, alpha=0.6, zorder=1)
ax.set_xscale("log"); ax.set(xlabel="solver steps consumed (training)", ylabel="closed-loop drag reduction (%)",
       title="Re 100 cylinder, $\\pm$90$^\\circ$ jets: DPC (certified adjoint) vs. PPO (RL)", ylim=(-6, 38), xlim=(2e3, 3e6))
ax.axhline(0, color="k", lw=0.6)
ax.text(0.98, 0.03, "circle = DPC (this work, unstructured replay adjoint)\nsquare = PPO\nred = free jets (unconstrained: both find steady suction, not wake control)\nblue/grey = zero-net-mass-flux (the Rabault benchmark)",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right", bbox=dict(fc="white", ec="0.7", alpha=0.9))
plt.tight_layout(); plt.savefig("figures/uadj_dpc_vs_ppo_cylinder.png", dpi=140); print("wrote figures/uadj_dpc_vs_ppo_cylinder.png")
