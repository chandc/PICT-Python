"""Square cylinder at Re = 100 against published data, at the SAME blockage.

The comparison is with Sohankar, Norberg & Davidson, IJNMF 26:39-56 (1998), whose Table III
case 5 is Re = 100, zero incidence, 5% blockage -- exactly our configuration. That matters more
than it sounds: the same paper shows the base suction changing 7.6% between 5% and 2.5%
blockage, so a comparison against a "low blockage" band is not a comparison at all.

Three panels:

  left    time-averaged surface C_p with its cycle rms, and the two points the reference
          actually tabulates -- stagnation and base. Note the reference stagnation is 1.052,
          NOT 1: at finite blockage the flow accelerates past the body, so C_p at the front
          exceeds 1 and "it must be exactly +1" is only true at zero blockage.
  middle  the integral quantities, ours against theirs, as relative error, with THEIR OWN
          spread across three grids drawn as the band. A discrepancy inside that band is not
          resolvable by this comparison.
  right   St and Re_c, each against the reference at matched blockage.
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

import reference_data as REF

cp = np.load("results/sq_surface_cp.npz")
S, CPm, CPr = cp["s"], cp["cp_mean"], cp["cp_rms"]
r = REF.re100_beta5()

OURS = dict(St=0.148811, C_D=1.4529, C_L_rms=0.1722, C_p_base=-0.7045,
            C_p_stag=1.0725, L_r=1.8865)

fig = plt.figure(figsize=(16.2, 6.4))

# ---------------------------------------------------------------- surface C_p
ax = fig.add_axes([0.045, 0.13, 0.36, 0.75])
ax.fill_between(S, CPm - CPr, CPm + CPr, color="#5b6c8f", alpha=0.3,
                label=r"ours, $\pm$ rms over one period")
ax.plot(S, CPm, color="#1f3b73", lw=1.8, label="ours, time mean (3500 samples)")
ax.plot([0, 4], [r["C_p_stag"]] * 2, color="#c0392b", ls="--", lw=1.2,
        label=fr"Sohankar $C_{{ps}}$ = {r['C_p_stag']:.3f}")
ax.plot([1.5, 2.5], [r["C_p_base"]] * 2, color="#c0392b", ls="-", lw=2.4,
        label=fr"Sohankar $C_{{pb}}$ = {r['C_p_base']:.3f} (base)")
ax.plot([0, 4], [1.0, 1.0], color="k", ls=":", lw=0.8)
ax.text(3.02, 1.02, r"$C_p=1$ — exact only at ZERO blockage", fontsize=7.5, ha="right")
for x in (0.5, 1.5, 2.5, 3.5):
    ax.axvline(x, color="k", lw=0.4, alpha=0.3)
ax.set_xlim(0, 4); ax.set_xlabel("perimeter s (anticlockwise from front stagnation)")
ax.set_ylabel(r"$C_p$"); ax.legend(fontsize=8, loc="lower left")
ax.set_title(r"time-averaged surface $C_p$", fontsize=11)

# ---------------------------------------------------------------- integral quantities
ax = fig.add_axes([0.455, 0.13, 0.24, 0.75])
names = [("C_D", r"$C_D$"), ("C_L_rms", r"$C_L^{rms}$"), ("C_p_base", r"$C_{pb}$"),
         ("C_p_stag", r"$C_{ps}$"), ("St", r"$St$"), ("L_r", r"$L_r$")]
y = np.arange(len(names))[::-1]
err = [100 * (OURS[k] - r[k]) / abs(r[k]) for k, _ in names]
# their own grid-to-grid spread, as a percentage of the reference, where available
spread = {"C_D": 100*(1.478-1.460)/1.460, "C_L_rms": 100*(0.156-0.139)/0.139,
          "C_p_base": 100*(0.678-0.661)/0.661, "C_p_stag": 100*(1.059-1.052)/1.052,
          "St": 0.0, "L_r": np.nan}
for yy, (k, lab) in zip(y, names):
    s = spread[k]
    if np.isfinite(s) and s > 0:
        ax.barh(yy, s, left=0, height=0.62, color="#cfd8e3",
                label="their grid-to-grid spread" if k == "C_D" else None)
ax.barh(y, err, height=0.34, color=["#1f3b73" if abs(e) < 10 else "#c0392b" for e in err])
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in names])
ax.set_xlabel("our value − reference, % of reference")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("integral quantities vs Sohankar\n(5% blockage, matched)", fontsize=10.5)
for yy, e in zip(y, err):
    ax.text(e + (0.6 if e >= 0 else -0.6), yy, f"{e:+.1f}%", va="center",
            ha="left" if e >= 0 else "right", fontsize=8)
ax.set_xlim(-12, 32)

# ---------------------------------------------------------------- St and Re_c
ax = fig.add_axes([0.755, 0.565, 0.225, 0.32])
# THE TWO TABLES HAVE DIFFERENT COLUMN ORDERS and mixing them up put drag values in the
# Strouhal band, 0.146-1.491. OUTLET_RE100 is (BC, X_d, St, ...) so St is index 2;
# REFINEMENT_RE100 is (Delta, N_b, BC, St, ...) so St is index 3.
sts = [row[2] for row in REF.OUTLET_RE100] + [row[3] for row in REF.REFINEMENT_RE100]
ax.axhspan(min(sts), max(sts), color="#cfd8e3",
           label=f"Sohankar, all cases {min(sts):.3f}–{max(sts):.3f}")
ax.axhline(r["St"], color="#c0392b", lw=1.6, label=f"matched case {r['St']:.3f}")
ax.errorbar([0.5], [OURS["St"]], yerr=[2e-6], fmt="o", ms=7, color="#1f3b73",
            capsize=4, label=f"ours {OURS['St']:.4f}")
ax.set_xlim(0, 1); ax.set_xticks([])
ax.set_ylim(0.128, 0.156)
ax.set_ylabel("St"); ax.legend(fontsize=7, loc="lower left")
ax.set_title("Strouhal number", fontsize=10.5)

ax = fig.add_axes([0.755, 0.13, 0.225, 0.32])
lo, hi = REF.RE_CR["alpha0_beta5"]
ax.axhspan(lo - hi, lo + hi, color="#f2c9c4", label=f"Sohankar 5% blockage {lo}±{hi}")
ax.axhline(lo, color="#c0392b", lw=1.6)
zlo, zhi = REF.RE_CR["zero_blockage_experiment"]
ax.axhspan(zlo - zhi, zlo + zhi, color="#cfd8e3", alpha=0.7,
           label=f"zero-blockage expt {zlo}±{zhi}")
ax.plot([0.35, 0.65], [52, 52], color="#1f3b73", lw=2)
ax.plot([0.35, 0.65], [55, 55], color="#1f3b73", lw=2)
ax.fill_between([0.35, 0.65], 52, 55, color="#1f3b73", alpha=0.25,
                label="ours, sign bracket 52–55")
ax.plot([0.5], [51.31], "o", ms=7, color="#1f3b73", label="ours, near-onset fit 51.3")
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_ylabel(r"$Re_c$")
ax.set_ylim(44, 57)
ax.legend(fontsize=6.5, loc="lower left"); ax.set_title("onset", fontsize=10.5)

fig.suptitle("square cylinder, Re = 100 — against " + REF.CITATION +
             ", matched at 5% blockage", fontsize=12.5)
out = "figures/sqcyl_vs_data.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print(f"  wrote {out}")
print(f"  {'quantity':<12}{'ours':>10}{'reference':>11}{'error':>9}{'their grid spread':>19}")
for k, lab in names:
    s = spread[k]
    ss = f"{s:.1f}%" if np.isfinite(s) else "--"
    print(f"  {k:<12}{OURS[k]:>10.4f}{r[k]:>11.4f}"
          f"{100*(OURS[k]-r[k])/abs(r[k]):>8.1f}%{ss:>19}")
