"""T8: perturbation-energy growth over time for every (grid, dt) run, on the same window.
E(t) = |a(t)|^2 with a(t) the complex amplitude of the seeded Orr-Sommerfeld mode (first streamwise
Fourier mode of u', v' projected on its initial shape), so d ln E / dt = 2 sigma. Left: ln E vs t with
the reference slope; middle: local sigma(t) from centred differences of ln|a|; right: table of the
window fit (t = 30..100) and the orders in dt (fixed 48x400) and in h (fixed dt = 0.0125)."""
import glob, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G_REF, C_REF, ALPHA, T0, T1 = 0.00223497, 0.24989154, 1.0, 30.0, 100.0
runs = {}
for f in sorted(glob.glob("results/t8_os_*.npz")):
    m = re.match(r".*t8_os_(?:nx(\d+)_)?ny(\d+)(?:_dt([0-9.]+))?(_fx)?\.npz", f)
    nx = int(m.group(1)) if m.group(1) else 48; ny = int(m.group(2)); dt = float(m.group(3)) if m.group(3) else 0.05; fx = bool(m.group(4)) or bool(m.group(1))
    d = np.load(f); runs[(nx, ny, dt, fx)] = (d["t"], d["a"])
fig, ax = plt.subplots(1, 3, figsize=(22, 6.5)); rows = []
cols = {100: "C0", 200: "C1", 400: "C2"}; styles = {0.1: ":", 0.05: "-", 0.025: "--", 0.0125: "-."}
for (nx, ny, dt, fx), (t, a) in sorted(runs.items()):
    lnE = 2 * np.log(np.abs(a)); sel = (t >= T0) & (t <= T1)
    sig = np.polyfit(t[sel], np.log(np.abs(a[sel])), 1)[0]; cph = -np.polyfit(t[sel], np.unwrap(np.angle(a[sel])), 1)[0] / ALPHA
    rows.append((nx, ny, dt, fx, sig, cph))
    lab = f"{nx}x{ny}, dt={dt}{', 2F-F' if fx else ', F^n'}: sigma={sig:.6f} ({(sig/G_REF-1)*100:+.1f}%)"
    ax[0].plot(t, lnE - np.interp(T0, t, lnE), color=cols[ny], ls=styles[dt], lw=1.4 if fx else 0.8, alpha=1 if fx else 0.5, label=lab)
    loc = np.gradient(np.log(np.abs(a)), t); ax[1].plot(t[1:-1], loc[1:-1], color=cols[ny], ls=styles[dt], lw=1.2 if fx else 0.7, alpha=1 if fx else 0.5)
ax[0].plot([T0, T1], [0, 2 * G_REF * (T1 - T0)], "k", lw=2.5, alpha=0.5, label=f"reference slope 2 sigma, sigma={G_REF}")
ax[0].axvspan(T0, T1, color="0.9", zorder=0); ax[0].set_xlabel("t"); ax[0].set_ylabel("ln E(t) - ln E(30)"); ax[0].set_title("perturbation energy of the OS mode"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
ax[1].axhline(G_REF, color="k", lw=2, alpha=0.5, label="reference sigma"); ax[1].axvspan(T0, T1, color="0.9", zorder=0); ax[1].set_ylim(0, 0.004); ax[1].set_xlabel("t"); ax[1].set_title("local growth rate d ln|a|/dt (same colours/styles)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
# table + orders
txt = [f"{'grid':>8} {'dt':>7} {'flux':>5} {'sigma':>9} {'err':>7} {'phase c':>9} {'err':>7}"]
for nx, ny, dt, fx, sig, cph in rows: txt.append(f"{nx}x{ny:<4d} {dt:7.4f} {'2F-F' if fx else 'F^n':>5} {sig:9.6f} {(sig/G_REF-1)*100:+6.1f}% {cph:9.5f} {(cph/C_REF-1)*100:+6.2f}%")
def order(vals, ratio=2):
    return np.log((vals[0] - vals[1]) / (vals[1] - vals[2])) / np.log(ratio) if len(vals) == 3 and (vals[0] - vals[1]) * (vals[1] - vals[2]) > 0 else float("nan")
lag = sorted([r for r in rows if r[0] == 48 and r[1] == 400 and not r[3]], key=lambda r: -r[2])
if len(lag) >= 3: txt.append(f"TEMPORAL, lagged flux F^n (48x400, dt {lag[0][2]}..{lag[-1][2]}): phase order {order([r[5] for r in lag[:3]]):.2f} -> first order")
fx200 = sorted([r for r in rows if r[0] == 48 and r[1] == 200 and r[3]], key=lambda r: -r[2])
if len(fx200) >= 3: txt.append(f"TEMPORAL, 2F-F (48x200, dt {fx200[0][2]}..{fx200[-1][2]}): phase {', '.join(f'{(r[5]/C_REF-1)*100:+.2f}%' for r in fx200)} -> dt-independent")
sp = sorted([r for r in rows if r[0] == 48 and abs(r[2] - 0.05) < 1e-9 and r[3]], key=lambda r: r[1])
if len(sp) == 3:
    e = [abs(r[4] - G_REF) for r in sp]; txt.append(f"SPATIAL wall-normal, 2F-F dt 0.05 (48 x 100/200/400): sigma error {np.log2(e[0]/e[1]):.2f}, {np.log2(e[1]/e[2]):.2f}; Richardson raw sigma {order([r[4] for r in sp]):.2f}, phase {order([r[5] for r in sp]):.2f}")
for ny_ in (200, 400):
    pair = {r[0]: r for r in rows if r[1] == ny_ and abs(r[2] - 0.05) < 1e-9 and r[3]}
    if 48 in pair and 96 in pair: txt.append(f"SPATIAL streamwise, ny={ny_}: phase err {(pair[48][5]/C_REF-1)*100:+.2f}% (48 cells/wavelength) -> {(pair[96][5]/C_REF-1)*100:+.2f}% (96); central-difference dispersion (kh)^2/6 predicts a 0.29% -> 0.07% contribution")
ax[2].axis("off"); ax[2].text(0, 1, "\n".join(txt), family="monospace", fontsize=9, va="top"); print("\n".join(txt))
plt.suptitle("T8 Orr-Sommerfeld Re=7500: energy growth of the seeded mode, all runs on the window t=30..100", fontsize=12); plt.tight_layout()
plt.savefig("figures/t8_os_growth.png", dpi=120); print("wrote figures/t8_os_growth.png")
