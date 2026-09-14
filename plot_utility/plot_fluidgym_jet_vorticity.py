"""v2: base-flow-subtracted vorticity (channel Poiseuille background removed)
on a single global triangulation so block seams blend smoothly."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

z = np.load("h80_fields.npz")
NB = int(z["nblocks"])
DT_CTRL = 0.25


def omega_z_block(x, y, u, v):
    x_xi, x_eta = np.gradient(x, axis=0), np.gradient(x, axis=1)
    y_xi, y_eta = np.gradient(y, axis=0), np.gradient(y, axis=1)
    J2 = x_xi * y_eta - x_eta * y_xi
    J2 = np.where(np.abs(J2) < 1e-300, 1e-300, J2)
    u_xi, u_eta = np.gradient(u, axis=0), np.gradient(u, axis=1)
    v_xi, v_eta = np.gradient(v, axis=0), np.gradient(v, axis=1)
    return (y_eta * v_xi - y_xi * v_eta) / J2 - (-x_eta * u_xi + x_xi * u_eta) / J2


def centers(xy):
    Xv, Yv = xy[0], xy[1]
    Xc = 0.25 * (Xv[:-1, :-1] + Xv[1:, :-1] + Xv[:-1, 1:] + Xv[1:, 1:])
    Yc = 0.25 * (Yv[:-1, :-1] + Yv[1:, :-1] + Yv[:-1, 1:] + Yv[1:, 1:])
    return Xc, Yc


# ---- channel base flow: parabolic fit to the inflow profile -> omega_base(y)
xy0 = z["u_xy0"][0]
Xc0, Yc0 = centers(xy0)
col = np.argmin(Xc0.mean(axis=0))
y_in, u_in = Yc0[:, col], z["u_vel0"][0][0][:, col]
c2, c1, c0 = np.polyfit(y_in, u_in, 2)          # u(y) ~ c2 y^2 + c1 y + c0
omega_base = lambda y: -(2.0 * c2 * y + c1)     # -du/dy of the fit
print(f"base-flow fit: u(y) = {c2:.4f} y^2 + {c1:.4f} y + {c0:.4f}  "
      f"(wall omega ~ {omega_base(-1.9):+.2f}/{omega_base(1.9):+.2f})")


def global_tri(prefix):
    xs, ys, os_ = [], [], []
    for b in range(NB):
        Xc, Yc = centers(z[f"{prefix}_xy{b}"][0])
        u, v = z[f"{prefix}_vel{b}"][0]
        om = omega_z_block(Xc, Yc, u, v) - omega_base(Yc)
        xs.append(Xc.ravel()); ys.append(Yc.ravel()); os_.append(om.ravel())
    x, y, o = np.concatenate(xs), np.concatenate(ys), np.concatenate(os_)
    tri = mtri.Triangulation(x, y)
    # mask triangles inside the cylinder or spanning unnaturally far
    xt = x[tri.triangles].mean(axis=1)
    yt = y[tri.triangles].mean(axis=1)
    r2 = xt ** 2 + yt ** 2
    ex = np.max(np.ptp(x[tri.triangles], axis=1))
    span = np.maximum(np.ptp(x[tri.triangles], axis=1),
                      np.ptp(y[tri.triangles], axis=1))
    tri.set_mask((r2 < 0.52 ** 2) | (span > 0.6))
    return tri, o


tri_u, om_u = global_tri("u")
tri_c, om_c = global_tri("c")

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 1.4], hspace=0.32, wspace=0.18)
lim = 2.0
levels = np.linspace(-lim, lim, 41)
for row, (tri, om, title, md) in enumerate((
        (tri_u, om_u, "uncontrolled", float(z["drag_u"].mean())),
        (tri_c, om_c, "jet-controlled (H=80 DPC policy)", float(z["drag_c"].mean())))):
    for col_i, (xlim, tag) in enumerate((((-2, 12), "near field"),
                                         ((-3, 21), "far field"))):
        ax = fig.add_subplot(gs[row, col_i])
        pc = ax.tricontourf(tri, np.clip(om, -lim, lim), levels=levels,
                            cmap="RdBu_r", extend="both")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.fill(0.5 * np.cos(th), 0.5 * np.sin(th), color="0.3", zorder=5)
        for s in (+1, -1):
            phi = np.radians(np.linspace(80, 100, 20)) * s
            ax.plot(0.52 * np.cos(phi), 0.52 * np.sin(phi), color="lime",
                    lw=3, zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(-2.1, 2.1)
        ax.set_aspect("equal")
        ax.set_title(f"{title} -- {tag}  (mean C_D {md:.3f})", fontsize=10)
        if col_i == 1:
            fig.colorbar(pc, ax=ax, label=r"$\omega_z - \omega_{base}(y)$",
                         shrink=0.8)

t = DT_CTRL * np.arange(len(z["drag_u"]))
ax1 = fig.add_subplot(gs[2, 0])
ax1.plot(t, z["drag_u"], "0.4", label="uncontrolled")
ax1.plot(t, z["drag_c"], "C3", label="controlled")
ax1.axhline(3.3281555, color="0.7", ls=":", lw=1)
ax1.set_xlabel("t  (time units)"); ax1.set_ylabel(r"$C_D$")
ax1.legend(fontsize=8); ax1.set_title("drag", fontsize=10)
ax2 = fig.add_subplot(gs[2, 1])
ax2.plot(t, z["lift_u"], "0.4", label=r"$C_L$ uncontrolled")
ax2.plot(t, z["lift_c"], "C3", label=r"$C_L$ controlled")
ax2.plot(t, z["act_c"], "C0", lw=1, label="action a(t)")
ax2.set_xlabel("t  (time units)")
ax2.legend(fontsize=8); ax2.set_title("lift and jet action", fontsize=10)

fig.suptitle("CylinderJet2D-easy (Re 100): wake vorticity with the channel "
             "base flow subtracted", fontsize=13)
fig.savefig("jet_vorticity_comparison_v2.png", dpi=150, bbox_inches="tight")
print("saved jet_vorticity_comparison_v2.png")
