"""Structured C-grid (all quads) around a NACA0012 at angle of attack, written as Gmsh 2.2 with the
physical names our cylinder driver uses (Inlet = far-field Dirichlet arc + top + bottom, Outlet = x = x_out,
Airfoil = no-slip). Chord 1, leading edge at the origin, free stream along +x, airfoil rotated clockwise
by alpha (nose up). Transfinite interpolation between the body+wake-cut curve and a C-shaped far boundary,
geometric stretching off the wall; the wake cut is merged so the two wake blocks are neighbours.

    python meshes/gen_naca_cgrid.py --alpha 20 --ns 121 --nw 61 --neta 71 --d0 0.005 --grow 1.08 \
        --xout 24 --R 8 --out meshes/naca0012_a20.msh
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=20.0); ap.add_argument("--split-pts", dest="split_pts", type=float, default=0.79, help="fraction of surface points in the nose segment (65% of the arc)"); ap.add_argument("--ns", type=int, default=193, help="surface points (odd, LE in the middle)")
ap.add_argument("--nw", type=int, default=141, help="wake points TE..outlet"); ap.add_argument("--h-le", dest="h_le", type=float, default=8e-4); ap.add_argument("--h-te", dest="h_te", type=float, default=0.008); ap.add_argument("--neta", type=int, default=71)
ap.add_argument("--d0", type=float, default=0.005, help="first cell height at the wall"); ap.add_argument("--grow", type=float, default=1.08, help="(unused: ratio is solved per line from d0 and the line length)")
ap.add_argument("--xout", type=float, default=24.0); ap.add_argument("--R", type=float, default=8.0, help="far-field radius / half-height")
ap.add_argument("--wake0", type=float, default=0.012, help="first wake-cut spacing at the TE"); ap.add_argument("--out", default="meshes/naca0012_a20.msh"); ap.add_argument("--iters", type=int, default=4000)
a = ap.parse_args()

def naca0012(x):   # closed trailing edge (Ladson coefficients with -0.1036)
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
def geo(n, first, L):
    # n points, first spacing `first`, total length L: solve ratio r
    from scipy.optimize import brentq
    f = lambda r: first * (r**(n - 1) - 1) / (r - 1) - L
    r = brentq(f, 1.0000001, 3.0); s = first * (r**np.arange(n - 1) ); return np.r_[0, np.cumsum(s)] / L, r

# surface: lower TE -> LE -> upper TE, distributed by ARC LENGTH with two geometric segments per side:
# from the nose (first spacing h_le) over the front 65% of the arc, and from the TE (first spacing h_te)
# over the rear 35%, so both the nose and the sharp trailing edge are resolved and the TE spacing matches
# the wake cut. (Nose-only cosine clustering left 0.026 cells at the TE against 0.005 wall layers, and the
# smoother opened them into a sparse fan behind the trailing edge.)
n_half = (a.ns + 1) // 2
xf = np.linspace(0, 1, 20001); yf = naca0012(xf); sf = np.r_[0, np.cumsum(np.hypot(np.diff(xf), np.diff(yf)))]; Stot = sf[-1]
nA = int(round(a.split_pts * (n_half - 1))) + 1; nB = n_half - nA + 1        # points split so the two segments' end spacings meet
tA, rA = geo(nA, a.h_le, 0.65 * Stot); tB, rB = geo(nB, a.h_te, 0.35 * Stot)
s_arc = np.r_[tA * 0.65 * Stot, Stot - (tB[::-1] * 0.35 * Stot)[1:]]          # 0 (LE) .. Stot (TE)
xc = np.interp(s_arc, sf, xf)
dA = np.diff(tA * 0.65 * Stot); dB = np.diff(tB * 0.35 * Stot)
print(f"surface: {n_half} pts/side, nose spacing {dA[0]:.1e} -> {dA[-1]:.4f} (ratio {rA:.3f}), join, {dB[-1]:.4f} <- TE spacing {dB[0]:.4f} (ratio {rB:.3f})")
lower = np.c_[xc[::-1], -naca0012(xc[::-1])]; upper = np.c_[xc, naca0012(xc)]
surf = np.vstack([lower, upper[1:]])                                    # ns points, TE at both ends, LE in the middle
al = np.radians(a.alpha); Rm = np.array([[np.cos(al), np.sin(al)], [-np.sin(al), np.cos(al)]])   # clockwise
surf = surf @ Rm.T
te = surf[0]                                                            # trailing edge (rotated)
# wake cut: from TE straight downstream to x_out, geometric stretching from wake0
tw, rw = geo(a.nw, a.wake0, a.xout - te[0])
xw = te[0] + tw * (a.xout - te[0])
# cut leaves the TE along the chord line (angle -alpha) and relaxes exponentially to horizontal, so the
# body/cut junction has no kink: y' = -tan(alpha) exp(-(x - x_te)/lam)
lam = 1.0; yw = te[1] - np.tan(al) * lam * (1 - np.exp(-(xw - te[0]) / lam))
wake = np.c_[xw, yw]
# inner curve xi: wake-lower (outlet -> TE), surface (TE lower -> LE -> TE upper), wake-upper (TE -> outlet)
inner = np.vstack([wake[::-1], surf[1:-1], wake])
nxi = len(inner)
# ---- grid generation: transfinite interpolation, then elliptic (Winslow) smoothing with control
# functions taken from the algebraic grid so the wall clustering survives (Thompson-Thames-Mastin with
# Sorenson-style sources). Straight TFI lines from the clustered nose fan to 10-degree cells; marching
# along normals folded at the 172-degree wedges the cut makes at the sharp trailing edge. The elliptic
# system has neither problem: it keeps the boundary point distributions and bends the lines smoothly.
nwl = a.nw; nsf = a.ns - 2
# far boundary: a C of radius R about the TE x, closed by lines to the outlet at y = +-R. Over the wake the
# far points sit directly above/below the cut points (vertical eta-lines, no shear: uniformly spaced far
# points against the geometric cut gave 22-degree cells around x ~ 11-15, |y| ~ 4 and the flow blew up there);
# around the arc the spacing grows geometrically from the junction value toward the front, so it is
# continuous at the junctions. The nose lines fan onto the coarse front arc; the elliptic smoother handles that.
cx = te[0]
outer = np.zeros_like(inner)
outer[:nwl] = np.c_[wake[::-1, 0], np.full(nwl, -a.R)]
outer[nwl + nsf:] = np.c_[wake[:, 0], np.full(nwl, a.R)]
Larc = np.pi * a.R; sj = wake[1, 0] - wake[0, 0]                                   # junction spacing = first cut spacing
nh = (nsf + 2 + 1) // 2                                                            # arc points incl. both junctions, per half
tj, rj = geo(nh, sj, Larc / 2); s_arc_out = np.r_[tj * Larc / 2, Larc - (tj[::-1] * Larc / 2)[1:]][1:-1]   # exclude junctions (they are the wake ends)
if len(s_arc_out) != nsf: s_arc_out = np.interp(np.linspace(0, 1, nsf), np.linspace(0, 1, len(s_arc_out)), s_arc_out)
th = -np.pi / 2 + s_arc_out / a.R
outer[nwl:nwl + nsf] = np.c_[cx - a.R * np.cos(th), a.R * np.sin(th)]
print(f"far arc: spacing {sj:.3f} at the junctions growing (ratio {rj:.3f}) to {np.diff(s_arc_out).max():.2f} at the front")
L = np.hypot(*(outer - inner).T)
dxi = np.r_[np.hypot(*(inner[1] - inner[0])), 0.5 * np.hypot(*(inner[2:] - inner[:-2]).T), np.hypot(*(inner[-1] - inner[-2]))]
on_body = np.zeros(nxi, bool); on_body[nwl - 1:nwl + nsf + 1] = True
d_first = np.where(on_body, a.d0, np.clip(0.3 * dxi, a.d0, 0.5))
d_first = np.minimum(d_first, 0.8 * L / (a.neta - 1))
S = np.zeros((a.neta, nxi)); ratios = np.zeros(nxi)
for i in range(nxi):
    S[:, i], ratios[i] = geo(a.neta, d_first[i], L[i])
print(f"first cell {a.d0} on the airfoil, {d_first[~on_body].min():.4f}..{d_first[~on_body].max():.3f} on the cut; growth {ratios.min():.4f}..{ratios.max():.4f}")
X0 = inner[None] + S[:, :, None] * (outer - inner)[None]                    # algebraic (TFI) grid, (neta, nxi, 2)
def d_xi(F):  return 0.5 * (F[:, 2:] - F[:, :-2])
def d_eta(F): return 0.5 * (F[2:, :] - F[:-2, :])
def dd_xi(F): return F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]
def dd_eta(F): return F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]
I = (slice(1, -1), slice(1, -1))
# control functions from the algebraic grid (interior nodes)
xk, yk = d_xi(X0[..., 0])[1:-1], d_xi(X0[..., 1])[1:-1]; xkk, ykk = dd_xi(X0[..., 0])[1:-1], dd_xi(X0[..., 1])[1:-1]
xe, ye = d_eta(X0[..., 0])[:, 1:-1], d_eta(X0[..., 1])[:, 1:-1]; xee, yee = dd_eta(X0[..., 0])[:, 1:-1], dd_eta(X0[..., 1])[:, 1:-1]
phi = -(xk * xkk + yk * ykk) / np.maximum(xk**2 + yk**2, 1e-30)
psi = -(xe * xee + ye * yee) / np.maximum(xe**2 + ye**2, 1e-30)
X = X0.copy(); omega = 1.0
for it in range(a.iters):
    x, y = X[..., 0], X[..., 1]
    xk, yk = d_xi(x)[1:-1], d_xi(y)[1:-1]; xe, ye = d_eta(x)[:, 1:-1], d_eta(y)[:, 1:-1]
    al_ = xe**2 + ye**2; be_ = xk * xe + yk * ye; ga_ = xk**2 + yk**2
    def upd(F, Fk, Fe):
        cross = 0.25 * (F[2:, 2:] - F[2:, :-2] - F[:-2, 2:] + F[:-2, :-2])
        num = al_ * (F[1:-1, 2:] + F[1:-1, :-2] + phi * Fk) + ga_ * (F[2:, 1:-1] + F[:-2, 1:-1] + psi * Fe) - 2 * be_ * cross
        return num / (2 * (al_ + ga_))
    xn, yn = upd(x, xk, xe), upd(y, yk, ye)
    dmax = max(np.abs(xn - x[I]).max(), np.abs(yn - y[I]).max())
    X[1:-1, 1:-1, 0] = (1 - omega) * x[I] + omega * xn; X[1:-1, 1:-1, 1] = (1 - omega) * y[I] + omega * yn
    if it % 500 == 0 or dmax < 1e-7: print(f"  elliptic smoothing it {it}: max move {dmax:.2e}")
    if dmax < 1e-7: break
# merge duplicate nodes (wake cut appears twice at eta=0; TE thrice)
pts = X.reshape(-1, 2); key = np.round(pts, 9); _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
nodes = pts[np.sort(idx)]; remap = {old: new for new, old in enumerate(np.sort(idx))}
order = np.argsort(idx); inv_sorted = np.empty(len(idx), int); inv_sorted[order] = np.arange(len(idx))
node_id = inv_sorted[inv].reshape(a.neta, nxi)
cells = []
for j in range(a.neta - 1):
    for i in range(nxi - 1):
        q = [node_id[j, i], node_id[j, i + 1], node_id[j + 1, i + 1], node_id[j + 1, i]]
        if len(set(q)) < 4: continue                                   # degenerate at the TE/cut merge
        cells.append(q)
cells = np.array(cells)
# orientation: make all CCW
P = nodes[cells]; area = 0.5 * ((P[:, 0, 0] * P[:, 1, 1] - P[:, 1, 0] * P[:, 0, 1]) + (P[:, 1, 0] * P[:, 2, 1] - P[:, 2, 0] * P[:, 1, 1]) + (P[:, 2, 0] * P[:, 3, 1] - P[:, 3, 0] * P[:, 2, 1]) + (P[:, 3, 0] * P[:, 0, 1] - P[:, 0, 0] * P[:, 3, 1]))
nneg = (area < 0).sum(); npos = (area > 0).sum(); print(f"  orientation: {npos} CCW, {nneg} CW (a mix means folded cells)"); cells[area < 0] = cells[area < 0][:, ::-1]; area = np.abs(area)
# boundary edges with tags: 2 Inlet (outer arc + top + bottom), 4 Outlet (xi = 0 and xi = nxi-1 columns), 5 Airfoil
edges = []
for i in range(nxi - 1): edges.append((node_id[-1, i], node_id[-1, i + 1], 2))
for j in range(a.neta - 1): edges.append((node_id[j, 0], node_id[j + 1, 0], 4)); edges.append((node_id[j, -1], node_id[j + 1, -1], 4))
for i in range(nwl - 1, nwl + nsf): edges.append((node_id[0, i], node_id[0, i + 1], 5))
edges = [(p, q, t) for p, q, t in edges if p != q]
# quality
d = [np.hypot(*(nodes[cells[:, k]] - nodes[cells[:, (k + 1) % 4]]).T) for k in range(4)]
asp = np.maximum.reduce(d) / np.minimum.reduce(d)
def angle(pa, pb, pc):
    u = pa - pb; v = pc - pb; return np.degrees(np.arccos(np.clip((u * v).sum(1) / np.hypot(*u.T) / np.hypot(*v.T), -1, 1)))
ang = np.stack([angle(nodes[cells[:, (k - 1) % 4]], nodes[cells[:, k]], nodes[cells[:, (k + 1) % 4]]) for k in range(4)])
print(f"NACA0012 alpha={a.alpha}: {len(nodes)} nodes, {len(cells)} quads; boundary edges: inlet {sum(t==2 for *_,t in edges)}, outlet {sum(t==4 for *_,t in edges)}, airfoil {sum(t==5 for *_,t in edges)}")
cc = nodes[cells].mean(axis=1); ia = np.argmax(asp); im = np.argmin(ang.min(axis=0))
print(f"  area min {area.min():.2e} max {area.max():.2e}; aspect p99 {np.percentile(asp,99):.1f} max {asp.max():.1f} at ({cc[ia,0]:.3f},{cc[ia,1]:.3f}); min angle {ang.min():.1f} deg at ({cc[im,0]:.3f},{cc[im,1]:.3f}), p1 {np.percentile(ang.min(axis=0),1):.1f}; wake stretch ratio {rw:.3f}")
print(f"  cells with min angle < 30 deg: {(ang.min(axis=0) < 30).sum()}, < 20 deg: {(ang.min(axis=0) < 20).sum()}; aspect > 100: {(asp > 100).sum()}")
with open(a.out, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n1 2 \"Inlet\"\n1 4 \"Outlet\"\n1 5 \"Airfoil\"\n2 1 \"Fluid\"\n$EndPhysicalNames\n")
    f.write(f"$Nodes\n{len(nodes)}\n"); f.writelines(f"{k+1} {x:.12g} {y:.12g} 0\n" for k, (x, y) in enumerate(nodes)); f.write("$EndNodes\n")
    f.write(f"$Elements\n{len(edges)+len(cells)}\n"); e = 1
    for p, q, t in edges: f.write(f"{e} 1 2 {t} {t} {p+1} {q+1}\n"); e += 1
    for c in cells: f.write(f"{e} 3 2 1 1 {c[0]+1} {c[1]+1} {c[2]+1} {c[3]+1}\n"); e += 1
    f.write("$EndElements\n")
print("wrote", a.out)
