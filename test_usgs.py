"""G3 unit checks for src/usgs.py on the 2.5D solver's fields.
  1 solid rotation u = omega x r: |S| = 0, Smagorinsky nu_t = 0, WALE = (C_w Delta)^2 ((2/3) omega^4)^(1/4) exactly
  2 filter width follows the cell: halving every cell dimension quarters nu_t (Delta^2)
  3 WALE ~ y^3 approaching a no-slip wall, on the mesh (gradient path included), exponent 3 +- 0.25
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np; sys.path.insert(0, ".")
from src.umesh import rect_mesh
from src.uops import DIRICHLET, NEUMANN
from src.upiso import BC
from src.upiso25 import PISO25
import src.usgs as usgs
def solver(nx, ny, nz, Lz=1.0, ylim=(0, 1), periodic_x=True, fields=None):
    m = rect_mesh(nx, ny, 0, 1, ylim[0], ylim[1], cells="quad")
    if periodic_x: m.make_periodic(1, 2, (1.0, 0.0))
    bf = m.fcentre[m.bfaces]; z = np.arange(nz) * Lz / nz
    bcs = []
    for comp in range(3):
        if fields is None: bcs.append(BC(m))
        else:
            vals = fields(bf[:, 0], bf[:, 1], np.zeros(len(bf)))[comp]   # boundary values at z = 0 (z-independent fields)
            bcs.append(BC(m, np.full(m.nbface, DIRICHLET), vals))
    s = PISO25(m, nz, Lz, 0.0, 0.01, bcs[0], bcs[1], bcs[2], BC(m))
    if fields is not None:
        x, y = m.centroid.T; X, Y, Z = x[:, None], y[:, None], z[None, :]
        u, v, w = fields(X, Y, Z); s.u[:] = u; s.v[:] = v; s.w[:] = w
    return s
ok_all = True
# 1 rotation
om = 1.7
s = solver(12, 12, 4, periodic_x=False, fields=lambda x, y, z: (-om * y + 0 * z, om * x + 0 * z, 0 * x + 0 * z))
g = usgs.velocity_gradient(s); mag = usgs.strain_magnitude(g)
nts = usgs.smagorinsky(s, g); ntw = usgs.wale(s, g); D = usgs.filter_width(s)
w_exact = (usgs.CW_WALE * D) ** 2 * ((2 / 3) * om ** 4) ** 0.25
ok = np.abs(mag).max() < 1e-9 and np.abs(nts).max() < 1e-12 and np.abs(ntw / w_exact[:, None] - 1).max() < 1e-6; ok_all &= ok
print(f"  [{'PASS' if ok else 'FAIL'}] solid rotation: |S| max {np.abs(mag).max():.1e}, Smagorinsky {np.abs(nts).max():.1e}, WALE / exact - 1 max {np.abs(ntw / w_exact[:, None] - 1).max():.1e}")
# 2 filter width
a = 2.5; vals = []
for n, nz in ((8, 4), (16, 8)):
    s = solver(n, n, nz, periodic_x=False, fields=lambda x, y, z: (a * y + 0 * z, 0 * y + 0 * z, 0 * y + 0 * z)); vals.append(float(usgs.smagorinsky(s).mean()))
ok = abs(vals[0] / vals[1] - 4) < 0.05; ok_all &= ok
print(f"  [{'PASS' if ok else 'FAIL'}] filter width follows the cell: nu_t coarse/fine {vals[0]/vals[1]:.4f} (4.00 for Delta^2)")
# 3 WALE wall exponent: u = A y, v = B y^2 on a wall-bounded strip, wall at y = 0
A, B = 1.0, 0.5; ny = 192
s = solver(4, ny, 4, ylim=(0, 1), periodic_x=True, fields=lambda x, y, z: (A * y + 0 * z, B * y ** 2 + 0 * z, 0 * y + 0 * z))
nt = usgs.wale(s)[:, 0]; y = s.m.centroid[:, 1]
sel = (y > 0.02) & (y < 0.12); p = np.polyfit(np.log(y[sel]), np.log(nt[sel]), 1)[0]
ok = abs(p - 3) < 0.25; ok_all &= ok
print(f"  [{'PASS' if ok else 'FAIL'}] WALE near-wall exponent {p:.3f} (expected 3), fit y in (0.02, 0.12) on {ny} cells")
print("ALL PASS" if ok_all else "FAILURES")
