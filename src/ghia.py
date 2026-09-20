"""Ghia, Ghia & Shin (1982) lid-driven cavity reference data, Re = 1000.

Transcribed from ~/Dropbox/Apple_MLX_CFD/sem_demo: `cavity_re1000_data.npz` (which carries
`Re = 1000.0` alongside the arrays) for the u-profile, and `scratch/plot_cavity.py`
(`ghia_v_1000`) for the v-profile.

THE NAMES CARRY THE REYNOLDS NUMBER, deliberately. That source repo's own notes record the
hazard:

    "The repo contains two different `ghia_v` arrays, both labelled only `ghia_v`"
        scratch/plot_cavity.py   -> -0.51550  (Re = 1000)
        lssem2d/tests/...        -> -0.2453   (Re = 100)

and the same applies to its two `ghia_u` arrays -- the one in `lssem2d/tests/verification.py`
begins 1.0, 0.8412, 0.7887, which is the Re = 100 profile, not this one. Comparing a Re = 1000
solution against the Re = 100 table would look like a solver error of roughly the right
magnitude to be believed, so the arrays here are never named bare `ghia_u` / `ghia_v`.

Convention: unit square, lid at y = 1 moving with u = 1.
  U_RE1000  -- u along the VERTICAL centreline x = 0.5, sampled at Y_RE1000
  V_RE1000  -- v along the HORIZONTAL centreline y = 0.5, sampled at X_RE1000
Both tables run from the far wall inwards, i.e. y (and x) DESCENDING.
"""
import numpy as np

RE = 1000.0

Y_RE1000 = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                     0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
U_RE1000 = np.array([1.0000, 0.6593, 0.5749, 0.5112, 0.4660, 0.3330, 0.1872, 0.0570, -0.0608,
                     -0.1065, -0.2781, -0.3829, -0.2973, -0.2222, -0.2020, -0.1811, 0.0000])

X_RE1000 = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                     0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
V_RE1000 = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                     -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                     0.27485, 0.0000])

# Sanity anchors, checked by test: the two extrema every published plot of this case shows.
U_MIN_RE1000 = -0.3829        # at y = 0.1719
V_MIN_RE1000 = -0.51550       # at x = 0.9063
V_MAX_RE1000 = 0.37095        # at x = 0.1563


def check():
    """Guard against a silent Re mix-up: these values are unique to Re = 1000."""
    problems = []
    if abs(U_RE1000.min() - U_MIN_RE1000) > 1e-12:
        problems.append("u-profile minimum is not the Re=1000 value")
    if abs(V_RE1000.min() - V_MIN_RE1000) > 1e-12:
        problems.append("v-profile minimum is not the Re=1000 value")
    if abs(U_RE1000[1] - 0.6593) > 1e-12:
        problems.append("u[1] is not 0.6593 -- this looks like the Re=100 table (0.8412)")
    for nm, a in (("Y", Y_RE1000), ("U", U_RE1000), ("X", X_RE1000), ("V", V_RE1000)):
        if len(a) != 17:
            problems.append(f"{nm} has {len(a)} entries, Ghia's tables have 17")
    return problems
