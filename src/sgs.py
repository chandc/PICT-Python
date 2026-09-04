"""Subgrid-scale eddy viscosity: the strain rate, and closures built on it.

WHAT WAS ALREADY HERE, AND WHY THAT MATTERS. The variable-viscosity operator is validated --
MMS at order 1.96-1.99 with a spatially varying nu, and a manufactured van Driest channel to
0.126%. The convective operator is discretely SKEW-SYMMETRIC, checked by
test_energy_conservation.py, which is the property that decides whether a scheme can carry a
cascade at all rather than bleed it through its own truncation error. Those are the two things
an LES needs from its host solver and they were the hard ones.

WHAT THIS FILE ADDS is the part that was missing: a strain rate on a curvilinear multi-block
grid, and closures that turn it into nu_t.

THE FILTER WIDTH IS THE CELL, NOT A CONSTANT. On a stretched, curvilinear grid the local filter
width is the cube root of the cell VOLUME, and the Jacobian is exactly that volume. A code that
hardwires a uniform Delta is silently wrong everywhere the grid stretches, which for the cylinder
O-grid is a factor of 47 between the wall cell and the outer cell.

SEAMS ARE HANDLED BY THE DOMAIN, not by this file. Every derivative goes through
`MultiBlock.gradient`, which resolves the neighbour across a seam. That is the same exchange the
solver's own operators use, and the same one a distributed version would send over MPI --
`src/sgs_net.py` was recently changed for exactly this reason.
"""
import numpy as np

# Constants and their sources; full citations in reference/bibliography.md.
CS_SMAGORINSKY = 0.17          # Lilly (1967), for isotropic turbulence
CW_WALE = 0.55                 # Nicoud & Ducros (1999)
A_PLUS = 26.0                  # van Driest (1956)
CV_VREMAN = 0.07               # Vreman (2004), ~2.5 C_s^2
CSIGMA = 1.35                  # Nicoud et al. (2011)


def velocity_gradient(d, u, v, w):
    """[block][i][j] = d u_i / d x_j, with seams resolved by the domain.

    u, v, w are dicts keyed by block, as everything that crosses a seam must be: `pad_field`
    reads the neighbour, so it needs every block's data, not one block's.
    """
    nb = len(d.blocks)
    out = []
    for b in range(nb):
        g = [d.gradient(b, f) for f in (u, v, w)]      # g[i][j] = d u_i / d x_j
        out.append(g)
    return out


def strain_rate(grad_b):
    """S_ij = 1/2 (du_i/dx_j + du_j/dx_i) for one block, as a 3x3 list of arrays."""
    return [[0.5 * (grad_b[i][j] + grad_b[j][i]) for j in range(3)] for i in range(3)]


def strain_magnitude(grad_b):
    """|S| = sqrt(2 S_ij S_ij).

    The factor 2 is the convention Smagorinsky is calibrated with; dropping it rescales C_s by
    sqrt(2) and the model then disagrees with every published constant.
    """
    S = strain_rate(grad_b)
    return np.sqrt(2.0 * sum(S[i][j] ** 2 for i in range(3) for j in range(3)))


def filter_width(d, b):
    """Delta = (cell volume)^(1/3), per cell.

    THE JACOBIAN ALONE IS NOT THE CELL VOLUME. `block_metrics` returns J as the ratio of
    physical to COMPUTATIONAL volume, and the computational grid is normalised, so on a uniform
    box J = 1 whatever the resolution. Using J^(1/3) as Delta made nu_t resolution-INDEPENDENT:
    refining 8 -> 16 left it unchanged where it must fall by 4. The physical volume is
    J * h_xi * h_eta * h_zeta, and h is the computational spacing the block carries.
    """
    J = d.block_metrics_cached(b)[0]
    h = d.blocks[b].h
    return (np.abs(J) * h[0] * h[1] * h[2]) ** (1.0 / 3.0)


def smagorinsky(d, u, v, w, cs=CS_SMAGORINSKY, damping=None):
    """nu_t = (C_s Delta)^2 |S| -- Smagorinsky (1963) -- optionally van Driest (1956) damped.

    `damping` is a dict {block: y_plus} if wall damping is wanted. WITHOUT damping this model
    does not vanish at a wall -- |S| is largest there -- which is why it needs one and why WALE
    below is usually preferred for wall-bounded flow.
    """
    G = velocity_gradient(d, u, v, w)
    out = {}
    for b in range(len(d.blocks)):
        D = filter_width(d, b)
        if damping is not None:
            D = D * (1.0 - np.exp(-damping[b] / A_PLUS))
        out[b] = (cs * D) ** 2 * strain_magnitude(G[b])
    return out


def wale(d, u, v, w, cw=CW_WALE):
    """The WALE model -- Nicoud & Ducros (1999), see reference/bibliography.md.

        S^d_ij = 1/2 (g2_ij + g2_ji) - 1/3 delta_ij g2_kk,     g2 = g g   (g_ij = du_i/dx_j)
        nu_t   = (C_w Delta)^2  (S^d:S^d)^{3/2} / ( (S:S)^{5/2} + (S^d:S^d)^{5/4} )

    nu_t ~ y^3 approaching a wall, which is the correct asymptotic and is why WALE needs no
    van Driest damping. Worth stating precisely because this repo got the neighbouring fact
    wrong once: van Driest damping applied to the MIXING LENGTH gives nu_t ~ y^4, not y^3
    (measured at 3.993), so "van Driest fixes the near-wall scaling" is false as usually said.
    Measured here: exponent 2.997.

    WALE DOES NOT VANISH IN SOLID-BODY ROTATION, and a first draft of this docstring claimed it
    did. S^d is the traceless symmetric part of g*g, and for u = omega x r that is
    diag(-1, -1, 2) omega^2 / 3, so S^d:S^d = (2/3) omega^4 while S:S = 0 -- the model responds
    to the ROTATION rate by design, which is the whole point of building it from g*g rather than
    from S. The test now checks the exact analytic value (C_w Delta)^2 ((2/3) omega^4)^(1/4)
    rather than zero, which is a stronger check than the wrong one it replaced.
    """
    G = velocity_gradient(d, u, v, w)
    out = {}
    for b in range(len(d.blocks)):
        g = G[b]
        g2 = [[sum(g[i][k] * g[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        tr = sum(g2[k][k] for k in range(3))
        Sd = [[0.5 * (g2[i][j] + g2[j][i]) - (tr / 3.0 if i == j else 0.0) for j in range(3)]
              for i in range(3)]
        S = strain_rate(g)
        SS = sum(S[i][j] ** 2 for i in range(3) for j in range(3))
        SdSd = sum(Sd[i][j] ** 2 for i in range(3) for j in range(3))
        D = filter_width(d, b)
        num = SdSd ** 1.5
        den = SS ** 2.5 + SdSd ** 1.25
        # den -> 0 only where BOTH invariants vanish, i.e. no velocity gradient at all, where
        # the numerator vanishes faster. The guard keeps 0/0 from becoming a nan.
        out[b] = (cw * D) ** 2 * np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)
    return out


def vreman(d, u, v, w, cv=CV_VREMAN):
    """Vreman (2004). VANISHES IN PURE SHEAR, which Smagorinsky and WALE do not.

        alpha_ij = du_j/dx_i                     (the TRANSPOSE of g used elsewhere here)
        beta_ij  = Delta^2 alpha_mi alpha_mj     (summed over m)
        B_beta   = b11 b22 - b12^2 + b11 b33 - b13^2 + b22 b33 - b23^2
        nu_t     = c sqrt( B_beta / (alpha_ij alpha_ij) ),   c ~ 2.5 C_s^2

    WHY IT IS WORTH HAVING HERE. In the transitional phase of Taylor-Green, measured against the
    SEM reference at Re = 800, WALE reached nu_t/nu = 1.1 at t = 4 -- an eddy viscosity exceeding
    the molecular one while the flow had not yet broken down, over-dissipating by -3.0% where the
    unmodelled run was +2.6%. A model that switches off when there is nothing to model is the
    property being bought.

    For pure shear u = (a y, 0, 0) every term of B_beta vanishes and nu_t = 0 EXACTLY. It does
    NOT vanish in solid-body rotation: there B_beta = Delta^4 omega^4 and alpha:alpha = 2 omega^2,
    so nu_t = c Delta^2 omega / sqrt(2). Both are asserted in test_sgs_models.py.
    """
    G = velocity_gradient(d, u, v, w)
    out = {}
    for b in range(len(d.blocks)):
        g = G[b]
        D2 = filter_width(d, b) ** 2
        al = [[g[j][i] for j in range(3)] for i in range(3)]      # alpha_ij = du_j/dx_i
        be = [[D2 * sum(al[m][i] * al[m][j] for m in range(3)) for j in range(3)]
              for i in range(3)]
        Bb = (be[0][0]*be[1][1] - be[0][1]**2 + be[0][0]*be[2][2] - be[0][2]**2
              + be[1][1]*be[2][2] - be[1][2]**2)
        aa = sum(al[i][j] ** 2 for i in range(3) for j in range(3))
        # B_beta can go slightly negative on round-off where it is analytically zero
        Bb = np.maximum(Bb, 0.0)
        out[b] = np.where(aa > 1e-300, cv * np.sqrt(Bb / np.maximum(aa, 1e-300)), 0.0)
    return out


def sigma_model(d, u, v, w, cs=CSIGMA):
    """The sigma model, Nicoud et al. (2011) -- built from the SINGULAR VALUES of grad u.

        sigma_1 >= sigma_2 >= sigma_3 >= 0   singular values of g_ij = du_i/dx_j
        D_sigma = sigma_3 (sigma_1 - sigma_2)(sigma_2 - sigma_3) / sigma_1^2
        nu_t    = (C_sigma Delta)^2 D_sigma,   C_sigma ~ 1.35

    THE CLEANEST LAMINAR BEHAVIOUR OF ANY STATIC MODEL. D_sigma vanishes identically for every
    two-dimensional flow, for pure shear, for solid-body rotation, and for axisymmetric
    expansion -- in each case two singular values coincide or the smallest is zero, and the
    product collapses. It is also cubic in the wall distance without damping.

    Singular values come from the eigenvalues of g^T g, which is symmetric positive
    semi-definite, so `eigvalsh` is both cheaper and better conditioned than a full SVD.
    """
    G = velocity_gradient(d, u, v, w)
    out = {}
    for b in range(len(d.blocks)):
        g = G[b]
        shp = g[0][0].shape
        M = np.empty(shp + (3, 3))
        for i in range(3):
            for j in range(3):
                M[..., i, j] = sum(g[m][i] * g[m][j] for m in range(3))   # (g^T g)_ij
        lam = np.linalg.eigvalsh(M)                     # ascending, >= 0 up to round-off
        sig = np.sqrt(np.maximum(lam, 0.0))
        s3, s2, s1 = sig[..., 0], sig[..., 1], sig[..., 2]
        D = np.where(s1 > 1e-300, s3 * (s1 - s2) * (s2 - s3) / np.maximum(s1 ** 2, 1e-300), 0.0)
        out[b] = (cs * filter_width(d, b)) ** 2 * np.maximum(D, 0.0)
    return out


MODELS = {"smagorinsky": smagorinsky, "wale": wale, "vreman": vreman, "sigma": sigma_model}


def effective_viscosity(d, u, v, w, nu_mol, model="wale", **kw):
    """nu_eff = nu_mol + nu_t, as the dict of per-block arrays the solver takes.

    nu_t is non-negative by construction in both models, which the solver needs: a negative
    total viscosity makes the diffusion operator indefinite and the momentum solve can then
    diverge in a way that looks like a physical instability. test_sgs_models.py checks the sign
    rather than assuming it.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    nt = MODELS[model](d, u, v, w, **kw)
    return {b: nu_mol + nt[b] for b in nt}, nt
