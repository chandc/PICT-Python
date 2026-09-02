"""Learn an eddy-viscosity profile that reproduces a target axial velocity profile.

THE INVERSE PROBLEM, which is the one that matters. `test_van_driest_channel.py` runs it
forwards: prescribe nu_t, get U, check it against the manufactured answer. That validates the
solver and teaches the network nothing. Here the target is the VELOCITY PROFILE and nu_t(y) is
what is learned:

    given   U_target(y)          the van Driest profile at Re_tau = 180
    find    nu_t(y) >= 0         such that the solved U matches it
    loss    || U(nu_t) - U_target ||^2

Nothing about nu_t appears in the loss. It is recovered only through the momentum balance, which
is exactly the position a closure network is in.

WHY THE GRADIENT EXISTS AT ALL. nu_t enters the MATRIX, not the right-hand side, so dL/dnu_t
runs through A. Freeze A -- the frozen-coefficient approximation Stage 4 measured as a mild bias
for a body force -- and this gradient is IDENTICALLY ZERO: the network sees nothing, the loss
does not move, and nothing errors. That is Stage 5c.4 in the plan, and it is why this file
assembles A differentiably in torch rather than reusing the frozen assembly.

POSITIVITY IS IN THE GRAPH, not applied afterwards. nu_t = softplus(theta) keeps the operator
positive definite for every theta the optimiser can reach; a clip would put a discontinuity in
the gradient exactly where it matters.

WHAT THIS CANNOT RECOVER, and it is not the optimiser's fault. At the centreline dU/dy -> 0 and
the total stress -> 0 together, so nu_t multiplies something that vanishes: the velocity is
insensitive to it and no amount of data or training fixes that. The run below reports the error
inside and outside that region separately, because a single number would hide it.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
KAPPA, APLUS, RE_TAU = 0.41, 26.0, 180.0
NY_HALF_RATIO = 1.05


def van_driest(n=400001):
    yp = np.linspace(0.0, RE_TAU, n)
    D = 1.0 - np.exp(-yp / APLUS)
    lm = KAPPA * yp * D
    tau = 1.0 - yp / RE_TAU
    dU = 2.0 * tau / (1.0 + np.sqrt(1.0 + 4.0 * lm ** 2 * tau))
    U = np.concatenate([[0.0], np.cumsum(0.5 * (dU[1:] + dU[:-1]) * np.diff(yp))])
    return yp, U, lm ** 2 * dU


def grid(dy0_plus=1.0, ratio=NY_HALF_RATIO):
    ys, dy = [0.0], dy0_plus / RE_TAU
    while ys[-1] < 1.0:
        ys.append(ys[-1] + dy)
        dy *= ratio
    ys = np.array(ys) / ys[-1]
    return np.concatenate([ys[:-1], 2.0 - ys[::-1]])


def solve(nu_t, yp_grid):
    """A(nu_t) U = b in wall units, with nu_t a torch tensor. Differentiable in nu_t."""
    n = len(yp_grid)
    nu = 1.0 + nu_t
    A = torch.zeros(n, n, dtype=torch.float64)
    b = -torch.ones(n, dtype=torch.float64) / RE_TAU
    rows = []
    for i in range(1, n - 1):
        hm = yp_grid[i] - yp_grid[i - 1]
        hp = yp_grid[i + 1] - yp_grid[i]
        hc = 0.5 * (hm + hp)
        cm = 0.5 * (nu[i - 1] + nu[i]) / hm          # the SAME face average the solver uses
        cp = 0.5 * (nu[i] + nu[i + 1]) / hp
        rows.append((i, cm / hc, cp / hc))
    idx_i = torch.tensor([r[0] for r in rows])
    A = A.index_put((idx_i, idx_i - 1), torch.stack([r[1] for r in rows]))
    A = A.index_put((idx_i, idx_i + 1), torch.stack([r[2] for r in rows]))
    A = A.index_put((idx_i, idx_i), -torch.stack([r[1] + r[2] for r in rows]))
    A = A.index_put((torch.tensor([0, n - 1]), torch.tensor([0, n - 1])),
                    torch.ones(2, dtype=torch.float64))
    b = b.index_put((torch.tensor([0, n - 1]),), torch.zeros(2, dtype=torch.float64))
    return torch.linalg.solve(A, b)


def main():
    yp_ref, U_ref, nut_ref = van_driest()
    y = grid()
    ypg = np.minimum(y, 2.0 - y) * RE_TAU
    yg = y * RE_TAU
    U_target = torch.tensor(np.interp(ypg, yp_ref, U_ref))
    nut_true = np.interp(ypg, yp_ref, nut_ref)

    # sanity: the true nu_t must reproduce the target through the SAME discrete operator
    U_check = solve(torch.tensor(nut_true), yg)
    print(f"  target from van Driest, {len(y)} points; the true nu_t reproduces it to "
          f"{float((U_check - U_target).abs().max()):.4f} in U+ (discretisation only)\n")

    # PARAMETERISATION AND OPTIMISER BOTH MATTER HERE, and the first attempt got both wrong.
    # nu_t = softplus(theta) from theta = 0 starts the profile at 0.69 everywhere and has to
    # climb to a peak of 27.6; with Adam at lr 0.15 that took 600 iterations to reach a 2.0
    # error in U+ and a 64% error in nu_t. Scaling by the expected peak puts the initial guess
    # in range, and L-BFGS suits a smooth deterministic problem in 97 parameters far better
    # than a stochastic optimiser designed for minibatches.
    SCALE = 30.0
    theta = torch.full((len(y),), -1.0, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=400, history_size=50,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nu_t = SCALE * torch.nn.functional.softplus(theta)
        loss = ((solve(nu_t, yg) - U_target) ** 2).sum()
        loss.backward()
        return loss

    print(f"  {'pass':>6}{'loss':>13}{'max |dU+|':>12}{'nu_t err 5<y+<108':>20}")
    for p_ in range(6):
        opt.step(closure)
        with torch.no_grad():
            nt = (SCALE * torch.nn.functional.softplus(theta)).numpy()
            U = solve(torch.tensor(nt), yg)
            good = (ypg > 5) & (ypg < 0.6 * RE_TAU)
            print(f"  {p_:>6}{float(((U-U_target)**2).sum()):>13.3e}"
                  f"{float((U - U_target).abs().max()):>12.6f}"
                  f"{np.abs(nt[good] - nut_true[good]).max():>20.4f}")

    with torch.no_grad():
        nt = (SCALE * torch.nn.functional.softplus(theta)).numpy()
        U = solve(torch.tensor(nt), yg).numpy()
    core = (ypg > 5) & (ypg < 0.6 * RE_TAU)
    near = ypg >= 0.85 * RE_TAU
    print(f"\n  velocity matched to {np.abs(U - U_target.numpy()).max():.5f} in U+")
    print(f"  nu_t recovered, 5 < y+ < {0.6*RE_TAU:.0f}: max error {np.abs(nt[core]-nut_true[core]).max():.3f} "
          f"of a peak {nut_true.max():.1f}  ({100*np.abs(nt[core]-nut_true[core]).max()/nut_true.max():.1f}%)")
    print(f"  nu_t recovered, y+ > {0.85*RE_TAU:.0f} (centreline): max error "
          f"{np.abs(nt[near]-nut_true[near]).max():.3f}  <-- UNIDENTIFIABLE, dU/dy -> 0 there")
    np.savez("results/nn_eddy_viscosity.npz", y=y, yp=ypg, U=U,
             U_target=U_target.numpy(), nut=nt, nut_true=nut_true)
    print("  saved results/nn_eddy_viscosity.npz")


if __name__ == "__main__":
    main()
