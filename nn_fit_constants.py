"""Fit the TWO van Driest constants, kappa and A+, from a target velocity profile.

The middle rung between field inversion and a network. `nn_eddy_viscosity.py` fits 97 free
values -- one per grid point -- which is exactly determined, cannot fail, and generalises
nowhere. Here the same loss is minimised over TWO numbers:

    nu_t+ = (kappa y+ D)^2 |dU+/dy+|,   D = 1 - exp(-y+/A+)

97 residuals, 2 parameters: overdetermined by a factor of 48, so a good fit is evidence about
the model rather than a consequence of counting. And the recovered constants transfer to any
Re_tau, which a free vector cannot do.

THE LOSS IS ON THE VELOCITY, WEIGHTED BY CELL WIDTH.

    L = sum_i w_i (U_i - U_target_i)^2 / sum_i w_i U_target_i^2,    w_i = cell width

An unweighted node sum is not a property of the profile. This grid is clustered, so 40 of its
97 points lie inside y+ = 30: a node sum weights the sublayer several times more heavily per
unit of channel than the core, and the fit ends up tuned by where the points were put. Weighting
by cell width makes the loss an integral norm and the answer grid-independent. Bulk velocity is
reported alongside as an independent scalar, not folded in.

THE CLOSURE IS COUPLED, NOT FROZEN. nu_t depends on dU/dy, which depends on the solution, so the
forward problem is nonlinear and is solved by Picard iteration -- unrolled, so the gradient
chains through every sweep. Computing nu_t from the TARGET's gradient instead would make the
problem linear and easier, and would also be a different model: it would never be usable in a
flow where the answer is not already known.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
RE_TAU = 180.0
KAPPA_TRUE, APLUS_TRUE = 0.41, 26.0


def grid(dy0_plus=1.0, ratio=1.05):
    ys, dy = [0.0], dy0_plus / RE_TAU
    while ys[-1] < 1.0:
        ys.append(ys[-1] + dy)
        dy *= ratio
    ys = np.array(ys) / ys[-1]
    return np.concatenate([ys[:-1], 2.0 - ys[::-1]]) * RE_TAU     # in wall units


def reference(n=400001):
    yp = np.linspace(0.0, RE_TAU, n)
    D = 1.0 - np.exp(-yp / APLUS_TRUE)
    lm = KAPPA_TRUE * yp * D
    tau = 1.0 - yp / RE_TAU
    dU = 2.0 * tau / (1.0 + np.sqrt(1.0 + 4.0 * lm ** 2 * tau))
    return yp, np.concatenate([[0.0], np.cumsum(0.5 * (dU[1:] + dU[:-1]) * np.diff(yp))])


def solve_coupled(kappa, aplus, yg, sweeps=60):
    """Picard: U -> nu_t(U) -> U, unrolled so the gradient chains through every sweep."""
    n = len(yg)
    ywall = torch.as_tensor(np.minimum(yg, 2 * RE_TAU - yg))
    tau = 1.0 - ywall / RE_TAU
    lm = kappa * ywall * (1.0 - torch.exp(-ywall / aplus))
    b = -torch.ones(n, dtype=torch.float64) / RE_TAU
    b = b.index_put((torch.tensor([0, n - 1]),), torch.zeros(2, dtype=torch.float64))
    hm = torch.as_tensor(yg[1:-1] - yg[:-2])
    hp = torch.as_tensor(yg[2:] - yg[1:-1])
    hc = 0.5 * (hm + hp)
    idx = torch.arange(1, n - 1)
    U = torch.zeros(n, dtype=torch.float64)
    for _ in range(sweeps):
        dU = torch.zeros(n, dtype=torch.float64)
        dU = dU.index_put((idx,), (U[2:] - U[:-2]) / (hm + hp))
        nu = 1.0 + lm ** 2 * dU.abs()
        cm = 0.5 * (nu[:-2] + nu[1:-1]) / hm
        cp = 0.5 * (nu[1:-1] + nu[2:]) / hp
        A = torch.zeros(n, n, dtype=torch.float64)
        A = A.index_put((idx, idx - 1), cm / hc)
        A = A.index_put((idx, idx + 1), cp / hc)
        A = A.index_put((idx, idx), -(cm + cp) / hc)
        A = A.index_put((torch.tensor([0, n - 1]), torch.tensor([0, n - 1])),
                        torch.ones(2, dtype=torch.float64))
        U = torch.linalg.solve(A, b)
    return U, lm ** 2 * dU.abs(), tau


def main():
    yg = grid()
    yp_ref, U_ref = reference()
    ywall = np.minimum(yg, 2 * RE_TAU - yg)
    U_target = torch.tensor(np.interp(ywall, yp_ref, U_ref))
    w = torch.tensor(np.gradient(yg))                       # cell width, the quadrature weight
    denom = float((w * U_target ** 2).sum())

    def loss_of(kappa, aplus):
        U, _, _ = solve_coupled(kappa, aplus, yg)
        return (w * (U - U_target) ** 2).sum() / denom, U

    L0, U0 = loss_of(torch.tensor(KAPPA_TRUE), torch.tensor(APLUS_TRUE))
    print(f"  ORACLE first: the true constants give L = {float(L0):.3e}, "
          f"max|dU+| = {float((U0-U_target).abs().max()):.4f}")
    print(f"  (not zero -- Picard on a discrete grid is not the continuous integration the\n"
          f"   target came from, so this is the floor no fit can beat)\n")

    p = torch.tensor([np.log(0.25), np.log(15.0)], requires_grad=True)   # deliberately off
    opt = torch.optim.LBFGS([p], lr=0.3, max_iter=120, history_size=40,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        k, a = torch.exp(p[0]), torch.exp(p[1])
        L, _ = loss_of(k, a)
        L.backward()
        return L

    print(f"  {'pass':>5}{'kappa':>10}{'A+':>9}{'loss':>13}{'max |dU+|':>12}")
    for it in range(4):
        opt.step(closure)
        with torch.no_grad():
            k, a = float(torch.exp(p[0])), float(torch.exp(p[1]))
            L, U = loss_of(torch.tensor(k), torch.tensor(a))
            print(f"  {it:>5}{k:>10.5f}{a:>9.4f}{float(L):>13.3e}"
                  f"{float((U-U_target).abs().max()):>12.5f}")

    k, a = float(torch.exp(p[0])), float(torch.exp(p[1]))
    L, U = loss_of(torch.tensor(k), torch.tensor(a))
    Ub = float((w * U).sum() / w.sum())
    Ub_t = float((w * U_target).sum() / w.sum())
    print(f"\n  recovered kappa = {k:.5f}   (true {KAPPA_TRUE})   error "
          f"{100*abs(k-KAPPA_TRUE)/KAPPA_TRUE:+.2f}%")
    print(f"  recovered A+    = {a:.4f}   (true {APLUS_TRUE})   error "
          f"{100*abs(a-APLUS_TRUE)/APLUS_TRUE:+.2f}%")
    print(f"  bulk velocity   U_b+ = {Ub:.4f} vs target {Ub_t:.4f} "
          f"({100*abs(Ub-Ub_t)/Ub_t:.3f}%)   [reported, not in the loss]")
    print(f"  final loss {float(L):.3e} against the oracle floor {float(L0):.3e}")
    np.savez("results/fit_constants.npz", yp=ywall, U=U.detach().numpy(),
             U_target=U_target.numpy(), kappa=k, aplus=a)

    # ------------------------------------------------------------------ the control
    # THE FIT BEAT THE ORACLE, which is the tell. The target came from a fine CONTINUOUS
    # integration and the forward model is a 97-point Picard solve; those are not the same
    # function, so the optimiser is free to move kappa and A+ to absorb the DISCRETISATION
    # error, and it does -- 68x lower loss than the true constants can achieve, at the price of
    # 10% and 13% errors in the physics.
    #
    # The control removes that freedom: regenerate the target with the SAME discrete solver at
    # the true constants, so the only thing separating target from model is the two numbers.
    # This is a deliberate inverse crime -- normally something to avoid, because it flatters the
    # method -- and it is exactly right here, because the question is whether the RECOVERY works,
    # not how big the model error is.
    print("\n  CONTROL: target regenerated by the same discrete solver at the true constants")
    with torch.no_grad():
        U_disc, _, _ = solve_coupled(torch.tensor(KAPPA_TRUE), torch.tensor(APLUS_TRUE), yg)
    denom2 = float((w * U_disc ** 2).sum())

    p2 = torch.tensor([np.log(0.25), np.log(15.0)], requires_grad=True)
    opt2 = torch.optim.LBFGS([p2], lr=0.3, max_iter=120, history_size=40,
                             line_search_fn="strong_wolfe")

    def closure2():
        opt2.zero_grad()
        U2, _, _ = solve_coupled(torch.exp(p2[0]), torch.exp(p2[1]), yg)
        L2 = (w * (U2 - U_disc) ** 2).sum() / denom2
        L2.backward()
        return L2

    for _ in range(4):
        opt2.step(closure2)
    with torch.no_grad():
        k2, a2 = float(torch.exp(p2[0])), float(torch.exp(p2[1]))
        U2, _, _ = solve_coupled(torch.tensor(k2), torch.tensor(a2), yg)
        L2 = float((w * (U2 - U_disc) ** 2).sum() / denom2)
    print(f"    kappa = {k2:.6f}  (true {KAPPA_TRUE},  error {100*abs(k2-KAPPA_TRUE)/KAPPA_TRUE:.4f}%)")
    print(f"    A+    = {a2:.5f}  (true {APLUS_TRUE},  error {100*abs(a2-APLUS_TRUE)/APLUS_TRUE:.4f}%)")
    print(f"    loss  = {L2:.3e}")
    print(f"\n  So the machinery recovers the constants when they are recoverable. The 10% and")
    print(f"  13% above are the price of fitting physics against a target the discretisation")
    print(f"  cannot reproduce -- the constants absorbed numerical error, and the loss got")
    print(f"  BETTER while the physics got worse.")


if __name__ == "__main__":
    main()
