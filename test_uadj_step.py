"""Gates U4-U5 of reference/unstructured_adjoint_plan.md on the fast-suite cases.

A8   torch step == production PISO.step, field for field, after 1 and 10 steps
A10  adjoint gradient == central finite difference through 1-3 steps, for the state, the body
     force, the boundary values and the viscosity (branch decisions and Poisson sweep counts
     replayed so both sides of each difference take the production branch)
A12  <w, J v> == <J^T w, v> through one step (forward-over-reverse vs reverse)

    python test_uadj_step.py            all cases
    python test_uadj_step.py cavity_tris
    python test_uadj_step.py --production cylinder_butterfly cylinder_tris   (production meshes)
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.uadj_cases import CASES, PRODUCTION
CASES = {**CASES}
from src.uadj_step import TorchUPISO, STATE_KEYS

torch.set_default_dtype(torch.float64)
FIELDS = ("u", "v", "p", "Ff")
FAILS = []
U_SCALE = [1.0]
SWEEPS = {}


def check(name, val, tol):
    ok = val <= tol
    print(f"    {'PASS' if ok else 'FAIL'}  {name:58s} {val:.2e}  (<= {tol:.0e})")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def gate_a8(case):
    s = CASES[case]()
    T = TorchUPISO(s)
    st = T.state_from_solver()
    worst1 = worst10 = 0.0
    with torch.no_grad():
        for k in range(10):
            st = T.step(st)
            s.step()
            e = max(rel(st[f].numpy(), getattr(s, f)) for f in FIELDS)
            dv = abs(float(T.tm.divergence(st["Ff"]).abs().max()) - float(np.abs(
                __import__("src.uops", fromlist=["divergence"]).divergence(s.m, s.Ff)).max()))
            if k == 0:
                worst1 = e
            worst10 = max(worst10, e)
    check(f"A8 {case}: u,v,p,F after 1 step", worst1, 1e-10)
    check(f"A8 {case}: u,v,p,F over 10 steps", worst10, 1e-9)
    check(f"A8 {case}: max|div F| torch vs production", dv, 1e-14)


def _loss_weights(T, seed):
    g = torch.Generator().manual_seed(seed)
    return {f: torch.randn(n, generator=g) for f, n in
            (("u", T.tm.ncell), ("v", T.tm.ncell), ("p", T.tm.ncell), ("Ff", T.tm.nface))}


def _rollout(T, st, nsteps):
    for _ in range(nsteps):
        st = T.step(st)
    return st


def gate_a10(case, nsteps):
    s = CASES[case]()
    T = TorchUPISO(s)
    W = _loss_weights(T, 11)

    def L(st):
        out = _rollout(T, st, nsteps)
        return sum((W[f] * out[f]).sum() for f in W)

    base = T.state_from_solver()
    U_SCALE[0] = float(base["u"].abs().max())
    # inputs to differentiate: the state leaves, body force, and nu enters through Lu/Lv values
    leaves = {k: base[k].clone().requires_grad_(True) for k in STATE_KEYS if base[k] is not None}
    fx = T.fx.clone().requires_grad_(True); T.fx = fx
    nu = T.nu_t.clone().requires_grad_(True); T.nu_t = nu
    T.record()
    loss = L({**base, **leaves})
    grads = torch.autograd.grad(loss, list(leaves.values()) + [fx, nu], allow_unused=True)
    g = dict(zip(list(leaves.keys()) + ["fx", "nu"], grads))
    params = {"fx": fx, "nu": nu}

    gen = torch.Generator().manual_seed(7)
    worst = {}
    for k in list(leaves.keys()) + ["fx", "nu"]:
        x0 = (leaves[k] if k not in params else params[k]).detach()
        d = torch.randn(x0.shape, generator=gen)
        # Step relative to the FIELD scale, not the input's own (a wall value or a zero body force
        # is identically zero). The FD error is truncation ~h^2 plus round-off ~eps/h; a sweep
        # over h and the best of it is the FD's own floor. A wrong adjoint shows at EVERY h.
        scale = max(float(x0.abs().max()), U_SCALE[0] if k != "fx" else U_SCALE[0] * T.nu)
        if k == "nu":
            d = torch.ones(())
        ad = float((g[k] * d).sum()) if g[k] is not None else 0.0
        errs = []
        for hr in (1e-3, 1e-4, 1e-5, 1e-6):
            h = hr * scale
            vals = []
            for sgn in (+1, -1):
                T.replay()
                st = {**base, **{kk: vv.detach() for kk, vv in leaves.items()}}
                if k == "fx":
                    T.fx = x0 + sgn * h * d
                elif k == "nu":
                    T.nu_t = x0 + sgn * h * d
                else:
                    st[k] = x0 + sgn * h * d
                with torch.no_grad():
                    vals.append(float(L(st)))
                T.fx, T.nu_t = fx.detach(), nu.detach()
            fd = (vals[0] - vals[1]) / (2 * h)
            errs.append(abs(fd - ad) / max(abs(fd), abs(ad), 1e-30))
        worst[k] = min(errs)
        SWEEPS[(case, nsteps, k)] = errs
    T.live()
    wk = max(worst, key=worst.get)
    check(f"A10 {case}: FD vs adjoint, {nsteps} step(s), worst input ({wk})", worst[wk], 1e-6)
    return worst


PATHS = ("conv_flux", "aP", "aC", "rc_history", "bdf2_history", "deferred_momentum", "deferred_poisson")


def gate_a11(case, nsteps=2):
    """Cut each gradient path in turn; the gradient must CHANGE (the path is live), while the
    forward values stay identical (the cut is gradient-only)."""
    s = CASES[case]()
    T = TorchUPISO(s)
    W = _loss_weights(T, 5)
    base = T.state_from_solver()
    keys = [k for k in STATE_KEYS if base[k] is not None]

    def grad_vec():
        leaves = {k: base[k].clone().requires_grad_(True) for k in keys}
        nu = T.nu_t.clone().requires_grad_(True); T.nu_t = nu
        out = _rollout(T, {**base, **leaves}, nsteps)
        L = sum((W[f] * out[f]).sum() for f in W)
        ins = list(leaves.values()) + [nu]
        g = torch.autograd.grad(L, ins, allow_unused=True)
        T.nu_t = nu.detach()
        return float(L.detach()), torch.cat([(x if x is not None else torch.zeros_like(i)).reshape(-1)
                                             for x, i in zip(g, ins)])
    L0, g0 = grad_vec()
    orthogonal = float(T.tm.Tf.abs().max()) < 1e-12 * float(T.tm.normal.abs().max())
    for path in PATHS:
        if path == "deferred_poisson" and orthogonal:
            # T_f == 0 on every face: the non-orthogonal term does not exist on this mesh, so
            # there is no path to exercise (it is exercised on cavity_tris and channel_open)
            print(f"    N/A   A11 {case}: cut {path:18s} T_f == 0 on every face (orthogonal mesh)")
            continue
        T.detach = {path}
        L1, g1 = grad_vec()
        T.detach = set()
        ch = float((g1 - g0).norm() / g0.norm())
        ok = ch >= 1e-6 and L1 == L0
        print(f"    {'PASS' if ok else 'FAIL'}  A11 {case}: cut {path:18s} gradient change {ch:.2e} (>= 1e-06), forward identical: {L1 == L0}")
        if not ok:
            FAILS.append(f"A11 {case} {path}")


def gate_a12(case):
    """<w, J v> (forward mode) == <J^T w, v> (reverse mode) through one full step."""
    import torch.autograd.forward_ad as fwAD
    s = CASES[case]()
    T = TorchUPISO(s)
    base = T.state_from_solver()
    keys = ("u", "v", "p", "Ff", "Ff_prev", "Ff_old", "Fbar_old", "u_old", "v_old")
    outk = ("u", "v", "p", "Ff")
    gen = torch.Generator().manual_seed(3)
    vdir = {k: torch.randn(base[k].shape, generator=gen) for k in keys}
    wdir = {k: torch.randn(base[k].shape, generator=gen) for k in outk}
    x = {k: base[k].clone().requires_grad_(True) for k in keys}
    T.record()
    out = T.step({**base, **x})
    JTw = torch.autograd.grad(sum((wdir[k] * out[k]).sum() for k in outk), list(x.values()))
    rev = float(sum((jt * vdir[k]).sum() for jt, k in zip(JTw, keys)))
    T.replay()
    with fwAD.dual_level():
        st = {**base, **{k: fwAD.make_dual(base[k], vdir[k]) for k in keys}}
        o = T.step(st)
        fwd = float(sum((wdir[k] * fwAD.unpack_dual(o[k]).tangent).sum() for k in outk))
    T.live()
    check(f"A12 {case}: <w, J v> vs <J^T w, v>, one step", abs(fwd - rev) / abs(rev), 1e-10)


if __name__ == "__main__":
    if "--production" in sys.argv:
        CASES.update(PRODUCTION)      # the T9 cylinder meshes: slower, run before a merge
    cases = [a for a in sys.argv[1:] if not a.startswith("-")] or list(CASES)
    t0 = time.time()
    for c in cases:
        print(f"  {c}")
        gate_a8(c)
        for n in (1, 2, 3):
            gate_a10(c, n)
        gate_a12(c)
        gate_a11(c)
    if "-v" in sys.argv or FAILS:
        for key, e in SWEEPS.items():
            print("   ", key, " ".join(f"{x:.1e}" for x in e))
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t0:.0f}s" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
