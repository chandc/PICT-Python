"""Gates A13-A19 of reference/unstructured_adjoint_plan.md: the rest of U5, U6 and U7.

A13  gradients in a recirculation bubble (Re 40 cylinder, faces with F ~ 0), switch counts reported
A14  a periodic seam: source on one side, loss on the other
A15  every boundary type: inflow, no-slip wall, symmetry (freestream), pressure outlet; and an
     exact zero where a value is not used (u on the Neumann freestream faces)
A16  wall forces: torch == production formula; dC_D/d(state) == FD
U6   dC_D/da and dC_L/da for the cylinder's +-90 degree jets (opposing and symmetric), jet ->
     wake transport, dC_L/domega for rotation, and HydroGym's NACA jets: production slots
     reproduced exactly, dC_L/da_j == FD for each jet
A17  replay == tape over a 5-step control window
A18  replay memory flat in the horizon (saved tensors + live LU factors), tape growing
A19  cost of the backward relative to the forward

    python test_uadj_control.py            (about 3 minutes)
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.uadj_cases import cylinder, channel_periodic
from src.uadj_step import TorchUPISO
from src.uadj_control import WallForces, SlotJets, Rotation
from src.uadj_replay import replay_grad, tape_grad, SavedBytes
from src.uadj_solve import LUFactor

torch.set_default_dtype(torch.float64)
FAILS = []


def check(name, val, tol, lower=False):
    ok = (val >= tol) if lower else (val <= tol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:64s} {val:.2e}  ({'>=' if lower else '<='} {tol:.0e})", flush=True)
    if not ok:
        FAILS.append(name)


def fd_vs_ad(T, f, x0, d, scale, hs=(1e-3, 1e-4, 1e-5, 1e-6)):
    """Best relative FD-vs-adjoint error over a step sweep, on the recorded branch."""
    x = x0.detach().clone().requires_grad_(True)
    T.record()
    y = f(x)
    g = torch.autograd.grad(y, x, allow_unused=True)[0]
    ad = 0.0 if g is None else float((g * d).sum())
    errs = []
    for hr in hs:
        h = hr * scale
        vals = []
        for sg in (1, -1):
            T.replay()
            with torch.no_grad():
                vals.append(float(f(x0.detach() + sg * h * d)))
        fd = (vals[0] - vals[1]) / (2 * h)
        errs.append(abs(fd - ad) / max(abs(fd), abs(ad), 1e-300))
    T.live()
    return min(errs), ad, g


def roll(T, st, n):
    for _ in range(n):
        st = T.step(st)
    return st


def main():
    t00 = time.time()
    # ------------------------------------------------------------------ the cylinder, Re 100
    print("  cylinder (butterfly quads), Re 100", flush=True)
    s = cylinder(nsteps=20)
    T = TorchUPISO(s)
    base = T.state_from_solver()
    m = s.m
    W = WallForces(T, s.wall_faces)
    cd_t, cl_t = W(base)
    cd_n, cl_n = W.numpy(s)
    check("A16 forces: torch vs production formula", max(abs(float(cd_t) - cd_n), abs(float(cl_t) - cl_n)) / abs(cd_n), 1e-14)
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    for key in ("u", "v", "p"):
        d = torch.randn(base[key].shape, generator=gen)
        e, _, _ = fd_vs_ad(T, lambda x: W({**base, key: x})[0], base[key], d, float(base["u"].abs().max()))
        worst = max(worst, e)
    check("A16 dC_D/d(u, v, p) vs FD", worst, 1e-8)

    # A15 -- boundary types
    bt = m.btag[m.bfaces]
    inv = {v: k for k, v in m.names.items()}
    Wl = torch.randn(m.ncell, generator=gen)

    def L2(st):
        out = roll(T, st, 2)
        return (Wl * out["u"]).sum() + (Wl * out["v"]).sum() + W(out)[0]
    rows = (("inflow u (Dirichlet)", "ub", "Inlet"), ("wall u (no-slip)", "ub", "Cylinder"),
            ("wall v (no-slip)", "vb", "Cylinder"), ("freestream v (symmetry, Dirichlet)", "vb", "Freestream"),
            ("outlet p (Dirichlet)", "pb", "Outlet"))
    for label, key, tagname in rows:
        sel = torch.as_tensor(bt == inv[tagname])
        d = torch.randn(m.nbface, generator=gen) * sel
        e, _, _ = fd_vs_ad(T, lambda x: L2({**base, key: x}), base[key], d, 1.0)
        check(f"A15 boundary {label}: FD vs adjoint, 2 steps", e, 1e-6)
    x = base["ub"].clone().requires_grad_(True)
    T.record(); g = torch.autograd.grad(L2({**base, "ub": x}), x)[0]; T.live()
    neu_fs = torch.as_tensor((bt == inv["Freestream"]) | (bt == inv["Outlet"]))
    check("A15 dL/d(u value) on Neumann faces (freestream, outlet) is exactly 0",
          float(g[neu_fs].abs().max()), 0.0)

    # U6 -- jets and rotation on the cylinder
    jets = SlotJets.cylinder(T, s.wall_faces, (90.0, -90.0), 10.0, vmax=1.0)
    print(f"    (cylinder jets: {[len(b) for b in jets.bidx]} faces per slot)")
    wake = torch.as_tensor((m.centroid[:, 0] > 2.0) & (m.centroid[:, 0] < 4.0) & (np.abs(m.centroid[:, 1]) < 1.0))
    for label, pat in (("opposing (+a, -a)", torch.tensor([1.0, -1.0])), ("symmetric (+a, +a)", torch.tensor([1.0, 1.0]))):
        for oname, obj in (("C_D", lambda o: W(o)[0]), ("C_L", lambda o: W(o)[1]),
                           ("wake v", lambda o: o["v"][wake].sum())):
            f = lambda a: obj(roll(T, jets.apply(base, a * pat), 2))
            e, ad, _ = fd_vs_ad(T, f, torch.tensor(0.05), torch.tensor(1.0), 1.0)
            check(f"U6 jets {label}: d{oname}/da, 2 steps", e, 1e-6)
    rot = Rotation(T, s.wall_faces)
    e, dcl, _ = fd_vs_ad(T, lambda w: W(roll(T, rot.apply(base, w), 2))[1], torch.tensor(0.0), torch.tensor(1.0), 1.0)
    check("U6 rotation: dC_L/domega, 2 steps", e, 1e-6)
    print(f"    (dC_L/domega after 2 steps = {dcl:+.4e})")

    # A17 -- replay == tape, 5 control steps, per-step drag objective, opposing jets
    pat = torch.tensor([1.0, -1.0])
    apply = lambda st, a: jets.apply(st, a * pat)
    sl = lambda st, k: W(st)[0]
    acts = [0.02 * (k + 1) for k in range(5)]
    lr, gr = replay_grad(T, base, acts, apply, step_loss=sl)
    lt, gt = tape_grad(T, base, acts, apply, step_loss=sl)
    err = max(abs(float(a) - float(b)) / max(abs(float(b)), 1e-30) for a, b in zip(gr, gt))
    check("A17 replay == tape, 5 control steps: loss", abs(lr - lt) / abs(lt), 1e-12)
    check("A17 replay == tape, 5 control steps: dL/da_k (worst)", err, 1e-9)

    # A18 / A19 -- memory and cost, tape vs replay
    print("    horizon  tape saved+LU MB   replay peak MB   fwd s/step  tape bwd/fwd  replay total/fwd")
    mem = {}
    for H in (3, 10, 30):
        acts = [0.01] * H
        LUFactor.reset_peak(); lu0 = LUFactor.live_bytes
        t0 = time.time()
        with SavedBytes() as sb:
            a_leaves = [torch.tensor(a, requires_grad=True) for a in acts]
            st = dict(base); loss = 0
            for k, a in enumerate(a_leaves):
                st = T.step(apply(st, a)); loss = loss + sl(st, k)
        tf = time.time() - t0
        tape_mb = (sb.total + LUFactor.peak_bytes - lu0) / 2 ** 20
        t1 = time.time(); torch.autograd.grad(loss, a_leaves); tb = time.time() - t1
        del loss, st, a_leaves
        LUFactor.reset_peak(); lu0 = LUFactor.live_bytes
        t2 = time.time()
        with SavedBytes() as pk:
            replay_grad(T, base, acts, apply, step_loss=sl)
        tr = time.time() - t2
        # saved bytes accumulate over the H rebuilt step graphs, each freed before the next is
        # built (the forward sweep is no_grad and saves nothing): one step's share is the peak
        rep_mb = (pk.total / H + LUFactor.peak_bytes - lu0) / 2 ** 20
        mem[H] = (tape_mb, rep_mb)
        print(f"    {H:7d}  {tape_mb:16.1f}   {rep_mb:14.1f}   {tf / H:10.3f}  {tb / tf:12.2f}  {tr / tf:16.2f}", flush=True)
    check("A18 replay peak memory, H = 30 vs H = 3 (ratio - 1)", abs(mem[30][1] / mem[3][1] - 1.0), 0.10)
    check("A18 tape memory grows with H (ratio H30 / H3)", mem[30][0] / mem[3][0], 5.0, lower=True)

    # ------------------------------------------------------------------ A13: Re 40 bubble
    print("  cylinder, Re 40, recirculation bubble", flush=True)
    s40 = cylinder(Re=40.0, dt=0.02, nsteps=1500)
    T40 = TorchUPISO(s40)
    b40 = T40.state_from_solver()
    F = b40["Ff"].abs(); fmax = float(F.max())
    n6, n3 = int((F < 1e-6 * fmax).sum()), int((F < 1e-3 * fmax).sum())
    c = s40.m.centroid
    bubble = torch.as_tensor((c[:, 0] > 0.5) & (c[:, 0] < 2.5) & (np.abs(c[:, 1]) < 0.5))
    print(f"    {int(bubble.sum())} bubble cells; faces with |F| < 1e-6 max: {n6}, < 1e-3 max: {n3}; "
          f"min u in the bubble {float(b40['u'][bubble].min()):+.3f}")
    d = torch.randn(s40.m.ncell, generator=gen) * bubble
    f = lambda x: roll(T40, {**b40, "u": x}, 2)["u"][bubble].sum()
    e, _, _ = fd_vs_ad(T40, f, b40["u"], d, float(b40["u"].abs().max()))
    check("A13 bubble: FD vs adjoint, 2 steps, on the recorded branch", e, 1e-6)
    # how many upwind decisions flip under the largest FD probe, when NOT replayed
    T40.record(); roll(T40, dict(b40), 2); rec = [m_ for n_, m_ in T40.masks.log if n_ == "upwind"]
    T40.record(); roll(T40, {**b40, "u": b40["u"] + 1e-3 * float(b40["u"].abs().max()) * d}, 2)
    new = [m_ for n_, m_ in T40.masks.log if n_ == "upwind"]; T40.live()
    flips = sum(int((a != b).sum()) for a, b in zip(rec, new))
    print(f"    upwind decisions that flip under a 1e-3 probe (live, not replayed): {flips}")

    # ------------------------------------------------------------------ A14: periodic seam
    # Seam invariance: the same periodic flow on the same box, with the seam at x = 0 and at x = pi.
    # Physics has no seam, so the gradient FIELD must be identical cell for cell once the cells are
    # matched modulo the period; any error in how the seam is adjointed breaks that at O(1).
    # (The first form of this gate asked for a large seam/mid-domain gradient ratio. That was
    # mis-posed: the pressure solve couples every cell in one step, so the ratio says nothing about
    # the seam. Recorded in reference/unstructured_adjoint.md, finding 9.)
    print("  periodic channel seam", flush=True)
    from src.umesh import rect_mesh
    from src.uops import DIRICHLET, NEUMANN
    from src.upiso import PISO, BC
    LX, nx, ny, nu = 2 * np.pi, 12, 8, 0.01

    def seam_grad(x0):
        mm = rect_mesh(nx, ny, x0, x0 + LX, -1.0, 1.0, cells="quad")
        mm.make_periodic(1, 2, (LX, 0.0))
        kd = np.full(mm.nbface, DIRICHLET)
        ss = PISO(mm, nu=nu, dt=0.02, bc_u=BC(mm, kd), bc_v=BC(mm, kd.copy()), bc_p=BC(mm, np.full(mm.nbface, NEUMANN)),
                  n_corr=2, n_nonorth=2, scheme="central", body_force=(np.full(mm.ncell, 2 * nu), np.zeros(mm.ncell)))
        xx, yy = mm.centroid.T
        ss.u[:] = 1 - yy ** 2 + 0.05 * np.sin(xx) * (1 - yy ** 2)
        ss.v[:] = 0.05 * np.cos(xx) * (1 - yy ** 2) ** 2
        ss.init_flux()
        for _ in range(3):
            ss.step()
        Tq = TorchUPISO(ss); bq = Tq.state_from_solver()
        probe = torch.as_tensor(np.exp(-((np.mod(xx, LX) - 1.0) ** 2 + yy ** 2)))   # a fixed physical weight
        leaves = {k: bq[k].clone().requires_grad_(True) for k in ("u", "v", "p")}
        out = roll(Tq, {**bq, **leaves}, 2)
        L = (probe * out["u"]).sum() + (probe * out["v"]).sum()
        g = torch.autograd.grad(L, list(leaves.values()))
        key = np.round(np.mod(xx, LX), 9) * 1e3 + np.round(yy, 9)            # physical position of each cell
        order = np.argsort(key)
        return [gi.numpy()[order] for gi in g], float(L)
    ga, La = seam_grad(0.0)
    gb, Lb = seam_grad(-np.pi)
    check("A14 seam: loss identical with the seam moved by half a period", abs(La - Lb) / abs(La), 1e-12)
    check("A14 seam: gradient field identical cell for cell (u, v, p)",
          max(float(np.abs(a - b).max() / np.abs(a).max()) for a, b in zip(ga, gb)), 1e-10)

    # ------------------------------------------------------------------ NACA jets (U8's target)
    print("  NACA0012 alpha 40, HydroGym's three jets", flush=True)
    from naca_env import NACAJetEnv
    env = NACAJetEnv(seed=0)
    env.reset(seed=0)
    Tn = TorchUPISO(env.s)
    bn = Tn.state_from_solver()
    nj = SlotJets.from_jetset(Tn, env.jets)
    a = np.array([0.3, -0.5, 0.2])
    env.jets.set_target(a); env.jets.a_prev = a.copy(); env.jets.ramp(1.0, env.s.bc_u, env.s.bc_v)
    st = nj.apply(bn, torch.tensor(a))
    par = max(float(np.abs(st["ub"].numpy() - env.s.bc_u.value).max()), float(np.abs(st["vb"].numpy() - env.s.bc_v.value).max()))
    check("U6 NACA jets: torch slot values == production JetSet.ramp", par, 1e-15)
    wall = env.m.bfaces[env.m.btag[env.m.bfaces] == env.T_W]
    Wn = WallForces(Tn, wall)
    check("U6 NACA forces: torch vs env.forces()", abs(float(Wn(Tn.state_from_solver())[1]) - env.forces()[1]), 1e-13)
    for j in range(3):
        dj = torch.zeros(3); dj[j] = 1.0
        f = lambda aa: Wn(roll(Tn, nj.apply(bn, aa), 2))[1]
        e, ad, _ = fd_vs_ad(Tn, f, torch.tensor(a), dj, 1.0)
        check(f"U6 NACA jet {j + 1}: dC_L/da_{j + 1}, 2 steps", e, 1e-6)
        print(f"      dC_L/da_{j + 1} = {ad:+.4e}")
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t00:.0f}s" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
