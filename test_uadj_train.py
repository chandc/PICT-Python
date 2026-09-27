"""Training bridge for the unstructured adjoint (record: "Training" open item before U8).
Gate A20: the policy-in-the-loop replay gradient equals the whole-window tape gradient, with the action
computed inside each replayed step from that step's observation (pressure at wake probes), on the coarse
butterfly cylinder at Re 100 with the +-90 degree jets. Also a finite-difference check of dL/dtheta for one
parameter and the replay's flat memory in the horizon.
    python test_uadj_train.py"""
import sys, os, time, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.uadj_cases import cylinder
from src.uadj_step import TorchUPISO
from src.uadj_control import WallForces, SlotJets
from src.uadj_replay import replay_policy_grad, tape_policy_grad, SavedBytes
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
FAILS = []
def check(name, val, tol, lower=False):
    ok = (val >= tol) if lower else (val <= tol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:64s} {val:.2e}  ({'>=' if lower else '<='} {tol:.0e})", flush=True)
    if not ok: FAILS.append(name)

class Policy(torch.nn.Module):
    """pressure at n wake probes -> 2 jet amplitudes (tanh-bounded), the DPC network shape in miniature"""
    def __init__(self, probes, hidden=16, amax=0.5):
        super().__init__(); self.probes = probes; self.amax = amax
        self.net = torch.nn.Sequential(torch.nn.Linear(len(probes), hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2))
        for p_ in self.net.parameters(): p_.data *= 0.3
    def forward(self, st): return self.amax * torch.tanh(self.net(st["p"][self.probes]))

def main():
    t00 = time.time()
    print("  cylinder (butterfly quads), Re 100, +-90 deg jets, pressure-probe policy", flush=True)
    s = cylinder(nsteps=20); T = TorchUPISO(s); base = T.state_from_solver(); m = s.m
    W = WallForces(T, s.wall_faces); jets = SlotJets.cylinder(T, s.wall_faces, (90.0, -90.0), 10.0, vmax=1.0)
    C = m.centroid; probes = torch.as_tensor(np.argsort(np.hypot(C[:, 0] - 1.5, np.abs(C[:, 1]) - 0.5))[:6])
    pol = Policy(probes); params = list(pol.parameters())
    apply = lambda st, a: jets.apply(st, a)
    sl = lambda st, k: W(st)[0] + 0.1 * (st["ub"] ** 2).sum()          # drag plus an actuation penalty through the boundary values
    for H, sub in ((3, 1), (5, 2)):
        for p_ in params: p_.grad = None
        lt = tape_policy_grad(T, base, pol, H, apply, sl, substeps=sub); gt = [p_.grad.detach().clone() for p_ in params]
        for p_ in params: p_.grad = None
        lr = replay_policy_grad(T, base, pol, H, apply, sl, substeps=sub); gr = [p_.grad.detach().clone() for p_ in params]
        gn = max(float(g.abs().max()) for g in gt)
        err = max(float((a - b).abs().max()) for a, b in zip(gr, gt)) / max(gn, 1e-300)
        check(f"A20 replay == tape, H {H} x {sub} substeps: loss", abs(lr - lt) / abs(lt), 1e-12)
        check(f"A20 replay == tape, H {H} x {sub} substeps: dL/dtheta (worst, rel. to max)", err, 1e-9)
        print(f"    (loss {lt:.6f}, |dL/dtheta|max {gn:.3e}, {sum(p_.numel() for p_ in params)} parameters)")
    # A21: FD check of dL/dtheta along the gradient direction, H 3 x 1, on the recorded branch
    for p_ in params: p_.grad = None
    T.record(); l0 = tape_policy_grad(T, base, pol, 3, apply, sl)                 # recorded branch; replay == tape by A20
    d = [p_.grad.detach().clone() for p_ in params]; dn = np.sqrt(sum(float((g ** 2).sum()) for g in d)); d = [g / dn for g in d]
    ad = dn                                                              # directional derivative along the unit gradient
    def loss_at(h):
        with torch.no_grad():
            for p_, g in zip(params, d): p_.add_(h * g)
            T.replay()
            st = dict(base); L = 0.0
            for k in range(3): st = T.step(apply(st, pol(st))); L += float(sl(st, k))
            for p_, g in zip(params, d): p_.sub_(h * g)
        return L
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        fd = (loss_at(h) - loss_at(-h)) / (2 * h); errs.append(abs(fd - ad) / max(abs(fd), abs(ad)))
    T.live(); check("A21 dL/dtheta along the gradient vs FD (best of 3 steps)", min(errs), 1e-6)
    # A22: memory flat in the horizon (policy in the loop)
    mem = {}
    for H in (3, 12):
        for p_ in params: p_.grad = None
        with SavedBytes() as sb: replay_policy_grad(T, base, pol, H, apply, sl)
        mem[H] = sb.total / H / 2 ** 20
    check("A22 replay peak memory per step, H 12 vs H 3 (ratio - 1)", abs(mem[12] / mem[3] - 1.0), 0.10)
    # one gradient step must lower the loss (sanity, not a gate)
    for p_ in params: p_.grad = None
    l_before = replay_policy_grad(T, base, pol, 5, apply, sl, substeps=2); g2 = sum(float((p_.grad ** 2).sum()) for p_ in params)
    with torch.no_grad():
        for p_ in params: p_.sub_(1e-3 / np.sqrt(g2) * p_.grad)
    with torch.no_grad():
        st = dict(base); l_after = 0.0
        for k in range(5):
            st = apply(st, pol(st))
            for _ in range(2): st = T.step(st)
            l_after += float(sl(st, k))
    print(f"    (one normalised gradient step, H 5 x 2: loss {l_before:.6f} -> {l_after:.6f}, change {l_after - l_before:+.2e})")
    print(f"\n  {len(FAILS)} failure(s), {time.time() - t00:.0f}s"); return len(FAILS)
if __name__ == "__main__": sys.exit(main())
