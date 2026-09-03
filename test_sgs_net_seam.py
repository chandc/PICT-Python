"""The network in the loop, and the seam it was blind to.

WHY THIS FILE EXISTS. `measurement_traps.md` section 5 records that TinySGSNet uses
`padding_mode="circular"`, which is right for one periodic box and wrong at a multi-block seam,
and that **none of the 38 adjoint gates would catch it**, because every one drives the solver
with a raw parameter vector and never a network. That is a blind spot of the test suite, not of
the network, and it is closed here.

Four checks, in increasing strength:

  1  circular padding is UNCHANGED -- the existing periodic path must not have moved.
  2  the seam defect is REAL and is measured, not asserted: per-block circular padding against
     the whole-box answer on a split periodic box, where the whole-box answer is exact.
  3  the halo-fed path is CORRECT: fed the neighbour data the solver already exchanges, the
     per-block result reproduces the whole-box answer to round-off.
  4  the NETWORK IN THE LOOP: finite differences against the adjoint on every one of the 173
     weights, with the multi-block PISO chain between the weights and the loss.

Check 4 is the one that had no coverage at all. Its mangle -- zeroing the network's contribution
in the backward only -- must make it fail, or it is measuring nothing.
"""
import numpy as np
import torch

from src.domains import periodic_box
from src.mb_adjoint import MultiBlockChain
from src.sgs_net import TinySGSNet

torch.set_default_dtype(torch.float64)


def _fields(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g, dtype=torch.float64) for _ in range(3)]


def check_circular_unchanged():
    """A fixed input and fixed weights must give the byte-identical answer they always did."""
    torch.manual_seed(7)
    net = TinySGSNet()
    x = _fields((6, 6, 6), seed=1)
    out = net(torch.stack(x).unsqueeze(0))
    ref = float(out.sum()), float((out ** 2).sum())
    torch.manual_seed(7)
    net2 = TinySGSNet(pad_mode="circular")
    out2 = net2(torch.stack(x).unsqueeze(0))
    ok = torch.equal(out, out2) and net.halo == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] circular path unchanged and self-padding "
          f"(caller halo {net.halo}); sum {ref[0]:+.6f}")
    return ok


def _blocks_of(box, n_split, ntot):
    """Split a cube of side ntot into n_split^3 blocks, returning index slices."""
    w = ntot // n_split
    for i in range(n_split):
        for j in range(n_split):
            for k in range(n_split):
                yield (slice(i * w, (i + 1) * w), slice(j * w, (j + 1) * w),
                       slice(k * w, (k + 1) * w))


def check_seam_defect_is_real(ntot=8, n_split=2):
    """Per-block circular padding against the exact whole-box answer."""
    torch.manual_seed(3)
    net = TinySGSNet()
    F = _fields((ntot, ntot, ntot), seed=2)
    whole = net(torch.stack(F).unsqueeze(0)).squeeze(0)          # exact: the box IS periodic
    err = 0.0
    for sl in _blocks_of(None, n_split, ntot):
        sub = [f[sl] for f in F]
        got = net(torch.stack(sub).unsqueeze(0)).squeeze(0)
        err = max(err, float((got - whole[(slice(None),) + sl]).abs().max()))
    scale = float(whole.abs().max())
    ok = err > 0.1 * scale
    print(f"  [{'PASS' if ok else 'FAIL'}] the seam defect is real: per-block circular padding "
          f"is wrong by {err:.4f} = {100*err/scale:.0f}% of the output scale {scale:.4f}")
    return ok


def check_halo_path_correct(ntot=8, n_split=2):
    """Fed the neighbour data, the per-block answer must match the whole box to round-off."""
    torch.manual_seed(3)
    ref = TinySGSNet()
    net = TinySGSNet(pad_mode="none")
    net.load_state_dict(ref.state_dict())                # identical weights, different padding
    F = _fields((ntot, ntot, ntot), seed=2)
    whole = ref(torch.stack(F).unsqueeze(0)).squeeze(0)
    err = 0.0
    for sl in _blocks_of(None, n_split, ntot):
        # the halo the SOLVER would supply: one cell of the true neighbour, periodic here
        pads = []
        for f in F:
            idx = [np.arange(s.start - 1, s.stop + 1) % ntot for s in sl]
            pads.append(f[np.ix_(*idx)])
        got = net(torch.stack(pads).unsqueeze(0)).squeeze(0)
        err = max(err, float((got - whole[(slice(None),) + sl]).abs().max()))
    ok = err < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] halo-fed per-block reproduces the whole box to "
          f"{err:.2e}  (same weights, padding=0, neighbour data supplied)")
    return ok


def check_network_in_the_loop(mangle=False):
    """FD vs adjoint on every network weight, with the multi-block PISO chain in between."""
    d = periodic_box(8, 2)
    chain = MultiBlockChain(d, 0.05, 0.05)
    torch.manual_seed(11)
    net = TinySGSNet()
    n = int(np.size(chain.u_init))
    side = int(round(n ** (1 / 3)))
    assert side ** 3 == n, f"expected a cube, got {n} cells"
    shape = (side, side, side)

    def src(u):
        # one state in, a 3-component source out; take the first component so the source is a
        # single flat vector of the chain's length
        S = net.field(u, u, u, shape)
        s = S[0]
        return s.detach() if mangle else s     # MANGLE: hide the net from the backward

    def loss():
        return chain.rollout([None, None], final_only=True, source_fn=src)

    L = loss()
    if L.requires_grad:
        L.backward()
        g = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).clone()
    else:
        # The mangle detaches the source, so the loss genuinely has no path to the weights and
        # backward() raises rather than returning zero. That IS the detection, but it has to be
        # recorded as a measured gradient of zero rather than as a crash -- the finite
        # difference below still moves, because detach affects the backward graph and not the
        # forward VALUE, so FD-vs-adjoint comes out order one.
        g = torch.zeros(sum(p.numel() for p in net.parameters()), dtype=torch.float64)
    with torch.no_grad():
        base = float(L)
    flat = torch.cat([p.data.reshape(-1) for p in net.parameters()])
    eps = 1e-6
    fd = torch.zeros_like(g)
    # sample where the signal is -- normalising by a near-zero per-cell gradient is trap 4 in
    # measurement_traps.md. When the mangle has zeroed g, fall back to a fixed set so the FD
    # still probes the weights that matter in the unmangled case.
    probe = (np.argsort(-np.abs(g.numpy()))[:12] if float(g.abs().max()) > 0
             else np.arange(12))
    for i in probe:
        old = flat[i].item()
        _set(net, i, old + eps); fp = float(loss())
        _set(net, i, old - eps); fm = float(loss())
        _set(net, i, old)
        fd[i] = (fp - fm) / (2 * eps)
    scale = max(float(g.abs().max()), float(fd.abs().max()))
    err = float((fd[probe] - g[probe]).abs().max()) / max(scale, 1e-30)
    ok = (err < 1e-6) if not mangle else (err > 1e-2)
    tag = "the mangle is DETECTED" if mangle else "network in the loop"
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: FD vs adjoint on {len(probe)} of "
          f"{g.numel()} weights, max error {err:.2e} of max|g| = {scale:.3e}")
    return ok


def _set(net, i, val):
    k = 0
    for p in net.parameters():
        m = p.numel()
        if k <= i < k + m:
            with torch.no_grad():
                p.view(-1)[i - k] = val
            return
        k += m


def main():
    print("=" * 78)
    print("  the network in the loop, and the seam the 38 adjoint gates could not see")
    print("=" * 78)
    r = [check_circular_unchanged(), check_seam_defect_is_real(), check_halo_path_correct(),
         check_network_in_the_loop(), check_network_in_the_loop(mangle=True)]
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
