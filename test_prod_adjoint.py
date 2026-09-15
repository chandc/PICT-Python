"""Stage 9.1 gates -- differentiable PRODUCTION momentum assembly.

The path no chain has certified: gradient flow THROUGH matrix assembly.
On the coarse butterfly (the production topology):

  9.1a  exactness/linearity: vals0 + T x reproduces the production
        assembler on random velocity fields to ~1e-12 relative. This
        MEASURES the affine claim (central convection); a hidden sign
        branch or limiter fails here.
  9.1b  pattern stability: the sparsity pattern is field-independent
        (asserted inside values(); gated on a second random field).
  9.1c  autograd through assembly: d<w, A(x) y>/dx vs FD.
  9.1d  through the SOLVE: d ||A(x)^{-1} b||^2 / dx via LinearSolve's
        matrix-values gradient chained with T -- assembly -> matrix ->
        solve -> loss, the Stage 9 spine in miniature.

Run:  .venv/bin/python test_prod_adjoint.py     (~5 min; T is cached)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cylinder_rect_bc import classify
from cylinder_ring_grid import ring_rect_domain
from src.adjoint_piso import LinearSolve
from src.prod_adjoint import MomentumAssembly, TorchMomentumAssembly

torch.set_default_dtype(torch.float64)
NU, DT = 0.01, 0.01                     # production values (Re 100, D=1, U=1)
PASS = FAIL = 0


def check(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}", flush=True)


def main():
    d, _ = ring_rect_domain(n_east=33, side_dt=0.08, nz=2, wake_dx=0.4,
                            wake_hold=8.0, wake_ratio=1.10)
    kindmap = {"inlet": "inflow", "outlet": "outflow",
               "lateral": "wall", "body": "wall"}
    for (b, fid), role in classify(d).items():
        d.blocks[b].faces[fid] = kindmap[role]

    t0 = time.time()
    asm = MomentumAssembly(d, NU, DT, verbose=True)
    print(f"  butterfly {d.n_cells:,} cells; T {asm.T.shape} nnz {asm.T.nnz:,} "
          f"({asm.ncolors} colors, {time.time()-t0:.0f}s incl. cache)", flush=True)

    rng = np.random.default_rng(0)
    N = asm.N

    # ------------------------------------------------------------------ 9.1a
    worst = 0.0
    for _ in range(3):
        x = rng.standard_normal(3 * N)
        ref = asm.values(x)
        pred = asm.vals0 + asm.T @ x
        worst = max(worst, np.abs(pred - ref).max() / np.abs(ref).max())
    check(worst < 1e-12, f"9.1a exact affine assembly on 3 random fields: "
                         f"worst rel {worst:.2e}")

    # ------------------------------------------------------------------ 9.1b
    try:
        asm.values(rng.standard_normal(3 * N))
        ok = True
    except AssertionError:
        ok = False
    check(ok, "9.1b sparsity pattern is field-independent")

    # ------------------------------------------------------------------ 9.1c
    tasm = TorchMomentumAssembly(asm)
    rows, cols = asm.idx
    w = torch.as_tensor(rng.standard_normal(asm.shape[0]))
    y = torch.as_tensor(rng.standard_normal(asm.shape[1]))
    wy = (w[torch.as_tensor(np.asarray(rows, dtype=np.int64))]
          * y[torch.as_tensor(np.asarray(cols, dtype=np.int64))])   # <w, A y>

    x = torch.tensor(rng.standard_normal(3 * N), requires_grad=True)
    (tasm.vals(x) * wy).sum().backward()
    g = x.grad.detach().numpy()
    ks = np.argsort(-np.abs(g))[:3]
    h, worst = 1e-3, 0.0
    x0 = x.detach().numpy()
    for k in ks:
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        Lp = float(((asm.vals0 + asm.T @ xp) * wy.numpy()).sum())
        Lm = float(((asm.vals0 + asm.T @ xm) * wy.numpy()).sum())
        worst = max(worst, abs((Lp - Lm) / (2 * h) - g[k]) / max(abs(g[k]), 1e-300))
    check(worst < 1e-9, f"9.1c autograd of <w, A(x) y> vs FD: worst rel {worst:.2e}")

    # ------------------------------------------------------------------ 9.1d
    b_rhs = torch.as_tensor(rng.standard_normal(asm.shape[0]))
    pattern = (asm.idx, asm.shape)

    def loss(xt):
        sol = LinearSolve.apply(tasm.vals(xt), b_rhs, pattern, False, False)
        return (sol ** 2).sum()

    x = torch.tensor(0.05 * rng.standard_normal(3 * N), requires_grad=True)
    L = loss(x)
    L.backward()
    g = x.grad.detach().numpy()
    ks = np.argsort(-np.abs(g))[:2]
    h, worst = 1e-4, 0.0
    x0 = x.detach().numpy()
    with torch.no_grad():
        for k in ks:
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            Lp = float(loss(torch.as_tensor(xp)))
            Lm = float(loss(torch.as_tensor(xm)))
            worst = max(worst, abs((Lp - Lm) / (2 * h) - g[k]) / max(abs(g[k]), 1e-300))
    check(worst < 1e-4,
          f"9.1d assembly -> solve -> loss gradient vs FD: worst rel {worst:.2e} "
          f"(L = {float(L):.4e})")

    # ================================================================== 9.2a
    from src.prod_adjoint import TorchFluxKernels
    fk = TorchFluxKernels(d)
    nb = len(d.blocks)
    xu = torch.as_tensor(rng.standard_normal(N))
    xv = torch.as_tensor(rng.standard_normal(N))
    xw = torch.as_tensor(rng.standard_normal(N))
    fu = {b: xu.numpy()[d.global_ids(b)] for b in range(nb)}
    fv = {b: xv.numpy()[d.global_ids(b)] for b in range(nb)}
    fw = {b: xw.numpy()[d.global_ids(b)] for b in range(nb)}
    with torch.no_grad():
        Ft = fk.face_fluxes_all(xu, xv, xw)
    worst_f = worst_d = 0.0
    for b in range(nb):
        Fr = d.face_fluxes(b, fu, fv, fw)
        for a in range(3):
            sc = max(np.abs(Fr[a]).max(), 1e-300)
            worst_f = max(worst_f, np.abs(Ft[b][a].numpy() - Fr[a]).max() / sc)
        dr = d.divergence(b, Fr, asm.Js[b])
        with torch.no_grad():
            dt_ = fk.divergence(b, Ft[b]).numpy()
        worst_d = max(worst_d, np.abs(dt_ - dr).max()
                      / max(np.abs(dr).max(), 1e-300))
    check(worst_f < 1e-13, f"9.2a torch face_fluxes == production on all "
                           f"{nb} blocks: worst rel {worst_f:.2e}")
    check(worst_d < 1e-13, f"9.2a torch divergence == production: "
                           f"worst rel {worst_d:.2e}")

    # autograd liveness through the flux path (linear -> FD exact)
    xg = torch.tensor(rng.standard_normal(N), requires_grad=True)
    L = sum((fk.divergence(b, Fb) ** 2).sum()
            for b, Fb in enumerate(fk.face_fluxes_all(xg, xv, xw)))
    L.backward()
    g = xg.grad.detach().numpy()
    k = int(np.argmax(np.abs(g)))
    h2 = 1e-4
    with torch.no_grad():
        def LL(val):
            xp = xg.detach().clone()
            xp[k] = val
            return float(sum((fk.divergence(b, Fb) ** 2).sum()
                             for b, Fb in enumerate(fk.face_fluxes_all(xp, xv, xw))))
        fd = (LL(float(xg[k]) + h2) - LL(float(xg[k]) - h2)) / (2 * h2)
    rel = abs(g[k] - fd) / max(abs(fd), 1e-300)
    check(rel < 1e-7, f"9.2a gradient through pad -> flux -> divergence: "
                      f"FD rel {rel:.2e}")

    print(f"\n  {PASS}/{PASS + FAIL} checks passed", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
