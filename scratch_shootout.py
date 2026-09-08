"""Preconditioner shootout on a DUMPED real system (see PICT_DUMP_SOLVE).

One config per process -- the AmgX binding keys configs per tolerance, so
different configs at one tolerance must not share a process.

usage: python scratch_shootout.py <system.npz> <config.json|scipy>
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

d = np.load(sys.argv[1])
A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                  shape=tuple(d["shape"]))
b = d["b"]
rtol = float(d["rtol"])
n = A.shape[0]
tag = sys.argv[2]

if tag == "scipy":
    from scipy.sparse.linalg import cg, LinearOperator
    M = LinearOperator(A.shape, lambda v: v / A.diagonal())
    it = [0]
    def cb(x):
        it[0] += 1
    t0 = time.perf_counter()
    x, info = cg(A, b, rtol=rtol, maxiter=20000, M=M, callback=cb)
    t1 = time.perf_counter()
    r = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(f"RESULT scipy_cg_jacobi n={n} setup=0.000 solve={t1-t0:.3f} "
          f"iters={it[0]} resid={r:.1e}")
    sys.exit(0)

import os
os.environ["AMGX_CONFIG"] = tag
from src.amgx.binding import AmgXSolver

t0 = time.perf_counter()
s = AmgXSolver(A, rtol=rtol)
t1 = time.perf_counter()
times, iters = [], []
for rep in range(3):
    ta = time.perf_counter()
    x = s.solve(A.data, b)
    times.append(time.perf_counter() - ta)
    iters.append(s.iterations)
r = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
name = os.path.basename(tag).replace(".json", "")
print(f"RESULT {name} n={n} setup={t1-t0:.3f} solve={min(times):.3f} "
      f"iters={iters[-1]} resid={r:.1e}", flush=True)
os._exit(0)  # skip the known teardown wart
