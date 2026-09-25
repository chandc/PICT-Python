import sys, time, numpy as np, scipy.sparse as sp; sys.path.insert(0,".")
import cupy as cp, cupyx.scipy.sparse as csp
from src.ucuda import DevCSR
N=65536; nk=66; rng=np.random.default_rng(0)
A=sp.random(N,N,density=5/N,format="csr",random_state=0)+sp.eye(N); X=rng.standard_normal((N,nk))
D=DevCSR(A); Xd=cp.asarray(X); Ac=csp.csr_matrix(A)
Y1=(D@Xd); Y2=(Ac@Xd); cp.cuda.Device().synchronize(); print("max diff kernel vs cupyx", float(cp.abs(Y1-Y2).max()), " vs scipy", float(np.abs(Y1.get()-A@X).max()))
for name,f in (("raw kernel",lambda: D@Xd),("cupyx",lambda: Ac@Xd)):
    f(); cp.cuda.Device().synchronize(); t0=time.time()
    for _ in range(20): f()
    cp.cuda.Device().synchronize(); t=(time.time()-t0)/20; print(f"  {name:10s} {t*1e3:6.2f} ms per matvec ({A.nnz*nk*8/t/1e9:.0f} GB/s of X reads)")
