"""CSR-times-dense-block on the GPU for the 2.5D solver (LES plan L4): a CuPy raw kernel for
Y = A @ X with X row-major (N, nk) -- the layout every field block in `PISO25` has. One thread per
(row, column): consecutive threads take consecutive columns of one row, so the reads of X[j, :] for
each nonzero are coalesced and the row's indices/values are broadcast within the warp. cupyx's
`csr @ dense` was measured at 17 ms for a 5-point 65k-row matrix times 66 columns on the GB10
(~25 GB/s); this kernel moves the same 210 MB at a bandwidth-bound rate.

`DevCSR` wraps a scipy CSR matrix on the device and dispatches: 2-D real float64 C-contiguous blocks
go through the kernel, complex blocks are split, anything else falls back to cupyx.
"""
import numpy as np
import scipy.sparse as sp

_KERNEL_SRC = r"""
extern "C" __global__
void spmm_rowmajor(const int nrow, const int nk, const int* __restrict__ indptr, const int* __restrict__ indices,
                   const double* __restrict__ vals, const double* __restrict__ X, double* __restrict__ Y)
{
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)nrow * nk;
    if (t >= total) return;
    int row = (int)(t / nk); int c = (int)(t - (long long)row * nk);
    double acc = 0.0;
    int k0 = indptr[row], k1 = indptr[row + 1];
    for (int k = k0; k < k1; ++k) acc += vals[k] * X[(long long)indices[k] * nk + c];
    Y[t] = acc;
}
"""
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy as cp
        _kernel = cp.RawKernel(_KERNEL_SRC, "spmm_rowmajor")
    return _kernel


class DevCSR:
    def __init__(self, A):
        import cupy as cp, cupyx.scipy.sparse as csp
        A = sp.csr_matrix(A); A.sort_indices()
        self.shape = A.shape; self.nnz = A.nnz
        self.indptr = cp.asarray(A.indptr.astype(np.int32)); self.indices = cp.asarray(A.indices.astype(np.int32)); self.vals = cp.asarray(A.data.astype(np.float64))
        self._csp = csp.csr_matrix(A)                       # fallback for shapes the kernel does not take
        self.cp = cp

    def _spmm(self, X):
        cp = self.cp; nrow = self.shape[0]; nk = X.shape[1]
        Y = cp.empty((nrow, nk), dtype=cp.float64); total = nrow * nk; bs = 256
        _get_kernel()(((total + bs - 1) // bs,), (bs,), (np.int32(nrow), np.int32(nk), self.indptr, self.indices, self.vals, X, Y))
        return Y

    def __matmul__(self, X):
        cp = self.cp
        if X.ndim == 2 and X.shape[1] >= 1:
            if X.dtype == cp.float64:
                return self._spmm(cp.ascontiguousarray(X))
            if X.dtype == cp.complex128:
                return self._spmm(cp.ascontiguousarray(X.real)) + 1j * self._spmm(cp.ascontiguousarray(X.imag))
        return self._csp @ X

    def diagonal(self):
        return self._csp.diagonal()

    def toarray(self):
        return self._csp.toarray()
