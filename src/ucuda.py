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
_SHIFT_SRC = r"""
extern "C" __global__
void spmm_shift(const int nrow, const int nk, const int* __restrict__ ipA, const int* __restrict__ jA, const double* __restrict__ vA,
                const int* __restrict__ ipD, const int* __restrict__ jD, const double* __restrict__ vD, const double* __restrict__ s,
                const double* __restrict__ X, double* __restrict__ Y)
{   // Y = A X + s_c (D X): the per-mode Helmholtz family in one pass
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; long long total = (long long)nrow * nk; if (t >= total) return;
    int row = (int)(t / nk); int c = (int)(t - (long long)row * nk);
    double acc = 0.0;
    for (int k = ipA[row]; k < ipA[row + 1]; ++k) acc += vA[k] * X[(long long)jA[k] * nk + c];
    double accd = 0.0;
    for (int k = ipD[row]; k < ipD[row + 1]; ++k) accd += vD[k] * X[(long long)jD[k] * nk + c];
    Y[t] = acc + s[c] * accd;
}
extern "C" __global__
void jacobi_shift(const int nrow, const int nk, const int* __restrict__ ipA, const int* __restrict__ jA, const double* __restrict__ vA,
                  const int* __restrict__ ipD, const int* __restrict__ jD, const double* __restrict__ vD, const double* __restrict__ s,
                  const double* __restrict__ diagA, const double* __restrict__ diagD, const double omega,
                  const double* __restrict__ X, const double* __restrict__ B, double* __restrict__ Xn)
{   // Xn = X + omega (B - (A + s D) X) / (diagA + s diagD): one damped-Jacobi sweep of the whole family
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; long long total = (long long)nrow * nk; if (t >= total) return;
    int row = (int)(t / nk); int c = (int)(t - (long long)row * nk);
    double acc = 0.0;
    for (int k = ipA[row]; k < ipA[row + 1]; ++k) acc += vA[k] * X[(long long)jA[k] * nk + c];
    double accd = 0.0;
    for (int k = ipD[row]; k < ipD[row + 1]; ++k) accd += vD[k] * X[(long long)jD[k] * nk + c];
    Xn[t] = X[t] + omega * (B[t] - acc - s[c] * accd) / (diagA[row] + s[c] * diagD[row]);
}
"""
_kernel = None; _kshift = None; _kjac = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy as cp
        _kernel = cp.RawKernel(_KERNEL_SRC, "spmm_rowmajor")
    return _kernel


def _get_shift():
    global _kshift, _kjac
    if _kshift is None:
        import cupy as cp
        _kshift = cp.RawKernel(_SHIFT_SRC, "spmm_shift"); _kjac = cp.RawKernel(_SHIFT_SRC, "jacobi_shift")
    return _kshift, _kjac


def spmm_shift(A, D, s, X):
    """Y = A @ X + s[None, :] * (D @ X) for DevCSR A, D and a row-major float64 block X (N, nk)."""
    cp = A.cp; nrow = A.shape[0]; nk = X.shape[1]; X = cp.ascontiguousarray(X); Y = cp.empty((nrow, nk), dtype=cp.float64); total = nrow * nk; bs = 256
    _get_shift()[0](((total + bs - 1) // bs,), (bs,), (np.int32(nrow), np.int32(nk), A.indptr, A.indices, A.vals, D.indptr, D.indices, D.vals, s, X, Y))
    return Y


def jacobi_shift(A, D, s, diagA, diagD, omega, X, B):
    """One damped-Jacobi sweep on the family (A + s D): returns X + omega (B - (A + s D) X) / (diagA + s diagD)."""
    cp = A.cp; nrow = A.shape[0]; nk = X.shape[1]; X = cp.ascontiguousarray(X); B = cp.ascontiguousarray(B); Xn = cp.empty_like(X); total = nrow * nk; bs = 256
    _get_shift()[1](((total + bs - 1) // bs,), (bs,), (np.int32(nrow), np.int32(nk), A.indptr, A.indices, A.vals, D.indptr, D.indices, D.vals, s, diagA, diagD, np.float64(omega), X, B, Xn))
    return Xn


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
