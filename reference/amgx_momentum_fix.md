# AmgX: the momentum system must not go to the GPU

**Symptom.** Long AmgX runs on the jet-resolved cylinder threw `AMGX REJECT: true resid
7.138e+06 > target 2.967e-06` with `status=3 (not converged) after 500 iters`, followed by
`RECONSTRUCT-RETRY`, `FULL RESET`, and a stream of `Caught amgx exception: Mode not found` at
`AMGX_solver_destroy`. GPU utilisation read 0% and the run fell back to scipy -- a 119-hour pace
wearing the label of a 25-hour one.

**Cause.** Every failing solve carried `rtol=1e-09`, the MOMENTUM tolerance; the pressure system
at 1e-6 never failed. The momentum operator is `J/dt + convection + diffusion` and is
DIAGONALLY DOMINANT -- the `J/dt` term grows as dt shrinks. Diagonal dominance means simple
relaxation already captures the operator, so there are no smooth error modes for a multilevel
method to damp. AMG has nothing to fix and its BiCGStab outer iteration breaks down chasing an
unreachable target, returning a vector whose residual is 2400x the RHS norm.

The `Mode not found` messages were a CONSEQUENCE, not the fault: the reset/retry path destroys
solver handles after a failed solve. They vanished with the failures.

**Fix.** `piso_multiblock` routes the momentum SolveCache to scipy whenever
`linear_backend == "amgx"`. `PICT_MOM_BACKEND` overrides.

**Measured, per step, 130,592 cells:**

| system | character | scipy | AmgX | |
|---|---|---|---|---|
| pressure | elliptic, NOT diagonally dominant | 2.816 s | **0.182 s** | **15x win** |
| momentum | DIAGONALLY DOMINANT | **0.125 s** | 0.149 s | AmgX SLOWER |

So the only system AmgX failed on is the one it made no faster. The shootout had already
measured the same thing from the other side: momentum under AMG diverges at 20,000 iterations,
under Jacobi converges in ONE at 0.014 s.

**After the fix:** `AMGX REJECT` 0, `unhealthy` 0, `RECONSTRUCT/FULL RESET` 0, `Mode not found`
0, over 40 steps, with the pressure solve still at 0.175-0.185 s.

## dt dependence, worth knowing before this is generalised

As `dt -> 0` the momentum matrix becomes MORE diagonally dominant and Jacobi gets relatively
better. At large dt, or for a steady-state solve where the `J/dt` term vanishes, momentum
becomes more elliptic and AMG could begin to pay. The rule is not "momentum never wants AMG", it
is "a diagonally dominant operator does not".

## What was ruled out first, each by its own run

The preconditioner (BLOCK_JACOBI and MULTICOLOR_DILU gave an IDENTICAL failure count), the
convergence criterion (RELATIVE_INI_CORE against RELATIVE_MAX, identical), the AmgX mode constant
(8193 is accepted by this build), the config files (all seven create solvers, rc=0), and drift
rebuilds (`PICT_AMGX_DRIFT=1e30`, identical). **The count being identical under every config is
what finally said the variable was not in the config.**

## Two environment traps found on the way

1. `pict-amgx:1.0` sets `AMGX_LIB=/tmp/AMGX/build/libamgxsh.so`, which DOES NOT EXIST -- `/tmp`
   did not survive the image build. The library lives on the HOST at `~/amgx/libamgxsh.so` and
   must be mounted (`-v $HOME/amgx:/amgx -e AMGX_LIB=/amgx/libamgxsh.so`). The build that
   produced the original shootout numbers is gone; the host copy is AmgX 2.5.0, Sep 2026.
2. **The image has TWO Pythons, and the one on PATH is the wrong one.**

   | | `/opt/cpn/bin/python3` (on PATH) | `/usr/bin/python3` |
   |---|---|---|
   | python | 3.13 | 3.12.3 |
   | torch | **absent** | **2.10.0a0+nv25.11** |
   | numpy | 2.4.6 | 2.1.0 |
   | scipy | 1.16.3 | **1.16.3** |
   | CUDA | -- | **True, NVIDIA GB10** |
   | mpi4py | present | absent |

   **`/usr/bin/python3` has the entire training stack: torch + numpy + scipy + working CUDA.**
   An earlier note here said no interpreter had all three; that was wrong -- it was written after
   checking `import torch` under the PATH interpreter and never checking numpy/scipy under the
   other one. There is no packaging blocker.

   The only gap is `mpi4py`, which does not matter: the discrete adjoint has NO MPI path (that is
   Gate 7 of the DD plan, unstarted), so any training loop is serial regardless.

   **Invoke `/usr/bin/python3` explicitly.** The PATH default silently gives the torch-less
   interpreter, and the failure is an ImportError several seconds into a job rather than at
   submission.

## And a note on how long this took

Five wrong diagnoses preceded the right one, and the log line that identified it --
`AMGX REJECT ... rtol=1e-09` -- was present in the FIRST failing run. The fallback prints the
failing system, its tolerance and its config precisely so this does not require guesswork. Grep
the log before forming a hypothesis.


---

# AmgX and PyTorch on the same GPU: VERIFIED to coexist

The training loop needs AmgX (forward pressure solve) and torch-CUDA (policy, and the adjoint
graph) alive in one process on one device. They are, and it was tested rather than assumed:
under `/usr/bin/python3`, torch reports CUDA available on the GB10, a GPU matmul succeeds, 40
AmgX solves run with the production configuration, and **a second GPU matmul still returns
correctly afterwards**. AmgX is loaded through `ctypes` from a `.so`, so it is interpreter-
agnostic; both simply hold CUDA contexts on the same device.

    torch 2.10.0a0+nv25.11  cuda_available=True   device: NVIDIA GB10
    torch GPU matmul OK, trace=5.0656e+03
    pressure backend=amgx  momentum backend=scipy
    AMGX REJECT 0 | unhealthy 0 | RECONSTRUCT 0 | FULL RESET 0
    torch GPU still OK AFTER 40 AmgX solves: 2.9745e+03

## PyTorch is NOT a scipy replacement, and does not need to be

What this solver uses from scipy is `scipy.sparse.linalg`: `splu`, `spilu`, `cg`, `bicgstab`.
PyTorch has sparse TENSORS (`torch.sparse`, CSR/COO, sparse-dense matmul) but **no Krylov solver
suite and no incomplete factorisation**. `torch.sparse.spsolve` is a direct solve via cuDSS, not
a preconditioned iterative solver; `torch.linalg.solve` is dense-only. Hand-writing CG over
`torch.sparse` is straightforward for the SYMMETRIC pressure operator; reproducing `spilu` is
not. Since one interpreter carries both libraries, the question is moot.

## `Mode not found` is cosmetic, and here is why it misled for so long

It appears **exactly 8 times** regardless of step count (40, 60, 80), of preconditioner, of
convergence criterion, and of whether any solve failed. It is AmgX C-layer teardown noise.

It also **cannot be captured by redirecting `sys.stderr`**: AmgX writes to fd 2 directly, so a
`contextlib.redirect_stderr` into a StringIO counts zero while the terminal shows eight. An
earlier claim here -- that the message vanished along with the solve failures -- came from
exactly that mistake.

The lasting lesson is the grep, not the message: `AMGX REJECT`/`unhealthy` are REAL failures and
`Mode not found` is NOT, and counting them together made a working fix look like a failed one
twice. Grep them separately.
