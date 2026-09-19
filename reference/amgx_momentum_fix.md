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
2. **The image has TWO Pythons.** `/opt/cpn/bin/python3` is on PATH and has numpy 2.4.6 /
   scipy 1.16.3 but NO torch; `/usr/bin/python3` has torch 2.10.0a0+nv25.11 but not our stack.
   AmgX is loaded by `ctypes` from a `.so`, so it is interpreter-agnostic -- nothing prevents
   AmgX and torch sharing the GPU in one process. **But the training loop needs numpy, scipy AND
   torch in ONE interpreter, and no interpreter in this image has all three.** That is the next
   blocker for G6, and it is a packaging fix, not a code fix.

## And a note on how long this took

Five wrong diagnoses preceded the right one, and the log line that identified it --
`AMGX REJECT ... rtol=1e-09` -- was present in the FIRST failing run. The fallback prints the
failing system, its tolerance and its config precisely so this does not require guesswork. Grep
the log before forming a hypothesis.
