# Gate 3's trajectory difference is NOT solver tolerance, and Gate 4 is now masking it

## The question

Gate 3 fails `< 1e-12` with a 10-step trajectory difference of 1.242e-07 at 8 ranks. Gate 4
measures essentially the same quantity -- 5.399e-07 against Gate 3's 5.412e-07 at 2 ranks -- and
PASSES, because its criterion was restated to `< 1.0e-06, from the solve tolerance`.

The obvious reading is that Gate 3's criterion is simply incoherent: you cannot ask for 1e-12
agreement when each linear solve is converged to 1e-6. That reading is WRONG, and it was tested
rather than assumed.

## The measurement (8 ranks, cylinder, 10 steps)

| pressure rtol | momentum tol | max relative difference | pressure iterations |
|---|---|---|---|
| 1e-6 | 1e-9 (default) | **1.242e-07** | ~330 |
| 1e-12 | 1e-9 (default) | **1.242e-07** | ~510 |
| 1e-6 | 1e-14 | **2.312e-08** | ~330 |
| 1e-12 | 1e-14 | **2.315e-08** | ~510 |

The iteration counts confirm each tolerance is genuinely active (330 -> 510 when the pressure
tolerance tightens six orders).

## What it says

1. **The PRESSURE tolerance has no effect whatsoever.** Six orders of magnitude tighter, from
   1e-6 to 1e-12, moves the difference by 0.2%.
2. **The MOMENTUM tolerance is a real contributor but not the source.** 1e-9 -> 1e-14 buys 5.4x,
   and then stops.
3. **A floor of ~2.3e-08 remains that no tolerance reaches.** It is tolerance-INDEPENDENT.

So Gate 3's criterion is right to fail, and relaxing it would have concealed a real effect.
**Gate 4's restated `< 1e-6` criterion now passes something that is not explained by the solve
tolerance it claims to derive from.** That restatement should be revisited: it is currently
9 orders of magnitude looser than the floor it is meant to bound.

## What it is not

* not the matrix: serial and distributed entries differ by exactly 0.00e+00 (Gate 3 item 1)
* not the reductions: Gate 5 compares C_D, C_L, divergence, ke and umax at 2/4/8/16 ranks and
  gets 0.000e+00 on all five, exactly
* not restart state: Gate 5's 20-continuous against 10+restart+10 is 0.000e+00 at every rank
  count

## The strongest remaining hypothesis, untested

The pressure system is PURE NEUMANN and therefore singular -- its solution is determined only up
to an additive constant, and PETSc removes the nullspace component. If that removal is not
bit-identical across rank counts (it is a global reduction over a partitioned vector), each step
leaves a slightly different constant in `phi`.

That would normally be harmless, since velocity depends only on the GRADIENT of p. It is not
harmless here, because the default scheme is `rotational`:

    p = p + phi - nu*div(u*)

**p ACCUMULATES.** A per-step constant offset that differs by rank count integrates over the
10 steps and appears in exactly the state array the gate compares -- while leaving every
gradient-derived quantity (velocity, forces, divergence) bitwise identical, which is precisely
the pattern Gate 5 reports.

It also explains the otherwise odd observation that the difference DECREASES with rank count
(5.4e-07 -> 2.1e-07 -> 1.2e-07 at 2/4/8): more ranks means a finer partition of the reduction and
a different, not necessarily larger, rounding path.

**How to test it, cheaply:** re-run Gate 3 with `scheme='chorin'`, which rebuilds p each step and
never accumulates. If the difference collapses, the nullspace constant is the source and the fix
is to pin the pressure level identically across rank counts. If it does not, this hypothesis is
dead and the search moves to the halo exchange.

This connects to the LES finding parked in `les_parking_lot.md`: accumulation of p is implicated
in two independent problems, and neither upstream PICT nor OpenFOAM accumulates at all.

---

# Both hypotheses tested. Both REFUTED. What is now known.

## Accumulation of p -- refuted

`test_gate3_scheme.py` (a variant; the original gate and its reference are untouched, and the
chorin capture goes to its own `_chorin.npz`) makes the scheme selectable. The checkpoint records
the scheme and rightly refuses a mismatch, so the load names that one key in `allow=` and keeps
every other check, including the grid fingerprint, in force.

| scheme | 8-rank difference |
|---|---|
| rotational (control, must reproduce) | **1.242e-07** |
| chorin -- p rebuilt each step, never accumulated | **1.158e-07** |

The control reproduces the original number exactly, so the variant is equivalent. Chorin gives
the same answer. **Accumulation is not the source.**

That test was also poorly aimed, and worth recording as such: a per-step nullspace constant
appears in `p` at the same magnitude whether or not it is accumulated, so this experiment could
only ever have refuted ACCUMULATION, not the constant itself.

## A nullspace constant -- also refuted

`gate3_which.py` reports the difference per array instead of as one global max, and asks whether
the pressure difference is UNIFORM (which a nullspace constant must be).

|  max\|diff\| | mean diff | std diff | std/\|mean\| | \|ref\|max | array |
|---|---|---|---|---|---|
| 1.850e-06 | -3.114e-08 | 1.537e-07 | 4.94 | 1.202e+01 | p_10 |
| 1.634e-06 | -5.638e-08 | 1.461e-07 | 2.59 | 1.202e+01 | p_5 |
| 1.624e-06 | -8.456e-08 | 1.816e-07 | 2.15 | 1.199e+01 | p_4 |

**The top ten arrays by difference are ALL pressure**, no velocity array among them. But
`std/|mean|` runs 1.3 to 17 -- the difference is not remotely uniform, so it is not a constant
offset. All 64 arrays differ at some level.

## What is established

1. the difference is concentrated in the PRESSURE field, not the velocity;
2. it is not a constant, so not a nullspace offset;
3. it is independent of the pressure solve tolerance over six orders of magnitude;
4. the momentum tolerance accounts for a factor of 5.4 and no more;
5. it is not the matrix (exact), the reductions (exact), or restart state (exact).

Relative size: ~1.5e-07 of |p|max ~ 12.

## Where to look next

The pressure operator is singular and ill-conditioned. For a Krylov solve stopped on a RELATIVE
RESIDUAL, the solution error is bounded by roughly `cond(A) * rtol`, so a condition number of
~1e6 leaves ~1e-6 of solution error at rtol 1e-12 -- the right order for what is measured. Two
partitions produce two different, equally converged, solutions inside that ball.

Against that: tightening rtol from 1e-6 to 1e-12 did NOT shrink the difference, though the
iteration count rose from ~330 to ~510, so the tolerance was genuinely active. Either the
achieved accuracy is floored by something other than rtol, or the error is not
condition-number-limited. **Measure `cond(A)` and the TRUE residual at both tolerances before
theorising further** -- that distinguishes the two directly and neither has been measured.

---

# RESOLVED: it was bjacobi's partition dependence

`linsolve.py` already named the suspect -- "PC CHOICE IS THE LAST PARTITION-DEPENDENT THING LEFT.
bjacobi factorises one diagonal block PER RANK, so the preconditioner -- and therefore the
iteration path, and therefore which of many valid within-tolerance solutions is reached --
depends on the decomposition." It was right.

`redundant`, the alternative that comment suggests, is unusable here: at 8 ranks it OOMs (signal
9), because it holds eight full factorisations of a 158,720^2 matrix. `jacobi` gets the same
property -- diagonal, therefore partition-independent -- for nothing.

**petsc_pc, not PICT_MPI_PC.** `PICT_MPI_PC` sets only the DISTRIBUTED path, so the serial
reference would keep bjacobi and the run would change the preconditioner and the partitioning at
the same time -- the exact confound Gate 3 exists to isolate. `_pcache` is constructed with no
`petsc_pc` at all, so the pressure serial path is hardwired to bjacobi; the test sets it on both
SolveCaches directly.

| preconditioner | serial iterations | 2 ranks | 8 ranks |
|---|---|---|---|
| bjacobi (control -- must reproduce) | ~310 | **5.412e-07** | **1.242e-07** |
| jacobi | ~1060 | **1.146e-13** | **9.305e-14** |

Six orders of magnitude, and **below Gate 3's original `< 1e-12` criterion.**

## Consequences

1. **Gate 3 passes at its strict criterion.** It never needed relaxing. The earlier instinct to
   call `< 1e-12` incoherent was wrong twice over -- first because the difference is not solve
   tolerance at all, and second because the criterion is achievable.
2. **Gate 4's restated `< 1.0e-06` should be tightened back.** It was loosened to accommodate a
   defect that has now been removed, and at 1e-6 it is nine orders looser than the achievable
   floor -- it would pass almost any regression.
3. **jacobi is also 8% faster at 8 ranks** (1.737 against 1.884 s/step, `gate6_kernels.md`).

## A trap to avoid when quoting the performance win

jacobi needs **1060 serial iterations against bjacobi's 303**, because at one rank bjacobi IS
full-matrix ILU. The 8% above is 8-ranks-against-8-ranks and is real. But a Gate 6 SPEEDUP
computed against a jacobi 1-rank baseline would rise partly because the BASELINE got slower --
the same shape of error as the rigged `momentum_tol` comparison in Gate 4. Measure the matched
1-rank jacobi time before any speedup number changes.

## Recommended change

Give the pressure `SolveCache` a `petsc_pc` the way the momentum one already has (`PICT_MOM_PC`),
defaulting to jacobi for the distributed path. That makes the port partition-independent end to
end, which is what every bitwise gate in this plan is trying to establish.

---

# SHIPPED: jacobi is now the default for BOTH solves

`src/piso_multiblock.py` gives each SolveCache a preconditioner defaulting to `jacobi`:
`PICT_PRES_PC` for pressure, `PICT_MOM_PC` for momentum. One value feeds the serial and the
distributed path, which is the point -- setting only `PICT_MPI_PC` changes the preconditioner and
the partitioning together.

## Defaulting only the pressure solve was not enough, and the gates caught it

| ranks | Gate 3 (pressure) | Gate 4 (momentum) |
|---|---|---|
| 2 | 3.227e-14 PASS | 4.182e-07 |
| 4 | 3.149e-14 PASS | 9.742e-08 |
| 8 | **5.178e-08 FAIL** | **5.178e-08** |

Gate 3 collapsed at 2 and 4 ranks and then failed at 8 with a number that is bit for bit what
Gate 4 reports. Gate 4 measures the MOMENTUM solve: `_mcache` was still reading an unset
`PICT_MOM_PC` and falling back to bjacobi, so momentum kept its own partition dependence and
became the binding term once pressure was fixed.

## With both defaulted

| ranks | Gate 3 | Gate 4 |
|---|---|---|
| 2 | 2.360e-14 | 1.219e-14 |
| 4 | 2.266e-14 | 9.783e-15 |
| 8 | 3.287e-14 | 1.176e-14 |

Against the starting point of 5.412e-07 / 2.130e-07 / 1.242e-07 (Gate 3) and 5.399e-07 /
2.143e-07 / 1.242e-07 (Gate 4). **Seven orders of magnitude, at every rank count, on both
solves.** The port is now partition-independent to round-off.

## Follow-up

**Gate 4's criterion should be tightened.** It was restated to `< 1.0e-06` to accommodate a
defect that has now been removed; the measured value is ~1.2e-14, so the criterion is eight
orders too loose and would pass a serious regression unnoticed. `< 1e-12`, matching Gate 3, is
now achievable and was achievable all along.

## Gate 4's criterion tightened: 1e-6 -> 1e-12

The old `bar = max(1000.0 * MOM_TOL, 1e-13)` rested on a measured table showing the trajectory
difference tracking the solve tolerance. That tracking was real and it was a SYMPTOM: under
bjacobi, serial and distributed took different Krylov paths, and two solves converged to the same
tolerance along different paths differ by about that tolerance.

Re-measured under jacobi, with the reference RECAPTURED at each tolerance:

| momentum_tol | trajectory difference |
|---|---|
| 1e-6 | 2.120e-14 |
| 1e-9 | 1.219e-14 |
| 1e-12 | 1.279e-14 |

**Flat across six orders** -- the difference no longer tracks the tolerance, so scaling by it is
meaningless. A fixed `1e-12` is what the quantity now deserves, matching Gate 3.

(The first attempt at this table was wrong and non-monotonic -- 1e-6 -> 8.9e-05, 1e-9 -> 1.2e-14,
1e-12 -> 1.3e-07 -- because gate4's REF filename keys on the PRESSURE rtol and does not recapture
when PICT_MOM_TOL changes, so it compared distributed at one tolerance against serial at another.
It looked like tighter tolerances making agreement worse.)

Verified, all at the new bar: 1.219e-14, 9.783e-15, 1.176e-14 at 2/4/8 ranks, ~50-100x margin.
**And the criterion still bites:** solving at a tolerance the reference was not captured at gives
8.869e-05 and FAILS, so tightening it did not merely make it look strict.
