# Scaling with both solves distributed — the Gate 6 decision number

Re=100 cylinder, 160,768 cells in 16 blocks of 157x16x4. Five timed steps after a warm-up.
One BLAS thread per rank, `--bind-to none`, PETSc backend, pressure rtol 1e-6, momentum 1e-14.

| ranks | s/step | speed-up | pressure | its | momentum | its | other |
|------:|-------:|---------:|---------:|----:|---------:|----:|------:|
| 1     |  3.735 |  1.00x   |  2.359   |1200 |  0.260   |  26 | 1.116 |
| 2     |  2.982 |  1.25x   |  1.537   |1323 |  0.807   |  70 | 0.639 |
| 4     |  1.724 |  2.17x   |  0.836   |1289 |  0.411   |  72 | 0.476 |
| 8     |  1.235 |  3.02x   |  0.535   |1302 |  0.242   |  72 | 0.457 |

**3.02x on 8 ranks, against Gate 6's abort threshold of 3x.** It clears, but barely.

Against the pre-port serial baseline of 6.865 s/step (SciPy, Gate 2 table) the end-to-end gain
is 5.6x: PETSc is 1.84x faster than SciPy at one rank, and scales 3.02x on top of that.

## Distributing the MOMENTUM solve is a bad trade

  * it is 7% of serial runtime (0.260 s of 3.735)
  * distributing it inflates iterations 26 -> 72, a 2.8x growth that breaches Gate 4's 2x abort
  * and it saves 0.018 s/step at 8 ranks -- 1.4% of the total

Those two Gate 4 failures therefore buy 1.4% performance. Leaving momentum on the replicated
path would restore trajectory equivalence to Gate 3's 7.8e-14, remove the iteration breach, and
cost 1.4%.

## Where the remaining time goes

`other` -- assembly, halo exchange and the Gate 2 allgather -- is 1.116 s at one rank and
0.457 s at eight: a 2.4x reduction, so it is scaling, but it has grown from 30% to **37% of
runtime** as the pressure solve shrank beneath it. It is now the second-largest term and the
allgather inside it is pure overhead that exists only because Gate 2 replicated the fields.
Removing it is the obvious next lever, and matters more than anything left in the solves.

## Comparability

The Gate 2 table measured a REPLICATED solve and showed wall time getting WORSE with rank count
(6.865 -> 12.702 s/step from 1 to 8). That table is not superseded by this one; it measures a
different thing, and the pair together is the argument for the port. Both use identical
conditions -- same case, same thread count, same binding -- because the Gate 2 exercise found
that `mpirun -n 1` pins to a single core and fakes a 2x speed-up when binding is left default.
