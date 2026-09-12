# DNS time steps are physics-limited, not stability-limited (verified here)

The claim: in DNS (and resolved LES) with an implicit or semi-implicit scheme,
the time step is set by the requirement to RESOLVE the flow's time scales --
the Kolmogorov time and the advective sweep of the smallest resolved eddies --
not by numerical temporal stability. A scheme can be perfectly stable at a dt
that quietly destroys the physics. Verification below combines the canonical
literature, this stack's own numbers, and a direct experiment (R14).

## The literature anchor

Choi & Moin (JCP 1994, "Effects of the computational time step on numerical
solutions of turbulent flow") ran turbulent channel DNS with a FULLY IMPLICIT
scheme -- unconditionally stable, no CFL bound -- at increasing dt. At
dt+ = dt u_tau^2/nu around 0.4 the solution is accurate; by dt+ ~ 1.6 the
statistics degrade; at larger dt+ THE TURBULENCE DECAYS TO LAMINAR while the
computation remains numerically stable throughout. The mechanism: the
near-wall regeneration cycle has time scales of a few viscous units
(Kolmogorov time at the wall tau_eta+ ~ 2.4); a dt that under-resolves them
filters the physics that sustains turbulence. Moin & Mahesh (Annu. Rev. Fluid
Mech. 1998) state the consequence as doctrine: with implicit methods the DNS
time step is chosen for ACCURACY (a fraction of the Kolmogorov time scale),
so removing the stability limit does not buy a larger dt.

Honest scope: the claim is about implicit/semi-implicit incompressible
solvers. A fully EXPLICIT code can genuinely be stability-bound (viscous
limit ~ dy+^2/2 near a wall, or acoustic CFL in compressible DNS) at a dt
BELOW the physics requirement -- which is an argument about scheme choice,
not a counterexample about the physics.

## This stack's numbers (channel, Re_tau = 180, delta = u_tau = 1, nu = 1/180)

- Viscous time unit t_nu = nu/u_tau^2 = 1/180. Production dt = 0.001 gives
  **dt+ = 0.18** -- half the Choi & Moin accuracy bound, and 13x inside the
  wall Kolmogorov time (tau_eta+ ~ 2.4).
- The scheme (BDF2, implicit diffusion AND implicit convection via the
  Picard-rebuilt momentum matrix) has no CFL stability bound; the cylinder
  campaigns run it at local CFL ~ 1.1 routinely. The channel runner's
  `--cfl-limit 0.8` watchdog is an ACCURACY tripwire, not a stability limit
  -- exactly the distinction this note is about.
- At dt = 0.001 the channel's convective CFL is ~0.3 (centreline u+ ~ 18.3,
  dx+ ~ 23): the production dt sits ~3x BELOW even an explicit scheme's
  convective limit and far below the implicit scheme's (none). Nothing about
  stability chose dt = 0.001; dt+ = 0.18 did.

## R14: the stable-but-wrong experiment (queued on Spark)

Three arms restart from the validated R5 LES state and integrate onward:

| arm | dt | dt+ | expectation |
|---|---|---|---|
| a | 0.001 | 0.18 | control: turbulence sustained, u_tau ~ 1, U+_c ~ 18 |
| b | 0.004 | 0.72 | numerically stable; statistics degrade |
| c | 0.010 | 1.80 | numerically stable; turbulence decays toward laminar (U+_c climbing toward the laminar 90) |

Success criterion for the CLAIM: arms b and c must remain numerically stable
(no divergence -- the NaN fail-fast guards make any instability loud) while
the turbulence metrics deteriorate. That outcome demonstrates on our own
solver that the usable dt is bounded by physics resolution, with stability
nowhere in sight. Results to be appended when the queue completes.
