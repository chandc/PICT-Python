"""Turbulent channel LES at Re_tau = 180, minimal box, against a reference DNS.

THE FIRST WALL-BOUNDED LES IN THIS REPO. Everything before it -- Taylor-Green at Re = 400 and
800 -- was a periodic box with no walls, no wall model and no sustained turbulence. Walls are
where LES is hardest and where most practical LES lives, so this is the case that tests what the
TGV study could not.

THE SETUP is the reference run's, exactly, so the comparison needs no translation:

    delta = u_tau = 1,   nu = 1/Re_tau = 1/180,   f_x = u_tau^2/delta = 1
    y in [0, 2],  Lx = pi (Lx+ = 565),  Lz = 0.34 pi (Lz+ = 192)

The box is DELIBERATELY ABOVE the Jimenez-Moin minimal unit (Lz+ ~ 100), not on it: at the
threshold a relaminarisation is indistinguishable from a solver bug.

THE FORCING IS A CONSTANT PRESSURE GRADIENT, which fixes u_tau = 1 by construction and makes
Re_tau = 180 exact rather than an outcome to be measured and hoped for. The measured wall
gradient then becomes a CHECK on the solution instead of a definition of it.

THE INITIAL CONDITION IS A TURBULENT DNS FIELD, interpolated. Plane Poiseuille is linearly
stable here, so transition must be bypassed, and a minimal box relaminarises easily enough that
a synthetic trip risks a failure that looks like a bug. See channel_ic.py.

THE DEFAULT MODEL IS SMAGORINSKY WITH VAN DRIEST DAMPING. Smagorinsky earned the default on the
Taylor-Green evidence in `les_model_study.md` -- at 48^3 it was the ONLY model that survived --
but that case was isotropic and had no walls, and undamped Smagorinsky is measurably wrong at
one. Measured on the interpolated DNS field, nu_t/nu averages 0.99 across the viscous sublayer
(y+ < 5) and peaks at 1.04 at y+ = 5: the model DOUBLES the viscosity exactly where the flow is
laminar and it should contribute nothing. WALE and sigma give 0.0002 and 0.0005 there.

The damping factor 1 - exp(-y+/A+) with A+ = 26 removes that. y+ is built from the wall distance
min(y, Ly - y) and u_tau = 1, which the constant-pressure-gradient forcing FIXES rather than
leaves to be measured -- so the damping needs no iteration on a friction velocity that is still
converging.

WHAT DAMPING DOES NOT DO, recorded because this repo got it wrong once: van Driest applied to
the mixing length gives nu_t ~ y^4 near the wall, not the correct y^3 (measured 3.993 in
`test_sgs_models.py`). So "van Driest fixes the near-wall scaling" is false as usually stated.
It fixes the MAGNITUDE -- the sublayer over-dissipation -- while WALE and sigma get the exponent
right by construction. `--model wale` and `--model sigma` remain the comparison.

A SECOND REASON DAMPING MATTERS HERE is that undamped Smagorinsky breaks the diagnostic, not
just the physics. `ChannelStats.u_tau` reads sqrt(nu du/dy) with the MOLECULAR nu; with
nu_t/nu = 0.92 at the wall the true stress is nearly double that, so the reported u_tau cannot
be compared against the 1.0 that CPG fixes, and equilibrium becomes unmeasurable. Damped, or
with WALE/sigma, nu_t(wall) -> 0 and the number means what it says.
"""
import argparse
import os
import time

import numpy as np

from channel_ic import interpolate_to
from src import checkpoint
from src import sgs
from src.channel_stats import ChannelStats
from src.domains import channel_box, clustered_y
from src.piso_multiblock import MultiBlockPISO

RE_TAU = 180.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=80)
    p.add_argument("--nz", type=int, default=24)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--model", choices=("none",) + tuple(sorted(sgs.MODELS)),
                   default="smagorinsky")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--t-end", type=float, default=30.0)
    p.add_argument("--t-stats", type=float, default=10.0,
                   help="start accumulating statistics after this much time")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--tag", default=None)
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="save a restartable field this often (0 disables)")
    p.add_argument("--restart", default=None,
                   help="resume from this checkpoint instead of interpolating the DNS field")
    p.add_argument("--cfl-limit", type=float, default=0.8,
                   help="abort if the wall-normal CFL exceeds this (it tightens as the flow "
                        "spins up; a start-up estimate cannot police it)")
    p.add_argument("--no-damping", action="store_true",
                   help="disable van Driest damping (Smagorinsky only); for the comparison run")
    a = p.parse_args()

    nu = 1.0 / RE_TAU
    tag = a.tag or f"chan_n{a.nx}x{a.ny}x{a.nz}_{a.model}"
    y = clustered_y(a.ny, re_tau=RE_TAU)
    Ly = y[-1]
    d = channel_box(a.nx, a.ny, a.nz, a.blocks, y_nodes=y)
    nb = len(d.blocks)

    m = MultiBlockPISO(d, nu, a.dt, 2, a.tol, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    # CONSTANT PRESSURE GRADIENT. `src` is an acceleration -- the solver multiplies it by the
    # Jacobian -- so f_x = u_tau^2/delta = 1 goes in directly.
    m.velocity_source = [1.0, 0.0, 0.0]

    # VAN DRIEST y+, from the wall distance and the u_tau the FORCING fixes at 1. Only
    # Smagorinsky takes it -- WALE and sigma vanish at a wall on their own and would be damped
    # twice.
    damp = None
    if a.model == "smagorinsky" and not a.no_damping:
        damp = {b: np.minimum(blk.y, Ly - blk.y) * 1.0 / nu
                for b, blk in enumerate(d.blocks)}

    # RESTART BEFORE INTERPOLATION. A 5.5-hour run that cannot resume is one power cut from
    # being repeated, and the DNS interpolation is the expensive part of startup (8.5 s) as
    # well as the part whose result must not silently change between segments.
    if a.restart:
        # THE CHECKPOINT RECORDS nu AS A FIELD, because the SGS model sets one every step; a
        # freshly built solver has a scalar, and the config check rightly refuses the mismatch.
        # Seeding a uniform field first makes the two agree WITHOUT resorting to strict=False,
        # which would also drop the grid-fingerprint guard -- the one check that stops a restart
        # onto the wrong mesh from producing a plausible wrong answer. The value seeded here is
        # irrelevant: the model overwrites it on the first step.
        if a.model != "none":
            m.set_nu({b: np.full_like(m.u[b], nu) for b in range(nb)})
        checkpoint.load(m, a.restart)
        print(f"  restarted from {a.restart}: t = {m.time:.3f}, step {m.nstep}", flush=True)
    else:
        uvw = interpolate_to(d)
        for b in range(nb):
            m.u[b][:], m.v[b][:], m.w[b][:] = uvw[b]
    for b in range(nb):                       # no-slip, enforced on the wall planes
        for arr, bc in ((m.u, m.u_bc), (m.v, m.v_bc), (m.w, m.w_bc)):
            arr[b][:, 0, :] = arr[b][:, -1, :] = 0.0
            bc[b][:, 0, :] = bc[b][:, -1, :] = 0.0

    dyp = np.diff(y) * RE_TAU
    print(f"  channel Re_tau = {RE_TAU:.0f}, {a.nx}x{a.ny}x{a.nz} in {nb} blocks "
          f"= {d.n_cells:,} cells, model = {a.model}"
          f"{' + van Driest' if damp is not None else ''}", flush=True)
    print(f"  dx+ {np.pi/a.nx*RE_TAU:.1f}   dz+ {0.34*np.pi/a.nz*RE_TAU:.1f}   "
          f"dy+ wall {dyp[0]:.2f} max {dyp.max():.2f}   {int((y*RE_TAU<10).sum())} points "
          f"under y+ = 10", flush=True)
    # CFL FROM THE ACTUAL FIELD, and re-checked every step. A hardcoded v_max = 3.0 sat here
    # and destroyed a 2.5-hour run: it matched the initial field, printed a comfortable
    # CFL_y = 0.54, and then went stale. v_max grew 2.8 -> 4.6 as the constant-pressure-gradient
    # forcing spun the flow up, CFL_y crossed 1 just after t = 4.95, and the solution exploded
    # from v_max 4.65 to 158 in fifty steps. The wall cell is dy+ = 1, i.e. 0.0056, so the
    # wall-normal CFL is the binding constraint here and it TIGHTENS as the run develops --
    # exactly the kind of limit a start-up estimate cannot police.
    dymin = float(np.diff(y).min())

    def cfl(m):
        u = max(float(np.abs(m.u[b]).max()) for b in range(nb))
        v = max(float(np.abs(m.v[b]).max()) for b in range(nb))
        w = max(float(np.abs(m.w[b]).max()) for b in range(nb))
        return (u * a.dt / (np.pi / a.nx), v * a.dt / dymin,
                w * a.dt / (0.34 * np.pi / a.nz))

    cx, cy, cz = cfl(m)
    print(f"  dt = {a.dt}:  CFL_x {cx:.2f}, CFL_y {cy:.2f}, CFL_z {cz:.2f}  "
          f"(dy_wall {dymin:.5f})", flush=True)
    if cy > a.cfl_limit:
        raise SystemExit(f"  ABORT before starting: CFL_y {cy:.2f} already exceeds "
                         f"{a.cfl_limit}. Reduce --dt to about "
                         f"{a.dt * a.cfl_limit / cy:.2e}.")
    print(f"  {'step':>8}{'t':>8}{'u_tau':>9}{'U+_c':>8}{'u_b+':>8}{'nu_t/nu':>9}"
          f"{'div':>10}{'s/step':>9}", flush=True)

    stats = ChannelStats(d)
    nsteps = int(round(a.t_end / a.dt))
    t0 = time.time()
    ratio = 0.0
    for i in range(1, nsteps + 1):
        if a.model != "none":
            U = {b: m.u[b] for b in range(nb)}
            V = {b: m.v[b] for b in range(nb)}
            W = {b: m.w[b] for b in range(nb)}
            kw = {"damping": damp} if damp is not None else {}
            nu_eff, nu_t = sgs.effective_viscosity(d, U, V, W, nu, model=a.model, **kw)
            worst = min(float(nu_eff[b].min()) for b in range(nb))
            if worst <= 0.0:
                raise SystemExit(f"  ABORT: nu_eff non-positive ({worst:.3e}) at step {i}")
            m.set_nu(nu_eff)
            ratio = float(np.mean([float(nu_t[b].mean()) for b in range(nb)])) / nu
        m.step()
        if i % 25 == 0:
            cx, cy, cz = cfl(m)
            if not (cy < a.cfl_limit and cx < 1.0 and cz < 1.0):
                checkpoint.save(m, f"results/fields/{tag}_CFL_ABORT.npz")
                raise SystemExit(
                    f"  ABORT at step {i}, t = {m.time:.3f}: CFL_x {cx:.2f} CFL_y {cy:.2f} "
                    f"CFL_z {cz:.2f} against limit {a.cfl_limit}. Field saved to "
                    f"results/fields/{tag}_CFL_ABORT.npz. Restart with a smaller --dt: "
                    f"about {a.dt * a.cfl_limit / max(cy, 1e-30):.2e}.")
        if m.time >= a.t_stats:
            stats.add(m)
        if i % 500 == 0:
            yv, U, up, vp, wp, uv = stats.profiles() if stats.nsamp else (y, None, None, None,
                                                                          None, None)
            ut = stats.u_tau(nu) if stats.nsamp else float("nan")
            uc = U[-1] if U is not None else float("nan")
            ub = float(np.trapz(U, yv) / (yv[-1] - yv[0])) if U is not None else float("nan")
            print(f"  {i:>8}{m.time:>8.2f}{ut:>9.4f}{uc:>8.2f}{ub:>8.2f}{ratio:>9.3f}"
                  f"{m.interior_divergence():>10.1e}{(time.time()-t0)/i:>9.3f}", flush=True)
            os.makedirs("results/fields", exist_ok=True)
            if stats.nsamp:
                stats.save(f"results/{tag}_stats.npz", nu)
        if a.checkpoint_every and i % a.checkpoint_every == 0:
            os.makedirs("results/fields", exist_ok=True)
            checkpoint.save(m, f"results/fields/{tag}.npz")
    if stats.nsamp:
        stats.save(f"results/{tag}_stats.npz", nu)
        print(f"\n  {stats.nsamp} samples over t = {stats.t0:.2f}-{stats.t1:.2f}", flush=True)
        print(f"  saved results/{tag}_stats.npz", flush=True)


if __name__ == "__main__":
    main()
