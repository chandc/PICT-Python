"""Why does the 2*dx mode grow in the Re_tau=180 channel? A guarded, attributable probe.

THE QUESTION. A 30-time-unit production run ended with 99.3% of its streamwise fluctuation
energy at exactly 2*dx -- a grid-scale checkerboard wearing the costume of turbulence. Every
integral diagnostic reported health throughout, divergence included, because such a mode can be
exactly divergence-free.

WHAT THIS PROBE IS FOR. Running variants that each remove one candidate cause, and measuring
the Nyquist mode's energy over time. x is simultaneously the PERIODIC direction, the FORCING
direction and the BLOCK-SPLIT direction, and the mode appeared in x alone.

TWO LESSONS FROM EARLIER ATTEMPTS ARE BUILT IN.

  * IT MUST RUN LONG ENOUGH. The mode DECAYS for the first ~2 time units and only then turns
    around; measured growth from t=8 to t=10 was 4.8x in energy, about 0.39 per time unit in
    amplitude. A 0.3-time-unit probe sampled the decay alone and concluded, wrongly, that the
    mode was damped in every configuration.

  * IT MUST BE GUARDED. The single-block control diverged near step 4000 and then spun at 98%
    CPU for 2.6 hours without writing a line -- a Krylov solve grinding on an already-destroyed
    field. The production driver has CFL and grid-mode guards; the first version of this probe
    did not, and lost an afternoon to exactly the failure those guards exist to catch. A
    diagnostic that can hang is not a diagnostic.

THIRD LESSON, 2026-09-06 (see reference/channel_checkerboard_remediation.md). The mode is
measured in VELOCITY, and an A/B that reports only the mode's amplitude can be won by an arm
whose turbulence collapsed: the chorin arm's streamwise fluctuation energy at y+=12 fell 10x
(1495 -> 141) exactly when its 2dx energy reached 5.8e-10. So every report line now also carries
the health of the field (u_tau, per-component fluctuation energy, Nyquist share of p, the
accumulated rotational drift |p - p_flux|, and the wide divergence of the CELL velocity), and
the report is read on the RATIO of mode to turbulence, not on the mode alone.
"""
import argparse
import time

import numpy as np

from src import checkpoint, sgs
from src.domains import channel_box, clustered_y
from src.piso_multiblock import MultiBlockPISO

RE_TAU = 180.0
NU = 1.0 / RE_TAU


def _cat(m, fld, nb):
    return np.concatenate([getattr(m, fld)[b] for b in range(nb)], axis=0)


def nyquist_share(arr, y, yplus=12.0):
    """(share, energy, total) of the streamwise Nyquist bin of `arr` at the plane y+ ~ yplus.

    `arr` is a full (nx, ny, nz) field. The plane mean is removed, the streamwise FFT is taken
    and averaged over z. `total` is the fluctuation energy summed over every non-zero
    wavenumber -- the number that must stay healthy for `share` to mean anything.
    """
    j = int(np.argmin(np.abs(y * RE_TAU - yplus)))
    up = arr[:, j, :] - arr[:, j, :].mean()
    S = (np.abs(np.fft.rfft(up, axis=0)) ** 2).mean(axis=1)
    tot = float(S[1:].sum())
    return (float(S[-1]) / tot if tot > 0 else 0.0), float(S[-1]), tot


def health(m, d, y, nb):
    """Everything an amplitude-only report leaves out."""
    U, V, W, P = (_cat(m, f, nb) for f in ("u", "v", "w", "p"))
    PF = np.concatenate([m.p_flux[b] for b in range(nb)], axis=0)
    # wall stress, both walls, one-sided over the first two cells (channel_stats.u_tau form)
    um = U.mean(axis=(0, 2))
    dudy0 = (um[1] - um[0]) / (y[1] - y[0])
    dudy1 = (um[-2] - um[-1]) / (y[-1] - y[-2])
    u_tau = float(np.sqrt(NU * 0.5 * (abs(dudy0) + abs(dudy1))))
    # per-component fluctuation energy, volume mean (u about its plane mean)
    up = U - um[None, :, None]
    e_u = float(np.mean(up ** 2)); e_v = float(np.mean(V ** 2)); e_w = float(np.mean(W ** 2))
    # rotational-term drift: p - p_flux is a pure running sum of -nu_eff*div(u*)
    drift = float(np.abs(P - PF).max())
    p_range = float(np.ptp(P))
    # Nyquist share of the PRESSURE at the same plane -- the pressure checkerboard family
    p_share, _, _ = nyquist_share(P, y)
    # WIDE divergence of the cell velocity: what the advective-form convection actually sees.
    # The projection makes the face flux solenoidal, not this.
    # physical spacings from the block geometry (x and z are uniform and periodic)
    b0 = d.blocks[0]
    dx = float(b0.x[1, 0, 0] - b0.x[0, 0, 0])
    dz = float(b0.z[0, 0, 1] - b0.z[0, 0, 0])
    dudx = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2 * dx)
    dvdy = np.gradient(V, y, axis=1, edge_order=2)
    dwdz = (np.roll(W, -1, axis=2) - np.roll(W, 1, axis=2)) / (2 * dz)
    divw = dudx + dvdy + dwdz
    div_rms = float(np.sqrt(np.mean(divw[:, 1:-1, :] ** 2)))
    grad_rms = float(np.sqrt(np.mean(dudx[:, 1:-1, :] ** 2)))
    return dict(u_tau=u_tau, e_u=e_u, e_v=e_v, e_w=e_w, drift=drift, p_range=p_range,
                p_share=p_share, div_rms=div_rms, grad_rms=grad_rms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=("rotational", "chorin", "incremental", "norc",
                                       "noforce", "uniform_y", "chorin_uniform"))
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--dt", type=float, default=0.0005)
    p.add_argument("--report", type=int, default=1000)
    p.add_argument("--cfl-limit", type=float, default=0.8)
    p.add_argument("--save-every", type=int, default=0,
                   help="write results/fields/probe_<variant>_<step>.npz every N steps")
    p.add_argument("--start", default="results/fields/chan_re180_t4.npz")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    # UNIFORM-y IS THE SHARPER TEST. The periodic Taylor-Green box runs the SAME rotational
    # scheme and is clean (0.03% Nyquist), so the scheme alone cannot produce the mode. What the
    # channel has and the box does not is walls and wall-normal STRETCHING -- and the stretching
    # varies the spacing 8.5x (dy+ 1.00 at the wall to 8.48 at the centreline), while
    # `pressure_checkerboard.md` records that the Rhie-Chow fix is ORTHOGONAL-ONLY and degrades
    # as skewness grows. Uniform nodes keep the walls and remove the stretching, which is the
    # one-variable version of the question.
    TAG = f"[{(a.tag or a.variant):<14}]"
    stretched = a.variant not in ("uniform_y", "chorin_uniform")
    y = clustered_y(80, re_tau=RE_TAU) if stretched else np.linspace(0.0, 2.0, 80)
    Ly = y[-1]
    # BLOCK COUNT IS FIXED AT 2, and no longer varies. It was the first hypothesis and it is
    # refuted: 1-block and 4-block runs were bit-identical at every reported step. Varying it
    # now would also force hand-restitching of p_flux and F_prev across decompositions -- the
    # staggered face fluxes have a duplicated seam face on axis 0 -- and that hand surgery is
    # what invalidated the earlier probes.
    nblk = 2
    d = channel_box(24, 80, 24, nblk, y_nodes=y)
    nb = len(d.blocks)
    rc = a.variant != "norc"
    scheme = ("chorin" if a.variant in ("chorin", "chorin_uniform")
              else "incremental" if a.variant == "incremental" else "rotational")
    m = MultiBlockPISO(d, NU, a.dt, 2, 1e-8, time_scheme="bdf2", scheme=scheme,
                       picard_iters=2, rhie_chow=rc, persistent_flux=True, ddt_corr=False)
    m.velocity_source = [0.0 if a.variant == "noforce" else 1.0, 0.0, 0.0]
    m.set_nu({b: np.full_like(m.u[b], NU) for b in range(nb)})

    # FULL restore, p_flux and F_prev included -- see the module docstring. Restitching fields
    # by hand is what made the earlier probes diverge at t ~ 2 while the production run reached
    # t = 28 from the same field.
    if stretched:
        checkpoint.load(m, a.start, allow=("dt", "scheme", "rhie_chow"))
    else:
        # The saved state lives on the STRETCHED mesh, so here it is interpolated in y. p_flux
        # and F_prev are mesh-specific and are NOT carried across; this variant therefore starts
        # without them, which is a real difference from the stretched runs and is why the
        # comparison is read on the TREND after the transient rather than the first time unit.
        f, _ = checkpoint.load_fields(a.start)
        y_src = clustered_y(80, re_tau=RE_TAU)
        for fld, arr in (("u", m.u), ("v", m.v), ("w", m.w), ("p", m.p)):
            for b in range(nb):
                src = f[fld][b]
                out = np.empty_like(arr[b])
                for i in range(src.shape[0]):
                    for k in range(src.shape[2]):
                        out[i, :, k] = np.interp(y, y_src, src[i, :, k])
                arr[b][:] = out
        print(f"  {TAG} uniform-y: interpolated from the stretched state; p_flux/F_prev not "
              f"carried (mesh-specific)", flush=True)
    # A scheme change is a change of what p MEANS: under rotational the saved p carries the
    # accumulated -nu*div(u*) that the flux never saw. incremental/chorin must start from the
    # projection pressure, or the first predictor feels a gradient no scheme of theirs produced.
    # ... and the saved p is itself contaminated: the production run that wrote this checkpoint
    # ran with the Picard-loop p_flux bug (see MultiBlockPISO._step_impl), so |p - p_flux| = 6.3
    # at t=4 against a rotational contribution of ~1e-2. Every arm therefore restarts from
    # p = p_flux; the rotational arm rebuilds its -nu*div(u*) sum from there.
    for b in range(nb):
        m.p[b][:] = m.p_flux[b]
    for arr, bc in ((m.u, m.u_bc), (m.v, m.v_bc), (m.w, m.w_bc)):
        for b in range(nb):
            arr[b][:, 0, :] = arr[b][:, -1, :] = 0.0
            bc[b][:, 0, :] = bc[b][:, -1, :] = 0.0

    damp = {b: np.minimum(blk.y, Ly - blk.y) / NU for b, blk in enumerate(d.blocks)}
    dymin = float(np.diff(y).min())
    frac0, e0, tot0 = nyquist_share(_cat(m, "u", nb), y)
    h0 = health(m, d, y, nb)
    print(f"  {TAG} blocks={nb} scheme={scheme} rhie_chow={rc} force={m.velocity_source[0]} "
          f"dt={a.dt} steps={a.steps} start={a.start}", flush=True)
    print(f"  {TAG} start: 2dx share {100*frac0:.3f}%, energy {e0:.4e}, total {tot0:.3e}  "
          f"u_tau {h0['u_tau']:.4f}  e_u/v/w {h0['e_u']:.3f}/{h0['e_v']:.4f}/{h0['e_w']:.4f}  "
          f"|p-pflux| {h0['drift']:.3e} (p range {h0['p_range']:.3e})  p2dx {100*h0['p_share']:.3f}%  "
          f"divw/|du/dx| {h0['div_rms']:.3e}/{h0['grad_rms']:.3e}", flush=True)

    t0 = time.time()
    for i in range(1, a.steps + 1):
        U = {b: m.u[b] for b in range(nb)}
        V = {b: m.v[b] for b in range(nb)}
        W = {b: m.w[b] for b in range(nb)}
        nu_eff, _ = sgs.effective_viscosity(d, U, V, W, NU, model="smagorinsky", damping=damp)
        m.set_nu(nu_eff)
        m.step()

        if i % 100 == 0:
            # GUARDS. Without these the probe hangs instead of failing, which is how the
            # previous attempt burned 2.6 hours at 98% CPU on a destroyed field.
            umax = max(float(np.abs(m.u[b]).max()) for b in range(nb))
            vmax = max(float(np.abs(m.v[b]).max()) for b in range(nb))
            if not np.isfinite(umax) or not np.isfinite(vmax):
                print(f"  {TAG} ABORT step {i}: field non-finite", flush=True)
                return 1
            cy = vmax * a.dt / dymin
            if cy > a.cfl_limit:
                print(f"  {TAG} ABORT step {i} t={m.time:.3f}: CFL_y {cy:.2f} exceeds "
                      f"{a.cfl_limit} (u_max {umax:.2f}, v_max {vmax:.2f})", flush=True)
                return 1

        if i % a.report == 0:
            frac, e, tot = nyquist_share(_cat(m, "u", nb), y)
            h = health(m, d, y, nb)
            print(f"  {TAG} step {i:>6} t={m.time:6.2f}  2dx share {100*frac:7.3f}%  "
                  f"energy {e:.4e}  amp {e/max(e0,1e-30):8.3f}x  total {tot:.3e}  "
                  f"u_tau {h['u_tau']:.4f}  e_u/v/w {h['e_u']:.3f}/{h['e_v']:.4f}/{h['e_w']:.4f}  "
                  f"|p-pflux| {h['drift']:.3e}  p2dx {100*h['p_share']:.3f}%  "
                  f"divw {h['div_rms']:.3e}  "
                  f"{(time.time()-t0)/i:.3f} s/step", flush=True)
        if a.save_every and i % a.save_every == 0:
            checkpoint.save(m, f"results/fields/probe_{a.tag or a.variant}_{i:06d}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
