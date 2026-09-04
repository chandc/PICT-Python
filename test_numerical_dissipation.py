"""How much energy the SCHEME removes, as opposed to viscosity — the LES prerequisite.

WHY THIS GATES EVERYTHING ELSE IN LES. A subgrid model exists to supply the dissipation the grid
cannot resolve. If the discretisation is already removing energy of its own, the model is being
fitted to the scheme's error rather than to the missing physics, and every calibration is
polluted. `run_tgv3d.py` makes this point for the single-block solver; this file measures it for
the MULTI-BLOCK path that the LES actually uses.

THE IDENTITY. For a fully periodic flow,

    dE/dt = -2 nu Z,        E = <|u|^2>/2,   Z = <|omega|^2>/2

exactly. Any gap is numerical.

TWO MEASUREMENT TRAPS, BOTH HIT WHILE WRITING THIS.

  Z is sampled EVERY STEP and integrated. Estimating its window mean as 0.5*(Z_first + Z_last)
  overestimates the mean of a CONVEX function -- TGV enstrophy grows -- and produced a spurious
  extra 2 percentage points of apparent NEGATIVE dissipation. The artefact was in the estimator.

  The remaining gap is still negative, and that is expected rather than alarming: Z is computed
  with a discrete gradient whose truncation error overestimates enstrophy on an under-resolved
  field, so 2 nu Z_discrete exceeds the true dissipation. What matters is that it CONVERGES.

WHAT IS ASSERTED. Not that the gap is zero at any given resolution -- it is not, and could not
be -- but that it falls at second order under refinement. A gap that did not converge would mean
the scheme has a dissipation error independent of resolution, which no amount of grid would fix
and which would make an SGS calibration meaningless.
"""
import numpy as np

from src.domains import periodic_box
from src.piso_multiblock import MultiBlockPISO

NU, DT = 1.0 / 800.0, 0.005
WARM, WINDOW = 20, 40


def tgv(blk):
    """On [0, 2 pi]^3, so E(0) = 1/8 and Z(0) = 3/8 exactly -- both checked below."""
    x, y, z = blk.x, blk.y, blk.z
    return (np.sin(x) * np.cos(y) * np.cos(z),
            -np.cos(x) * np.sin(y) * np.cos(z),
            np.zeros_like(x))


def energy_enstrophy(d, m, nb):
    U = {b: m.u[b] for b in range(nb)}
    V = {b: m.v[b] for b in range(nb)}
    W = {b: m.w[b] for b in range(nb)}
    E = Z = vol = 0.0
    for b in range(nb):
        J = d.block_metrics_cached(b)[0]
        h = d.blocks[b].h
        dV = np.abs(J) * h[0] * h[1] * h[2]
        gu, gv, gw = d.gradient(b, U), d.gradient(b, V), d.gradient(b, W)
        wx, wy, wz = gw[1] - gv[2], gu[2] - gw[0], gv[0] - gu[1]
        E += float(np.sum(0.5 * (m.u[b] ** 2 + m.v[b] ** 2 + m.w[b] ** 2) * dV))
        Z += float(np.sum(0.5 * (wx ** 2 + wy ** 2 + wz ** 2) * dV))
        vol += float(np.sum(dV))
    return E / vol, Z / vol


def run(n, ns):
    d = periodic_box(n, ns, L=2 * np.pi)
    nb = len(d.blocks)
    m = MultiBlockPISO(d, NU, DT, 2, 1e-10, time_scheme="bdf2", scheme="rotational",
                       picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
    for b in range(nb):
        m.u[b][:], m.v[b][:], m.w[b][:] = tgv(d.blocks[b])
    for _ in range(WARM):
        m.step()
    Es, Zs = [], []
    e, z = energy_enstrophy(d, m, nb)
    Es.append(e); Zs.append(z)
    for _ in range(WINDOW):
        m.step()
        e, z = energy_enstrophy(d, m, nb)
        Es.append(e); Zs.append(z)
    dEdt = -(Es[-1] - Es[0]) / (WINDOW * DT)
    phys = 2 * NU * np.trapz(Zs, dx=1.0) / WINDOW
    return dEdt, phys, dEdt - phys


def check_initial_state():
    """E(0) and Z(0) against their exact values -- the check that caught a two-part bug.

    Writing TGV as sin(2 pi x) on a UNIT box gives the same velocities at wavenumber 2 pi
    instead of 1. E(0) still reads 0.125, so an energy check alone passes; the ENSTROPHY is
    (2 pi)^2 = 39.5 times too large, and with it the dissipation. Measured: Z(0) = 14.06.

    Scaling the box exposed a second unit-box assumption inherited from Block, whose `period`
    defaults to (1, 1, 1) and is what pad_coords adds when it wraps a periodic ghost. With the
    coordinates scaled and the period left at 1, the wrapped ghost landed a whole box short: the
    Jacobian came out at 1/(2 pi)^3 and E(0) read 0.147 -- an error that did NOT converge with
    refinement, which is the signature of a geometry bug rather than a discretisation one.
    """
    print("  initial state against exact values, E(0) = 0.125 and Z(0) = 0.375")
    ok = True
    prev = None
    for n in (32, 48, 64):
        d = periodic_box(n, 2, L=2 * np.pi)
        nb = len(d.blocks)
        m = MultiBlockPISO(d, NU, DT, 2, 1e-10, time_scheme="bdf2", scheme="rotational",
                           picard_iters=2, rhie_chow=True, persistent_flux=True, ddt_corr=False)
        for b in range(nb):
            m.u[b][:], m.v[b][:], m.w[b][:] = tgv(d.blocks[b])
        E, Z = energy_enstrophy(d, m, nb)
        eE, eZ = abs(E - 0.125) / 0.125, abs(Z - 0.375) / 0.375
        good = eE < 1e-12 and eZ < 0.06
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {n}^3: E(0) err {100*eE:.3e}%, "
              f"Z(0) err {100*eZ:.3f}%" + (f", falls {prev/eZ:.2f}x" if prev else ""))
        prev = eZ
    return ok


def main():
    print("=" * 78)
    print(f"  numerical dissipation of the multi-block solver, TGV at Re = {1/NU:.0f}")
    print("=" * 78)
    print(f"  {'grid':>8}{'blocks':>8}{'-dE/dt':>12}{'2 nu Z':>12}{'numerical':>12}{'share':>10}")
    grids = ((24, 2), (32, 2), (40, 2))
    share = []
    for n, ns in grids:
        dEdt, phys, num = run(n, ns)
        share.append(abs(num) / dEdt)
        print(f"  {str(n)+'^3':>8}{ns:>8}{dEdt:>12.3e}{phys:>12.3e}{num:>12.3e}"
              f"{100*num/dEdt:>9.1f}%")
    r = [check_initial_state()]
    print()
    # second order: halving the error needs (n2/n1)^2. Compare measured to expected.
    for i in range(len(grids) - 1):
        f = grids[i + 1][0] / grids[i][0]
        got, want = share[i] / share[i + 1], f ** 2
        ok = 0.55 * want <= got <= 1.6 * want
        r.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {grids[i][0]}^3 -> {grids[i+1][0]}^3: error "
              f"falls {got:.2f}x, second order predicts {want:.2f}x")
    small = share[-1] < 0.10
    r.append(small)
    print(f"  [{'PASS' if small else 'FAIL'}] at the finest grid the scheme's own dissipation is "
          f"{100*share[-1]:.1f}% of the total (bar: under 10%)")
    print("=" * 78)
    print(f"  {sum(r)}/{len(r)} checks passed")
    print("=" * 78)
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
