"""Gate 0 of the PETSc port: reference trajectories and a profile, captured before any change.

WHY THIS EXISTS. Gate 1's success criterion is not "close" -- it is **bitwise identical** to
these files. That criterion only means anything if the reference is captured from unmodified
serial code, on the machine the comparison will run on, with the grid recorded alongside so a
later run cannot silently compare against a different mesh.

WHAT IS CAPTURED, per case:
  * the full field after each of 10 steps, so a divergence can be located in TIME rather than
    only detected at the end -- a port that drifts at step 7 and a port that is wrong at step 1
    need different debugging, and a final-state-only reference cannot tell them apart;
  * scalar reductions per step (kinetic energy, max divergence, max |u|), which are what a
    distributed run gets wrong first, because they are where MPI reduction order enters;
  * the grid fingerprint, so `src/checkpoint.py` refuses a mismatched mesh rather than
    producing a plausible wrong answer;
  * a per-step split of solve time against assembly time. Gate 6 has to decide whether the port
    is worth continuing, and that decision needs to know what fraction of the runtime is even
    addressable: if assembly dominates, distributing the SOLVE cannot deliver the speed-up the
    plan projects, and that is better known now than after Gates 1-5 are paid for.

THE REDUCTIONS ARE STORED IN FULL PRECISION as raw float64 bytes, not printed and re-parsed.
Bitwise comparison against a decimal round-trip is not bitwise comparison.

WHAT GOES INTO GIT IS THE DIGEST FILE, NOT THE 35 MB OF FIELDS. That is safe for one specific
reason, established rather than hoped: this capture is bitwise reproducible. Two independent
runs of unmodified serial code produced 640/640 identical state arrays and identical reductions,
and adding the timing instrumentation left all 640 identical again. So the .npz is derived data
that can be RECREATED EXACTLY by re-running this script, and a per-array BLAKE2 digest is
exactly as strong a bitwise test as the arrays themselves while being four orders of magnitude
smaller. Had reproducibility failed, the fields would have had to be committed -- and Gate 1's
criterion would have been unachievable anyway.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from src import checkpoint


def capture(name, build, nsteps=10, out_dir="reference/gate0"):
    """Run `nsteps` from a built solver, recording state and per-step reductions."""
    os.makedirs(out_dir, exist_ok=True)
    d, m = build()
    nb = len(d.blocks)
    fp = checkpoint.grid_fingerprint(m)
    rows, t_solve, t_total = [], 0.0, 0.0
    states = {}
    for i in range(nsteps):
        t0 = time.perf_counter()
        m.step()
        dt_step = time.perf_counter() - t0
        t_total += dt_step
        # `t_solve` is CUMULATIVE, so the per-step cost is the difference. Recording the
        # cumulative value per step would make every step after the first look progressively
        # more solve-bound, which is exactly the wrong signal for the Gate 6 decision.
        ts = float(getattr(m, "t_solve", float("nan")))
        dt_solve = ts - t_solve
        t_solve = ts
        ke = sum(float(np.sum(m.u[b] ** 2 + m.v[b] ** 2 + m.w[b] ** 2)) for b in range(nb))
        umax = max(float(np.abs(m.u[b]).max()) for b in range(nb))
        div = float(m.interior_divergence())
        rows.append((i + 1, m.time, ke, umax, div, dt_step, dt_solve))
        for b in range(nb):
            states[f"s{i+1}_u_{b}"] = m.u[b].copy()
            states[f"s{i+1}_v_{b}"] = m.v[b].copy()
            states[f"s{i+1}_w_{b}"] = m.w[b].copy()
            states[f"s{i+1}_p_{b}"] = m.p[b].copy()
    # per-array digest, the thing that actually goes into git
    digests = {k: hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).hexdigest()
               for k, v in states.items()}
    red = np.array([r[1:5] for r in rows], dtype=np.float64)
    digests["__reductions__"] = hashlib.blake2b(red.tobytes(), digest_size=16).hexdigest()
    with open(f"{out_dir}/{name}.digests.json", "w") as fh:
        json.dump(digests, fh, indent=0, sort_keys=True)

    path = f"{out_dir}/{name}.npz"
    np.savez_compressed(path, reductions=np.array([r[1:] for r in rows], dtype=np.float64),
                        grid=np.array(fp if fp is not None else ""), nblocks=np.array(nb),
                        ncells=np.array(d.n_cells), nsteps=np.array(nsteps), **states)
    frac = t_solve / t_total if t_total > 0 else float("nan")
    meta = dict(name=name, nblocks=nb, ncells=int(d.n_cells), nsteps=nsteps,
                grid_fingerprint=fp, seconds_per_step=t_total / nsteps,
                total_seconds=t_total, solve_seconds=t_solve,
                reductions_digest=digests["__reductions__"],
                assembly_seconds=t_total - t_solve, solve_fraction=frac,
                n_solves=int(getattr(m, "n_solve", 0)))
    print(f"  {name:<16} {d.n_cells:>9,} cells  {nb:>3} blocks  "
          f"{t_total/nsteps:>7.3f} s/step   div {rows[-1][4]:.2e}")
    print(f"  {'':16} solve {t_solve:.2f}s ({100*frac:.1f}%)  assembly "
          f"{t_total-t_solve:.2f}s ({100*(1-frac):.1f}%)  over "
          f"{getattr(m, 'n_solve', 0)} solves")
    print(f"  {'':16} fingerprint {str(fp)[:16]}...  -> {path}")
    print(f"  {'':16} {len(digests)} digests -> {out_dir}/{name}.digests.json")
    return meta


def _cylinder():
    from cylinder_grid import cylinder_domain
    from src.piso_multiblock import MultiBlockPISO
    d, _, _ = cylinder_domain(nz=4)   # the checkpoint is nz = 4; the builder defaults to 8
    m = MultiBlockPISO(d, 1.0 / 100.0, 0.005, 2, 1e-6, time_scheme="bdf2",
                       scheme="rotational", picard_iters=2, rhie_chow=True,
                       persistent_flux=True, ddt_corr=False)
    # PIN THE MOMENTUM TOLERANCE, matching test_mpi_equivalence. The reference and the test
    # that consumes it must build the SAME solver, and they had drifted: Gate 4 tightened
    # momentum_tol to 1e-14 to stop partition-dependent iteration paths diverging, the test was
    # pinned back to `tol` so it could still compare against the original Gate 0 digests, and
    # this generator was left on the default. A reference captured with different settings from
    # the run being checked reports a configuration difference as a regression -- which is
    # exactly what "0/640 digests" meant here, twice, on two machines.
    m.momentum_tol = m.tol
    checkpoint.load(m, "results/fields/cyl_shed_mac.npz")
    return d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="reference/gate0")
    a = ap.parse_args()
    print("=" * 78)
    print("  Gate 0 — reference trajectories, captured from unmodified serial code")
    print("=" * 78)
    metas = [capture("cylinder_re100", _cylinder, a.steps, a.out)]
    with open(f"{a.out}/manifest.json", "w") as fh:
        json.dump(dict(cases=metas, host=os.uname().nodename,
                       captured=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), fh,
                  indent=2)
    print("=" * 78)
    print(f"  wrote {a.out}/manifest.json")


if __name__ == "__main__":
    main()
