"""
Checkpoint / restart for the PISO solvers, single-block and multi-block alike.

A restart is only worth having if it is EXACT: continuing from a checkpoint must reproduce, to
round-off, what an uninterrupted run would have produced. Two things make that true, and both
are easy to get wrong:

  * u_prev, the BDF2 history, must be saved. Without it the first step after a restart falls
    back to backward Euler -- the run continues, looks stable, and is quietly first-order
    accurate across the seam. Nothing in the output announces this.
  * The pressure must be saved. The incremental and rotational schemes carry p forward as a
    running total; a restart from p = 0 re-derives it over several steps and perturbs the
    velocity meanwhile.
  * With Rhie-Chow enabled there are two MORE pieces of running state: p_flux, the projection
    pressure the face flux actually carries (it differs from p under the rotational scheme),
    and F_prev, the previous step's face flux that ddt_corr re-injects. Neither is
    reconstructible from u and p, so a restart without them silently loses the checkerboard
    damping for a step and then limps back.

Restarting into a solver configured differently from the one that wrote the file (different nu,
dt, time scheme, or grid) is refused rather than silently accepted, because the resulting run
would be neither the old case nor a clean new one. Pass strict=False to override deliberately.
The grid is checked by a fingerprint of the node coordinates, because block count and field
shapes are not a grid: rotating the cylinder O-grid by half a cell changes every coordinate and
no shape. Checkpoints written before the fingerprint existed carry none and still load.

Fields are stored flat with a per-block prefix, so one .npz serves both solver types and can be
read for post-processing without constructing a solver at all -- see load_fields().
"""
import hashlib

import numpy as np

FORMAT = 3                      # bump when the on-disk layout changes incompatibly
FIELDS = ("u", "v", "w", "p")
# Config that changes the meaning of the state. A restart that disagrees on any of these is
# not a continuation of the same simulation.
CONFIG = ("nu", "dt", "time_scheme", "scheme", "picard_iters", "corrector_steps",
          "implicit_cross", "rhie_chow", "persistent_flux", "ddt_corr")


def grid_fingerprint(solver):
    """Hash of the node coordinates, or None if the solver exposes no domain.

    SHAPES ARE NOT A GRID. `load` checked block count and field shapes, and the module docstring
    claimed a grid change was refused; it was not. Rotating the cylinder O-grid by half a cell
    changes every node coordinate and no shape at all, so a checkpoint written on the old grid
    loaded silently onto the new one and would have been restarted as though it were a
    continuation. This closes that hole for anything carrying a domain.

    Old checkpoints have no fingerprint and are still readable -- absence is not a mismatch.

    QUANTISED BEFORE HASHING, because a hash of raw float64 bytes is not a fingerprint of the
    GRID, it is a fingerprint of the grid AND the machine that built it. The identical
    cylinder_grid.py gives coordinates differing by 3.553e-15 -- one ULP at r = 20 -- between
    numpy 2.0.2 on the Mac and 2.4.6 on the GB10, which flipped every bit of the digest and
    refused a checkpoint that was perfectly valid. That blocks moving a run between machines,
    which is the whole point of writing one.

    So the coordinates are scaled by the domain extent and rounded to 12 decimals before
    hashing: 12 significant figures, three orders finer than the round-off being tolerated and
    nine orders coarser than any grid change anyone would make on purpose. The half-cell
    rotation this function was written to catch moves nodes by 0.0123 in a domain of 20, which
    is 1e9 times the quantum.
    """
    d = getattr(solver, "d", None)
    blocks = getattr(d, "blocks", None)
    if not blocks:
        return None
    scale = 0.0
    for blk in blocks:
        for axis in ("x", "y", "z"):
            a = np.asarray(getattr(blk, axis), dtype=np.float64)
            if a.size:
                scale = max(scale, float(np.abs(a).max()))
    scale = scale or 1.0
    h = hashlib.blake2b(digest_size=16)
    for blk in blocks:
        for axis in ("x", "y", "z"):
            a = np.ascontiguousarray(getattr(blk, axis), dtype=np.float64) / scale
            h.update(np.ascontiguousarray(np.round(a, 12)).tobytes())
    return h.hexdigest()


def _legacy_grid_fingerprint(solver):
    """The pre-quantisation digest: raw float64 bytes, machine-dependent.

    Kept only so checkpoints written before the quantised fingerprint still verify on the
    machine that wrote them. New files carry the quantised digest; `load` accepts either.
    """
    d = getattr(solver, "d", None)
    blocks = getattr(d, "blocks", None)
    if not blocks:
        return None
    h = hashlib.blake2b(digest_size=16)
    for blk in blocks:
        for axis in ("x", "y", "z"):
            h.update(np.ascontiguousarray(getattr(blk, axis), dtype=np.float64).tobytes())
    return h.hexdigest()


def _config_repr(v):
    """A savable, comparable descriptor for a config value.

    `nu` MAY BE A FIELD -- a dict of per-block arrays -- for an eddy-viscosity closure. Storing
    that with np.array() produces an OBJECT array, which saves without complaint and then fails
    to load with `allow_pickle=False`:

        ValueError: Object arrays cannot be loaded when allow_pickle=False

    So an LES would checkpoint happily for hours and be unrestartable, with nothing wrong until
    the moment it was needed. That is the worst shape a bug can have on a long run.

    Only the KIND is recorded, not the values. An eddy viscosity is recomputed from the velocity
    at every step, so demanding that the restored field match the saved one would refuse every
    legitimate restart. What the check must still catch is a solver built with a SCALAR nu
    resuming a run that used a field, or vice versa -- a genuine configuration change that would
    silently produce a different simulation.

    The field itself is not restored. For an LES it does not need to be; for a prescribed field
    such as a sponge, the caller sets it before or after loading, exactly as it was set before
    the first step.
    """
    if isinstance(v, dict):
        return f"field[{len(v)}]"
    if isinstance(v, (list, tuple)):
        return f"field[{len(v)}]"
    return v


def _is_multiblock(s):
    return isinstance(getattr(s, "u", None), dict)


def _blocks(s):
    """Field state as {name: {block: array}}, single-block presented as block 0."""
    if _is_multiblock(s):
        return {f: dict(getattr(s, f)) for f in FIELDS}
    return {f: {0: getattr(s, f)} for f in FIELDS}


def save(solver, path, **extra):
    """Write the full restart state. `extra` is stored alongside for post-processing."""
    out = {"__format__": np.array(FORMAT),
           "nstep": np.array(getattr(solver, "nstep", 0)),
           "time": np.array(getattr(solver, "time", 0.0)),
           "multiblock": np.array(_is_multiblock(solver))}
    for k in CONFIG:
        if hasattr(solver, k):
            out[f"cfg_{k}"] = np.array(_config_repr(getattr(solver, k)))

    fp = grid_fingerprint(solver)
    if fp is not None:
        out["grid"] = np.array(fp)

    st = _blocks(solver)
    nb = len(st["u"])
    out["nblocks"] = np.array(nb)
    for f in FIELDS:
        for b, arr in st[f].items():
            out[f"{f}_{b}"] = np.asarray(arr)

    # BDF2 history -- absent only before the first step has been taken
    prev = getattr(solver, "u_prev", None)
    out["has_prev"] = np.array(prev is not None)
    if prev is not None:
        for f, part in zip(("u", "v", "w"), prev):
            part = part if isinstance(part, dict) else {0: part}
            for b, arr in part.items():
                out[f"prev{f}_{b}"] = np.asarray(arr)

    # Rhie-Chow running state (absent unless the option is on)
    pf = getattr(solver, "p_flux", None)
    if pf is not None:
        part = pf if isinstance(pf, dict) else {0: pf}
        for b, arr in part.items():
            out[f"pflux_{b}"] = np.asarray(arr)
    Fp = getattr(solver, "F_prev", None)
    out["has_Fprev"] = np.array(Fp is not None)
    if Fp is not None:
        per_block = Fp if isinstance(Fp, dict) else {0: Fp}
        for b, axes in per_block.items():
            for a, arr in enumerate(axes):
                out[f"Fprev_{b}_{a}"] = np.asarray(arr)

    for k, v in extra.items():
        out[f"x_{k}"] = np.asarray(v)
    np.savez_compressed(path, **out)
    return path


def load_fields(path):
    """Read a checkpoint for post-processing, without needing a solver.

    Returns (fields, meta) where fields is {name: {block: array}} and meta carries nstep,
    time, the saved config and any extras.
    """
    d = np.load(path, allow_pickle=False)
    fmt = int(d["__format__"])
    if fmt != FORMAT:
        raise ValueError(f"checkpoint format {fmt}, this build reads {FORMAT}")
    nb = int(d["nblocks"])
    fields = {f: {b: d[f"{f}_{b}"] for b in range(nb)} for f in FIELDS}
    meta = {"nstep": int(d["nstep"]), "time": float(d["time"]),
            "multiblock": bool(d["multiblock"]), "nblocks": nb,
            "config": {k: d[f"cfg_{k}"].item() for k in CONFIG if f"cfg_{k}" in d},
            "grid": (str(d["grid"]) if "grid" in d.files else None),
            "extra": {k[2:]: d[k] for k in d.files if k.startswith("x_")}}
    return fields, meta


def load(solver, path, strict=True):
    """Restore a checkpoint into `solver`, which must already be built on the same grid."""
    fields, meta = load_fields(path)

    if meta["multiblock"] != _is_multiblock(solver):
        raise ValueError("checkpoint/solver disagree on multi-block vs single-block")
    nb_solver = len(_blocks(solver)["u"])
    if meta["nblocks"] != nb_solver:
        raise ValueError(f"checkpoint has {meta['nblocks']} block(s), "
                         f"solver has {nb_solver}")
    for f in FIELDS:
        for b, arr in fields[f].items():
            want = _blocks(solver)[f][b].shape
            if arr.shape != want:
                raise ValueError(f"block {b} field {f}: checkpoint {arr.shape} "
                                 f"vs solver {want}")
    if strict and meta["grid"] is not None:
        now = grid_fingerprint(solver)
        # Either digest is acceptable: the quantised one for files written since the fix, the
        # raw-bytes one for files written before it AND read back on the same machine. A file
        # carrying only the legacy digest still cannot cross machines, which is correct -- it
        # genuinely does not record enough to tell round-off from a real change.
        ok = {now, _legacy_grid_fingerprint(solver)} - {None}
        if now is not None and meta["grid"] not in ok:
            raise ValueError(
                f"checkpoint was written on a DIFFERENT grid (fingerprint {meta['grid'][:12]}, "
                f"solver {now[:12]}). Block count and field shapes match, so nothing else here "
                f"would have caught it; pass strict=False if the change is intended.")
    if strict:
        bad = {k: (v, _config_repr(getattr(solver, k))) for k, v in meta["config"].items()
               if hasattr(solver, k) and _config_repr(getattr(solver, k)) != v}
        if bad:
            detail = ", ".join(f"{k}: file={a!r} solver={b!r}" for k, (a, b) in bad.items())
            raise ValueError(
                f"solver is configured differently from the checkpoint ({detail}). "
                "This would be neither a continuation nor a clean new run; "
                "pass strict=False if the change is intended.")

    mb = meta["multiblock"]
    for f in FIELDS:
        if mb:
            getattr(solver, f).update({b: a.copy() for b, a in fields[f].items()})
        else:
            setattr(solver, f, fields[f][0].copy())

    d = np.load(path, allow_pickle=False)
    if bool(d["has_prev"]):
        parts = []
        for f in ("u", "v", "w"):
            per = {b: d[f"prev{f}_{b}"].copy() for b in range(meta["nblocks"])}
            parts.append(per if mb else per[0])
        solver.u_prev = tuple(parts)
    else:
        solver.u_prev = None

    if f"pflux_0" in d.files:
        per = {b: d[f"pflux_{b}"].copy() for b in range(meta["nblocks"])
               if f"pflux_{b}" in d.files}
        solver.p_flux = per if mb else per[0]
    if bool(d["has_Fprev"]) if "has_Fprev" in d.files else False:
        per = {}
        for b in range(meta["nblocks"]):
            axes = [d[f"Fprev_{b}_{a}"].copy() for a in range(3)
                    if f"Fprev_{b}_{a}" in d.files]
            if axes:
                per[b] = axes
        solver.F_prev = per if mb else per.get(0)
    else:
        solver.F_prev = None

    solver.nstep, solver.time = meta["nstep"], meta["time"]
    return meta
