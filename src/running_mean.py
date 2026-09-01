"""Time-average of the solver fields, accumulated as the run goes.

WHY. Every field we save is INSTANTANEOUS, so the quantities the literature actually tabulates
for a shedding cylinder -- mean recirculation length, base pressure coefficient, mean separation
angle, the mean streamline pattern -- could not be computed at all, however long the run. A
snapshot of a limit cycle is a snapshot at one phase; the mean wake is a different object, and
for a circular cylinder at Re = 100 the mean bubble is ~0.9 D while the instantaneous one swings
between roughly half and twice that.

The cost is one array add per field per step and no extra solve, so it is worth doing on every
run rather than as a separate averaging pass -- a second pass would mean either storing hundreds
of snapshots or running the case twice.

A PLAIN SUM, NOT WELFORD. Welford's algorithm exists for the variance and for cases where the
mean is large relative to the spread; here the fields are O(1), the count is O(10^4), and
float64 carries ~16 digits, so a running sum loses nothing measurable. Using it keeps the state
to one array per field, which is what makes the checkpoint cheap.

START IT AFTER THE TRANSIENT. Averaging from t = 0 folds the settle and the growth phase into
the answer. The drivers start it at the kick, and `t_start` records where, so a mean can never
be quoted without the window it came from.
"""
import numpy as np


class RunningMean:
    """Running sum of named per-block fields, plus the window it covers."""

    FIELDS = ("u", "v", "w", "p")

    def __init__(self, d, t_start, fields=FIELDS):
        self.fields = tuple(fields)
        self.n = 0
        self.t_start = float(t_start)
        self.t_end = float(t_start)
        self.sums = {f: [np.zeros(blk.shape) for blk in d.blocks] for f in self.fields}

    def add(self, m):
        for f in self.fields:
            src = getattr(m, f)
            for b, acc in enumerate(self.sums[f]):
                acc += src[b]
        self.n += 1
        self.t_end = float(m.time)

    def mean(self):
        if not self.n:
            raise ValueError("no samples accumulated")
        return {f: [acc / self.n for acc in self.sums[f]] for f in self.fields}

    def save(self, path):
        out = {"n": np.array(self.n), "t_start": np.array(self.t_start),
               "t_end": np.array(self.t_end), "nblocks": np.array(len(self.sums["u"])),
               "names": np.array(self.fields)}
        for f, blocks in self.mean().items():
            for b, arr in enumerate(blocks):
                out[f"{f}_{b}"] = arr
        np.savez_compressed(path, **out)
        return path


def load_mean(path):
    """(fields, meta) with fields = {name: {block: array}}, mirroring checkpoint.load_fields."""
    d = np.load(path, allow_pickle=False)
    nb = int(d["nblocks"])
    names = [str(x) for x in d["names"]]
    fields = {f: {b: d[f"{f}_{b}"] for b in range(nb)} for f in names}
    meta = {"n": int(d["n"]), "t_start": float(d["t_start"]), "t_end": float(d["t_end"]),
            "nblocks": nb, "names": names}
    return fields, meta
