"""Plane-and-time averaged statistics for a channel, with the two halves folded.

WHAT IS ACCUMULATED. At each wall-normal station: the mean streamwise velocity and the four
non-trivial Reynolds stresses, averaged over the two HOMOGENEOUS directions (x and z) and over
time. Those five are exactly what the reference DNS archive stores, so the comparison needs no
translation.

    U, <u'u'>, <v'v'>, <w'w'>, <u'v'>

THE HALVES ARE FOLDED, and it is free sampling rather than a tidy-up. A channel's statistics are
symmetric about the centreline -- U, u', v', w' even in (y - delta), <u'v'> ODD -- so averaging
y with its mirror DOUBLES the effective sample count at every station. For a 15-time-unit window
that is worth as much as running twice as long. The reference archive does the same and says so.

WHAT IS STORED IS THE RAW SECOND MOMENT -- <u^2>, not <u'^2> -- and the fluctuation is formed at
the END by subtracting the square of the converged time mean. This is the reference archive's
convention, and adopting it is not merely for convenience.

The alternative, subtracting each SNAPSHOT's own plane mean, was written first and is wrong for
this case. It measures only the SPATIAL fluctuation within a plane and throws away the temporal
wandering of the plane mean itself. In a large box that wandering is negligible. In a MINIMAL
box it is a substantial part of the turbulence: the box holds roughly one streak pair, so the
plane-averaged streamwise velocity genuinely oscillates as that structure cycles, and discarding
it would bias every stress DOWNWARD by an amount that grows as the box shrinks -- silently, and
in the direction that flatters an under-resolved LES.

Raw moments have their own trap, which is why the mean is subtracted only at the end: a RUNNING
mean would make early samples fluctuate about an unconverged average, biasing the stresses up
while the flow is still developing. The fix is not a different moment, it is to start
accumulating after the transient, which is what `--t-stats` is for.
"""
import numpy as np


class ChannelStats:
    """Accumulate plane averages on a channel whose wall-normal axis is index 1."""

    def __init__(self, domain, y_tol=1e-9):
        self.d = domain
        ys = np.concatenate([b.y.ravel() for b in domain.blocks])
        self.y = np.unique(np.round(ys, 9))
        self.n = len(self.y)
        # index map: for each block, which station each node belongs to
        self.idx = [np.searchsorted(self.y, np.round(b.y, 9)) for b in domain.blocks]
        self.counts = np.zeros(self.n)
        for b, blk in enumerate(domain.blocks):
            np.add.at(self.counts, self.idx[b].ravel(), 1.0)
        self.sums = np.zeros((5, self.n))
        self.nsamp = 0
        self.t0 = None
        self.t1 = None

    def _plane(self, field):
        out = np.zeros(self.n)
        for b in range(len(self.d.blocks)):
            np.add.at(out, self.idx[b].ravel(), field[b].ravel())
        return out / np.maximum(self.counts, 1)

    def add(self, m):
        """One snapshot: plane averages of U and of the RAW moments u^2, v^2, w^2, uv."""
        acc = np.zeros((5, self.n))
        for b in range(len(self.d.blocks)):
            i = self.idx[b].ravel()
            u, v, w = m.u[b].ravel(), m.v[b].ravel(), m.w[b].ravel()
            np.add.at(acc[0], i, u)
            np.add.at(acc[1], i, u * u)
            np.add.at(acc[2], i, v * v)
            np.add.at(acc[3], i, w * w)
            np.add.at(acc[4], i, u * v)
        acc /= np.maximum(self.counts, 1)
        # V and W have zero mean by symmetry, so <v'^2> = <v^2> and <u'v'> = <uv> - U V ~ <uv>.
        # U does NOT, and its square must come off explicitly -- at the centreline <u^2> = 348.5
        # against U^2 = 348.2, so the fluctuation is 0.1% of the raw moment. Forgetting the
        # subtraction would report u' ~ 18.7 instead of 0.6.
        self.sums += acc
        self.nsamp += 1
        if self.t0 is None:
            self.t0 = m.time
        self.t1 = m.time

    def profiles(self, fold=True):
        """(y, U, u', v', w', -<u'v'>) on the lower half, folded unless told not to."""
        if self.nsamp == 0:
            raise RuntimeError("no samples accumulated")
        p = self.sums / self.nsamp
        p = np.array([p[0], p[1] - p[0] ** 2, p[2], p[3], p[4]])   # raw moments -> central
        y = self.y
        Ly = y[-1]
        if not fold:
            return y, p[0], np.sqrt(p[1]), np.sqrt(p[2]), np.sqrt(p[3]), -p[4]
        # mirror: U, uu, vv, ww are EVEN about the centreline, <u'v'> is ODD
        ym = Ly - y
        order = np.argsort(ym)
        q = np.empty_like(p)
        for k in range(4):
            q[k] = np.interp(y, ym[order], p[k][order])
        q[4] = -np.interp(y, ym[order], p[4][order])
        f = 0.5 * (p + q)
        half = y <= 0.5 * Ly + 1e-12
        return (y[half], f[0][half], np.sqrt(np.maximum(f[1][half], 0)),
                np.sqrt(np.maximum(f[2][half], 0)), np.sqrt(np.maximum(f[3][half], 0)),
                -f[4][half])

    def u_tau(self, nu):
        """From the wall gradient of the mean profile, one-sided over the first two cells."""
        p = self.sums[0] / max(self.nsamp, 1)
        dudy = (p[1] - p[0]) / (self.y[1] - self.y[0])
        return float(np.sqrt(nu * abs(dudy)))

    def save(self, path, nu):
        np.savez(path, y=self.y, sums=self.sums, nsamp=self.nsamp, nu=nu,
                 t0=self.t0, t1=self.t1)
