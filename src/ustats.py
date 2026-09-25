"""Running statistics for the 2.5D solver, binned by a cell coordinate (wall-normal y for a channel):
means, second moments and the Reynolds shear stress, accumulated over the cells of each bin, all
planes and all sampled steps. Nothing is assumed about the cell ordering; the bins come from the
unique centroid coordinate (rounded), so a stretched or perturbed mesh works.
"""
import numpy as np


class Stats:
    def __init__(self, s, coord=1, decimals=10, edges=None):
        """`edges`: bin edges along `coord` for an irregular (e.g. triangle) mesh; the bin coordinate
        is then the volume-weighted mean centroid coordinate of its cells. Default: one bin per
        distinct centroid coordinate (a structured-in-y mesh)."""
        self.s = s; m = s.m
        if edges is None:
            key = np.round(m.centroid[:, coord], decimals)
            self.y, inv = np.unique(key, return_inverse=True); self.inv = inv
        else:
            inv = np.clip(np.digitize(m.centroid[:, coord], edges) - 1, 0, len(edges) - 2); self.inv = inv
            self.y = np.bincount(inv, weights=m.vol * m.centroid[:, coord], minlength=len(edges) - 1) / np.maximum(np.bincount(inv, weights=m.vol, minlength=len(edges) - 1), 1e-300)
        self.wsum = np.bincount(inv, weights=m.vol, minlength=len(self.y)) * s.nz
        self.n = 0
        self.acc = {k: np.zeros(len(self.y)) for k in ("u", "v", "w", "uu", "vv", "ww", "uv", "p", "pp")}

    def _binsum(self, f):
        f = self.s.host(f) if hasattr(self.s, "host") else f                     # statistics are accumulated on the host
        return np.bincount(self.inv, weights=(self.s.m.vol[:, None] * f).sum(axis=1), minlength=len(self.y))

    def sample(self):
        s = self.s; u, v, w, p = s.u, s.v, s.w, s.p
        for k, f in (("u", u), ("v", v), ("w", w), ("uu", u * u), ("vv", v * v), ("ww", w * w), ("uv", u * v), ("p", p), ("pp", p * p)):
            self.acc[k] += self._binsum(f) / self.wsum
        self.n += 1

    def profiles(self):
        n = max(self.n, 1); a = {k: v / n for k, v in self.acc.items()}
        U, V, W = a["u"], a["v"], a["w"]
        return dict(y=self.y, U=U, V=V, W=W, uu=a["uu"] - U * U, vv=a["vv"] - V * V, ww=a["ww"] - W * W, uv=a["uv"] - U * V, P=a["p"], pp=a["pp"] - a["p"] ** 2, nsamples=self.n)
