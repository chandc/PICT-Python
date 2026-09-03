"""Tiny 3D CNNs for subgrid closure, small enough that finite differences on EVERY weight is cheap.

PADDING IS A BOUNDARY CONDITION, AND THE RIGHT ONE DEPENDS ON THE DOMAIN.

`padding_mode="circular"` is CORRECT for the periodic box these were trained on and WRONG at a
multi-block seam, where the neighbour is another block rather than the opposite face. Applied
per block on a split periodic box, the interior planes agree with the whole-box answer to
1.1e-16 and the planes adjacent to a seam are wrong by O(1) -- 146% of the output scale. None of
the 38 adjoint gates would have caught it, because every one of them drives the solver with a
raw parameter vector and never a network.

So the mode is now explicit rather than assumed:

    pad_mode="circular"   one periodic box. Unchanged, still the default, still correct there.
    pad_mode="none"       convolutions with padding=0. The CALLER supplies a halo, which for a
                          multi-block domain means MultiBlock.pad_field -- the same exchange
                          the solver already uses for its own operators, and the same one a
                          distributed version would send over MPI.

The second mode is the one that generalises: it makes the network consume the solver's halo
rather than invent its own boundary.
"""
import torch
import torch.nn as nn


class TinySGSNet(nn.Module):
    """
    velocity (1,3,n,n,n) -> momentum source (1,3,n,n,n).

    Two layers, 173 parameters:  conv3d(3->2, k=3) = 164,  conv3d(2->3, k=1) = 9.

    pad_mode: see the module docstring. "circular" for one periodic box, "none" when the caller
    supplies a halo (multi-block). The layer count and parameter count are identical either way,
    so weights are interchangeable between the two modes.
    """

    def __init__(self, hidden=2, pad_mode="circular"):
        super().__init__()
        if pad_mode not in ("circular", "none"):
            raise ValueError(f"pad_mode must be 'circular' or 'none', got {pad_mode!r}")
        self.pad_mode = pad_mode
        # halo the CALLER must supply. Zero for circular, which pads itself; one for "none",
        # which expects the neighbour data already attached.
        self.halo = 1 if pad_mode == "none" else 0
        pad = 0 if pad_mode == "none" else 1
        pm = "zeros" if pad_mode == "none" else pad_mode
        self.c1 = nn.Conv3d(3, hidden, 3, padding=pad, padding_mode=pm)
        self.c2 = nn.Conv3d(hidden, 3, 1)

    def forward(self, x):
        return self.c2(torch.tanh(self.c1(x)))

    def field(self, u, v, w, shape):
        """
        Fields in, a (3, N) source tensor out. Accepts numpy arrays or torch tensors --
        passing TORCH tensors keeps the network input inside the autograd graph, which is
        what makes a multi-step rollout gradient faithful rather than per-step.
        """
        cols = []
        for f in (u, v, w):
            t = f if torch.is_tensor(f) else torch.as_tensor(f)
            cols.append(t.reshape(shape))
        return self(torch.stack(cols).unsqueeze(0)).squeeze(0).reshape(3, -1)

    def field_halo(self, u, v, w, padded_shape):
        """Fields WITH a one-cell halo in, the interior source out, as (3, N_interior).

        Only valid for pad_mode="none". `padded_shape` is the shape INCLUDING the halo, so the
        result is (3, prod(padded_shape - 2)). The halo must come from the domain -- for a
        multi-block grid that is MultiBlock.pad_field(b, fields, width=1), which resolves the
        seam from the real neighbour instead of wrapping around the block.
        """
        if self.pad_mode != "none":
            raise ValueError("field_halo needs pad_mode='none'; with circular padding the "
                             "network invents its own boundary and the halo is ignored")
        cols = []
        for f in (u, v, w):
            t = f if torch.is_tensor(f) else torch.as_tensor(f)
            cols.append(t.reshape(padded_shape))
        out = self(torch.stack(cols).unsqueeze(0)).squeeze(0)
        return out.reshape(3, -1)


class SGSNet(nn.Module):
    """
    Larger network for the closure task. Stage 2's 173-parameter net was sized so that finite
    differences on EVERY weight stayed affordable; that constraint does not apply here, where
    the gate is statistical (correlation on held-out data) rather than exact.

    Still deliberately small -- ~10k parameters. A closure that only works with a large network
    on 24 snapshots would be memorising, and the held-out correlation is what would catch it.
    """

    def __init__(self, width=24, depth=3):
        super().__init__()
        layers, c_in = [], 3
        for _ in range(depth):
            layers += [nn.Conv3d(c_in, width, 3, padding=1, padding_mode="circular"),
                       nn.GELU()]
            c_in = width
        layers += [nn.Conv3d(width, 3, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
