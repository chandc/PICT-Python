"""U7 of the unstructured adjoint plan: memory-flat adjoint replay over `TorchUPISO` rollouts.

The port of `src/prod_replay.py`: forward under no_grad storing DETACHED state snapshots; backward
walks the steps in reverse and rebuilds ONE step's graph at a time (leaf state + leaf action ->
apply -> step), seeds it with the accumulated adjoint of the next state, and takes one
autograd.grad. Peak graph memory is one step's, whatever the horizon; the price is one extra
forward per step. The LU factors live on the step's graph (in the solve's ctx), so they are
freed with it: that is the other half of the memory argument, since a tape keeps every factor.

Branch decisions and Poisson sweep counts are recorded on the no-grad forward and replayed in the
reverse sweep, step by step, so the rebuilt graph is of the forward that actually ran (it would
otherwise re-decide on bitwise-identical values, and agree, but that is luck, not construction).

A state entry that is None (history before the first step) stays None; its cotangent is dropped.
"""
import torch

from src.uadj_step import STATE_KEYS


def _detach(st):
    return {k: (None if v is None else v.detach()) for k, v in st.items()}


def _leaf(snap):
    return {k: (None if v is None else v.detach().clone().requires_grad_(True)) for k, v in snap.items()}


def replay_grad(T, st0, actions, apply_action, step_loss=None, final_loss=None, substeps=1,
                want_state_grad=False):
    """Gradient of  sum_k step_loss(st_k+1, k) + final_loss(st_N)  w.r.t. per-control-step action
    tensors (any shape), at one control step's graph memory. Each control step is
    apply_action(st, a_k) followed by `substeps` solver steps.
    Returns (loss_value, [dL/da_k], optionally dL/dst0)."""
    n = len(actions)
    snaps, logs = [_detach(st0)], []
    loss_val = 0.0
    with torch.no_grad():
        st = _detach(st0)
        for k in range(n):
            T.record()
            st = apply_action(st, torch.as_tensor(actions[k], dtype=torch.float64))
            for _ in range(substeps):
                st = T.step(st)
            logs.append((T.masks.log, list(T.poisson_counts)))
            snaps.append(_detach(st))
            if step_loss is not None:
                loss_val += float(step_loss(st, k))
        if final_loss is not None:
            loss_val += float(final_loss(st))

    lam, a_grads, st0_grads = None, [None] * n, None
    for k in range(n - 1, -1, -1):
        leaf = _leaf(snaps[k])
        a_leaf = torch.as_tensor(actions[k], dtype=torch.float64).clone().requires_grad_(True)
        T.replay_from(logs[k])
        out = apply_action(leaf, a_leaf)
        for _ in range(substeps):
            out = T.step(out)
        sur = torch.zeros((), dtype=torch.float64)
        if step_loss is not None:
            sur = sur + step_loss(out, k)
        if k == n - 1 and final_loss is not None:
            sur = sur + final_loss(out)
        if lam is not None:
            for key in STATE_KEYS:
                if out[key] is not None and lam.get(key) is not None:
                    sur = sur + (out[key] * lam[key]).sum()
        keys = [key for key in STATE_KEYS if leaf[key] is not None]
        ins = [a_leaf] + [leaf[key] for key in keys]
        g = torch.autograd.grad(sur, ins, allow_unused=True)
        a_grads[k] = torch.zeros_like(a_leaf) if g[0] is None else g[0].detach()
        lam = {key: (torch.zeros_like(leaf[key]) if gi is None else gi.detach()) for key, gi in zip(keys, g[1:])}
        if k == 0:
            st0_grads = lam
    T.live()
    return (loss_val, a_grads, st0_grads) if want_state_grad else (loss_val, a_grads)


def tape_grad(T, st0, actions, apply_action, step_loss=None, final_loss=None, substeps=1):
    """The same gradient with one graph over the whole window: the reference replay must equal."""
    a_leaves = [torch.as_tensor(a, dtype=torch.float64).clone().requires_grad_(True) for a in actions]
    st = dict(st0)
    loss = torch.zeros((), dtype=torch.float64)
    for k, a in enumerate(a_leaves):
        st = apply_action(st, a)
        for _ in range(substeps):
            st = T.step(st)
        if step_loss is not None:
            loss = loss + step_loss(st, k)
    if final_loss is not None:
        loss = loss + final_loss(st)
    g = torch.autograd.grad(loss, a_leaves)
    return float(loss.detach()), [x.detach() for x in g]


class SavedBytes:
    """Bytes autograd holds for backward, via saved-tensor hooks (plan gate A18). Counts tensors
    saved on the graph; the LU factors are counted separately through `LUFactor` bookkeeping."""

    def __init__(self):
        self.total = 0

    def __enter__(self):
        def pack(t):
            self.total += t.numel() * t.element_size()
            return t
        self._ctx = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        self._ctx.__exit__(*a)
