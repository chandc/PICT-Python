"""Stage 9.5: MEMORY-FLAT adjoint replay over production-step rollouts.

The tape's memory grows with every step of a rollout (the M0 lesson: H=80
held 8 GiB of tape in FluidGym's stack, and horizon was the variable that
decided training success). This replaces the across-step tape with the
classic checkpoint-and-replay adjoint:

  forward   : run the rollout under no_grad, saving DETACHED state
              snapshots -- plain arrays, no graph;
  backward  : walk the steps in REVERSE; for step k, rebuild the graph for
              THAT STEP ONLY (leaf state + leaf action -> one step()call),
              seed it with the accumulated adjoint state lambda_{k+1} plus
              any direct loss cotangent, and take one autograd.grad --
              yielding lambda_k and dL/da_k.

Peak autograd memory is ONE step's graph regardless of window length; the
compute price is one extra forward per step (the standard 2x). Within a
step, gradients still flow through LinearSolve's exact adjoint solves and
the probed assembly -- nothing about the per-step gradient changes, which
is what the equivalence gate (replay == tape on the same window) certifies.

State keys carried across steps: u, v, w, p, p_flux, ubc, vbc, wbc, and
the BDF2 history u_prev (whose cotangent flows back through the identity
out["u_prev"] = [in u, v, w]).
"""
import torch

_KEYS = ("u", "v", "w", "p", "p_flux", "ubc", "vbc", "wbc")


def _leaf_state(snap):
    """Detached leaf copies of a snapshot, requires_grad on."""
    st = {}
    for k in _KEYS:
        st[k] = snap[k].detach().clone().requires_grad_(True)
    if snap["u_prev"] is None:
        st["u_prev"] = None
    else:
        st["u_prev"] = [c.detach().clone().requires_grad_(True)
                        for c in snap["u_prev"]]
    return st


def _leaves(st):
    out = [st[k] for k in _KEYS]
    if st["u_prev"] is not None:
        out += list(st["u_prev"])
    return out


def replay_rollout_grad(tps, st0, actions, final_loss, step_loss=None,
                        apply_action=None, want_state_grad=False):
    """Gradient of  sum_k step_loss(st_{k+1}, k) + final_loss(st_N)  with
    respect to per-step scalar `actions` (and optionally st0), at O(1)
    graph memory.

    tps          : TorchProductionStep (jets registered if apply_action
                   defaults)
    st0          : starting state dict (as from state_from_solver())
    actions      : list of python floats / 0-d tensors, one per step
    final_loss   : callable st -> scalar tensor (applied to the last state)
    step_loss    : optional callable (st, k) -> scalar tensor or None
    apply_action : callable (st, a) -> st; defaults to tps.apply_jets
    Returns (loss_value, action_grads[, st0_grads])."""
    if apply_action is None:
        apply_action = tps.apply_jets
    n = len(actions)

    # ---------------- forward: detached snapshots, no graph
    snaps = [ {k: (None if st0[k] is None else
                   ([c.detach() for c in st0[k]] if isinstance(st0[k], list)
                    else st0[k].detach())) for k in st0} ]
    loss_val = 0.0
    with torch.no_grad():
        st = tps._clone(st0)
        for k in range(n):
            st = apply_action(st, torch.as_tensor(float(actions[k])))
            st = tps.step(st)
            snaps.append(tps._clone(st))
            if step_loss is not None:
                sl = step_loss(st, k)
                if sl is not None:
                    loss_val += float(sl)
        loss_val += float(final_loss(st))

    # ---------------- backward: one step's graph at a time, in reverse
    lam = None                                   # dict key -> cotangent
    a_grads = [0.0] * n
    st0_grads = None
    for k in range(n - 1, -1, -1):
        leaf = _leaf_state(snaps[k])
        a_leaf = torch.tensor(float(actions[k]), dtype=torch.float64,
                              requires_grad=True)
        st_in = apply_action(dict(leaf, u_prev=leaf["u_prev"]), a_leaf)
        out = tps.step(st_in)

        surrogate = 0.0
        if k == n - 1:
            surrogate = surrogate + final_loss(out)
        if step_loss is not None:
            sl = step_loss(out, k)
            if sl is not None:
                surrogate = surrogate + sl
        if lam is not None:
            for key in _KEYS:
                surrogate = surrogate + (out[key] * lam[key]).sum()
            if out["u_prev"] is not None and lam.get("u_prev") is not None:
                for c, lc in zip(out["u_prev"], lam["u_prev"]):
                    surrogate = surrogate + (c * lc).sum()

        inputs = [a_leaf] + _leaves(leaf)
        grads = torch.autograd.grad(surrogate, inputs, allow_unused=True)
        a_grads[k] = float(grads[0]) if grads[0] is not None else 0.0
        gs = [g if g is not None else torch.zeros_like(x)
              for g, x in zip(grads[1:], _leaves(leaf))]
        lam = {key: gs[i].detach() for i, key in enumerate(_KEYS)}
        if leaf["u_prev"] is not None:
            lam["u_prev"] = [g.detach() for g in gs[len(_KEYS):]]
        else:
            lam["u_prev"] = None
        if k == 0 and want_state_grad:
            st0_grads = lam

    if want_state_grad:
        return loss_val, a_grads, st0_grads
    return loss_val, a_grads


def replay_policy_grad(tps, st0, policy_action, n_ctrl, substeps, step_loss,
                       final_loss=None, apply_action=None):
    """Memory-flat policy gradient over a control window.

    policy_action(st) -> 0-d action tensor, computed from st's tensors and
    the policy parameters -- called INSIDE each replayed control step on
    the leaf state, so d(loss)/d(theta) carries both the action path and
    the observation path (full BPTT semantics, not obs-detached).
    Each control step = apply_action(st, a) then `substeps` solver steps.
    step_loss(st, k) -> per-control-step scalar (the discounted reward
    term); final_loss(st) optional terminal term.

    Parameter gradients ACCUMULATE into .grad of whatever tensors
    policy_action closes over (zero them before calling); returns the
    forward loss value."""
    if apply_action is None:
        apply_action = tps.apply_jets

    snaps = [tps._clone(st0)]
    loss_val = 0.0
    with torch.no_grad():
        st = tps._clone(st0)
        for k in range(n_ctrl):
            st = apply_action(st, policy_action(st))
            for _ in range(substeps):
                st = tps.step(st)
            snaps.append(tps._clone(st))
            loss_val += float(step_loss(st, k))
        if final_loss is not None:
            loss_val += float(final_loss(st))

    lam = None
    for k in range(n_ctrl - 1, -1, -1):
        leaf = _leaf_state(snaps[k])
        st_in = apply_action(dict(leaf, u_prev=leaf["u_prev"]),
                             policy_action(leaf))
        out = st_in
        for _ in range(substeps):
            out = tps.step(out)

        surrogate = step_loss(out, k)
        if k == n_ctrl - 1 and final_loss is not None:
            surrogate = surrogate + final_loss(out)
        if lam is not None:
            for key in _KEYS:
                surrogate = surrogate + (out[key] * lam[key]).sum()
            if out["u_prev"] is not None and lam.get("u_prev") is not None:
                for c, lc in zip(out["u_prev"], lam["u_prev"]):
                    surrogate = surrogate + (c * lc).sum()

        surrogate.backward()          # accumulates policy-parameter grads
        lam = {key: (leaf[key].grad.detach() if leaf[key].grad is not None
                     else torch.zeros_like(leaf[key])) for key in _KEYS}
        if leaf["u_prev"] is not None:
            lam["u_prev"] = [(c.grad.detach() if c.grad is not None
                              else torch.zeros_like(c))
                             for c in leaf["u_prev"]]
        else:
            lam["u_prev"] = None
    return loss_val
