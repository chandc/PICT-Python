"""PICT-Python backend for HydroGym: the validated butterfly cylinder as a
`PDEBase`/`FlowEnv` environment. See reference/fluidgym_parity.md (HydroGym
section) and test_hydrogym_pict.py for the gates.

    from hydrogym_pict import make_env
    env = make_env(mesh="coarse", num_substeps=25, restart="results/fields/....npz")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step([0.3])
"""
from hydrogym_pict.core_shim import HYDROGYM_SOURCE, FlowEnv
from hydrogym_pict.flow import RotaryCylinder
from hydrogym_pict.solver import PICTTransientSolver


def make_env(mesh="coarse", dt=1e-2, tol=1e-6, backend="scipy", restart=None,
             num_substeps=1, reward_aggregation="mean", max_steps=int(1e6)):
    """A HydroGym FlowEnv running the PICT-Python production solver."""
    flow_config = dict(mesh=mesh, dt=dt, tol=tol, backend=backend)
    if restart is not None:
        flow_config["restart"] = restart
    return FlowEnv({
        "flow": RotaryCylinder,
        "flow_config": flow_config,
        "solver": PICTTransientSolver,
        "solver_config": dict(dt=dt),
        "actuation_config": dict(num_substeps=num_substeps,
                                 reward_aggregation=reward_aggregation),
        "max_steps": max_steps,
    })


__all__ = ["RotaryCylinder", "PICTTransientSolver", "FlowEnv", "make_env",
           "HYDROGYM_SOURCE"]
