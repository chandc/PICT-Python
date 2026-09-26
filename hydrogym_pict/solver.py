"""TransientSolver for the PICT backend: one production PISO step per step().

Order per step mirrors HydroGym's semi-implicit solvers: integrate the
actuator model (flow.advance_time -- the damped first-order lag toward the
commanded value), write the resulting rotation into the body Dirichlet
arrays, then advance the flow one dt.
"""
from hydrogym_pict.core_shim import TransientSolver


class PICTTransientSolver(TransientSolver):
    def __init__(self, flow, dt=None):
        super().__init__(flow, dt)
        if self.dt != flow.m.dt:
            raise ValueError(
                f"solver dt {self.dt} != flow dt {flow.m.dt}; the momentum "
                "matrix is assembled for the flow's dt -- pass matching values")

    def step(self, iter, control=None, **kwargs):
        flow = self.flow
        flow.advance_time(self.dt, control)
        flow._apply_rotation(float(flow.actuators[0].state))
        flow.m.step()
        flow.t = flow.m.time
        return flow
