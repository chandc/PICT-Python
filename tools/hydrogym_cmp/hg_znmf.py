"""Zero-net-mass-flux variant of HydroGym's jet Cylinder: the same two 10-degree cosine slots at +-90
degrees and the same single scalar a in [-0.1, 0.1], but opposed -- the top slot blows while the bottom
sucks (a > 0), as Rabault et al. (2019) and HydroGym's own paper describe. Only the sign of A_lo differs
from the shipped class, so the flux through the top slot is +a and through the bottom -a: net zero.
Restarts from the same published developed-shedding checkpoint as the shipped Cylinder (the auto-loader
keys on the class name, so the path is passed explicitly)."""
import glob, os
import firedrake as fd, ufl
from ufl import atan2
import hydrogym.firedrake as hgym
from hydrogym.firedrake.envs.cylinder.flow import RADIUS

from pathlib import Path
# same selection rule as HydroGym's _resolve_single_checkpoint (glob order, first file), so the ZNMF run
# starts from the identical checkpoint the shipped Cylinder resolves to in the same container
_CKPT = [str(p.resolve()) for p in Path(os.path.expanduser("~/.cache/hydrogym/Cylinder_2D_Re100_medium_FD")).glob("*.ckpt")]

class CylinderZNMF(hgym.Cylinder):
    def __init__(self, **config):
        if config.get("restart") is None and _CKPT:
            config["restart"] = _CKPT[0]
        super().__init__(**config)

    @property
    def cyl_velocity_field(self):
        theta = atan2(ufl.real(self.y), ufl.real(self.x)); pi = ufl.pi
        self.rad = fd.Constant(RADIUS); omega = pi / 18
        def slot(theta_c):
            return ufl.conditional(abs(theta - theta_c) < omega / 2, pi / (2 * omega * self.rad**2) * ufl.cos((pi / omega) * (theta - theta_c)), 0.0)
        return ufl.as_tensor((self.x, self.y)) * (slot(0.5 * pi) - slot(-0.5 * pi))     # opposed: ZNMF
