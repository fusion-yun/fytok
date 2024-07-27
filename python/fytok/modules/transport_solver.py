""" 输运求解器 """

import typing
from scipy import constants

from spdm.utils.type_hint import array_type, ArrayType
from spdm.core.htree import List, Dict
from spdm.core.expression import Expression
from spdm.core.sp_tree import annotation, SpTree, AttributeTree

from spdm.model.process import Process


from fytok.utils.logger import logger
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium
from fytok.utils.base import IDS, FyEntity


EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverEquation(SpTree):
    """Profile and derivatives a the primary quantity for a 1D transport equation"""

    identifier: str
    """ Identifier of the primary quantity of the transport equation. The description
        node contains the path to the quantity in the physics IDS (example:
        core_profiles/profiles_1d/ion/D/density)"""

    profile: array_type | Expression
    """ Profile of the primary quantity"""

    flux: array_type | Expression
    """ Flux of the primary quantity"""

    units: typing.Tuple[float, float]

    d_dr: array_type | Expression
    """ Radial derivative with respect to the primary coordinate"""

    dflux_dr: array_type | Expression
    """ Radial derivative of Flux of the primary quantity"""

    d2_dr2: array_type | Expression
    """ Second order radial derivative with respect to the primary coordinate"""

    d_dt: array_type | Expression
    """ Time derivative"""

    d_dt_cphi: array_type | Expression
    """ Derivative with respect to time, at constant toroidal flux (for current
        diffusion equation)"""

    d_dt_cr: array_type | Expression
    """ Derivative with respect to time, at constant primary coordinate coordinate (for
        current diffusion equation)"""

    coefficient: typing.List[typing.Any]
    """ Set of numerical coefficients involved in the transport equation
       
        [d_dt,D,V,RHS]
        
        d_dt + flux'= RHS  
        
        flux =-D y' + V y

        u * y + v* flux - w =0 
    """

    boundary_condition_type: int = 1

    boundary_condition_value: tuple
    """ [u,v,v] 
    
    u * profile + v* flux - w =0"""

    convergence: AttributeTree
    """ Convergence details"""


class TransportSolver(
    FyEntity,
    IDS,
    Process,
    plugin_prefix="transport_solver/",
    plugin_default="fy_trans",
):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    class InPorts(Process.InPorts):
        equilibrium: Equilibrium
        core_profiles: CoreProfiles
        core_transport: CoreTransport
        core_sources: CoreSources

    class OutPorts(Process.OutPorts):
        core_profiles: CoreProfiles

    in_ports: InPorts  # type:ignore

    out_ports: OutPorts  # type:ignore

    # profiles_1d: CoreProfiles.Profiles1D

    # drho_tor_dt: ArrayType = annotation(units="m.s^-1")
    # """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    # d_dvolume_drho_tor_dt: ArrayType = annotation(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""

    # solver: str = "ion_solver"
    # ion_thermal: set
    # ion_non_thermal: set
    # impurities: set
    # neutral: set
    # equations: List[TransportSolverEquation]
    # variables: Dict[Expression]

    primary_coordinate: str = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    def execute(self, *args, **kwargs):

        logger.info(f"Solve transport equations : { '  ,'.join([equ.identifier for equ in self.equations])}")

        res = super().execute(*args, **kwargs)

        equilibrium = self.in_ports.equilibrium

        core_profiles_in = self.in_ports.core_profiles

        core_profiles_out = self.out_ports.core_profiles

        core_profiles_out.time = equilibrium.time
        core_profiles_out.vacuum_toroidal_field.r0 = equilibrium.vacuum_toroidal_field.r0
        core_profiles_out.vacuum_toroidal_field.b0 = equilibrium.vacuum_toroidal_field.b0

        # rho_tor_norm = core_profiles_in.profiles_1d.grid.rho_tor_norm
        # grid = equilibrium.profiles_1d.grid.remesh(rho_tor_norm=rho_tor_norm)
        # core_profiles_out.profiles_1d.grid = grid

        core_profiles_out.profiles_1d.ion = [ion.label for ion in core_profiles_in.profiles_1d.ion]

        return res
