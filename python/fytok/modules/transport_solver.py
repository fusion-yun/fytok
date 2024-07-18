import typing
from scipy import constants

from spdm.utils.tags import _not_found_
from spdm.utils.type_hint import array_type

from spdm.core.htree import List, Dict, HTree
from spdm.core.expression import Expression
from spdm.core.sp_tree import sp_property, SpTree, AttributeTree

from spdm.model.process import Process
from spdm.model.port import Ports

from fytok.utils.logger import logger
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSourcesSource
from fytok.modules.core_transport import CoreTransportModel
from fytok.modules.equilibrium import Equilibrium
from fytok.utils.base import IDS, FyModule

from fytok.modules.utilities import CoreRadialGrid

# from ..ontology import transport_solver_numerics

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverNumericsEquation(SpTree):
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
    IDS,
    FyModule,
    Process,
    plugin_prefix="transport_solver_numerics/",
    plugin_default="fy_trans",
):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    class InPorts(Ports):
        equilibrium: Equilibrium
        core_profiles: CoreProfiles
        core_transport: List[CoreTransportModel]
        core_sources: List[CoreSourcesSource]

    class OutPorts(Ports):
        core_profiles: CoreProfiles

    in_ports: InPorts

    out_ports: OutPorts

    profiles_1d: CoreProfiles.Profiles1D

    equations: List[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: AttributeTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: array_type | Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: array_type | Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""

    solver: str = "ion_solver"

    ion_thermal: set

    ion_non_thermal: set

    impurities: set

    neutral: set

    primary_coordinate: str = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    equations: List[TransportSolverNumericsEquation]

    variables: Dict[Expression]

    def execute(self, *args, **kwargs) -> CoreProfiles:

        logger.info(f"Solve transport equations : { '  ,'.join([equ.identifier for equ in self.equations])}")

        res = CoreProfiles(super().execute(*args, **kwargs))

        res.profiles_1d = self.in_ports.core_profiles.profiles_1d.__getstate__()

        rho_tor_norm = res.profiles_1d.rho_tor_norm

        grid = self.in_ports.equilibrium.profiles_1d.grid.remesh(rho_tor_norm=rho_tor_norm)

        res.profiles_1d.grid = grid

        return res
