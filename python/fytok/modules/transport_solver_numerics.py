from __future__ import annotations

from scipy import constants
from copy import copy
import math
from spdm.core.expression import Expression, Variable, zero
from spdm.core.sp_property import sp_tree, sp_property, PropertyTree
from spdm.core.time_series import TimeSlice, TimeSeriesAoS
from spdm.core.aos import AoS
from spdm.utils.tags import _not_found_
from spdm.utils.typing import array_type

from .core_profiles import CoreProfiles
from .core_sources import CoreSources
from .core_transport import CoreTransport
from .equilibrium import Equilibrium
from .utilities import *
from ..utils.logger import logger

# from ..ontology import transport_solver_numerics

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


@sp_tree
class TransportSolverNumericsEquation:
    """Profile and derivatives a the primary quantity for a 1D transport equation"""

    identifier: str = sp_property(alias="@name")
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

    convergence: PropertyTree
    """ Convergence details"""


@sp_tree(coordinate1="grid/rho_tor_norm")
class TransportSolverNumericsTimeSlice(TimeSlice):
    """Numerics related to 1D radial solver for a given time slice"""

    grid: CoreRadialGrid

    equations: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: PropertyTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: array_type | Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: array_type | Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""


@sp_tree
class TransportSolverNumerics(IDS):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    code: Code = {"name": "fy_trans"}

    solver: str = "ion_solver"

    ion_thermal: set = set()

    ion_non_thermal: set = set()

    impurities: set = set()

    neutral: set = set()

    primary_coordinate: str = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    equations: AoS[TransportSolverNumericsEquation] = []

    variables: typing.Dict[str, Expression] = {}

    profiles_1d: CoreProfiles.TimeSlice.Profiles1D = {}

    TimeSlice = TransportSolverNumericsTimeSlice

    time_slice: TimeSeriesAoS[TransportSolverNumericsTimeSlice] = []

    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)

    def preprocess(self, *args, **kwargs) -> TransportSolverNumericsTimeSlice:
        eq_grid: CoreRadialGrid = self.inports["equilibrium/time_slice/0/profiles_1d/grid"].fetch()

        rho_tor_norm = kwargs.get("rho_tor_norm", _not_found_)

        if rho_tor_norm is _not_found_:
            rho_tor_norm = self.code.parameters.rho_tor_norm

        new_grid = eq_grid.remesh(rho_tor_norm)

        self.profiles_1d["grid"] = new_grid

        current = super().preprocess(*args, **kwargs)

        current["grid"] = new_grid

        return current

    def execute(
        self,
        current: TransportSolverNumericsTimeSlice,
        *previous: TransportSolverNumericsTimeSlice,
    ) -> TransportSolverNumericsTimeSlice:
        logger.info(f"Solve transport equations : { '  ,'.join([equ.identifier for equ in self.equations])}")
        return super().execute(current, *previous)

    def postprocess(self, current: TransportSolverNumericsTimeSlice) -> TransportSolverNumericsTimeSlice:
        return super().postprocess(current)

    def refresh(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_transport: CoreTransport = None,
        core_sources: CoreSources = None,
        core_profiles: CoreProfiles = None,
        **kwargs,
    ) -> TransportSolverNumericsTimeSlice:
        return super().refresh(
            *args,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
            core_profiles=core_profiles,
            **kwargs,
        )

    def fetch(self, *args, **kwargs) -> CoreProfiles.TimeSlice.Profiles1D:
        """获得 CoreProfiles.TimeSlice.Profiles1D 形式状态树。"""

        current: TransportSolverNumericsTimeSlice = self.time_slice.current

        X = current.grid.rho_tor_norm

        Y = sum([[equ.profile, equ.flux] for equ in current.equations], [])

        self.profiles_1d["grid"] = current.grid

        profiles_1d = self.profiles_1d.fetch(X, *Y)

        profiles_1d[self.primary_coordinate] = current.grid.get(self.primary_coordinate)

        return profiles_1d
