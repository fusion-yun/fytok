from functools import cached_property

from spdm.core.actor import Actor
from spdm.core.sp_property import sp_tree
from .modules.pulse_schedule import PulseSchedule
from .modules.transport_solver_numerics import TransportSolverNumerics

from .utils.logger import logger
from .tokamak import Tokamak


@sp_tree
class Scenario(Actor):
    """
    Scenario

    """

    tokamak: Tokamak

    pulse_schedule: PulseSchedule
