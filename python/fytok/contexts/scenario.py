from functools import cached_property

from spdm.model.context import Context
from spdm.core.sp_tree import sp_tree

from .pulse_schedule import PulseSchedule

from ..utils.logger import logger

from .tokamak import Tokamak


@sp_tree
class Scenario(Context):
    """
    Scenario

    """

    tokamak: Tokamak

    pulse_schedule: PulseSchedule
