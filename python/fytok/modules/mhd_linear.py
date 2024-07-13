import typing
from spdm.model.time_sequence import TimeSlice
from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyModule

from fytok.ontology import mhd_linear

_TMHDLinearTimeSlice = typing.TypeVar("_TMHDLinearTimeSlice", bound=TimeSlice)


class MHDLinear(IDS, FyModule, Actor[_TMHDLinearTimeSlice], mhd_linear.MHDLinear):
    """线性磁流体"""
