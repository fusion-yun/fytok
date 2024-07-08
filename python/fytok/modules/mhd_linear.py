import typing
from spdm.core.time_sequence import TimeSlice
from fytok.utils.base import IDS, FyActor

from fytok.ontology import mhd_linear

_TMHDLinearTimeSlice = typing.TypeVar("_TMHDLinearTimeSlice", bound=TimeSlice)


class MHDLinear(IDS, FyActor[_TMHDLinearTimeSlice], mhd_linear.MHDLinear):
    """线性磁流体"""
