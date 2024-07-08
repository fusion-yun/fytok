import typing
from spdm.core.time_sequence import TimeSlice
from fytok.utils.base import IDS, FyActor

from fytok.ontology import mhd

_TMHDTimeSlice = typing.TypeVar("_TMHDTimeSlice", bound=TimeSlice)


class MHD(IDS, FyActor[_TMHDTimeSlice], mhd.MHD):
    """磁流体"""
