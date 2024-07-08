import typing
from spdm.core.time_sequence import TimeSlice
from spdm.core.actor import Actor
from fytok.utils.base import IDS, FyModule

from fytok.ontology import mhd

_TMHDTimeSlice = typing.TypeVar("_TMHDTimeSlice", bound=TimeSlice)


class MHD(IDS, FyModule, Actor[_TMHDTimeSlice], mhd.MHD):
    """磁流体"""
