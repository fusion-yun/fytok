import typing
from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyModule

from fytok.ontology import gyrokinetics

_TGyrokineticsTimeSlice = typing.TypeVar("_TGyrokineticsTimeSlice")


class Gyrokinetics(IDS, FyModule, Actor[_TGyrokineticsTimeSlice], gyrokinetics.Gyrokinetics):
    """回旋动理学"""
