import typing
from fytok.utils.base import IDS, FyActor

from fytok.ontology import gyrokinetics

_TGyrokineticsTimeSlice = typing.TypeVar("_TGyrokineticsTimeSlice")


class Gyrokinetics(IDS, FyActor[_TGyrokineticsTimeSlice], gyrokinetics.Gyrokinetics):
    """回旋动理学"""
