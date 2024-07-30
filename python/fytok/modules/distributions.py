import typing
from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import distributions

_TSlice = typing.TypeVar("_TSlice")


class Distributions(FyEntity, IDS, Actor[_TSlice], distributions._T_distributions):
    pass
