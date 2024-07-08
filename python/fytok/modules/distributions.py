import typing
from spdm.core.actor import Actor
from fytok.utils.base import IDS, FyModule

from fytok.ontology import distributions

_TSlice = typing.TYpeVar("_TSlice")


class Distributions(IDS, FyModule, Actor[_TSlice], distributions._T_distributions):
    pass
