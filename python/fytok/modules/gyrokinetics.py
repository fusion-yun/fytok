import typing
from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyModule

from fytok.ontology import gyrokinetics


class Gyrokinetics(IDS, FyModule, Actor, gyrokinetics.Gyrokinetics):
    """回旋动理学"""
