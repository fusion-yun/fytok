import typing
from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import gyrokinetics


class Gyrokinetics(FyEntity, IDS, Actor, gyrokinetics.Gyrokinetics):
    """回旋动理学"""
