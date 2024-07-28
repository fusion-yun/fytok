"""磁流体"""

from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import mhd


class MHD(FyEntity, IDS, Actor, mhd.MHD):
    """磁流体"""
