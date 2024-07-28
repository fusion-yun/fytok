""" MHD Linear """

from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import mhd_linear


class MHDLinear(FyEntity, IDS, Actor, mhd_linear.MHDLinear):
    """线性磁流体"""
