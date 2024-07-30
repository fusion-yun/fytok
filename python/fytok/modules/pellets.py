""" 弹丸注入 """

from spdm.model.component import Component
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import pellets


class Pellets(FyEntity, IDS, Component, pellets.pellets):
    """弹丸注入"""
