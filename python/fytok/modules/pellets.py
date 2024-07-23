from spdm.model.component import Component
from fytok.utils.base import IDS, FyEntity

from fytok.ontology import pellets


class Pellets(IDS, FyEntity, Component, pellets.pellets):
    def __view__(self, view="RZ", **styles):
        return {}, styles
