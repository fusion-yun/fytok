from spdm.core.component import Component
from fytok.utils.base import IDS, FyModule

from fytok.ontology import pellets


class Pellets(IDS, FyModule, Component, pellets.pellets):
    def __view__(self, view="RZ", **styles):
        return {}, styles
