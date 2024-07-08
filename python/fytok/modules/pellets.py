from fytok.utils.base import IDS, FyComponent

from fytok.ontology import pellets


class Pellets(IDS, FyComponent, pellets.pellets):
    def __view__(self, view="RZ", **styles):
        return {}, styles
