from spdm.model.component import Component
from fytok.utils.base import IDS, FyEntity


from fytok.ontology import ic_antennas


class ICAntennas(FyEntity, IDS, Component, ic_antennas.ic_antennas):
    def __view__(self, view="RZ", **styles):

        geo = {}
        if view != "RZ":
            geo["antenna"] = [antenna.name for antenna in self.antenna]
            styles["antenna"] = {"$matplotlib": {"color": "blue"}, "text": True}

        geo["$styles"] = styles
        return geo
