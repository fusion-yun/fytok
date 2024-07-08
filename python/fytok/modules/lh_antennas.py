from spdm.core.component import Component
from fytok.utils.base import IDS, FyModule
from fytok.ontology import lh_antennas


class LHAntennas(IDS, FyModule, Component, lh_antennas.lh_antennas):
    def __view__(self, view_point="RZ", **styles):

        geo = {}
        match view_point.lower():
            case "top":
                geo["antenna"] = [antenna.name for antenna in self.antenna]
                styles["antenna"] = {"$matplotlib": {"color": "blue"}, "text": True}

        return geo, styles
