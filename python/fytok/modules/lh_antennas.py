from fytok.utils.base import IDS, FyComponent
from fytok.ontology import lh_antennas


class LHAntennas(IDS, FyComponent, lh_antennas.lh_antennas):
    def __view__(self, view_point="RZ", **styles) :

        geo = {}
        match view_point.lower():
            case "top":
                geo["antenna"] = [antenna.name for antenna in self.antenna]
                styles["antenna"] = {"$matplotlib": {"color": "blue"}, "text": True}

        return geo,styles
