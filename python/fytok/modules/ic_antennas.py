from fytok.utils.base import IDS, FyComponent

from spdm.core.geo_object import GeoObject

from fytok.ontology import ic_antennas


class ICAntennas(IDS, FyComponent, ic_antennas.ic_antennas):
    def __view__(self, view="RZ", **styles) -> GeoObject:

        geo = {}
        if view != "RZ":
            geo["antenna"] = [antenna.name for antenna in self.antenna]
            styles["antenna"] = {"$matplotlib": {"color": "blue"}, "text": True}

        return geo, styles
