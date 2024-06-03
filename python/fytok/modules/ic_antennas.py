from __future__ import annotations

from spdm.core.geo_object import GeoObject

from ..ontology import ic_antennas 


class ICAntennas(ic_antennas._T_ic_antennas):
    def __view__(self, view="RZ", **kwargs) -> GeoObject:

        geo = {}
        styles = {}
        if view != "RZ":
            geo["antenna"] = [antenna.name for antenna in self.antenna]
            styles["antenna"] = {"$matplotlib": {"color": 'blue'}, "text": True}

        return geo 
