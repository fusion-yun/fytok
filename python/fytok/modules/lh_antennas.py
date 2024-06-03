from __future__ import annotations

from spdm.core.geo_object import GeoObject

from ..ontology import lh_antennas


class LHAntennas(lh_antennas._T_lh_antennas):
    def __view__(self, view_point="RZ", **kwargs) -> GeoObject:

        geo = {}
        styles = {}
        match view_point.lower():
            case "top":
                geo["antenna"] = [antenna.name for antenna in self.antenna]
                styles["antenna"] = {"$matplotlib": {"color": 'blue'}, "text": True}

        return geo 
