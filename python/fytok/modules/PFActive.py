import typing

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Rectangle
from ..utils.logger import logger

from ..schema import pf_active


class PFActive(pf_active._T_pf_active):
    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}

        match view_point.lower():
            case "rz":
                geo_coils = []
                for coil in self.coil:
                    rect = coil.element[0].geometry.rectangle
                    geo_coils.append(Rectangle(rect.r - rect.width / 2.0,  rect.z -
                                               rect.height / 2.0,   rect.width,  rect.height,
                                               name=coil.name))

                geo["coil"] = geo_coils
                styles["coil"] = {"$matplotlib": {"color": 'black'}, "text": True}

        return geo, styles
