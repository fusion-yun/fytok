from spdm.core.geo_object import GeoObject
from spdm.geometry.polygon import Rectangle
from ..utils.logger import logger
from ..ontology import pf_active


class PFActive(pf_active._T_pf_active):
    def __view__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}

        match view_point.lower():
            case "rz":
                geo_coils = []
                for coil in self.coil:
                    rect = coil.element[0].geometry.rectangle
                    geo_coils.append(
                        Rectangle(
                            rect.r - rect.width / 2.0,
                            rect.z - rect.height / 2.0,
                            rect.width,
                            rect.height,
                            name=coil.name,
                            styles={"$matplotlib": {"color": "black"}, "text": True},
                        )
                    )

                geo["coil"] = geo_coils

        return geo 
