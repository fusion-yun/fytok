from spdm.geometry.polygon import Rectangle

from spdm.model.component import Component
from fytok.utils.base import IDS, FyModule
from fytok.ontology import pf_active


class PFActive(IDS, FyModule, Component, pf_active.pf_active):
    def __view__(self, view_point="RZ", **styles):
        geo = {"$styles": styles}

        match view_point.lower():
            case "rz":
                geo_coils = []
                for coil in self.coil:
                    # for element in coil.element:
                    rect = coil.element[0].geometry.rectangle
                    geo_coils.append(
                        Rectangle(
                            (rect.r - rect.width / 2.0, rect.r + rect.width / 2.0),
                            (rect.z - rect.height / 2.0, rect.z + rect.height / 2.0),
                            name=coil.name,
                            styles={"$matplotlib": {"color": "black"}, "text": True},
                        )
                    )

                geo["coil"] = geo_coils

        return geo
