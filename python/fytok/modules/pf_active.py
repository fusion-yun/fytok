""" 极向场线圈 """

from spdm.geometry.polygon import RectangleRZ

from spdm.model.component import Component
from fytok.utils.base import IDS, FyEntity
from fytok.ontology import pf_active


class PFActive(IDS, Component,FyEntity,  pf_active.pf_active):
    """极向场线圈"""

    def __view__(self, view_point="RZ", **styles):
        geo = {"$styles": styles}

        match view_point.lower():
            case "rz":
                geo_coils = []
                for coil in self.coil:
                    # for element in coil.element:
                    rect = coil.element[0].geometry.rectangle
                    geo_coils.append(
                        RectangleRZ(
                            (rect.r - rect.width / 2.0, rect.r + rect.width / 2.0),
                            (rect.z - rect.height / 2.0, rect.z + rect.height / 2.0),
                            name=coil.name,
                            styles={"$matplotlib": {"color": "black"}, "text": True},
                        )
                    )

                geo["coil"] = geo_coils

        return geo
