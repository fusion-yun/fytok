import typing
from spdm.core.aos import AoS
from spdm.geometry.circle import Circle
from spdm.core.geo_object import GeoObject
from spdm.geometry.polyline import Polyline
from spdm.utils.tags import _not_found_

from ..ontology import wall

from ..utils.logger import logger


class Wall(wall._T_wall):
    """Description of the torus wall and its interaction with the plasma"""

    Description2D = wall._T_wall_2d

    description_2d: AoS[Description2D]

    def __view__(self, view_point="RZ", **kwargs) -> GeoObject | typing.List[GeoObject]:
        geo = {}

        desc = self.description_2d[0]  # 0 for equilibrium codes

        match view_point.lower():
            case "top":
                vessel_r = desc.vessel.unit[0].annular.outline_outer.r
                # vessel_z = desc.vessel.unit[0].annular.outline_outer.z
                geo["vessel_outer"] = [Circle(0.0, 0.0, vessel_r.min()), Circle(0.0, 0.0, vessel_r.max())]

            case "rz":
                if desc.limiter.unit[0].outline.r is not _not_found_:
                    geo["limiter"] = Polyline(
                        desc.limiter.unit[0].outline.r,
                        desc.limiter.unit[0].outline.z,
                        styles={"$matplotlib": {"edgecolor": "green"}},
                    )
                else:
                    units = []
                    for unit in desc.vessel.unit:
                        if unit.annular is not _not_found_:
                            units.append(
                                {
                                    "annular": {
                                        "vessel_inner": Polyline(
                                            unit.annular.outline_inner.r,
                                            unit.annular.outline_inner.z,
                                            styles={"$matplotlib": {"edgecolor": "blue"}},
                                        ),
                                        "vessel_outer": Polyline(
                                            unit.annular.outline_outer.r,
                                            unit.annular.outline_outer.z,
                                            styles={"$matplotlib": {"edgecolor": "blue"}},
                                        ),
                                    }
                                }
                            )

                        else:
                            elements = []
                            for element in unit.element:
                                elements.append(Polyline(element.outline.r, element.outline.z, name=element.name))
                            units.append({"element": elements})

                        geo["unit"] = units

        return geo
