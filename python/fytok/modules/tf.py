from __future__ import annotations

import numpy as np
from spdm.core.geo_object import GeoObject
from spdm.geometry.polygon import Polygon
from spdm.utils.constants import TWOPI
from spdm.utils.typing import _not_found_
from ..utils.logger import logger
from ..ontology import tf


class TF(tf._T_tf):

    def __view__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}
        r0 = self.r0

        match view_point.lower():
            case "rz_":
                conductor = self.coil[0].conductor[0]
                geo["coils"] = Polygon(conductor.elements.start_points.r,
                                       conductor.elements.start_points.z,
                                       name=self.coil[0].name)

            case "top":
                if self.is_periodic == 0:
                    coils_n = self.coils_n
                    d_phi = TWOPI/self.coils_n

                    cross_section = self.coil[0].conductor[0].cross_section
                    r = cross_section.delta_r
                    phi = cross_section.delta_phi
                    name = self.coil[0].name
                    geo["coils"] = [
                        Polygon((r0+r)*np.cos(phi+d_phi*i),
                                (r0+r)*np.sin(phi+d_phi*i),
                                name=name+f"{i}") for i in range(coils_n)]

                else:
                    geo["coils"] = [
                        Polygon(
                            (r0 + coil.conductor[0].cross_section.delta_r) *
                            np.cos(coil.conductor[0].cross_section.delta_phi),
                            (r0 + coil.conductor[0].cross_section.delta_r) *
                            np.sin(coil.conductor[0].cross_section.delta_phi),
                            name=coil.name) for coil in self.coil]

        return geo
