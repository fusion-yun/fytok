
from ..ontology import magnetics
from spdm.core.geo_object import GeoObject
from spdm.geometry.point import Point
from spdm.utils.tags import _not_found_


class Magnetics(magnetics._T_magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """

    def __view__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}
        match view_point.lower():
            case "rz":
                if self.b_field_tor_probe is not _not_found_:
                    geo["b_field_tor_probe"] = [Point(p.position[0].r,  p.position[0].z, name=p.name)
                                                for p in self.b_field_tor_probe]
                if self.flux_loop is not _not_found_:
                    geo["flux_loop"] = [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.flux_loop]

        return geo 
