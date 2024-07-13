from spdm.utils.tags import _not_found_

from spdm.geometry.point import Point
from spdm.model.component import Component

from fytok.utils.base import IDS, FyModule
from fytok.ontology import magnetics


class Magnetics(IDS, FyModule, Component, magnetics.magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control."""

    def __view__(self, view_point="RZ", **styles):
        geo = {"$styles": styles}
        match view_point.lower():
            case "rz":
                if self.b_field_tor_probe is not _not_found_:
                    geo["b_field_tor_probe"] = [
                        Point(p.position[0].r, p.position[0].z, name=p.name) for p in self.b_field_tor_probe
                    ]
                if self.flux_loop is not _not_found_:
                    geo["flux_loop"] = [Point(p.position[0].r, p.position[0].z, name=p.name) for p in self.flux_loop]

        return geo
