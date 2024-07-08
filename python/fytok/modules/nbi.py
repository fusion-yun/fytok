import typing

from spdm.core.component import Component
from spdm.geometry.line import Line
from spdm.geometry.polygon import Rectangle
from fytok.utils.base import IDS, FyModule

from fytok.ontology import nbi


def draw_nbi_unit(unit: nbi._T_nbi_unit, name: str):
    geo = None
    if unit.source.geometry_type == 3:
        geo = [Line(), Rectangle(name=unit.name)]

    else:
        pass

    return geo


class NBI(IDS, FyModule, Component, nbi.nbi):
    def __view__(self, view_point="RZ", **styles):
        geo = {}

        match view_point.lower():
            case "top":
                geo["unit"] = [draw_nbi_unit(unit) for unit in self.unit]

        return geo, styles
