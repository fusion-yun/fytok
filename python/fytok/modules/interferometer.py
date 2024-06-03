from __future__ import annotations

from ..utils.logger import logger

from spdm.core.geo_object import GeoObject
from spdm.geometry.line import Line
from spdm.utils.tags import _not_found_
from ..ontology import interferometer


class Interferometer(interferometer._T_interferometer):
    def __view__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}
        match view_point.lower():
            case "rz":
                if self.channel is not _not_found_:
                    geo["channel"] = [
                        Line(
                            [channel.line_of_sight.first_point.r, channel.line_of_sight.first_point.z],
                            [channel.line_of_sight.second_point.r, channel.line_of_sight.second_point.z],
                            name=channel.name,
                            styles={"$matplotlib": {"color": "blue"}, "text": True},
                        )
                        for channel in self.channel
                    ]

        return geo
