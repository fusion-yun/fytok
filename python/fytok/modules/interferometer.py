""" Interferometer module """

from spdm.geometry.line import Line
from spdm.utils.tags import _not_found_
from spdm.model.component import Component

from fytok.utils.base import IDS, FyEntity
from fytok.ontology import interferometer


class Interferometer(FyEntity, IDS, Component, interferometer.Interferometer):
    def __view__(self, view_point="RZ", **kwargs) -> dict:
        geo = {"$styles": kwargs}
        match view_point.lower():
            case "rz":
                if self.channel is not _not_found_:
                    geo["channel"] = [
                        Line(
                            [
                                [
                                    channel.line_of_sight.first_point.r,
                                    channel.line_of_sight.first_point.z,
                                ],
                                [
                                    channel.line_of_sight.second_point.r,
                                    channel.line_of_sight.second_point.z,
                                ],
                            ],
                            name=channel.name,
                            styles={"$matplotlib": {"color": "blue"}, "text": True},
                        )
                        for channel in self.channel
                    ]
        return geo
