from spdm.core.component import Component
from fytok.utils.base import IDS, FyModule
from fytok.ontology import ec_launchers


class ECLaunchers(IDS, FyModule, Component, ec_launchers.ec_launchers):
    def __view__(self, view_point="RZ", **styles):
        geo = {}

        match view_point.lower():
            case "top":
                geo["beam"] = [beam.name for beam in self.beam]
                styles["beam"] = {"$matplotlib": {"color": "blue"}, "text": True}

        return geo
