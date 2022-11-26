
from functools import cached_property

from spdm.data import Dict, File, Link, List, Node, Path, Query, sp_property,Function

from ...IDS import IDS
from ..common.Misc import Identifier, VacuumToroidalField
from ..transport.CoreProfiles import CoreProfiles1D


class CoreInstantChange(Dict[Node]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def identifier(self) -> Identifier:
        r"""
            Instant change term identifier. Available options (refer to the children of this identifier structure) :

            Name           | Index       | Description
            ---------------+-------------+-----------------------------------------------------
            unspecified    | 0           | Unspecified instant changes
            total          | 1           | Total instant changes; combines all types of events
            pellet         | 2           | Instant changes from a pellet
            sawtooth       | 3           | Instant changes from a sawtooth
            elm            | 4           | Instant changes from an edge localised mode
        """
        return self["identifier"]

    @cached_property
    def profiles_1d(self) -> CoreProfiles1D:
        r"""
            Changes in 1D core profiles for various time slices. This structure mirrors core_profiles/profiles_1d and 
            describes instant changes to each of these physical quantities (i.e. a signed difference quantity after change - quantity before change) {dynamic}
        """
        return self["profiles_1d"]


class CoreInstantChanges(IDS):
    r"""
        Instant changes of the radial core plasma profiles due to pellet, MHD, ...
    """
    _IDS = "core_instant_changes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def vacuum_toroidal_field(self):
        return VacuumToroidalField(**self.get("vacuum_toroidal_field", _not_found_)._as_dict())