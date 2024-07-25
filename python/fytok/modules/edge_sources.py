from copy import copy, deepcopy
import math
from spdm.utils.tags import _not_found_
from spdm.core.htree import List, Dict
from spdm.core.sp_tree import annotation, sp_property, SpTree

from spdm.core.aos import AoS
from spdm.core.expression import Expression, Variable, zero, one

from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.utilities import FyActor
from fytok.utils.atoms import atoms

from fytok.ontology import edge_sources

from functools import cached_property


class CoreSourcesSource(FyActor):
    _plugin_prefix = "fytok.modules.core_sources.source."

    identifier: str

    species: DistributionSpecies

    TimeSlice = CoreSourcesTimeSlice

    time_slice: StateTreeSequence[CoreSourcesTimeSlice]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, *args, **kwargs) -> CoreSourcesTimeSlice:
        current: CoreSourcesTimeSlice = super().preprocess(*args, **kwargs)

        grid = current.get_cache("profiles_1d/grid", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            eq_grid: CoreRadialGrid = self.inports["/equilibrium/time_slice/0/profiles_1d/grid"].fetch()

            if isinstance(grid, dict):
                new_grid = grid
                if not isinstance(grid.get("psi_axis", _not_found_), float):
                    new_grid["psi_axis"] = eq_grid.psi_axis
                    new_grid["psi_boundary"] = eq_grid.psi_boundary
                    new_grid["rho_tor_boundary"] = eq_grid.rho_tor_boundary
                # new_grid = {
                #     **eq_grid._cache,
                #     **{k: v for k, v in grid.items() if v is not _not_found_ and v is not None},
                # }
            else:
                rho_tor_norm = kwargs.get("rho_tor_norm", _not_found_)

                if rho_tor_norm is _not_found_:
                    rho_tor_norm = self.code.parameters.rho_tor_norm

                new_grid = eq_grid.remesh(rho_tor_norm)

            current["profiles_1d/grid"] = new_grid

        return current

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args, **kwargs) -> CoreSourcesTimeSlice:
        return super().fetch(profiles_1d.rho_tor_norm, *args, **kwargs)

    def flush(self) -> CoreSourcesTimeSlice:
        super().flush()

        current: CoreSourcesTimeSlice = self.time_slice.current

        profiles_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports[
            "core_profiles/time_slice/0/profiles_1d"
        ].fetch()
        # eq_grid: CoreRadialGrid = self.inports["equilibrium/time_slice/0/profiles_1d/grid"].fetch()

        current.update(self.fetch(profiles_1d)._cache)

        return current

    def refresh(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_profiles: CoreProfiles = None,
        **kwargs,
    ) -> CoreSourcesTimeSlice:
        return super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)


@sp_tree
class EdgeSources:

    Source = CoreSourcesSource

    source: AoS[EdgeSources.Source]
