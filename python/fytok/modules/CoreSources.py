from __future__ import annotations
from copy import copy, deepcopy
import math
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression, Variable, zero, one
from spdm.utils.tags import _not_found_

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import *
from ..utils.atoms import atoms

from ..ontology import core_sources


@sp_tree
class CoreSourcesSpecies(SpTree):
    """Source terms related to electrons"""

    @sp_tree
    class _Decomposed(SpTree):
        """Source terms decomposed for the particle transport equation, assuming
        core_radial_grid 3 levels above"""

        implicit_part: Expression
        """ Implicit part of the source term, i.e. to be multiplied by the equation's
        primary quantity"""

        explicit_part: Expression
        """ Explicit part of the source term"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.z is _not_found_:
            ion = atoms[self.label]
            self.z = ion.z
            self.a = ion.a

    label: str = sp_property(alias="@name")
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    z: int
    """ Charge number of the neutral species"""

    a: float
    """ Mass number of the neutral species"""

    particles: Expression = sp_property(units="s^-1.m^-3", default_value=0)
    """Source term for electron density equation"""

    particles_decomposed: _Decomposed

    @sp_property(units="s^-1")
    def particles_inside(self) -> Expression:
        """Electron source inside the flux surface. Cumulative volume integral of the
        source term for the electron density equation."""
        return self.particles.I

    energy: Expression = sp_property(units="W.m^-3", default_value=0)
    """Source term for the electron energy equation"""

    energy_decomposed: _Decomposed

    @sp_property(units="W")
    def power_inside(self) -> Expression:
        """Power coupled to electrons inside the flux surface. Cumulative volume integral
        of the source term for the electron energy equation"""
        return self.energy.I

    momentum: CoreVectorComponents = sp_property(units="kg.m^-1.s^-2")


@sp_tree
class CoreSourcesElectrons(CoreSourcesSpecies):
    label: str = "electron"
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""


@sp_tree
class CoreSourcesNeutral(core_sources._T_core_sources_source_profiles_1d_neutral):
    pass


@sp_tree(coordinate1="grid/rho_tor_norm")
class CoreSourcesProfiles1D(core_sources._T_core_sources_source_profiles_1d):
    grid: CoreRadialGrid
    """ Radial grid"""

    total_ion_energy: Expression = sp_property(units="W.m^-3")
    """Total ion energy source"""

    @sp_property(units="W")
    def total_ion_power_inside(self) -> Expression:
        return self.torque_tor_inside.I

    momentum_tor: Expression

    torque_tor_inside: Expression

    momentum_tor_j_cross_b_field: Expression

    j_parallel: Expression

    current_parallel_inside: Expression

    conductivity_parallel: Expression

    Electrons = CoreSourcesElectrons
    electrons: CoreSourcesElectrons

    Ion = CoreSourcesSpecies
    ion: AoS[CoreSourcesSpecies]

    Neutral = CoreSourcesNeutral
    neutral: AoS[CoreSourcesNeutral]


@sp_tree
class CoreSourcesGlobalQuantities(core_sources._T_core_sources_source_global):
    pass


@sp_tree
class CoreSourcesTimeSlice(TimeSlice):
    Profiles1D = CoreSourcesProfiles1D

    GlobalQuantities = CoreSourcesGlobalQuantities

    profiles_1d: CoreSourcesProfiles1D

    global_quantities: CoreSourcesGlobalQuantities


@sp_tree
class CoreSourcesSource(Module):
    _plugin_prefix = "fytok.plugins.core_sources.source."

    identifier: str

    species: DistributionSpecies

    TimeSlice = CoreSourcesTimeSlice

    time_slice: TimeSeriesAoS[CoreSourcesTimeSlice]

    def preprocess(self, *args, **kwargs) -> CoreSourcesTimeSlice:
        current = super().preprocess(*args, **kwargs)

        grid = current.get_cache("profiles_1d/grid", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            eq_grid: CoreRadialGrid = self.inports["equilibrium/time_slice/0/profiles_1d/grid"].fetch()

            if isinstance(grid, dict):
                new_grid = {
                    **eq_grid._cache,
                    **{k: v for k, v in grid.items() if v is not _not_found_ and v is not None},
                }
            else:
                rho_tor_norm = kwargs.get("rho_tor_norm", _not_found_)

                if rho_tor_norm is _not_found_:
                    rho_tor_norm = self.code.parameters.rho_tor_norm

                new_grid = eq_grid.remesh(rho_tor_norm)

            current["profiles_1d/grid"] = new_grid
            
        return current

    def fetch(self, first=None, *args, **kwargs) -> CoreSourcesTimeSlice:
        if isinstance(first, array_type):
            rho_tor_norm = first
        elif isinstance(first, CoreProfiles.TimeSlice.Profiles1D):
            rho_tor_norm = first.rho_tor_norm
        elif first is None:
            rho_tor_norm = kwargs.get("rho_tor_norm", None)
        else:
            raise TypeError(f"Unknown argument {first}")

        return super().fetch(rho_tor_norm)

    def flush(self) -> CoreSourcesTimeSlice:
        super().flush()

        current: CoreSourcesTimeSlice = self.time_slice.current

        profiles_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

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
class CoreSources(IDS):
    Source = CoreSourcesSource

    source: AoS[CoreSourcesSource]

    def initialize(self, *args, **kwargs) -> None:
        super().initialize(*args, **kwargs)
        for source in self.source:
            source.initialize()

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, **kwargs)

        for source in self.source:
            source.refresh(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

    def advance(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().advance(*args, **kwargs)

        for source in self.source:
            source.advance(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

    def flush(self):
        super().flush()

        for source in self.source:
            source.flush()
