"""Source terms for the core transport equations"""

import typing

from spdm.core.path import Path
from spdm.core.sp_tree import annotation, sp_property, SpTree
from spdm.core.htree import Set
from spdm.core.expression import Expression
from spdm.core.mesh import Mesh
from spdm.core.time import WithTime
from spdm.core.domain import WithDomain
from spdm.model.process import ProcessBundle
from spdm.model.actor import Actor
from spdm.model.context import Context

from fytok.utils.base import IDS, FyEntity, WithIdentifier

from fytok.modules.utilities import CoreVectorComponents, CoreRadialGrid, DistributionSpecies, Species
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from fytok.ontology import core_sources


class CoreSourcesSpecies(Species, SpTree):
    """Source terms related to electrons"""

    class _Decomposed(SpTree):
        """Source terms decomposed for the particle transport equation, assuming
        core_radial_grid 3 levels above"""

        implicit_part: Expression
        """ Implicit part of the source term, i.e. to be multiplied by the equation's
        primary quantity"""

        explicit_part: Expression
        """ Explicit part of the source term"""

    particles: Expression = annotation(units="s^-1.m^-3", default_value=0)
    """Source term for electron density equation"""

    particles_decomposed: _Decomposed

    @sp_property(units="s^-1")
    def particles_inside(self) -> Expression:
        """Electron source inside the flux surface. Cumulative volume integral of the
        source term for the electron density equation."""
        return self.particles.I

    energy: Expression = annotation(units="W.m^-3", default_value=0)
    """Source term for the electron energy equation"""

    energy_decomposed: _Decomposed

    @sp_property(units="W")
    def power_inside(self) -> Expression:
        """Power coupled to electrons inside the flux surface. Cumulative volume integral
        of the source term for the electron energy equation"""
        return self.energy.I

    momentum: CoreVectorComponents = annotation(units="kg.m^-1.s^-2")


class CoreSourcesElectrons(CoreSourcesSpecies, default_value={"label": "electron"}):
    """String identifying the neutral species (e.g. H, D, T, He, C, ...)"""


class CoreSourcesNeutral(CoreSourcesSpecies):
    pass


class CoreSourcesGlobalQuantities(core_sources.core_sources_source_global):
    pass


class CoreSourcesProfiles1D(WithDomain, core_sources.core_sources_source_profiles_1d, domain="rho_tor_norm"):
    """Source terms profiles 1D"""

    grid: CoreRadialGrid
    """ Radial grid"""

    total_ion_energy: Expression = annotation(units="W.m^-3")
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
    ion: Set[CoreSourcesSpecies]

    Neutral = CoreSourcesNeutral
    neutral: Set[CoreSourcesNeutral]


class CoreSourcesProfiles2D(WithDomain, core_sources.core_sources_source_profiles_2d, domain="grid"):
    grid: Mesh


class CoreSourcesSource(
    WithIdentifier,
    Actor,
    FyEntity,
    plugin_prefix="core_sources/source/",
):
    """Source model for the core transport equations"""

    class InPorts(Actor.InPorts):
        equilibrium: Equilibrium
        core_profiles: CoreProfiles

    species: DistributionSpecies

    GlobalQuantities = CoreSourcesGlobalQuantities
    global_quantities: CoreSourcesGlobalQuantities

    Profiles1D = CoreSourcesProfiles1D
    profiles_1d: CoreSourcesProfiles1D

    Profiles2D = CoreSourcesProfiles2D
    profiles_2d: CoreSourcesProfiles2D

    def execute(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs) -> typing.Self:
        return CoreSources.Source(
            Path().update(
                super().execute(*args, **kwargs),
                {
                    "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
                    "profiles_1d": {
                        "grid": core_profiles.profiles_1d.grid,
                        "electrons": {},
                        "ion": [ion.label for ion in core_profiles.profiles_1d.ion],
                    },
                },
            ),
        )


class CoreSources(IDS, Context, FyEntity, code={"name": "core_sources"}):
    """Source terms for the core transport equations"""

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            args = ({"source": args[0]},)

        super().__init__(*args, **kwargs)

    InPorts = CoreSourcesSource.InPorts

    Source = CoreSourcesSource
    source: ProcessBundle[CoreSourcesSource]

    def __str__(self) -> str:
        return str(self.source)

    def execute(self, *args, **kwargs) -> typing.Any:
        return {"source": self.source.execute(*args, **kwargs)}
