import typing

from spdm.utils.tags import _not_found_
from spdm.core.sp_tree import sp_property, SpTree
from spdm.core.htree import List, Set
from spdm.core.expression import Expression
from spdm.core.mesh import Mesh
from spdm.core.category import WithCategory
from spdm.core.domain import WithDomain
from spdm.core.history import WithHistory

from spdm.model.entity import Entity
from spdm.model.port import Ports
from spdm.model.actor import Actor

from fytok.utils.atoms import atoms
from fytok.utils.base import IDS, FyModule

from fytok.modules.utilities import CoreVectorComponents, CoreRadialGrid, DistributionSpecies, Species
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from fytok.ontology import core_sources


class CoreSourcesSpecies(Species):
    """Source terms related to electrons"""

    class _Decomposed(SpTree):
        """Source terms decomposed for the particle transport equation, assuming
        core_radial_grid 3 levels above"""

        implicit_part: Expression
        """ Implicit part of the source term, i.e. to be multiplied by the equation's
        primary quantity"""

        explicit_part: Expression
        """ Explicit part of the source term"""

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


class CoreSourcesElectrons(CoreSourcesSpecies, label="electron"):
    """String identifying the neutral species (e.g. H, D, T, He, C, ...)"""


class CoreSourcesNeutral(CoreSourcesSpecies):
    pass


class CoreSourcesGlobalQuantities(core_sources.core_sources_source_global):
    pass


class CoreSourcesProfiles1D(WithDomain, core_sources.core_sources_source_profiles_1d, domain="rho_tor_norm"):
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
    ion: Set[CoreSourcesSpecies]

    Neutral = CoreSourcesNeutral
    neutral: Set[CoreSourcesNeutral]


class CoreSourcesProfiles2D(WithDomain, core_sources.core_sources_source_profiles_2d, domain="grid"):
    grid: Mesh


class CoreSourcesSource(
    FyModule,
    WithHistory,
    WithCategory,
    Actor,
    plugin_prefix="core_sources/source/",
):

    class InPorts(Ports):
        equilibrium: Equilibrium
        core_profiles: CoreProfiles

    in_ports: InPorts

    species: DistributionSpecies

    GlobalQuantities = CoreSourcesGlobalQuantities
    global_quantities: CoreSourcesGlobalQuantities

    Profiles1D = CoreSourcesProfiles1D
    profiles_1d: CoreSourcesProfiles1D

    Profiles2D = CoreSourcesProfiles2D
    profiles_2d: CoreSourcesProfiles2D

    def refresh(self, *args, psi_norm=None, radial_grid=None, **kwargs) -> typing.Self:

        if psi_norm is None:
            psi_norm = self.in_ports.core_profiles.profiles_1d.psi_norm

        if radial_grid is None:
            radial_grid = self.in_ports.equilibrium.profiles_1d.grid.fetch(psi_norm)

        return super().refresh({"profiles_1d": {"psi_norm": psi_norm, "grid": radial_grid}}, *args, **kwargs)


class CoreSources(IDS, Entity, plugin_name="core_sources"):
    source: Set[CoreSourcesSource]
