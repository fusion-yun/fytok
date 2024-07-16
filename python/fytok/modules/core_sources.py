import typing

from spdm.utils.tags import _not_found_
from spdm.core.sp_tree import sp_property, SpTree
from spdm.core.htree import List
from spdm.core.expression import Expression
from spdm.core.domain import WithDomain

from spdm.model.port import Ports
from spdm.model.actor import Actor

from fytok.utils.atoms import atoms
from fytok.utils.base import IDS, FyActor

from fytok.modules.utilities import CoreVectorComponents, CoreRadialGrid, DistributionSpecies
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from fytok.ontology import core_sources


class CoreSourcesSpecies(SpTree):
    """Source terms related to electrons"""

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


class CoreSourcesElectrons(CoreSourcesSpecies):
    label: str = "electron"
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""


class CoreSourcesNeutral(core_sources.core_sources_source_profiles_1d_neutral):
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
    ion: List[CoreSourcesSpecies]

    Neutral = CoreSourcesNeutral
    neutral: List[CoreSourcesNeutral]


class CoreSourcesProfiles2D(WithDomain, core_sources.core_sources_source_profiles_2d, domain="grid"):
    pass


class CoreSourcesSource(FyActor, plugin_prefix="core_sources/source/"):

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
            psi_norm = self.in_ports.core_profiles.fetch("profiles_1d/psi_norm")
        if radial_grid is None:
            radial_grid = self.in_ports.equilibrium.fetch("profiles_1d/grid", psi_norm)

        return super().refresh({"profiles_1d": {"psi_norm": psi_norm, "grid": radial_grid}}, *args, **kwargs)


class CoreSources(IDS):

    Source = CoreSourcesSource

    source: List[CoreSourcesSource]

    def initialize(self, *args, **kwargs):
        for m in self.model:
            m.initialize(*args, **kwargs)
