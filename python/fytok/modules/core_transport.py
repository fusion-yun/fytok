import abc
import typing
import numpy as np

from spdm.utils.tags import _not_found_
from spdm.core.htree import Set
from spdm.core.sp_tree import sp_property
from spdm.core.expression import Expression
from spdm.core.domain import WithDomain
from spdm.core.category import WithCategory

from spdm.core.mesh import Mesh
from spdm.model.actor import Actor

from fytok.utils.base import IDS, FyEntity

from fytok.modules.utilities import CoreRadialGrid, VacuumToroidalField, Species
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from fytok.ontology import core_transport


class CoreTransportModelParticles(core_transport.core_transport_model_2_density):
    d: Expression = sp_property(domain=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(domain=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(domain=".../grid_flux/rho_tor_norm")


class CoreTransportModelEnergy(core_transport.core_transport_model_2_energy):
    d: Expression = sp_property(domain=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(domain=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(domain=".../grid_flux/rho_tor_norm")


class CoreTransportModelMomentum(core_transport.core_transport_model_4_momentum):
    d: Expression = sp_property(domain=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(domain=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(domain=".../grid_flux/rho_tor_norm")


class CoreTransportElectrons(
    Species, core_transport.core_transport_model_electrons, default_value={"label": "electron"}
):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


class CoreTransportIon(Species, core_transport.core_transport_model_ions):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


class CoreTransportNeutral(Species, core_transport.core_transport_model_neutral):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy


class CoreTransportProfiles1D(
    WithDomain, core_transport.core_transport_model_profiles_1d, domain="grid/rho_tor_norm"
):

    grid: CoreRadialGrid
    """ Radial grid"""

    grid_d: CoreRadialGrid = sp_property(alias="grid")

    grid_v: CoreRadialGrid = sp_property(alias="grid")

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid.rho_tor_norm
        return self.grid.remesh(rho_tor_norm=0.5 * (rho_tor_norm[:-1] + rho_tor_norm[1:]))

    electrons: CoreTransportElectrons
    ion: Set[CoreTransportIon]
    neutral: Set[CoreTransportNeutral]

    rho_tor_norm: Expression = sp_property(alias="grid/rho_tor_norm", label=r"$\bar{\rho}_{tor}$", units="-")

    conductivity_parallel: Expression = sp_property(label=r"$\sigma_{\parallel}$", units=r"$\Omega^{-1}\cdot m^{-1}$")


class CoreTransportProfiles2D(WithDomain, core_transport.core_transport_model_profiles_2d, domain="grid"):
    grid: Mesh


class CoreTransportModel(
    FyEntity,
    WithCategory,
    Actor,
    plugin_prefix="core_transport/model/",
):

    class InPorts(Actor.InPorts):
        core_profiles: CoreProfiles
        equilibrium: Equilibrium

    in_ports: InPorts

    flux_multiplier: float = 0.0

    vacuum_toroidal_field: VacuumToroidalField

    Profiles1D = CoreTransportProfiles1D
    profiles_1d: CoreTransportProfiles1D

    Profiles2D = CoreTransportProfiles2D
    profiles_2d: CoreTransportProfiles2D

    def execute(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs) -> typing.Self:
        current = super().execute(*args, **kwargs)

        current.vacuum_toroidal_field = equilibrium.vacuum_toroidal_field

        current.profiles_1d.grid = equilibrium.profiles_1d.grid.remesh(
            rho_tor_norm=core_profiles.profiles_1d.grid.rho_tor_norm
        )

        return current

    @staticmethod
    def _flux2DV(
        spec: CoreProfiles.Profiles1D.Ion,
        ion: CoreProfiles.Profiles1D.Ion,
        R0: float,
        rho_tor_boundary,
    ):
        """Convert flux to d,v ,
        @ref https://wpcd-workflows.github.io/ets.html#ds-and-vs-from-turbulence-codes-to-transport-solvers
        """
        inv_Ln = 1 / R0  # np.max(1 / R0, ion.density.dln / rho_tor_boundary)
        inv_LT = 1 / R0  # np.max(1 / R0, ion.temperature.dln / rho_tor_boundary)
        D_ = np.abs(spec.particles.flux) / inv_Ln
        Chi_ = np.abs(spec.energy.flux) / inv_LT
        D = np.maximum(D_, Chi_ / 5)
        Chi = np.maximum(Chi_, D_ / 5)
        spec.particles.d = D
        spec.particles.v = spec.particles.flux + D * ion.density.dln / rho_tor_boundary
        spec.energy.d = Chi
        spec.energy.v = spec.energy.flux + Chi * ion.temperature.dln / rho_tor_boundary


class CoreTransport(IDS):
    Model = CoreTransportModel
    model: Set[CoreTransportModel]
