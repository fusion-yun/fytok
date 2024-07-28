import abc
import typing
import numpy as np

from spdm.utils.tags import _not_found_
from spdm.core.path import Path
from spdm.core.htree import Set
from spdm.core.sp_tree import annotation, sp_property
from spdm.core.expression import Expression
from spdm.core.time import WithTime
from spdm.core.domain import WithDomain
from spdm.core.category import WithCategory

from spdm.core.mesh import Mesh
from spdm.model.actor import Actor
from spdm.model.context import Context
from spdm.model.process import ProcessBundle

from fytok.utils.base import IDS, FyEntity

from fytok.modules.utilities import CoreRadialGrid, VacuumToroidalField, Species
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from fytok.ontology import core_transport


class CoreTransportModelParticles(core_transport.core_transport_model_2_density):
    d: Expression = annotation(domain=".../grid_d/rho_tor_norm")
    v: Expression = annotation(domain=".../grid_v/rho_tor_norm")
    flux: Expression = annotation(domain=".../grid_flux/rho_tor_norm")


class CoreTransportModelEnergy(core_transport.core_transport_model_2_energy):
    d: Expression = annotation(domain=".../grid_d/rho_tor_norm")
    v: Expression = annotation(domain=".../grid_v/rho_tor_norm")
    flux: Expression = annotation(domain=".../grid_flux/rho_tor_norm")


class CoreTransportModelMomentum(core_transport.core_transport_model_4_momentum):
    d: Expression = annotation(domain=".../grid_d/rho_tor_norm")
    v: Expression = annotation(domain=".../grid_v/rho_tor_norm")
    flux: Expression = annotation(domain=".../grid_flux/rho_tor_norm")


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


class CoreTransportProfiles1D(WithDomain, domain="grid/rho_tor_norm"):
    """Profiles of core transport"""

    grid: CoreRadialGrid
    """ Radial grid"""

    grid_d: CoreRadialGrid = annotation(alias="grid")

    grid_v: CoreRadialGrid = annotation(alias="grid")

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid.rho_tor_norm
        return self.grid.remesh(rho_tor_norm=0.5 * (rho_tor_norm[:-1] + rho_tor_norm[1:]))

    electrons: CoreTransportElectrons
    ion: Set[CoreTransportIon]
    neutral: Set[CoreTransportNeutral]

    rho_tor_norm: Expression = annotation(alias="grid/rho_tor_norm", label=r"$\bar{\rho}_{tor}$", units="-")

    conductivity_parallel: Expression = annotation(label=r"$\sigma_{\parallel}$", units=r"$\Omega^{-1}\cdot m^{-1}$")


class CoreTransportProfiles2D(WithDomain, domain="grid"):
    grid: Mesh


class CoreTransportModel(
    WithCategory,
    Actor,
    FyEntity,
    plugin_prefix="core_transport/model/",
):
    """CoreTransport Model"""

    class InPorts(Actor.InPorts):
        core_profiles: CoreProfiles
        equilibrium: Equilibrium

    in_ports: InPorts  # type:ignore

    flux_multiplier: float = 0.0

    vacuum_toroidal_field: VacuumToroidalField

    Profiles1D = CoreTransportProfiles1D
    profiles_1d: CoreTransportProfiles1D

    Profiles2D = CoreTransportProfiles2D
    profiles_2d: CoreTransportProfiles2D

    def execute(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs) -> typing.Self:
        return self.__class__(
            Path().update(
                super().execute(*args, **kwargs),
                {
                    "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
                    "profiles_1d": {
                        "grid": equilibrium.profiles_1d.grid.remesh(
                            rho_tor_norm=core_profiles.profiles_1d.grid.rho_tor_norm
                        ),
                        "ion": [ion.label for ion in core_profiles.profiles_1d.ion],
                    },
                },
            ),
        )

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


class CoreTransport(WithTime, IDS, Context, FyEntity, code={"name": "core_transport"}):
    """芯部输运"""

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            args = ({"model": args[0]},)
        super().__init__(*args, **kwargs)

    Model = CoreTransportModel

    in_ports: CoreTransportModel.InPorts  # type:ignore

    model: ProcessBundle[CoreTransportModel]

    def __str__(self) -> str:
        return str(self.model)
