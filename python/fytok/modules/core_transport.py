import numpy as np

from spdm.utils.tags import _not_found_
from spdm.core.htree import List
from spdm.core.sp_tree import sp_property, SpTree
from spdm.core.expression import Expression
from spdm.core.domain import WithDomain
from spdm.core.time import WithTime
from spdm.core.mesh import Mesh

from spdm.model.actor import Actor

from fytok.utils.atoms import atoms
from fytok.utils.base import IDS, FyModule, FyActor
from fytok.modules.utilities import CoreRadialGrid, VacuumToroidalField
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


class CoreTransportElectrons(core_transport.core_transport_model_electrons):
    label: str = "electrons"
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


class CoreTransportIon(core_transport.core_transport_model_ions):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ion = atoms[self.label]
        self.z = ion.z
        self.a = ion.a

    label: str = sp_property(alias="@name")
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    z: int
    """ Charge number of the neutral species"""

    a: float
    """ Mass number of the neutral species"""

    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


class CoreTransportNeutral(core_transport.core_transport_model_neutral):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy


class CoreTransportProfiles1D(WithDomain, core_transport.core_transport_model_profiles_1d, domain="grid"):

    grid: CoreRadialGrid
    """ Radial grid"""

    grid_d: CoreRadialGrid = sp_property(alias="grid")

    grid_v: CoreRadialGrid = sp_property(alias="grid")

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid.rho_tor_norm
        return self.grid.fetch(0.5 * (rho_tor_norm[:-1] + rho_tor_norm[1:]))

    Electrons = CoreTransportElectrons
    electrons: CoreTransportElectrons

    Ion = CoreTransportIon
    ion: List[CoreTransportIon]

    Neutral = CoreTransportNeutral
    neutral: List[CoreTransportNeutral]


class CoreTransportProfiles2D(WithDomain, core_transport.core_transport_model_profiles_2d, domain="grid"):
    grid: Mesh


class CoreTransportModel(FyActor, plugin_prefix="core_transport/model/"):

    vacuum_toroidal_field: VacuumToroidalField

    flux_multiplier: float = 0.0

    Profiles1D = CoreTransportProfiles1D
    profiles_1d: CoreTransportProfiles1D

    Profiles2D = CoreTransportProfiles2D
    profiles_2d: CoreTransportProfiles2D

    def preprocess(self, *args, **kwargs):
        current = super().preprocess(*args, **kwargs)

        current["vacuum_toroidal_field"] = self.inports["/equilibrium/time_slice/0/vacuum_toroidal_field"].fetch()

        grid = current.get_cache("profiles_1d/grid_d", _not_found_)

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
                rho_tor_norm = kwargs.get("rho_tor_norm", self.code.parameters.rho_tor_norm)
                new_grid = eq_grid.remesh(rho_tor_norm)

            current["profiles_1d/grid_d"] = new_grid

        return current

    def execute(self, current, *previous):
        return super().execute(current, *previous)

    def postprocess(self, current):
        return super().postprocess(current)

    def fetch(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], CoreProfiles.Profiles1D):
            args = (args[0].rho_tor_norm, *args[1:])

        return super().fetch(*args, **kwargs)

    def flush(self):
        super().flush()

        profiles_1d: CoreProfiles.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

        current.update(self.fetch(profiles_1d)._cache)

        return current

    def refresh(
        self,
        *args,
        core_profiles: CoreProfiles = None,
        equilibrium: Equilibrium = None,
        **kwargs,
    ):
        return super().refresh(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)

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

    model: List[CoreTransportModel]
    """ Core transport model"""

    def initialize(self, *args, **kwargs):
        for m in self.model:
            m.initialize(*args, **kwargs)
