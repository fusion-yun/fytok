import numpy as np
import scipy.constants


from spdm.utils.type_hint import array_type
from spdm.core.htree import Set, List
from spdm.core.sp_tree import sp_property, SpTree
from spdm.core.expression import Expression
from spdm.core.field import Field
from spdm.core.mesh import Mesh
from spdm.core.domain import WithDomain
from spdm.core.history import WithHistory
from spdm.model.entity import Entity

from fytok.utils.logger import logger
from fytok.utils.base import IDS, FyModule


from fytok.modules.utilities import (
    CoreVectorComponents,
    PlasmaCompositionSpecies,
    CoreRadialGrid,
    VacuumToroidalField,
    Species,
)

from fytok.ontology import core_profiles

PI = scipy.constants.pi
TWOPI = 2.0 * PI


class CoreProfilesSpecies(Species):

    temperature: Expression = sp_property(units="eV")

    density: Expression = sp_property(units="m^-3")
    # return self.density_thermal + self.density_fast

    density_thermal: Expression = sp_property(units="m^-3")

    density_fast: Expression = sp_property(units="m^-3")

    @sp_property
    def pressure(self) -> Expression:
        # FIXME: coefficient on pressure fast
        return self.density * self.temperature * scipy.constants.electron_volt

    pressure_thermal: Expression = sp_property(units="Pa")

    pressure_fast_perpendicular: Expression = sp_property(units="Pa")

    pressure_fast_parallel: Expression = sp_property(units="Pa")

    rotation_frequency_tor: Expression = sp_property(units="rad.s^-1")

    velocity: CoreVectorComponents = sp_property(units="m.s^-1")


class CoreProfilesState(CoreProfilesSpecies):
    label: str
    """ String identifying state"""

    electron_configuration: str
    """ Configuration of atomic orbitals of this state, e.g. 1s2-2s1"""

    vibrational_level: float = sp_property(units="Elementary Charge Unit")
    """ Vibrational level (can be bundled)"""

    vibrational_mode: str
    """ Vibrational mode of this state, e.g. _A_g_. Need to define, or adopt a standard
        nomenclature."""

    neutral_type: int
    """ Neutral type (if the considered state is a neutral), in terms of energy. ID =1:
        cold; 2: thermal; 3: fast; 4: NBI"""


class CoreProfilesIon(CoreProfilesSpecies):
    z_ion_1d: Expression = sp_property(unit="C")

    @sp_property
    def z_ion_square_1d(self) -> Expression:
        return self.z_ion_1d * self.z_ion_1d

    # velocity: core_profiles_vector_components_2 = sp_property(units="m.s^-1")

    @sp_property(unit="s^-1", default_value=0.1)
    def collision_frequency(self) -> Expression:
        r"""
        collision frequency
        $$
            \tau_{ss}^{-1} = \frac{\sqrt{2} \pi e^4 z_s^4 n_{0s}}{m_s^{1/2} T_{0s}^{3/2}} {\rm ln} \Lambda
        $$
        """
        return (
            np.sqrt(2)
            * PI
            * scipy.constants.elementary_charge**4
            * self.z**4
            * self.density_thermal
            / np.sqrt(self.mass)
            / self.temperature**1.5
            / self._parent.coulomb_logarithm
        )


class CoreProfilesNeutral(CoreProfilesSpecies):
    element: List[PlasmaCompositionSpecies]
    """ List of elements forming the atom or molecule"""

    multiple_states_flag: int
    """ Multiple states calculation flag : 0-Only one state is considered; 1-Multiple
        states are considered and are described in the state structure"""

    state: List[CoreProfilesState]
    """ Quantities related to the different states of the species (energy, excitation,...)"""


class CoreProfilesElectrons(CoreProfilesSpecies):

    def __init__(self, *args, label="e", **kwargs):
        super().__init__(*args, label=label, **kwargs)

    @sp_property(units="-")
    def collisionality_norm(self) -> Expression:
        raise NotImplementedError("collisionality_norm")

    @sp_property
    def tau(self):
        return 1.09e16 * ((self.temperature / 1000) ** (3 / 2)) / self.density / self._parent.coulomb_logarithm

    @sp_property
    def vT(self):
        return np.sqrt(self.temperature * scipy.constants.electron_volt / scipy.constants.electron_mass)


class CoreGlobalQuantities(core_profiles.core_profiles_global_quantities):
    vacuum_toroidal_field: VacuumToroidalField

    ip: float = sp_property(units="A")

    current_non_inductive: float = sp_property(units="A")

    current_bootstrap: float = sp_property(units="A")

    v_loop: float = sp_property(units="V")

    li_3: float = sp_property(units="-")

    beta_tor: float = sp_property(units="-")

    beta_tor_norm: float = sp_property(units="-")

    beta_pol: float = sp_property(units="-")

    energy_diamagnetic: float = sp_property(units="J")

    z_eff_resistive: float = sp_property(units="-")

    t_e_peaking: float = sp_property(units="-")

    t_i_average_peaking: float = sp_property(units="-")

    resistive_psi_losses: float = sp_property(units="Wb")

    ejima: float = sp_property(units="-")

    t_e_volume_average: float = sp_property(units="eV")

    n_e_volume_average: float = sp_property(units="m^-3")

    class GlobalQuantitiesIon:
        t_i_volume_average: float = sp_property(units="eV")
        n_i_volume_average: float = sp_property(units="m^-3")

    ion: List[GlobalQuantitiesIon]

    ion_time_slice: float = sp_property(units="s")


class CoreProfiles1D(WithDomain, core_profiles.core_profiles_profiles_1d, domain="psi_norm"):

    grid: CoreRadialGrid = {"extrapolate": 0}

    Electrons = CoreProfilesElectrons
    electrons: CoreProfilesElectrons

    Ion = CoreProfilesIon
    ion: Set[CoreProfilesIon]

    Neutral = CoreProfilesNeutral
    neutral: Set[CoreProfilesNeutral]

    rho_tor_norm: array_type | Expression = sp_property(
        label=r"\bar{\rho}_{tor}", units="-", alias="grid/rho_tor_norm"
    )

    rho_tor: Expression = sp_property(label=r"\rho_{tor}", units="m", alias="grid/rho_tor")

    psi_norm: array_type | Expression = sp_property(label=r"\bar{\psi}", units="-")

    psi: Expression = sp_property(label=r"\psi", units="Wb")

    @sp_property
    def zeff(self) -> Expression:
        return sum([((ion.z_ion_1d**2) * ion.density) for ion in self.ion]) / self.n_i_total

    @sp_property
    def pressure(self) -> Expression:
        return sum([ion.pressure for ion in self.ion], self.electrons.pressure)

    @sp_property
    def pprime(self) -> Expression:
        return self.pressure.d

    @sp_property
    def pressure_thermal(self) -> Expression:
        return sum([ion.pressure_thermal for ion in self.ion], self.electrons.pressure_thermal)

    @sp_property
    def t_i_average(self) -> Expression:
        return sum([ion.z_ion_1d * ion.temperature * ion.density for ion in self.ion]) / self.n_i_total

    @sp_property
    def n_i_total(self) -> Expression:
        return sum([(ion.z_ion_1d * ion.density) for ion in self.ion])

    @sp_property
    def n_i_total_over_n_e(self) -> Expression:
        return self.n_i_total / self.electrons.density

    @sp_property
    def n_i_thermal_total(self) -> Expression:
        return sum([ion.z * ion.density_thermal for ion in self.ion])

    # t_i_average: Expression = sp_property(units="eV")

    # t_i_average_fit: core_profiles_1D_fit = sp_property(units="eV")

    # n_i_total_over_n_e: Expression = sp_property(units="-")

    # n_i_thermal_total: Expression = sp_property(units="m^-3")

    momentum_tor: Expression = sp_property(units="kg.m^-1.s^-1")

    zeff: Expression = sp_property(units="-")

    # zeff_fit: core_profiles_1D_fit = sp_property(units="-")

    pressure_ion_total: Expression = sp_property(units="Pa")

    pressure_thermal: Expression = sp_property(units="Pa")

    pressure_perpendicular: Expression = sp_property(units="Pa")

    pressure_parallel: Expression = sp_property(units="Pa")

    j_total: Expression = sp_property(units="A/m^2")

    @sp_property(units="A")
    def current_parallel_inside(self) -> Expression:
        return self.j_total.I

    j_tor: Expression = sp_property(units="A/m^2")

    j_ohmic: Expression = sp_property(units="A/m^2")

    @sp_property(units="A/m^2")
    def j_non_inductive(self) -> Expression:
        return self.j_total - self.j_ohmic

    j_bootstrap: Expression = sp_property(units="A/m^2")

    @sp_property(units="ohm^-1.m^-1")
    def conductivity_parallel(self) -> Expression:
        return self.j_ohmic / self.e_field.parallel

    class EFieldVectorComponents(SpTree):
        radial: Expression = sp_property(default_value=0.0)

        diamagnetic: Expression

        parallel: Expression

        poloidal: Expression

        toroidal: Expression

        @sp_property
        def parallel(self) -> Expression:
            vloop = self._parent.get("vloop", None)
            if vloop is None:
                logger.error(f"Can not calculate E_parallel from vloop!")
                e_par = 0.0
            else:
                e_par = vloop / (TWOPI * self._parent.grid.r0)
            return e_par

    e_field: EFieldVectorComponents = sp_property(units="V.m^-1")

    phi_potential: Expression = sp_property(units="V")

    rotation_frequency_tor_sonic: Expression

    q: Expression = sp_property(units="-")

    @sp_property(units="-")
    def magnetic_shear(self) -> Expression:
        return self.grid.rho_tor_norm * self.q.dln

    @sp_property
    def beta_pol(self) -> Expression:
        return (
            4 * self.pressure.I / (self._parent.vacuum_toroidal_field.r0 * scipy.constants.mu_0 * (self.j_total**2))
        )

    # if isinstance(d, np.ndarray) or (hasattr(d.__class__, 'empty') and not d.empty):
    #     return d

    # else:
    #     Te = self.electrons.temperature
    #     ne = self.electrons.density

    #     # Electron collisions: Coulomb logarithm
    #     # clog = np.asarray([
    #     #     (24.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 2.30*np.log10(Te[idx]))
    #     #     if Te[idx] >= 10 else (23.0 - 1.15*np.log10(ne[idx]*1.0e-6) + 3.45*np.log10(Te[idx]))
    #     #     for idx in range(len(ne))
    #     # ])
    #     clog = self.coulomb_logarithm
    #     # electron collision time:
    #     # tau_e = (np.sqrt(2.*constants.electron_mass)*(Te**1.5)) / 1.8e-19 / (ne * 1.0e-6) / clog

    #     # Plasma electrical conductivity:
    #     return 1.96e0 * constants.elementary_charge**2   \
    #         * ((np.sqrt(2.*constants.electron_mass)*(Te**1.5)) / 1.8e-19 / clog) \
    #         / constants.m_e

    @sp_property
    def coulomb_logarithm(self) -> Expression:
        """Coulomb logarithm,
        @ref: Tokamaks 2003  Ch.14.5 p727 ,2003
        """
        Te = self.electrons.temperature
        Ne = self.electrons.density

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003

        return (14.9 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)) * (Te < 10) + (
            15.2 - 0.5 * np.log(Ne / 1e20) + np.log(Te / 1000)
        ) * (Te >= 10)

    @sp_property
    def electron_collision_time(self) -> Expression:
        """electron collision time ,
        @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        Ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16 * ((Te / 1000.0) ** (3 / 2)) / Ne / lnCoul

    ffprime: Expression = sp_property(label="$ff^{\prime}$")

    pprime: Expression = sp_property(label="$p^{\prime}$")


class CoreProfilesIon2D(WithDomain, Species, domain=".../grid"):
    temperature: Expression
    """Temperature (average over charge states when multiple charge states are considered) {dynamic} [eV]"""
    density: Expression
    """Density (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]"""
    density_thermal: Expression
    """Density (thermal) (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]"""
    density_fast: Expression
    """Density of fast (non-thermal) particles (sum over charge states when multiple charge states are considered) {dynamic} [m^-3]"""
    pressure: Expression
    """Pressure (thermal+non-thermal) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]"""
    pressure_thermal: Expression
    """Pressure (thermal) associated with random motion ~average((v-average(v))^2) (sum over charge states when multiple charge states are considered) {dynamic} [Pa]"""
    pressure_fast_perpendicular: Expression
    """Fast (non-thermal) perpendicular pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]"""
    pressure_fast_parallel: Expression
    """Fast (non-thermal) parallel pressure (sum over charge states when multiple charge states are considered) {dynamic} [Pa]"""
    rotation_frequency_tor: Expression
    """Toroidal rotation frequency (i.e. toroidal velocity divided by the major radius at which the toroidal velocity is taken) (average over charge states when multiple charge states are considered) {dynamic} [rad.s^-1]"""
    velocity: Expression
    """Velocity (average over charge states when multiple charge states are considered) at the position of maximum major radius on every flux surface [m.s^-1]	structure	"""
    multiple_states_flag: Expression
    """Multiple states calculation flag : 0-Only the 'ion' level is considered and the 'state' array of structure is empty; 1-Ion states are considered and are described in the 'state' array of structure {dynamic}"""


class CoreProfiles2D(WithDomain, domain="grid"):
    grid: Mesh

    ion: Set[CoreProfilesIon2D]

    t_i_average: Expression
    """Ion temperature (averaged on states and ion species) {dynamic} [eV] """
    n_i_total_over_n_e: Expression
    """Ratio of total ion density (sum over species and charge states) over electron density. (thermal+non-thermal) {dynamic} [-] """
    n_i_thermal_total: Expression
    """Total ion thermal density (sum over species and charge states) {dynamic} [m^-3] """
    momentum_tor: Expression
    """Total plasma toroidal momentum, summed over ion species and electrons weighted by their density and major radius, i.e. sum_over_species(n*R*m*Vphi) {dynamic} [kg.m^-1.s^-1] """
    zeff: Expression
    """Effective charge {dynamic} [-]	FLT_2D	1- profiles_2d(itime)/grid/dim1 """
    pressure_ion_total: Expression
    """Total (sum over ion species) thermal ion pressure {dynamic} [Pa] """
    pressure_thermal: Expression
    """Thermal pressure (electrons+ions) {dynamic} [Pa] """
    pressure_perpendicular: Expression
    """Total perpendicular pressure (electrons+ions, thermal+non-thermal) {dynamic} [Pa] """
    pressure_parallel: Expression
    """Total parallel pressure (electrons+ions, thermal+non-thermal) {dynamic} [Pa]"""


class CoreProfiles(IDS, FyModule, WithHistory, Entity, plugin_name="core_profiles"):
    """
    Core plasma profiles
    """

    vacuum_toroidal_field: VacuumToroidalField

    GlobalQuantities = CoreGlobalQuantities
    global_quantities: CoreGlobalQuantities

    Profiles1D = CoreProfiles1D
    profiles_1d: CoreProfiles1D

    Profiles2D = CoreProfiles2D
    profiles_2d: CoreProfiles2D
