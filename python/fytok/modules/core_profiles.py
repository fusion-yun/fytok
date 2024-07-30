import numpy as np
import scipy.constants

from spdm.utils.tags import _not_found_
from spdm.core.htree import Set, List
from spdm.core.sp_tree import annotation, sp_property, SpTree
from spdm.core.expression import Expression
from spdm.core.mesh import Mesh
from spdm.core.domain import WithDomain
from spdm.core.time import WithTime

from fytok.utils.logger import logger
from fytok.utils.base import IDS, FyEntity


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


class CoreProfilesSpecies(Species, SpTree):
    """Core Profiles Species"""

    is_thermal: bool = True

    density: Expression = annotation(units=r"m^{-3}")

    temperature: Expression = annotation(units=r"eV")

    @sp_property(units="Pa")
    def pressure_thermal(self) -> Expression:
        return self.density * self.temperature * scipy.constants.electron_volt

    pressure_fast_perpendicular: Expression = annotation(units="Pa", default_value=0.0)

    pressure_fast_parallel: Expression = annotation(units="Pa", default_value=0.0)

    @sp_property(units="Pa")
    def pressure(self) -> Expression:
        return self.pressure_fast_perpendicular + self.pressure_fast_parallel + self.pressure_thermal

    rotation_frequency_tor: Expression = annotation(units=r"rad \cdot s^{-1}", default_value=0.0)

    velocity: CoreVectorComponents = annotation(units=r"m \cdot s^{-1}")


class CoreProfilesState(CoreProfilesSpecies):
    """Core Profiles State"""

    electron_configuration: str
    """ Configuration of atomic orbitals of this state, e.g. 1s2-2s1"""

    vibrational_level: float = annotation(units="Elementary Charge Unit")
    """ Vibrational level (can be bundled)"""

    vibrational_mode: str
    """ Vibrational mode of this state, e.g. _A_g_. Need to define, or adopt a standard
        nomenclature."""

    neutral_type: int
    """ Neutral type (if the considered state is a neutral), in terms of energy. ID =1:
        cold; 2: thermal; 3: fast; 4: NBI"""


class CoreProfilesIon(CoreProfilesSpecies):
    """Core Profiles Ion"""

    is_impurity: bool = False

    z_ion_1d: Expression = annotation(units="C")

    @sp_property
    def z_ion_square_1d(self) -> Expression:
        return self.z_ion_1d * self.z_ion_1d

    @sp_property(units=r"s^{-1}", default_value=0.1)
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
            * self.density
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


class CoreProfilesElectrons(CoreProfilesSpecies, label="electron"):
    """Core Profiles Electrons"""

    @sp_property(units="-")
    def collisionality_norm(self) -> Expression:
        raise NotImplementedError("collisionality_norm")

    @sp_property(units="s")
    def tau(self):
        return 1.09e16 * ((self.temperature / 1000) ** (3 / 2)) / self.density / self._parent.coulomb_logarithm

    @sp_property(units=r"m/s")
    def vT(self):
        return np.sqrt(self.temperature * scipy.constants.electron_volt / scipy.constants.electron_mass)


class CoreGlobalQuantities(core_profiles.core_profiles_global_quantities):
    """Core Global Quantities"""

    vacuum_toroidal_field: VacuumToroidalField

    ip: float = annotation(units="A")

    current_non_inductive: float = annotation(units="A")

    current_bootstrap: float = annotation(units="A")

    v_loop: float = annotation(units="V")

    li_3: float = annotation(units="-")

    beta_tor: float = annotation(units="-")

    beta_tor_norm: float = annotation(units="-")

    beta_pol: float = annotation(units="-")

    energy_diamagnetic: float = annotation(units="J")

    z_eff_resistive: float = annotation(units="-")

    t_e_peaking: float = annotation(units="-")

    t_i_average_peaking: float = annotation(units="-")

    resistive_psi_losses: float = annotation(units="Wb")

    ejima: float = annotation(units="-")

    t_e_volume_average: float = annotation(units="eV")

    n_e_volume_average: float = annotation(units="m^-3")

    class GlobalQuantitiesIon:
        t_i_volume_average: float = annotation(units="eV")
        n_i_volume_average: float = annotation(units="m^-3")

    ion: List[GlobalQuantitiesIon]

    ion_time_slice: float = annotation(units="s")


class CoreProfiles1D(WithDomain, SpTree, domain="grid/rho_tor_norm"):
    """Core Profiles 1D"""

    grid: CoreRadialGrid = {"primary_coordinate": "rho_tor_norm"}

    Electrons = CoreProfilesElectrons
    electrons: CoreProfilesElectrons

    Ion = CoreProfilesIon
    ion: Set[CoreProfilesIon]

    Neutral = CoreProfilesNeutral
    neutral: Set[CoreProfilesNeutral]

    rho_tor_norm: Expression = annotation(label=r"\bar{\rho}_{tor}", units="-")
    rho_tor: Expression = annotation(label=r"\rho_{tor}", units="m")
    psi_norm: Expression = annotation(label=r"\bar{\psi}", units="-")
    psi: Expression = annotation(label=r"\psi", units="Wb")

    @sp_property
    def zeff(self) -> Expression:
        return sum(((ion.z_ion_1d**2) * ion.density) for ion in self.ion) / self.n_i_total

    @sp_property
    def pressure(self) -> Expression:
        return sum([ion.pressure for ion in self.ion], self.electrons.pressure)

    @sp_property
    def pprime(self) -> Expression:
        return self.pressure.d

    @sp_property
    def pressure_thermal(self) -> Expression:
        return sum(ion.pressure_thermal for ion in self.ion) + self.electrons.pressure_thermal

    @sp_property
    def t_i_average(self) -> Expression:
        return sum(ion.z_ion_1d * ion.temperature * ion.density for ion in self.ion) / self.n_i_total

    @sp_property
    def n_i_total(self) -> Expression:
        return sum((ion.z_ion_1d * ion.density) for ion in self.ion)

    @sp_property
    def n_i_total_over_n_e(self) -> Expression:
        return self.n_i_total / self.electrons.density

    @sp_property
    def n_i_thermal_total(self) -> Expression:
        return sum(ion.z * ion.density_thermal for ion in self.ion)

    momentum_tor: Expression = annotation(units=r"kg \cdot m^{-1} \cdot s^{-1}")

    zeff: Expression = annotation(units="-")

    # zeff_fit: core_profiles_1D_fit = annotation(units="-")

    pressure_ion_total: Expression = annotation(units="Pa")

    pressure_thermal: Expression = annotation(units="Pa")

    pressure_perpendicular: Expression = annotation(units="Pa")

    pressure_parallel: Expression = annotation(units="Pa")

    j_total: Expression = annotation(units="A/m^2")

    @sp_property(units="A")
    def current_parallel_inside(self) -> Expression:
        return self.j_total.I

    j_tor: Expression = annotation(units=r"A/m^2")

    j_ohmic: Expression = annotation(units=r"A/m^2")

    @sp_property(units=r"A/m^2")
    def j_non_inductive(self) -> Expression:
        return self.j_total - self.j_ohmic

    j_bootstrap: Expression = annotation(units=r"A/m^2")

    @sp_property(units=r"ohm^-1.m^-1")
    def conductivity_parallel(self) -> Expression:
        return self.j_ohmic / self.e_field.parallel

    class EFieldVectorComponents(SpTree):
        radial: Expression = annotation(default_value=0.0)

        diamagnetic: Expression

        # parallel: Expression

        poloidal: Expression

        toroidal: Expression

        @sp_property
        def parallel(self) -> Expression:
            vloop = self._parent.get("vloop", _not_found_)
            if vloop is _not_found_:
                logger.error("Can not calculate E_parallel from vloop!")
                e_par = 0.0
            else:
                e_par = vloop / (TWOPI * self._parent.grid.r0)
            return e_par

    e_field: EFieldVectorComponents = annotation(units=r"V.m^{-1}")

    phi_potential: Expression = annotation(units="V")

    rotation_frequency_tor_sonic: Expression

    q: Expression = annotation(units="-")

    @sp_property(units="-")
    def magnetic_shear(self) -> Expression:
        return self.grid.rho_tor_norm * self.q.dln

    @sp_property(units="-")
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
        ne = self.electrons.density

        # Coulomb logarithm
        #  Ch.14.5 p727 Tokamaks 2003

        return (14.9 - 0.5 * np.log(ne / 1e20) + np.log(Te / 1000)) * (Te < 10) + (
            15.2 - 0.5 * np.log(ne / 1e20) + np.log(Te / 1000)
        ) * (
            Te >= 10
        )  # type:ignore

    @sp_property
    def electron_collision_time(self) -> Expression:
        """electron collision time ,
        @ref: Tokamak 2003, eq 14.6.1
        """
        Te = self.electrons.temperature(self.grid.rho_tor_norm)
        ne = self.electrons.density(self.grid.rho_tor_norm)
        lnCoul = self.coulomb_logarithm(self.grid.rho_tor_norm)
        return 1.09e16 * ((Te / 1000.0) ** (3 / 2)) / ne / lnCoul

    ffprime: Expression = annotation(label=r"$ff^{\prime}$")

    pprime: Expression = annotation(label=r"$p^{\prime}$")


class CoreProfilesElectrons2D(WithDomain, Species, domain=".../grid"):
    pass


class CoreProfilesIon2D(WithDomain, Species, domain=".../grid"):
    """Ion Profiles"""

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
    """Core Profiles"""

    grid_type: str = annotation(alias="grid/type")

    grid: Mesh

    electrons: CoreProfilesElectrons2D
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


class CoreProfiles(WithTime, IDS, FyEntity, code={"name": "core_profiles"}):
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
