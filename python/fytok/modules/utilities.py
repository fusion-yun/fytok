import abc
import typing
import numpy as np

from spdm.utils.type_hint import array_type, ArrayType
from spdm.utils.tags import _not_found_
from spdm.core.htree import List
from spdm.core.sp_tree import SpTree, sp_property
from spdm.core.domain import DomainPPoly
from spdm.core.expression import Expression
from spdm.core.function import Function


class Species(SpTree):
    label: str
    a: float
    z: float

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.label is _not_found_ or self.label is None:
            self.label = self._metadata.get("label", None)

        atom_desc = atoms[self.label]

        self._cache["z"] = atom_desc.z
        self._cache["a"] = atom_desc.a

    def __hash__(self) -> int:
        return hash(self.label)


class VacuumToroidalField(SpTree):
    r0: float
    b0: float


class CoreRadialGrid(DomainPPoly, plugin_name="core_radial"):
    """芯部径向坐标"""

    # PPolyDomain.__init__(self, self._cache["psi_norm"])
    # assert isinstance(self.psi_axis, float), f"psi_axis must be specified  {self.psi_axis}"
    # assert isinstance(self.psi_boundary, float), f"psi_boundary must be specified {self.psi_boundary}"
    # assert isinstance(self.rho_tor_boundary, float), f"rho_tor_boundary must be specified {self.rho_tor_boundary}"
    # assert self.rho_tor_norm[0] >= 0 and self.rho_tor_norm[-1] <= 1.0, f"illegal rho_tor_norm {self.rho_tor_norm}"
    # assert self.psi_norm[0] >= 0 and self.psi_norm[-1] <= 1.0, f"illegal psi_norm {self.psi_norm}"

    def remesh(self, rho_tor_norm=None, psi_norm=None, **kwargs) -> typing.Self:
        """Duplicate the grid with new rho_tor_norm or psi_norm"""

        if isinstance(rho_tor_norm, array_type):
            psi_norm = Function(self.rho_tor_norm, self.psi_norm)(rho_tor_norm)
            if psi_norm[0] < 0:
                psi_norm[0] = 0.0
        elif isinstance(psi_norm, array_type):
            rho_tor_norm = Function(self.psi_norm, self.rho_tor_norm)(psi_norm)
            if rho_tor_norm[0] < 0:
                psi_norm[0] = 0.0

        else:
            rho_tor_norm = self.rho_tor_norm
            psi_norm = self.psi_norm
        # if rho_tor_norm is None or rho_tor_norm is _not_found_:
        #     if psi_norm is _not_found_ or psi_norm is None:
        #         psi_norm = self.psi_norm
        #         rho_tor_norm = self.rho_tor_norm
        #     else:
        #         rho_tor_norm = Function(
        #             self.psi_norm,
        #             self.rho_tor_norm,
        #             name="rho_tor_norm",
        #             label=r"\bar{\rho}",
        #         )(psi_norm)
        # else:
        #     rho_tor_norm = np.asarray(rho_tor_norm)

        # if psi_norm is _not_found_ or psi_norm is None:
        #     psi_norm = Function(
        #         self.rho_tor_norm,
        #         self.psi_norm,
        #         name="psi_norm",
        #         label=r"\bar{\psi}",
        #     )(rho_tor_norm)

        # else:
        #     psi_norm = np.asarray(psi_norm)

        return CoreRadialGrid(
            {
                "rho_tor_norm": rho_tor_norm,
                "psi_norm": psi_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            }
        )

    def fetch(self, first=None, psi_norm=None, **kwargs) -> typing.Self:
        if isinstance(first, array_type):
            rho_tor_norm = first
        else:
            rho_tor_norm = getattr(first, "rho_tor_norm", kwargs.pop("rho_tor_norm", None))

        if psi_norm is None and isinstance(first, SpTree):
            psi_norm = getattr(first, "psi_norm", None)

        return self.remesh(rho_tor_norm, psi_norm=psi_norm, **kwargs)

    psi_axis: float
    psi_boundary: float
    psi_norm: array_type

    phi_boundary: float
    phi_norm: array_type

    rho_tor_boundary: float
    rho_tor_norm: array_type

    @sp_property
    def psi(self) -> array_type:
        return self.psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis

    @sp_property
    def phi(self) -> array_type:
        return self.phi_norm * self.phi_boundary

    @sp_property
    def rho_tor(self) -> array_type:
        return self.rho_tor_norm * self.rho_tor_boundary

    @sp_property
    def rho_pol_norm(self) -> array_type:
        return np.sqrt(self.psi_norm)

    @property
    def coordinates(self) -> typing.Tuple[ArrayType, ...]:
        return (self.psi_norm,)


class CoreVectorComponents(SpTree):
    """Vector components in predefined directions"""

    radial: Expression
    """ Radial component"""

    diamagnetic: Expression
    """ Diamagnetic component"""

    parallel: Expression
    """ Parallel component"""

    poloidal: Expression
    """ Poloidal component"""

    toroidal: Expression
    """ Toroidal component"""


class DetectorAperture(SpTree):  # (utilities._T_detector_aperture):
    def __view__(self, view="RZ", **styles):
        geo = {}
        return geo, styles


class PlasmaCompositionIonState(SpTree):
    label: str
    z_min: float = sp_property(units="Elementary Charge Unit")
    z_max: float = sp_property(units="Elementary Charge Unit")
    electron_configuration: str
    vibrational_level: float = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str


class PlasmaCompositionSpecies(SpTree):
    label: str
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )


class PlasmaCompositionNeutralElement(SpTree):
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )
    atoms_n: int


class PlasmaCompositionIons(SpTree):
    label: str
    element: List[PlasmaCompositionNeutralElement]
    z_ion: float  # = sp_property( units="Elementary Charge Unit")
    state: PlasmaCompositionIonState


class PlasmaCompositionNeutralState(SpTree):
    label: str
    electron_configuration: str
    vibrational_level: float  # = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str
    neutral_type: str


class PlasmaCompositionNeutral(SpTree):
    label: str
    element: List[PlasmaCompositionNeutralElement]
    state: PlasmaCompositionNeutralState


class DistributionSpecies(SpTree):
    type: str
    ion: PlasmaCompositionIons
    neutral: PlasmaCompositionNeutral


# __all__ = ["IDS", "Module", "Code", "Library",
#            "DetectorAperture", "CoreRadialGrid",
#            "array_type", "Function", "Field",
#            "HTree", "List", "Dict", "SpTree", "sp_property",
#            "List", "TimeSeriesList", "TimeSlice",
#            "Signal", "SignalND", "Identifier"
#            "IntFlag"]
