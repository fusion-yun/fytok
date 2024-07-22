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
from fytok.utils.atoms import atoms


class Species(abc.ABC):
    label: str
    a: float
    z: float

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            args = ({"label": args[0]},)

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

    def __init__(self, *args, primary_coordinate: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_coordinate = primary_coordinate or self._metadata.get("primary_coordinate", "psi_norm")

    def remesh(self, primary_coordinate: str = None, **kwargs) -> typing.Self:
        """Duplicate the grid with new rho_tor_norm or psi_norm"""

        axis_name, x1 = next(iter(kwargs.items()))
        x0 = getattr(self, axis_name)

        return CoreRadialGrid(
            psi_norm=Function(x0, self.psi_norm)(x1) if self.psi_norm is not _not_found_ else _not_found_,
            rho_tor_norm=Function(x0, self.rho_tor_norm)(x1) if self.rho_tor_norm is not _not_found_ else _not_found_,
            phi_norm=Function(x0, self.phi_norm)(x1) if self.phi_norm is not _not_found_ else _not_found_,
            psi_axis=self.psi_axis,
            psi_boundary=self.psi_boundary,
            phi_boundary=self.phi_boundary,
            rho_tor_boundary=self.rho_tor_boundary,
            primary_coordinate=primary_coordinate or self.primary_coordinate,
        )

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
        return (getattr(self, self.primary_coordinate),)


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


class DetectorAperture(SpTree):
    def __view__(self, **styles):
        return {"$styles": styles}


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
