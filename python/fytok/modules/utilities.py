""" Utilities/Common modules """

import abc
import typing
import numpy as np

from spdm.utils.type_hint import array_type, ArrayType
from spdm.utils.tags import _not_found_
from spdm.core.htree import List
from spdm.core.sp_tree import SpTree, annotation, sp_property
from spdm.core.domain import DomainPPoly
from spdm.core.expression import Expression
from spdm.core.function import Function

from fytok.utils.logger import logger
from fytok.utils.atoms import Atom


class Species(abc.ABC):
    """Species of particles"""

    label: str
    a: float
    z: float

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            args = ({"label": args[0]},)

        super().__init__(*args, **kwargs)
        if self.label is _not_found_ or self.label is None:
            self.label = self._metadata.get("label", None) or self._metadata.get("name", None)

        atom = Atom(self.label)

        self._cache["z"] = atom.z
        self._cache["a"] = atom.a

    def __hash__(self) -> int:
        return hash(self.label)


class VacuumToroidalField(SpTree):
    r0: float
    b0: float


class CoreRadialGrid(DomainPPoly, plugin_name="core_radial"):
    """芯部径向坐标"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._coordinates is None:
            self._coordinates = (getattr(self, self.primary_coordinate),)

    def remesh(self, primary_coordinate: str = None, **kwargs) -> typing.Self:
        """Duplicate the grid with new rho_tor_norm or psi_norm"""

        new_axis = _not_found_

        if primary_coordinate is None:
            primary_coordinate, new_axis = next(iter(kwargs.items()))

        if isinstance(new_axis, array_type):
            old_axis = getattr(self, primary_coordinate)

            assert isinstance(old_axis, array_type), f"Can not get x-axis {primary_coordinate}"

            return CoreRadialGrid(
                psi_norm=(
                    Function(old_axis, self.psi_norm)(new_axis) if self.psi_norm is not _not_found_ else _not_found_
                ),
                rho_tor_norm=(
                    Function(old_axis, self.rho_tor_norm)(new_axis)
                    if self.rho_tor_norm is not _not_found_
                    else _not_found_
                ),
                phi_norm=(
                    Function(old_axis, self.phi_norm)(new_axis) if self.phi_norm is not _not_found_ else _not_found_
                ),
                psi_axis=self.psi_axis,
                psi_boundary=self.psi_boundary,
                phi_boundary=self.phi_boundary,
                rho_tor_boundary=self.rho_tor_boundary,
                primary_coordinate=primary_coordinate,
            )  # type:ignore

        else:
            return CoreRadialGrid(**(self._cache | {"primary_coordinate": primary_coordinate}))

    primary_coordinate: str = "psi_norm"

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
    z_min: float = annotation(units="Elementary Charge Unit")
    z_max: float = annotation(units="Elementary Charge Unit")
    electron_configuration: str
    vibrational_level: float = annotation(units="Elementary Charge Unit")
    vibrational_mode: str


class PlasmaCompositionSpecies(SpTree):
    label: str
    a: float  # = annotation(units="Atomic Mass Unit", )
    z_n: float  # = annotation(units="Elementary Charge Unit", )


class PlasmaCompositionNeutralElement(SpTree):
    a: float  # = annotation(units="Atomic Mass Unit", )
    z_n: float  # = annotation(units="Elementary Charge Unit", )
    atoms_n: int


class PlasmaCompositionIons(SpTree):
    label: str
    element: List[PlasmaCompositionNeutralElement]
    z_ion: float  # = annotation( units="Elementary Charge Unit")
    state: PlasmaCompositionIonState


class PlasmaCompositionNeutralState(SpTree):
    label: str
    electron_configuration: str
    vibrational_level: float  # = annotation(units="Elementary Charge Unit")
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
#            "HTree", "List", "Dict", "SpTree", "annotation",
#            "List", "TimeSeriesList", "TimeSlice",
#            "Signal", "SignalND", "Identifier"
#            "IntFlag"]
