from __future__ import annotations
import collections
import functools
import typing
from dataclasses import dataclass
from enum import IntFlag
import numpy as np

from spdm.core.path import Path
from spdm.core.service import Service
from spdm.core.aos import AoS
from spdm.core.field import Field
from spdm.core.expression import Expression, zero
from spdm.core.function import Function
from spdm.core.htree import Dict, HTree, List
from spdm.core.signal import Signal, SignalND
from spdm.core.sp_tree import SpTree, sp_property
from spdm.core.sp_object import SpObject
from spdm.core.property_tree import PropertyTree
from spdm.core.actor import Actor
from spdm.core.component import Component
from spdm.core.processor import Processor
from spdm.core.time_sequence import TimeSlice
from spdm.core.time_sequence import TimeSlice, TimeSequence

from spdm.geometry.curve import Curve

from spdm.utils.type_hint import array_type, is_array, as_array
from spdm.utils.tags import _not_found_
from spdm.view import sp_view as sp_view

from fytok.utils.logger import logger
from fytok.utils.envs import FY_JOBID


class IDSProperties(SpTree):
    comment: str
    homogeneous_time: int
    provider: str
    creation_date: str
    version_put: PropertyTree
    provenance: PropertyTree


class Library(SpTree):
    name: str
    commit: str
    version: str = "0.0.0"
    repository: str = ""
    parameters: PropertyTree


class Code(SpTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cache = Path().update(
            self._cache,
            {
                "name": self._parent.__class__.__name__,
                "module_path": self._parent.__module__ + "." + self._parent.__class__.__name__,
            },
        )

    name: str
    """代码名称，也是调用 plugin 的 identifier"""

    module_path: str
    """模块路径， 可用于 import 模块"""

    commit: str
    version: str = "0.0.0"
    copyright: str = "NO_COPYRIGHT_STATEMENT"
    repository: str = ""
    output_flag: array_type
    library: List[Library]

    parameters: PropertyTree = {}
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    def __str__(self) -> str:
        return "-".join([s for s in [self.name, self.version.replace(".", "_")] if isinstance(s, str)])

    def __repr__(self) -> str:
        desc = {
            "name": self.name,
            "version": self.version,
            "copyright": self.copyright,
        }

        return ", ".join(
            [
                f"{key}='{value}'"
                for key, value in desc.items()
                if value is not _not_found_ and value is not None and value != ""
            ]
        )


class Identifier(SpTree):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            pass
        elif isinstance(args[0], str):
            args = ({"name": args[0]}, *args[1:])
        elif isinstance(args[0], int):
            args = ({"int": args[0]}, *args[1:])
        super().__init__(*args, **kwargs)

    name: str
    index: int
    description: str


class IDS(SpTree):
    def __init__(self, *args, _entry=None, **kwargs):
        if len(args) > 0 and isinstance(args[0], str) and _entry is None:
            _entry = args[0]
            args = args[1:]

        cache = {k: kwargs.pop(k) for k in list(kwargs.keys()) if not k.startswith("_")}

        cache = Path().update(*args, cache)

        super().__init__(cache, _entry=_entry, **kwargs)

    ids_properties: IDSProperties


class FyModule(SpObject):

    _plugin_prefix = "fytok.modules."
    _plugin_registry = {}

    identifier: str

    code: Code

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self.code.module_path}"


_TSlice = typing.TypeVar("_TSlice")


class FyActor(FyModule, Actor[_TSlice]):

    def refresh(self, *args, **kwargs) -> typing.Type[_TSlice]:
        """更新当前 Actor 的状态。
        更新当前状态树 （time_slice），并执行 self.iteration+=1

        """
        logger.verbose(f"Refresh module {self.code.module_path}")

        current = super().refresh(*args, **kwargs)

        return current


class FyComponent(FyModule, Component):
    pass


class FyProcessor(FyModule, Processor):
    pass


class RZTuple(SpTree):
    r: typing.Any
    z: typing.Any


class PointRZ(SpTree):  # utilities._T_rz0d_dynamic_aos
    r: float
    z: float


class CurveRZ(SpTree):  # utilities._T_rz1d_dynamic_aos
    r: array_type
    z: array_type


class VacuumToroidalField(SpTree):
    r0: float
    b0: float


class CoreRadialGrid(SpTree):
    def __init__(self, *args, **kwargs) -> None:
        SpTree.__init__(self, *args, **kwargs)
        # PPolyDomain.__init__(self, self._cache["psi_norm"])
        # assert isinstance(self.psi_axis, float), f"psi_axis must be specified  {self.psi_axis}"
        # assert isinstance(self.psi_boundary, float), f"psi_boundary must be specified {self.psi_boundary}"
        # assert isinstance(self.rho_tor_boundary, float), f"rho_tor_boundary must be specified {self.rho_tor_boundary}"
        # assert self.rho_tor_norm[0] >= 0 and self.rho_tor_norm[-1] <= 1.0, f"illegal rho_tor_norm {self.rho_tor_norm}"
        # assert self.psi_norm[0] >= 0 and self.psi_norm[-1] <= 1.0, f"illegal psi_norm {self.psi_norm}"

    def __copy__(self) -> CoreRadialGrid:
        return CoreRadialGrid(
            {
                "psi_norm": self.psi_norm,
                "rho_tor_norm": self.rho_tor_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            }
        )

    def __serialize__(self, dumper=None):
        return HTree._do_serialize(
            {
                "psi_norm": self.psi_norm,
                "rho_tor_norm": self.rho_tor_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            },
            dumper,
        )

    def remesh(self, rho_tor_norm=None, *args, psi_norm=None, **kwargs) -> CoreRadialGrid:
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

    def fetch(self, first=None, *args, psi_norm=None, **kwargs) -> CoreRadialGrid:
        if isinstance(first, array_type):
            rho_tor_norm = first
        else:
            rho_tor_norm = getattr(first, "rho_tor_norm", kwargs.pop("rho_tor_norm", None))

        if psi_norm is None and isinstance(first, SpTree):
            psi_norm = getattr(first, "psi_norm", None)

        return self.remesh(rho_tor_norm, *args, psi_norm=psi_norm, **kwargs)

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
    def __view__(self, view="RZ", **kwargs):
        geo = {}
        styles = {}
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
    element: AoS[PlasmaCompositionNeutralElement]
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
    element: AoS[PlasmaCompositionNeutralElement]
    state: PlasmaCompositionNeutralState


class DistributionSpecies(SpTree):
    type: str
    ion: PlasmaCompositionIons
    neutral: PlasmaCompositionNeutral


# __all__ = ["IDS", "Module", "Code", "Library",
#            "DetectorAperture", "CoreRadialGrid", "PointRZ",   "CurveRZ",
#            "array_type", "Function", "Field",
#            "HTree", "List", "Dict", "SpTree", "sp_property",
#            "AoS", "TimeSeriesAoS", "TimeSlice",
#            "Signal", "SignalND", "Identifier"
#            "IntFlag"]
