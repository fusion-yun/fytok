from __future__ import annotations
import collections
import functools
import typing
from dataclasses import dataclass
from enum import IntFlag
import numpy as np

from spdm.core.path import Path
from spdm.core.actor import Actor
from spdm.core.aos import AoS
from spdm.core.field import Field
from spdm.core.expression import Expression, zero
from spdm.core.function import Function
from spdm.core.htree import Dict, HTree, List
from spdm.core.signal import Signal, SignalND
from spdm.core.sp_property import SpTree, sp_property, sp_tree, PropertyTree
from spdm.core.time_series import TimeSeriesAoS, TimeSlice
from spdm.core.domain import PPolyDomain

from spdm.geometry.curve import Curve
from spdm.utils.typing import array_type, is_array, as_array
from spdm.utils.tags import _not_found_
from spdm.view import sp_view as sp_view

from ..utils.logger import logger
from ..utils.envs import FY_JOBID


@sp_tree
class IDSProperties:
    comment: str
    homogeneous_time: int
    provider: str
    creation_date: str
    version_put: SpTree
    provenance: SpTree


@sp_tree
class Library:
    name: str
    commit: str
    version: str
    repository: str
    parameters: SpTree


@sp_tree
class Code:
    name: str = "default"
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


@sp_tree
class Identifier:
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            pass
        elif isinstance(args[0], str):
            args[0] = {"name": args[0]}
        elif isinstance(args[0], int):
            args[0] = {"int": args[0]}
        super().__init__(*args, **kwargs)

    name: str
    index: int
    description: str


@sp_tree
class Module(Actor):

    def __new__(cls, *args, **kwargs) -> typing.Type[typing.Self]:
        pth = Path("code/name")
        plugin_name = None
        if len(args) > 0 and isinstance(args[0], dict):
            plugin_name = pth.get(args[0], None)
        if plugin_name is None:
            plugin_name = pth.get(kwargs, None)
        if plugin_name is None:
            plugin_name = Path("default_value/name").get(cls.code, None)
        if plugin_name == "default":
            plugin_name = None
        return super().__new__(cls, {"$class": plugin_name})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.code.name:
            self.code.name = self.__class__.__name__

        self.code.module_path = self.__module__ + "." + self.__class__.__name__

        logger.verbose(f"Initialize module {self.code} ")

    code: Code = {}
    """ 对于 Module 的一般性说明。 
        @note code 在 __init__ 时由初始化参数定义。"""

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self.code.module_path}"

    def refresh(self, *args, **kwargs) -> typing.Type[TimeSlice]:
        """更新当前 Actor 的状态。
        更新当前状态树 （time_slice），并执行 self.iteration+=1

        """
        logger.verbose(f"Refresh module {self.code.module_path}")

        current = super().refresh(*args, **kwargs)

        return current


class IDS(Module):
    """Base class of IDS"""

    ids_properties: IDSProperties

    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    def _repr_svg_(self) -> str:
        if hasattr(self.__class__, "__view__"):
            try:
                res = sp_view.display(self.__view__(), output="svg")
            except Exception as error:
                raise RuntimeError(f"{self}") from error
        else:
            res = None
        return res


@sp_tree
class RZTuple:
    r: typing.Any
    z: typing.Any


@sp_tree
class PointRZ:  # utilities._T_rz0d_dynamic_aos
    r: float
    z: float


@sp_tree
class CurveRZ:  # utilities._T_rz1d_dynamic_aos
    r: array_type
    z: array_type


@sp_tree
class VacuumToroidalField:
    r0: float
    b0: float


@sp_tree
class CoreRadialGrid:
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


@sp_tree
class CoreVectorComponents:
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


class DetectorAperture:  # (utilities._T_detector_aperture):
    def __view__(self, view="RZ", **kwargs):
        geo = {}
        styles = {}
        return geo, styles


@sp_tree
class PlasmaCompositionIonState:
    label: str
    z_min: float = sp_property(units="Elementary Charge Unit")
    z_max: float = sp_property(units="Elementary Charge Unit")
    electron_configuration: str
    vibrational_level: float = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str


@sp_tree
class PlasmaCompositionSpecies:
    label: str
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )


@sp_tree
class PlasmaCompositionNeutralElement(SpTree):
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )
    atoms_n: int


@sp_tree
class PlasmaCompositionIons:
    label: str
    element: AoS[PlasmaCompositionNeutralElement]
    z_ion: float  # = sp_property( units="Elementary Charge Unit")
    state: PlasmaCompositionIonState


class PlasmaCompositionNeutralState:
    label: str
    electron_configuration: str
    vibrational_level: float  # = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str
    neutral_type: str


class PlasmaCompositionNeutral:
    label: str
    element: AoS[PlasmaCompositionNeutralElement]
    state: PlasmaCompositionNeutralState


@sp_tree
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
