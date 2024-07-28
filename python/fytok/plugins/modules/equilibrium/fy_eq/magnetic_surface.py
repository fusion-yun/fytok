import typing
import numpy as np
import scipy.constants


from spdm.core.expression import Variable
from spdm.core.sp_tree import annotation, sp_property, sp_tree
from spdm.geometry.curve import Curve
from spdm.core.geo_object import GeoObject, GeoObjectSet
from spdm.geometry.point import Point

from spdm.utils.type_hint import array_type

from fytok.modules.utilities import *


PI = scipy.constants.pi

_R = Variable(0, "R")
_Z = Variable(1, "Z")

_T = typing.TypeVar("_T", float, array_type)


@sp_tree(coordiante1="psi_norm")
class MageticSurface(typing.Generic[_T]):
    @staticmethod
    def _shape_box(s: GeoObject):
        if isinstance(s, Point):
            r, z = s.points
            rmin = r
            rmax = r
            zmin = z
            zmax = z
            r_inboard = r
            r_outboard = r
            rzmin = r
            rzmax = r
        elif isinstance(s, Curve):
            r, z = s.points
            (rmin, zmin) = s.bbox.origin
            (rmax, zmax) = s.bbox.origin + s.bbox.dimensions
            rzmin = r[np.argmin(z)]
            rzmax = r[np.argmax(z)]

            # FIXME: 仅仅对当前磁面查找算法有效
            r_inboard = s.coordinates(0.5)[0]
            r_outboard = s.coordinates(0)[0]
        else:
            raise TypeError(f"Invalid type {type(s)}")
        return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

    def __init__(self, psi_norm: _T, curves: GeoObject | GeoObjectSet):
        self.psi_norm = psi_norm

        sbox = np.asarray([[*MageticSurface[_T]._shape_box(s)] for s in curves], dtype=float)
        (self.rmin, self.zmin, self.rmax, self.zmax, self.rzmin, self.rzmax, self.r_inboard, self.r_outboard) = sbox.T

    Rmin: _T
    Zmin: _T
    Rmax: _T
    Zmax: _T
    Rzmin: _T
    Rzmax: _T
    r_inboard: _T
    r_outboard: _T

    psi_norm: _T

    psi: _T = annotation(units="Wb")

    phi: _T

    rho_tor: _T

    outline: Curve | GeoObjectSet

    x_point: GeoObjectSet = annotation(alias="../coordinate_system/x_point")

    @sp_property
    def strike_point(self) -> GeoObjectSet:
        return

    @sp_property
    def active_limiter_point(self) -> GeoObjectSet:
        return NotImplemented

    @sp_property
    def geometric_axis(self) -> Point:
        return (self.Rmin + self.Rmax) * 0.5, (self.Zmin + self.Zmax) * 0.5

    @sp_property
    def minor_radius(self) -> _T:
        return (self.Rmax - self.Rmin) * 0.5

    @sp_property
    def elongation(self) -> _T:
        return (self.Zmax - self.Zmin) / (self.Rmax - self.Rmin)

    @sp_property
    def elongation_upper(self) -> _T:
        return (self.Zmax - (self.Zmax + self.Zmin) * 0.5) / (self.Rmax - self.Rmin)

    @sp_property
    def elongation_lower(self) -> _T:
        return ((self.Zmax + self.Zmin) * 0.5 - self.Zmin) / (self.Rmax - self.Rmin)

    @sp_property
    def triangularity(self) -> _T:
        return (self.Rzmax - self.Rzmin) / (self.Rmax - self.Rmin) * 2

    @sp_property
    def triangularity_upper(self) -> _T:
        return ((self.Rmax + self.Rmin) * 0.5 - self.Rzmax) / (self.Rmax - self.Rmin) * 2

    @sp_property
    def triangularity_lower(self) -> _T:
        return ((self.Rmax + self.Rmin) * 0.5 - self.Rzmin) / (self.Rmax - self.Rmin) * 2
