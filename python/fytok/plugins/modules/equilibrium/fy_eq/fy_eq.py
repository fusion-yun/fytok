import collections
import collections.abc
import functools
import typing
import numpy as np
import scipy.constants
from dataclasses import dataclass


from spdm.core.htree import List
from spdm.core.sp_tree import sp_property
from spdm.core.expression import Expression, Variable
from spdm.core.field import Field
from spdm.core.function import Function
from spdm.core.geo_object import GeoObject, GeoObjectSet
from spdm.geometry.point import PointRZ, Point
from spdm.geometry.point_set import PointSetRZ
from spdm.geometry.curve import Curve, CurveRZ
from spdm.core.mesh import Mesh
from spdm.mesh.mesh_curvilinear import CurvilinearMesh
from spdm.utils.tags import _not_found_
from spdm.utils.type_hint import ArrayLike, NumericType, array_type, scalar_type

from fytok.utils.envs import FY_VERSION, FY_COPYRIGHT
from fytok.utils.logger import logger
from fytok.modules import equilibrium
from fytok.modules.utilities import CoreRadialGrid, VacuumToroidalField

from .contours import find_critical_points, find_contours


PI = scipy.constants.pi

_R = Variable(0, "R")
_Z = Variable(1, "Z")


TOLERANCE = 1.0e-6


# fmt:off
COCOS_TABLE = [
    # e_Bp ,    $\sigma_{Bp}$,    $\sigma_{R\varphi\Z}$,  $\sigma_{\rho\theta\varphi}$
    None,                                                                                    # 0
    (0,         +1,                  +1,                       +1                   ),       # 1
    (0,         +1,                  -1,                       +1                   ),       # 2
    (0,         -1,                  +1,                       -1                   ),       # 3
    (0,         -1,                  -1,                       -1                   ),       # 4
    (0,         +1,                  +1,                       -1                   ),       # 5
    (0,         +1,                  -1,                       -1                   ),       # 6
    (0,         -1,                  +1,                       +1                   ),       # 7
    (0,         -1,                  -1,                       +1                   ),       # 8
    None,                                                                                    # 9
    None,                                                                                    # 10
    (1,         +1,                  +1,                       +1                   ),       # 11
    (1,         +1,                  -1,                       +1                   ),       # 12
    (1,         -1,                  +1,                       -1                   ),       # 13
    (1,         -1,                  -1,                       -1                   ),       # 14
    (1,         +1,                  +1,                       -1                   ),       # 15
    (1,         +1,                  -1,                       -1                   ),       # 16
    (1,         -1,                  +1,                       +1                   ),       # 17
    (1,         -1,                  -1,                       +1                   ),       # 18
]
# fmt:on

OXPoint = typing.Tuple[Point, float]


class FyEqCoordinateSystem(equilibrium.EquilibriumCoordinateSystem):
    r"""
    Flux surface coordinate system on a square grid of flux and poloidal angle
    默认采用磁面坐标

    $$
        V^{\prime}\left(\rho\right)=\frac{\partial V}{\partial\rho}=2\pi\int_{0}^{2\pi}\sqrt{g}d\theta=2\pi\oint\frac{R}{\left|\nabla\rho\right|}dl
    $$

    $$
        \left\langle\alpha\right\rangle\equiv\frac{2\pi}{V^{\prime}}\int_{0}^{2\pi}\alpha\sqrt{g}d\theta=\frac{2\pi}{V^{\prime}}\varoint\alpha\frac{R}{\left|\nabla\rho\right|}dl
    $$

    Magnetic Flux Coordinates
    psi         :                     ,  flux function , $B \cdot \nabla \psi=0$ need not to be the poloidal flux funcion $\Psi$
    theta       : 0 <= theta   < 2*pi ,  poloidal angle
    phi         : 0 <= phi     < 2*pi ,  toroidal angle
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._s_B0 = np.sign(self.b0)
        self._s_Ip = np.sign(self.ip)
        self._e_Bp, self._s_Bp, self._s_RpZ, self._s_rtp = COCOS_TABLE[5]
        self._s_eBp_2PI = 1.0 if self._e_Bp == 0 else (2.0 * scipy.constants.pi)

        # if not np.all(np.diff(self.psi_norm) > 0):
        #     raise RuntimeError("psi_norm is not monotonically increasing!")
        # if np.isclose(self.psi_norm[0], 0.0) and np.isclose(self.psi_norm[-1], 1.0):
        #     logger.warning(
        #         f"Singular values are caused when psi_norm takes values of 0.0 or 1.0.! {self.psi_norm[0]} {self.psi_norm[-1]}"
        #     )

    b0: float = sp_property(alias="../vacuum_toroidal_field/b0")
    r0: float = sp_property(alias="../vacuum_toroidal_field/r0")
    ip: float = sp_property(alias="../global_quantities/ip")

    # 磁面坐标
    psirz: Field = sp_property(alias="../profiles_2d/psi")

    psi_norm: array_type = sp_property(alias="../profiles_1d/psi_norm")

    @sp_property
    def critical_points(self) -> typing.Tuple[List[OXPoint], List[OXPoint]]:
        return find_critical_points(self.psirz)

    o_points: List[OXPoint] = sp_property(alias="critical_points/0")

    x_points: List[OXPoint] = sp_property(alias="critical_points/1")

    magnetic_axis: PointRZ = sp_property(alias="critical_points/0/0/0")

    psi_axis: float = sp_property(alias="critical_points/0/0/1")

    psi_boundary: float = sp_property(alias="critical_points/1/0/1")

    @sp_property(label=r"\frac{d V}{d\psi}")
    def dvolume_dpsi(self) -> Expression:
        return Function(*self._surface_integral(1.0))

    @sp_property
    def Bpol(self) -> Expression:
        r"""$B_{pol}= \left|\nabla \psi \right|/2 \pi R $"""
        return (
            np.sqrt(self.psirz.pd(1, 0) ** 2 + self.psirz.pd(0, 1) ** 2)
            / _R
            * np.abs(self._s_RpZ * self._s_Bp / self._s_eBp_2PI)
        )

    # 磁面坐标的函数，ffprime，pprime
    ffprime: Expression = sp_property(alias="../profiles_1d/f_df_dpsi", label=r" \frac{f df}{d\psi}")
    pprime: Expression = sp_property(alias="../profiles_1d/dpressure_dpsi", label=r" \frac{dP}{d\psi}")
    #################################
    # Profiles 2D

    # def grid(self) -> Mesh:
    #     theta = _not_found_
    #     psi_norm = _not_found_
    #     raw_grid = self.get_cache("grid", _not_found_)
    #     if raw_grid is not _not_found_:
    #         theta = raw_grid.get("dim1", _not_found_)
    #         psi_norm = raw_grid.get("dim2", _not_found_)
    #     if theta is _not_found_:
    #         theta = self.get(".../code/parameters/theta", _not_found_)
    #         if theta is _not_found_:
    #             ntheta = self.get(".../code/parameters/num_of_theta", 64)
    #             theta = np.linspace(0, 2.0 * scipy.constants.pi, ntheta, endpoint=False)
    #     if psi_norm is _not_found_:
    #         psi_norm = self.get(".../code/parameters/psi_norm", (0.0, 0.995, 128))
    #         if isinstance(psi_norm, tuple):
    #             psi_norm = np.linspace(*psi_norm)
    #     if not isinstance(psi_norm, np.ndarray) or not isinstance(theta, np.ndarray):
    #         raise RuntimeError(f"Can not create grid! psi_norm={psi_norm} theta={theta}")
    #     # if not (isinstance(theta, np.ndarray) and theta.ndim == 1):
    #     #     raise ValueError(f"Can not create grid! theta={theta}")
    #     surfs = GeoObjectSet([surf for _, surf in self.find_surfaces(psi_norm)])
    #     return CurvilinearMesh(psi_norm, theta, geometry=surfs, periods=[False, 2.0 * scipy.constants.pi])

    @sp_property(mesh="grid")
    def r(self) -> Field:
        return self.grid.points[0]

    @sp_property(mesh="grid")
    def z(self) -> Field:
        return self.grid.points[1]

    @sp_property(mesh="grid")
    def jacobian(self) -> Field:
        raise NotImplementedError(f"")

    @sp_property(mesh="grid")
    def tensor_covariant(self) -> Field:
        raise NotImplementedError(f"")

    @sp_property(mesh="grid")
    def tensor_contravariant(self) -> Field:
        raise NotImplementedError(f"")

    ###############################
    # surface integral
    def find_surfaces_by_psi(
        self, psi, enclose_axis=True
    ) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:

        psi_axis = self.psi_axis
        psi_boundary = self.psi_boundary
        R = self.magnetic_axis[0]
        Z = self.magnetic_axis[1]

        for psi_val, surfs in find_contours(self.psirz, values=psi):

            if np.isclose(psi_val, psi_axis):
                yield psi_val, PointRZ(R, Z)

            # elif np.isclose(psi_val, psi_boundary):
            #     yield psi_val, next(surfs)

            else:
                for surf in surfs:
                    if enclose_axis and not (isinstance(surf, Curve) and surf.is_closed and surf.enclose(R, Z)):
                        continue

                    yield psi_val, surf

                # else:
                #     logger.warning(f"Can not find surf at {psi_val}  ")

    def find_surfaces(self, psi_norm, **kwargs) -> typing.Generator[typing.Tuple[float, GeoObject], None, None]:
        psi = psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis
        for p, surf in self.find_surfaces_by_psi(psi, **kwargs):
            yield (p - self.psi_axis) / (self.psi_boundary - self.psi_axis), surf

    @dataclass
    class ShapeProperty:
        psi: float | np.ndarray
        Rmin: float | np.ndarray
        Zmin: float | np.ndarray
        Rmax: float | np.ndarray
        Zmax: float | np.ndarray
        Rzmin: float | np.ndarray
        Rzmax: float | np.ndarray
        r_inboard: float | np.ndarray
        r_outboard: float | np.ndarray

    def shape_property(self, psi_norm: typing.Union[float, typing.Sequence[float]] = None) -> ShapeProperty:
        def shape_box(s: GeoObject):
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
            elif isinstance(s, GeoObject):
                r, z = s.points
                (rmin, zmin) = s.bbox.origin
                (rmax, zmax) = s.bbox.origin + s.bbox.dimensions
                rzmin = r[np.argmin(z)]
                rzmax = r[np.argmax(z)]
                r_inboard = s.coordinates(0.5)[0]
                r_outboard = s.coordinates(0)[0]
            else:
                raise TypeError(f"Invalid type {type(s)}")
            return rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard

        if psi_norm is None:
            psi_norm = self.psi_norm
        elif not isinstance(psi_norm, (np.ndarray, collections.abc.MutableSequence)):
            psi_norm = [psi_norm]

        sbox = np.asarray([[p, *shape_box(s)] for p, s in self.find_surfaces(psi_norm)], dtype=float)

        if sbox.shape[0] == 1:
            psi_norm, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox[0]
        else:
            psi_norm, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard = sbox.T
        if np.isscalar(psi_norm):
            return FyEqCoordinateSystem.ShapeProperty(
                psi_norm, rmin, zmin, rmax, zmax, rzmin, rzmax, r_inboard, r_outboard
            )
        else:
            return FyEqCoordinateSystem.ShapeProperty(
                psi_norm,
                Function(psi_norm, rmin, name="rmin"),
                Function(psi_norm, zmin, name="zmin"),
                Function(psi_norm, rmax, name="rmax"),
                Function(psi_norm, zmax, name="zmax"),
                Function(psi_norm, rzmin, name="rzmin"),
                Function(psi_norm, rzmax, name="rzmax"),
                Function(psi_norm, r_inboard, name="r_inboard"),
                Function(psi_norm, r_outboard, name="r_outboard"),
            )

    _surf_list = None

    def _surface_integral(self, func: Expression, psi_norm: array_type = None) -> typing.Tuple[ArrayLike, ArrayLike]:
        r"""
        $ V^{\prime} =  2 \pi  \int{ R / \left|\nabla \psi \right| * dl }$
        $ V^{\prime}(psi)= 2 \pi  \int{ dl * R / \left|\nabla \psi \right|}$
        """

        if psi_norm is None:
            psi_norm = self.psi_norm
            if self._surf_list is None:
                self._surf_list = [*self.find_surfaces(psi_norm)]
            surfs_list = self._surf_list
        else:
            if isinstance(psi_norm, scalar_type):
                psi_norm = [psi_norm]
            surfs_list = self.find_surfaces(psi_norm)

        x: list = []
        y = []
        for p, surf in surfs_list:
            if isinstance(surf, Curve):
                v = surf.integral(func / self.Bpol)
            elif isinstance(surf, Point):  # o-point
                # R, Z = surf.points
                # v = (func(R, Z) if callable(func) else func) * self.ddpsi(R, Z)
                # logger.debug((R, Z, v))
                # # v *= r**2/self.ddpsi(r, z)
                # logger.warning(f"Found point pos={surf.points}  psi={p}")
                v = 0  # np.nan
            else:
                continue
                logger.warning(f"Found an island at psi={p} pos={surf}")
            y.append(v)
            x.append(p)

        match len(psi_norm):
            case 0:
                y = None, None
            case 1:
                y = x[0], y[0]
            case _:
                y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        return y

    def surface_integral(self, func: Expression, psi_norm: NumericType = None) -> Expression | float:
        psi_norm, value = self._surface_integral(func, psi_norm)

        if np.isscalar(psi_norm):
            return value
        else:
            label = getattr(func, "__label__", func)
            return Function(
                psi_norm,
                value,
                name=f"surface_integral({label})",
                label=rf"<{func.__repr__()}>",
            )

    def surface_average(self, func: Expression, *xargs) -> Expression | ArrayLike:
        r"""
        $\left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}$
        """
        return self.surface_integral(func, *xargs) / self.dvolume_dpsi(*xargs)


class FyEqProfiles2D(equilibrium.EquilibriumProfiles2D):
    _coord: FyEqCoordinateSystem = sp_property(alias="../coordinate_system")
    _profiles_1d: equilibrium.EquilibriumProfiles1D = sp_property(alias="../profiles_1d")
    _global_quantities: equilibrium.EquilibriumGlobalQuantities = sp_property(alias="../global_quantities")

    psi: Field

    j_parallel: Field

    grid: Mesh

    r: array_type = sp_property(alias=["grid/points", (..., 0)])

    z: array_type = sp_property(alias=["grid/points", (..., 1)])

    @sp_property
    def psi_norm(self) -> Expression:
        """normalized psirz"""
        return (self.psi - self._coord.psi_axis) / (self._coord.psi_boundary - self._coord.psi_axis)

    @sp_property
    def phi(self) -> Expression:
        return self._profiles_1d.phi(self.psi_norm)

    @sp_property
    def theta(self) -> Expression:
        return self._coord.theta

    @sp_property
    def j_tor(self) -> Expression:
        return _R * self._profiles_1d.dpressure_dpsi(self.psi_norm) + self._profiles_1d.f_df_dpsi(self.psi_norm) / (
            _R * scipy.constants.mu_0
        )

    @sp_property(label="B_{r}")
    def b_field_r(self) -> Expression:
        """COCOS Eq.19 [O. Sauter and S.Yu. Medvedev, Computer Physics Communications 184 (2013) 293]"""
        return self.psi.pd(0, 1) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property(label="B_{z}")
    def b_field_z(self) -> Expression:
        return -self.psi.pd(1, 0) / _R * (self._coord._s_RpZ * self._coord._s_Bp / self._coord._s_eBp_2PI)

    @sp_property(label="B_{tor}")
    def b_field_tor(self) -> Expression:
        return self._profiles_1d.f(self.psi_norm) / _R

    @sp_property
    def Bpol2(self) -> Expression:
        r"""$B_{pol}= \left|\nabla \psi \right|/2 \pi R $"""
        return self.b_field_r**2 + self.b_field_z**2

    @sp_property(label="B^2")
    def B2(self) -> Expression:
        return self.b_field_r**2 + self.b_field_z**2 + self.b_field_tor**2

    @sp_property
    def grad_psi2(self) -> Expression:
        return self.psi.pd(1, 0) ** 2 + self.psi.pd(0, 1) ** 2

    @sp_property
    def grad_psi(self) -> Expression:
        return np.sqrt(self.grad_psi2)

    @sp_property
    def ddpsi(self) -> Expression:
        return np.sqrt(self.psi.pd(2, 0) * self.psi.pd(0, 2) + self.psi.pd(1, 1) ** 2)


class FyEqProfiles1D(equilibrium.EquilibriumProfiles1D):
    _root: equilibrium.EquilibriumTimeSlice = sp_property(alias="../")

    _profiles_2d: FyEqProfiles2D = sp_property(alias="../profiles_2d")

    _coord: FyEqCoordinateSystem = sp_property(alias="../coordinate_system")

    psi_norm: array_type

    f_df_dpsi: Expression

    ffprime: Expression = sp_property(alias="f_df_dpsi")

    dpressure_dpsi: Expression

    pprime: Expression = sp_property(alias="dpressure_dpsi")

    @sp_property
    def grid(self) -> CoreRadialGrid:
        psi_norm = self.psi_norm
        rho_tor_norm = self.rho_tor_norm(self.psi_norm)
        if rho_tor_norm[0] < 0:
            rho_tor_norm[0] = 0.0
        return CoreRadialGrid(
            {
                "psi_norm": psi_norm,
                "psi_axis": self._coord.psi_axis,
                "psi_boundary": self._coord.psi_boundary,
                "rho_tor_norm": rho_tor_norm,
                "rho_tor_boundary": np.sqrt(
                    np.abs(self.phi(self.psi_norm[-1]) / (scipy.constants.pi * self._coord.b0))
                ),
            }
        )

    @sp_property(label=r"\psi")
    def psi(self) -> Expression:
        return self.psi_norm * (self._coord.psi_boundary - self._coord.psi_axis) + self._coord.psi_axis

    @sp_property(label="f")
    def f(self) -> Expression:
        return np.sqrt(
            2.0 * (self._coord.psi_boundary - self._coord.psi_axis) * self.f_df_dpsi.I
            + (self._coord.b0 * self._coord.r0) ** 2
        )

    @sp_property(label=r"\phi")
    def phi(self) -> Expression:
        return self.dphi_dpsi.I * (self._coord.psi_boundary - self._coord.psi_axis)

    @sp_property(label=r"\rho_{tor}")
    def rho_tor(self) -> Expression:
        return np.sqrt(np.abs(self.phi / (scipy.constants.pi * self._coord.b0)))

    @sp_property(label=r"\bar{\rho}_{tor}")
    def rho_tor_norm(self) -> Expression:
        phi = np.asarray(self.phi)

        if np.isclose(self.psi_norm[-1], 1.0):
            phi_boundary = phi[-1]
        else:
            phi_boundary = self.phi(1.0)
        r_ = np.sqrt(phi / phi_boundary)
        if r_[0] < 0:
            r_[0] = 0.0
        return r_

    @sp_property(label=r"\frac{d V}{d\psi}")
    def dvolume_dpsi(self) -> Expression:
        return self._coord.dvolume_dpsi

    @sp_property(label=r"q")
    def q(self) -> Expression:
        return self.dphi_dpsi * (self._coord._s_eBp_2PI / (2.0 * scipy.constants.pi))

    @sp_property
    def magnetic_shear(self) -> Expression:
        return self.rho_tor * self.q.d / self.q * self.dpsi_drho_tor

    @sp_property(label=r"\frac{d\phi}{d\psi}")
    def dphi_dpsi(self) -> Expression:
        return self.f * self._coord.surface_integral(1.0 / (_R**2))

    @sp_property
    def drho_tor_dpsi(self) -> Expression:
        r"""
        $\frac{d\rho_{tor}}{d\psi} 
            =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
            =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
            =\frac{q}{2\pi B_{0}\rho_{tor}}
        $
        """
        return 1.0 / self.dpsi_drho_tor

    @sp_property
    def dpsi_drho_tor(self) -> Expression:
        return np.abs(self._coord.b0) * self.rho_tor / self.q

    @sp_property
    def dpsi_drho_tor_norm(self) -> Expression:
        return self.dpsi_drho_tor * self._coord.rho_tor_boundary

    @sp_property
    def dpsi_norm_drho_tor_norm(self) -> Expression:
        return self.dpsi_drho_tor * self._coord.rho_tor_boundary / (self._coord.psi_boundary - self._coord.psi_axis)

    @sp_property(label=r"\frac{dV}{d\rho_{tor}}")
    def dvolume_drho_tor(self) -> Expression:
        return self._coord._s_eBp_2PI * np.abs(self._coord.b0) * self.dvolume_dpsi * self.dpsi_drho_tor

    @sp_property
    def volume(self) -> Expression:
        return self.dvolume_dpsi.I * (self.grid.psi_boundary - self.grid.psi_axis)

    @sp_property
    def area(self) -> Expression:
        return self.darea_dpsi.I * (self.grid.psi_boundary - self.grid.psi_axis)

    @sp_property
    def darea_dpsi(self) -> Expression:
        """FIXME: just a simple approximation!"""
        return self.dvolume_dpsi / ((2.0 * scipy.constants.pi) * self._coord.r0)

    @sp_property
    def darea_drho_tor(self) -> Expression:
        return self.darea_dpsi * self.dpsi_drho_tor

    @sp_property
    def surface(self) -> Expression:
        return self.dvolume_drho_tor * self.gm7

    @sp_property
    def dphi_dvolume(self) -> Expression:
        return self.f * self.gm1

    @sp_property
    def gm1(self) -> Expression:
        return self._coord.surface_average(1.0 / (_R**2))

    @sp_property
    def gm2(self) -> Expression:
        return self._coord.surface_average(self._profiles_2d.grad_psi2 / (_R**2)) / (self.dpsi_drho_tor**2)

    @sp_property
    def gm3(self) -> Expression:
        return self._coord.surface_average(self._profiles_2d.grad_psi2) / (self.dpsi_drho_tor**2)

    @sp_property
    def gm4(self) -> Expression:
        return self._coord.surface_average(1.0 / self._profiles_2d.B2)

    @sp_property
    def gm5(self) -> Expression:
        return self._coord.surface_average(self._profiles_2d.B2)

    @sp_property
    def gm6(self) -> Expression:
        return self._coord.surface_average(self._profiles_2d.grad_psi2 / self._profiles_2d.B2) / (
            self.dpsi_drho_tor**2
        )

    @sp_property
    def gm7(self) -> Expression:
        return self._coord.surface_average(np.sqrt(self._profiles_2d.grad_psi2)) / self.dpsi_drho_tor

    @sp_property
    def gm8(self) -> Expression:
        return self._coord.surface_average(_R)

    @sp_property
    def gm9(self) -> Expression:
        return self._coord.surface_average(1.0 / _R)

    # 描述磁面形状
    @functools.cached_property
    def _shape_property(self) -> FyEqCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def minor_radius(self) -> Expression:
        return (self._shape_property.Rmax - self._shape_property.Rmin) * 0.5

    @sp_property
    def major_radius(self) -> Expression:
        return (self._shape_property.Rmax + self._shape_property.Rmin) * 0.5

    @sp_property
    def magnetic_z(self) -> Expression:
        return (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5

    @sp_property
    def r_inboard(self) -> Expression:
        return self._shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Expression:
        return self._shape_property.r_outboard

    @sp_property
    def elongation(self) -> Expression:
        return (self._shape_property.Zmax - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_upper(self) -> Expression:
        return (self._shape_property.Zmax - (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_lower(self) -> Expression:
        return ((self._shape_property.Zmax + self._shape_property.Zmin) * 0.5 - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def triangularity_upper(self) -> Expression:
        return (
            ((self._shape_property.Rmax - self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmax)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_lower(self) -> Expression:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity(self) -> Expression:
        psi = self.grid.psi[1:]
        res = (
            (self._shape_property.Rzmax(psi) - self._shape_property.Rzmin(psi))
            / (self._shape_property.Rmax(psi) - self._shape_property.Rmin(psi))
            * 2
        )
        return Function(psi, res, name="triangularity")

    @sp_property
    def squareness(self) -> Expression:
        return 0.0

    @sp_property
    def trapped_fraction(self) -> Expression:
        """Trapped particle fraction[-]
        Tokamak 3ed, 14.10
        """
        epsilon = self.rho_tor(self.psi) / self._coord._R0
        return 1.0 - (1 - epsilon) ** 2 / np.sqrt(1.0 - epsilon**2) / (1 + 1.46 * np.sqrt(epsilon))

    @sp_property
    def plasma_current(self) -> Expression:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor / scipy.constants.mu_0

    @sp_property
    def j_tor(self) -> Expression:
        return self.plasma_current.d / self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Expression:
        return self._coord.surface_average(dot(self._coord.j, self._coord.B) / np.sqrt(self._coord.B2))

        # fvac = self._coord._fvac
        # d = np.asarray(function_like(np.asarray(self.volume),
        #                              np.asarray(fvac*self.plasma_current/self.f)).pd())
        # return self._coord._R0*(self.f / fvac)**2 * d


class FyEqGlobalQuantities(equilibrium.EquilibriumGlobalQuantities):
    _root: equilibrium.EquilibriumTimeSlice = sp_property(alias="../")

    _coord: FyEqCoordinateSystem = sp_property(alias="../coordinate_system")

    psi_axis: float

    psi_boundary: float

    magnetic_axis: PointRZ = sp_property(alias="../coordinate_system/magnetic_axis")

    @sp_property
    def b_field_tor_axis(self) -> float:
        return (self._root.profiles_2d.b_field_tor(self.magnetic_axis.r, self.magnetic_axis.z),)

    @sp_property
    def q_axis(self) -> float:
        return self._root.profiles_1d.q(0.0)

    @sp_property
    def q_95(self) -> float:
        return self._root.profiles_1d.q(0.95)

    @sp_property
    def q_min(self) -> typing.Tuple[float, float]:
        q = np.asarray(self._root.profiles_1d.q)
        idx = np.argmin(q)
        return q[idx], self._root.profiles_1d.rho_tor_norm[idx]


class FyEqBoundary(equilibrium.EquilibriumBoundary):
    _coord: FyEqCoordinateSystem = sp_property(alias="../coordinate_system")

    # psi_norm: float

    psi: float

    phi: float

    rho_tor_norm: float

    @sp_property
    def outline(self) -> CurveRZ:
        try:
            _, surf = next(self._coord.find_surfaces(self.psi_norm))
        except StopIteration:
            raise RuntimeError(f"Can not find surface at psi_norm={self.psi_norm} ")
        return surf

    @functools.cached_property
    def _shape_property(self) -> FyEqCoordinateSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> PointRZ:
        return {
            "r": (self._shape_property.Rmin + self._shape_property.Rmax) * 0.5,
            "z": (self._shape_property.Zmin + self._shape_property.Zmax) * 0.5,
        }

    @sp_property
    def minor_radius(self) -> float:
        return (self._shape_property.Rmax - self._shape_property.Rmin) * 0.5

    @sp_property
    def elongation(self) -> float:
        return (self._shape_property.Zmax - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_upper(self) -> float:
        return (self._shape_property.Zmax - (self._shape_property.Zmax + self._shape_property.Zmin) * 0.5) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property
    def elongation_lower(self) -> float:
        return ((self._shape_property.Zmax + self._shape_property.Zmin) * 0.5 - self._shape_property.Zmin) / (
            self._shape_property.Rmax - self._shape_property.Rmin
        )

    @sp_property(coordinate1="../psi")
    def triangularity(self) -> float:
        return (
            (self._shape_property.Rzmax - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_upper(self) -> float:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmax)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def triangularity_lower(self) -> float:
        return (
            ((self._shape_property.Rmax + self._shape_property.Rmin) * 0.5 - self._shape_property.Rzmin)
            / (self._shape_property.Rmax - self._shape_property.Rmin)
            * 2
        )

    @sp_property
    def x_point(self) -> PointSetRZ:
        return [pv[0] for pv in self._coord.x_points]

    @sp_property
    def strike_point(self) -> PointSetRZ:
        return

    @sp_property
    def active_limiter_point(self) -> PointSetRZ:
        return NotImplemented


class FyEqBoundarySeparatrix(FyEqBoundary):
    _coord: FyEqCoordinateSystem = sp_property(alias="../coordinate_system")

    @sp_property
    def outline(self) -> GeoObjectSet:
        """RZ outline of the plasma boundary"""
        return GeoObjectSet([surf for _, surf in self._coord.find_surfaces(self.psi_norm, enclose_axis=False)])


class FyEqTimeSlice(equilibrium.EquilibriumTimeSlice):
    vacuum_toroidal_field: VacuumToroidalField

    global_quantities: FyEqGlobalQuantities

    profiles_1d: FyEqProfiles1D

    profiles_2d: FyEqProfiles2D

    boundary: FyEqBoundary

    boundary_separatrix: FyEqBoundarySeparatrix

    coordinate_system: FyEqCoordinateSystem = sp_property(default_value={})


class FyEq(
    equilibrium.Equilibrium[FyEqTimeSlice],
    code={
        "name": "fy_eq",
        "version": FY_VERSION,
        "copyright": FY_COPYRIGHT,
    },
):
    """
    Magnetic surface analyze 磁面分析工具
    =============================
    input:
        - vacuum_toroidal_field.b0, vacuum_toroidal_field.r0
        - f, Diamagnetic function (F=R B_Phi)
        - profiles_2d.psi (RZ 2D)

    output：
        - 识别 O,X point
        - 识别 Separatrix, boundary
        - Surface average

    """
