from __future__ import annotations
import numpy as np
from typing_extensions import Self

from spdm.core.aos import AoS
from spdm.core.expression import Expression
from spdm.core.sp_property import sp_property
from spdm.core.time_series import TimeSeriesAoS, TimeSlice
from spdm.core.geo_object import GeoObject, GeoObjectSet
from spdm.core.mesh import Mesh

from spdm.geometry.curve import Curve
from spdm.geometry.point import Point

from spdm.utils.tags import _not_found_

from .utilities import *

from ..utils.logger import logger
from ..ontology import equilibrium


@sp_tree(domain="grid")
class EquilibriumCoordinateSystem(equilibrium._T_equilibrium_coordinate_system):
    grid_type: Identifier

    grid: Mesh

    r: Field = sp_property(units="m")

    z: Field = sp_property(units="m")

    jacobian: Field = sp_property(units="mixed")

    tensor_covariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")

    tensor_contravariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")


@sp_tree
class EquilibriumGlobalQuantities(equilibrium._T_equilibrium_global_quantities):
    psi_axis: float = sp_property(units="Wb")

    psi_boundary: float = sp_property(units="Wb")

    b_field_tor_axis: float = sp_property(units="T")

    magnetic_axis: Point
    """ Magnetic axis position and toroidal field"""

    ip: float = sp_property(units="A")

    beta_pol: float

    beta_tor: float

    beta_normal: float

    li_3: float

    volume: float = sp_property(units="m^3")

    area: float = sp_property(units="m^2")

    surface: float = sp_property(units="m^2")

    length_pol: float = sp_property(units="m")

    @sp_tree
    class CurrentCentre:
        r: float = sp_property(units="m")
        z: float = sp_property(units="m")
        velocity_z: float = sp_property(units="m.s^-1")

    current_centre: CurrentCentre

    q_axis: float

    q_95: float

    @sp_tree
    class Qmin:
        value: float
        rho_tor_norm: float

    q_min: Qmin

    energy_mhd: float = sp_property(units="J")

    psi_external_average: float = sp_property(units="Wb")

    v_external: float = sp_property(units="V")

    plasma_inductance: float = sp_property(units="H")

    plasma_resistance: float = sp_property(units="ohm")


@sp_tree(domain="psi_norm")
class EquilibriumProfiles1D(equilibrium._T_equilibrium_profiles_1d):
    """
    1D profiles of the equilibrium quantities
    NOTE:
        - psi_norm is the normalized poloidal flux
        - psi is the poloidal flux,
        - 以psi_norm为主坐标, 是因为 psi_norm 时必定单调增的，而 psi 由于符号的原因，不一定时单调增的。
          scipy.interpolate 在一维插值时，要求 x 为单增。以 psi 为 磁面坐标，在插值时会造成问题。
          profiles_1d 中涉及对磁面坐标的求导和积分时需注意修正 ！！！！

    """

    @property
    def _root(self):  # -> EquilibriumTimeSlice:
        return self._parent

    @sp_property
    def grid(self) -> CoreRadialGrid:
        L = (self.psi[-1] - self.psi[0]) / (self.psi_norm[-1] - self.psi_norm[0])
        psi_axis = self.psi[-1] - L * self.psi_norm[-1]
        psi_boundary = L + psi_axis
        psi_norm = self.psi_norm
        rho_tor_norm = _not_found_
        rho_tor_boundary = _not_found_
        if self.rho_tor_norm is not _not_found_:
            rho_tor_norm = self.rho_tor_norm(self.psi_norm)
            if self.rho_tor is not _not_found_:
                rho_tor_boundary = self.rho_tor[-1] / self.rho_tor_norm[-1]

        return CoreRadialGrid(
            {
                "psi_norm": psi_norm,
                "rho_tor_norm": rho_tor_norm,
                "psi_axis": psi_axis,
                "psi_boundary": psi_boundary,
                "rho_tor_boundary": rho_tor_boundary,
            }
        )

    psi_norm: array_type | Expression = sp_property(units="-", label=r"\bar{\psi}")

    psi: array_type | Expression = sp_property(units="Wb", label=r"\psi")

    dphi_dpsi: Expression = sp_property(label=r"\frac{d\phi}{d\psi}", units="-")

    phi: Expression = sp_property(units="Wb", label=r"\phi")

    q: Expression = sp_property(units="-", label="q")

    pressure: Expression = sp_property(units="Pa", label="P")

    dpressure_dpsi: Expression = sp_property(units="Pa.Wb^-1", label=r"\frac{dP}{d\psi}")

    f: Expression = sp_property(units="T.m")

    f_df_dpsi: Expression = sp_property(units="T^2.m^2/Wb", label=r"\frac{f d f}{d \psi}")

    j_tor: Expression = sp_property(units="A \cdot m^{-2}")

    j_parallel: Expression = sp_property(units="A/m^2")

    magnetic_shear: Expression = sp_property(units="-")

    r_inboard: Expression = sp_property(units="m")

    r_outboard: Expression = sp_property(units="m")

    rho_tor: Expression = sp_property(units="m", label=r"\rho_{tor}")

    rho_tor_norm: Expression = sp_property(units="m", label=r"\bar{\rho_{tor}}")

    dpsi_drho_tor: Expression = sp_property(units="Wb/m", label=r"\frac{d\psi}{d\rho_{tor}}")

    @sp_property
    def geometric_axis(self) -> Point:
        return (self.major_radius, self.magnetic_z)

    minor_radius: Expression = sp_property(units="m")

    major_radius: Expression = sp_property(units="m")  # R0

    magnetic_z: Expression = sp_property(units="m")  # Z0

    elongation: Expression

    triangularity_upper: Expression

    triangularity_lower: Expression

    triangularity: Expression

    squareness_upper_inner: Expression

    squareness_upper_outer: Expression

    squareness_lower_inner: Expression

    squareness_lower_outer: Expression

    squareness: Expression = sp_property(default_value=zero)

    volume: Expression = sp_property(units="m^3")

    rho_volume_norm: Expression

    dvolume_dpsi: Expression = sp_property(units="m^3 \cdot Wb^{-1}")

    dvolume_drho_tor: Expression = sp_property(units="m^2", label=r"V^{\prime}")

    area: Expression = sp_property(units="m^2")

    darea_dpsi: Expression = sp_property(units="m^2 \cdot Wb^{-1}")

    darea_drho_tor: Expression = sp_property(units="m")

    surface: Expression = sp_property(units="m^2")

    trapped_fraction: Expression

    gm1: Expression
    gm2: Expression
    gm3: Expression
    gm4: Expression
    gm5: Expression
    gm6: Expression
    gm7: Expression
    gm8: Expression
    gm9: Expression

    b_field_average: Expression = sp_property(units="T")

    b_field_min: Expression = sp_property(units="T")

    b_field_max: Expression = sp_property(units="T")

    beta_pol: Expression

    mass_density: Expression = sp_property(units="kg \cdot m^{-3}")


@sp_tree(domain="grid")
class EquilibriumProfiles2D(equilibrium._T_equilibrium_profiles_2d):
    type: Identifier

    grid: Mesh

    r: Field = sp_property(units="m")

    z: Field = sp_property(units="m")

    psi: Field = sp_property(units="Wb")

    theta: Field = sp_property(units="rad")

    phi: Field = sp_property(units="Wb")

    j_tor: Field = sp_property(units="A.m^-2")

    j_parallel: Field = sp_property(units="A.m^-2")

    b_field_r: Field = sp_property(units="T")

    b_field_z: Field = sp_property(units="T")

    b_field_tor: Field = sp_property(units="T")


@sp_tree
class EquilibriumBoundary(equilibrium._T_equilibrium_boundary):
    type: int

    outline: Curve

    psi_norm: float = sp_property(default_value=0.995)

    psi: float = sp_property(units="Wb")

    geometric_axis: Point

    minor_radius: float = sp_property(units="m")

    elongation: float

    elongation_upper: float

    elongation_lower: float

    triangularity: float

    triangularity_upper: float

    triangularity_lower: float

    squareness_upper_inner: float

    squareness_upper_outer: float

    squareness_lower_inner: float

    squareness_lower_outer: float

    x_point: AoS[Point]

    strike_point: AoS[Point]

    active_limiter_point: Point


@sp_tree
class EquilibriumBoundarySeparatrix(equilibrium._T_equilibrium_boundary_separatrix):
    type: int

    outline: GeoObjectSet

    psi: float = sp_property(units="Wb")

    geometric_axis: Point

    minor_radius: float = sp_property(units="m")

    elongation: float

    elongation_upper: float

    elongation_lower: float

    triangularity: float

    triangularity_upper: float

    triangularity_lower: float

    squareness_upper_inner: float

    squareness_upper_outer: float

    squareness_lower_inner: float

    squareness_lower_outer: float

    x_point: AoS[Point]

    strike_point: AoS[Point]

    active_limiter_point: Point


@sp_tree
class EequilibriumConstraints(equilibrium._T_equilibrium_constraints):
    pass


@sp_tree
class EquilibriumGGD(equilibrium._T_equilibrium_ggd):
    pass


@sp_tree
class EquilibriumTimeSlice(equilibrium._T_equilibrium_time_slice):
    vacuum_toroidal_field: VacuumToroidalField

    Boundary = EquilibriumBoundary
    boundary: EquilibriumBoundary

    BoundarySeparatrix = EquilibriumBoundarySeparatrix
    boundary_separatrix: EquilibriumBoundarySeparatrix

    Constraints = EequilibriumConstraints
    constraints: EequilibriumConstraints

    GlobalQuantities = EquilibriumGlobalQuantities
    global_quantities: EquilibriumGlobalQuantities

    Profiles1D = EquilibriumProfiles1D
    profiles_1d: EquilibriumProfiles1D

    Profiles2D = EquilibriumProfiles2D
    profiles_2d: EquilibriumProfiles2D

    CoordinateSystem = EquilibriumCoordinateSystem
    coordinate_system: EquilibriumCoordinateSystem

    GGD = EquilibriumGGD
    ggd: GGD

    def __view__(self, view_point="RZ", **kwargs):
        """
        plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        geo = {}

        match view_point.lower():
            case "rz":
                geo["psi"] = self.profiles_2d.psi.__view__()

                try:
                    geo["o_points"] = Point(
                        self.global_quantities.magnetic_axis.r,
                        self.global_quantities.magnetic_axis.z,
                        styles={"$matplotlib": {"color": "red", "marker": ".", "linewidths": 0.5}},
                    )

                    geo["x_points"] = [
                        Point(
                            p.r,
                            p.z,
                            name=f"{idx}",
                            styles={"$matplotlib": {"color": "blue", "marker": "x", "linewidths": 0.5}},
                        )
                        for idx, p in enumerate(self.boundary.x_point)
                    ]

                    # geo["strike_points"] = [
                    #     Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(self.boundary.strike_point)
                    # ]

                    geo["boundary"] = self.boundary.outline
                    geo["boundary"]._metadata["styles"] = {
                        "$matplotlib": {"color": "blue", "linestyle": "dotted", "linewidth": 0.5}
                    }
                    # geo["boundary_separatrix"] = self.boundary_separatrix.outline
                    # if geo["boundary_separatrix"] is not _not_found_:
                    #     geo["boundary_separatrix"]._metadata["styles"] = {
                    #         "$matplotlib": {"color": "red", "linestyle": "dashed", "linewidth": 0.25}
                    #     }
                except Exception as error:
                    raise error

        geo["styles"] = kwargs

        return geo

    # def __view__(self, view_port="RZ", **kwargs) -> GeoObject:
    #     geo = {}

    #     if view_port == "RZ":
    #         o_points, x_points = self.coordinate_system.critical_points

    #         geo["o_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(o_points)]
    #         geo["x_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(x_points)]

    #         geo["boundary"] = Curve(self.boundary.outline.r.__array__(), self.boundary.outline.z.__array__())

    #         geo["boundary_separatrix"] = Curve(
    #             self.boundary_separatrix.outline.r.__array__(),
    #             self.boundary_separatrix.outline.z.__array__(),
    #         )

    #     geo["psi"] = self.profiles_2d.psi.__view__()

    #     styles = {
    #         "o_points": {"$matplotlib": {"c": "red", "marker": "."}},
    #         "x_points": {"$matplotlib": {"c": "blue", "marker": "x"}},
    #         "boundary": {"$matplotlib": {"color": "red", "linewidth": 0.5}},
    #         "boundary_separatrix": {
    #             "$matplotlib": {
    #                 "color": "red",
    #                 "linestyle": "dashed",
    #                 "linewidth": 0.25,
    #             }
    #         },
    #     }
    #     styles = update_tree(styles, kwargs)

    #     return geo, styles


from .wall import Wall
from .tf import TF
from .magnetics import Magnetics
from .pf_active import PFActive


@sp_tree
class Equilibrium(IDS):
    r"""
    Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

    Reference:

    - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.
    """

    code: Code = {"name": "fy_eq"}  # default plugin

    ids_properties: IDSProperties

    TimeSlice = EquilibriumTimeSlice

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice]

    def __view__(self, *args, **kwargs):
        current = self.time_slice.current
        return current.__view__(*args, **kwargs) if current is not _not_found_ else {}

    def refresh(
        self,
        *args,
        time=None,
        wall: Wall = None,
        tf: TF = None,
        magnetics: Magnetics = None,
        pf_active: PFActive = None,
        **kwargs,
    ) -> None:
        return super().refresh(
            *args,
            time=time,
            tf=tf,
            wall=wall,
            magnetics=magnetics,
            pf_active=pf_active,
            **kwargs,
        )


r"""
  COCOS  11
    ```{text}
        Top view
                ***************
                *               *
            *   ***********   *
            *   *           *   *
            *   *             *   *
            *   *             *   *
        Ip  v   *             *   ^  \phi
            *   *    Z o--->R *   *
            *   *             *   *
            *   *             *   *
            *   *     Bpol    *   *
            *   *     o     *   *
            *   ***********   *
                *               *
                ***************
                Bpol x
                Poloidal view
            ^Z
            |
            |       ************
            |      *            *
            |     *         ^    *
            |     *   \rho /     *
            |     *       /      *
            +-----*------X-------*---->R
            |     *  Ip, \phi   *
            |     *              *
            |      *            *
            |       *****<******
            |       Bpol,\theta
            |
                Cylindrical coordinate      : $(R,\phi,Z)$
        Poloidal plane coordinate   : $(\rho,\theta,\phi)$
    ```
"""
