""" Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
"""

from spdm.utils.tags import _not_found_
from spdm.utils.type_hint import array_type, ArrayType

from spdm.core.expression import Expression, zero
from spdm.core.sp_tree import annotation, sp_property, SpTree
from spdm.core.mesh import Mesh
from spdm.core.field import Field
from spdm.core.time import WithTime
from spdm.core.domain import WithDomain
from spdm.core.geo_object import GeoObject
from spdm.geometry.curve import Curve
from spdm.geometry.point import PointRZ
from spdm.geometry.point_set import PointSetRZ


from fytok.utils.base import IDS, Identifier, FyEntity
from fytok.modules.utilities import CoreRadialGrid, VacuumToroidalField


class EquilibriumBoundary(SpTree):
    """Boundary of the equilibrium"""

    type: int

    outline: GeoObject

    psi_norm: float = 0.995

    psi: float = annotation(units="Wb")

    geometric_axis: PointRZ

    minor_radius: float = annotation(units="m")

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

    x_point: PointSetRZ

    strike_point: PointSetRZ

    active_limiter_point: PointRZ


class EquilibriumBoundarySeparatrix(SpTree):
    """Boundary of the equilibrium"""

    type: int

    outline: Curve

    psi_norm: float = 1.0

    psi: float = annotation(units="Wb")

    geometric_axis: PointRZ

    minor_radius: float = annotation(units="m")

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

    x_point: PointSetRZ

    strike_point: PointSetRZ

    active_limiter_point: PointRZ


class EquilibriumGlobalQuantities(SpTree):
    """Global quantities of the equilibrium"""

    psi_axis: float = annotation(units="Wb")

    psi_boundary: float = annotation(units="Wb")

    b_field_tor_axis: float = annotation(units="T")

    magnetic_axis: PointRZ
    """ Magnetic axis position and toroidal field"""

    ip: float = annotation(units="A")

    beta_pol: float

    beta_tor: float

    beta_normal: float

    li_3: float

    volume: float = annotation(units="m^3")

    area: float = annotation(units="m^2")

    surface: float = annotation(units="m^2")

    length_pol: float = annotation(units="m")

    class CurrentCentre(SpTree):
        r: float = annotation(units="m")
        z: float = annotation(units="m")
        velocity_z: float = annotation(units="m.s^-1")

    current_centre: CurrentCentre

    q_axis: float

    q_95: float

    class Qmin(SpTree):
        value: float
        rho_tor_norm: float

    q_min: Qmin

    energy_mhd: float = annotation(units="J")

    psi_external_average: float = annotation(units="Wb")

    v_external: float = annotation(units="V")

    plasma_inductance: float = annotation(units="H")

    plasma_resistance: float = annotation(units="ohm")


class EquilibriumCoordinateSystem(WithDomain, SpTree, domain="grid"):
    """Coordinate system of the equilibrium"""

    grid: Mesh

    r: Field = annotation(units="m")

    z: Field = annotation(units="m")

    jacobian: Field = annotation(units="mixed")

    tensor_covariant: Field = annotation(coordinate3="1...3", coordinate4="1...3", units="mixed")

    tensor_contravariant: Field = annotation(coordinate3="1...3", coordinate4="1...3", units="mixed")


class EquilibriumProfiles1D(WithDomain, SpTree, domain="psi_norm"):
    """
    1D profiles of the equilibrium quantities
    NOTE:
        - psi_norm is the normalized poloidal flux
        - psi is the poloidal flux,
        - 以psi_norm为主坐标, 是因为 psi_norm 时必定单调增的，而 psi 由于符号的原因，不一定时单调增的。
          scipy.interpolate 在一维插值时，要求 x 为单增。以 psi 为 磁面坐标，在插值时会造成问题。
          profiles_1d 中涉及对磁面坐标的求导和积分时需注意修正 ！！！！

    """

    grid: CoreRadialGrid = {"primary_coordinate": "psi_norm"}

    psi_norm: ArrayType = annotation(units="-", label=r"\bar{\psi}")

    psi: Expression = annotation(units="Wb", label=r"\psi")

    dphi_dpsi: Expression = annotation(units="-", label=r"\frac{d\phi}{d\psi}")

    phi: Expression = annotation(units="Wb", label=r"\phi")

    q: Expression = annotation(units="-", label="q")

    pressure: Expression = annotation(units="Pa", label="P")

    dpressure_dpsi: Expression = annotation(units=r"Pa.Wb^-1", label=r"\frac{dP}{d\psi}")

    f: Expression = annotation(units="T.m")

    f_df_dpsi: Expression = annotation(units=r"T^2.m^2/Wb", label=r"\frac{f d f}{d \psi}")

    j_tor: Expression = annotation(units=r"A \cdot m^{-2}")

    j_parallel: Expression = annotation(units=r"A/m^2")

    magnetic_shear: Expression = annotation(units="-")

    r_inboard: Expression = annotation(units="m")

    r_outboard: Expression = annotation(units="m")

    rho_tor: Expression = annotation(units="m", label=r"\rho_{tor}")

    rho_tor_norm: Expression = annotation(units="-", label=r"\bar{\rho_{tor}}")

    dpsi_drho_tor: Expression = annotation(units="Wb/m", label=r"\frac{d\psi}{d\rho_{tor}}")

    @sp_property
    def geometric_axis(self) -> PointRZ:
        return PointRZ(self.major_radius, self.magnetic_z)

    minor_radius: Expression = annotation(units="m")

    major_radius: Expression = annotation(units="m")  # R0

    magnetic_z: Expression = annotation(units="m")  # Z0

    elongation: Expression

    triangularity_upper: Expression

    triangularity_lower: Expression

    triangularity: Expression

    squareness_upper_inner: Expression

    squareness_upper_outer: Expression

    squareness_lower_inner: Expression

    squareness_lower_outer: Expression

    squareness: Expression = zero

    volume: Expression = annotation(units=r"m^3")

    rho_volume_norm: Expression

    dvolume_dpsi: Expression = annotation(units=r"m^3 \cdot Wb^{-1}")

    dvolume_drho_tor: Expression = annotation(units=r"m^2", label=r"V^{\prime}")

    area: Expression = annotation(units=r"m^2")

    darea_dpsi: Expression = annotation(units=r"m^2 \cdot Wb^{-1}")

    darea_drho_tor: Expression = annotation(units=r"m")

    surface: Expression = annotation(units=r"m^2")

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

    b_field_average: Expression = annotation(units="T")

    b_field_min: Expression = annotation(units="T")

    b_field_max: Expression = annotation(units="T")

    beta_pol: Expression

    mass_density: Expression = annotation(units=r"kg \cdot m^{-3}")


class EquilibriumProfiles2D(WithDomain, SpTree, domain="grid"):
    """Profiles 2D"""

    type: Identifier

    grid: Mesh

    grid_type: str = annotation(alias="grid/type")

    r: Field = annotation(units="m", alias=["grid/coordinates", (0,)])

    z: Field = annotation(units="m", alias=["grid/coordinates", (1,)])

    psi: Field = annotation(units="Wb")

    theta: Expression = annotation(units="rad")

    phi: Expression = annotation(units="Wb")

    j_tor: Expression = annotation(units=r"A.m^{-2}")

    j_parallel: Expression = annotation(units=r"A.m^-2", label=r"$j_{\parallel}$")

    b_field_r: Expression = annotation(units="T", label=r"B_{r}")

    b_field_z: Expression = annotation(units="T", label=r"B_{z}")

    b_field_tor: Expression = annotation(units="T", label=r"B_{tor}")


class EquilibriumGGD(WithDomain, SpTree, domain="grid"):
    grid: Mesh


class Equilibrium(
    WithTime,
    IDS,
    FyEntity,
    plugin_prefix="equilibrium/",
    plugin_default="fy_eq",
    code={"name": "equilibrium"},
):
    r"""
    Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

    Reference:

    - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS",
      Computer Physics Communications 184, 2 (2013), pp. 293--302.

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

    vacuum_toroidal_field: VacuumToroidalField

    Boundary = EquilibriumBoundary
    boundary: EquilibriumBoundary

    BoundarySeparatrix = EquilibriumBoundarySeparatrix
    boundary_separatrix: EquilibriumBoundarySeparatrix

    GlobalQuantities = EquilibriumGlobalQuantities
    global_quantities: EquilibriumGlobalQuantities

    Profiles1D = EquilibriumProfiles1D
    profiles_1d: EquilibriumProfiles1D

    Profiles2D = EquilibriumProfiles2D
    profiles_2d: EquilibriumProfiles2D

    CoordinateSystem = EquilibriumCoordinateSystem
    coordinate_system: EquilibriumCoordinateSystem

    def __view__(self, view_point="RZ", **kwargs):
        """
        plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        geo = {}

        match view_point.lower():
            case "rz":
                geo["psi"] = self.profiles_2d.psi.__view__()

                try:
                    geo["o_points"] = PointRZ(
                        self.global_quantities.magnetic_axis[0],
                        self.global_quantities.magnetic_axis[1],
                        styles={
                            "$matplotlib": {
                                "color": "red",
                                "marker": ".",
                                "linewidths": 0.5,
                            }
                        },
                    )

                    geo["x_points"] = [
                        PointRZ(
                            p[0],
                            p[1],
                            name=f"{idx}",
                            styles={
                                "$matplotlib": {
                                    "color": "blue",
                                    "marker": "x",
                                    "linewidths": 0.5,
                                }
                            },
                        )
                        for idx, p in enumerate(self.boundary.x_point)
                    ]

                    # geo["strike_points"] = [
                    #     PointRZ(p.r, p.z, name=f"{idx}") for idx, p in enumerate(self.boundary.strike_point)
                    # ]
                    if self.boundary is not _not_found_:
                        geo["boundary"] = self.boundary.outline
                        geo["boundary"]._metadata["styles"] = {
                            "$matplotlib": {
                                "color": "blue",
                                "linestyle": "dotted",
                                "linewidth": 0.5,
                            }
                        }
                    if self.boundary_separatrix is not _not_found_:
                        geo["boundary_separatrix"] = self.boundary_separatrix.outline
                        geo["boundary_separatrix"]._metadata["styles"] = {
                            "$matplotlib": {
                                "color": "red",
                                "linestyle": "dashed",
                                "linewidth": 0.25,
                            }
                        }
                except Exception as error:
                    raise error

        geo["styles"] = kwargs

        return geo
