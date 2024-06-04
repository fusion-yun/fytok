# from __future__ import annotations
# @NOTE：
#   在插件中 from __future__ import annotations 会导致插件无法加载，
#   故障点是：typing.get_type_hints() 找不到类型， i.e. Code,TimeSeriesAoS

import typing
import numpy as np
import scipy.constants

from spdm.core.expression import Variable, Expression, Scalar, one, zero, derivative
from spdm.core.sp_property import sp_tree
from spdm.core.path import as_path
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_


from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.transport_solver_numerics import TransportSolverNumerics, TransportSolverNumericsTimeSlice
from fytok.modules.utilities import *

from fytok.utils.atoms import atoms
from fytok.utils.logger import logger

from .bvp import solve_bvp

EPSILON = 1.0e-32


def derivative_(y: array_type, x: array_type, dc_index=None):
    res = derivative(y, x)

    if dc_index is not None:
        res[dc_index - 1 : dc_index + 1] = 0.5 * (res[dc_index - 1] + res[dc_index + 1])
        # res = np.zeros_like(x)
        # # res[:dc_index] = InterpolatedUnivariateSpline(x[:dc_index], y[:dc_index], ext=0).derivative()(x[:dc_index])
        # # res[dc_index:] = InterpolatedUnivariateSpline(x[dc_index:], y[dc_index:], ext=0).derivative()(x[dc_index:])
        # res[:dc_index] = (y[1 : dc_index + 1] - y[:dc_index]) / (x[1 : dc_index + 1] - x[:dc_index])
        # res[dc_index:-1] = (y[dc_index + 1 :] - y[dc_index:-1]) / (x[dc_index + 1 :] - x[dc_index:-1])
        # res[-1] = res[-2]

    return res


@sp_tree
class FyTrans(TransportSolverNumerics):
    r"""
    Solve transport equations $\rho=\sqrt{ \Phi/\pi B_{0}}$
    See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`
    
        Solve transport equations :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
        See  :cite:`hinton_theory_1976,coster_european_2010,pereverzev_astraautomated_1991`

            Solve transport equations

            Current Equation

            Args:
                core_profiles       : profiles at :math:`t-1`
                equilibrium         : Equilibrium
                transports          : CoreTransport
                sources             : CoreSources
                boundary_condition  :

            Note:
                .. math ::  \sigma_{\parallel}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho} \right) \psi= \
                            \frac{F^{2}}{\mu_{0}B_{0}\rho}\frac{\partial}{\partial\rho}\left[\frac{V^{\prime}}{4\pi^{2}}\left\langle \left|\frac{\nabla\rho}{R}\right|^{2}\right\rangle \
                            \frac{1}{F}\frac{\partial\psi}{\partial\rho}\right]-\frac{V^{\prime}}{2\pi\rho}\left(j_{ni,exp}+j_{ni,imp}\psi\right)
                    :label: transport_current


                if :math:`\psi` is not solved, then

                ..  math ::  \psi =\int_{0}^{\rho}\frac{2\pi B_{0}}{q}\rho d\rho

            Particle Transport
            Note:

                .. math::
                    \left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                    \left(V^{\prime}n_{s}\right)+\frac{\partial}{\partial\rho}\Gamma_{s}=\
                    V^{\prime}\left(S_{s,exp}-S_{s,imp}\cdot n_{s}\right)
                    :label: particle_density_transport

                .. math::
                    \Gamma_{s}\equiv-D_{s}\cdot\frac{\partial n_{s}}{\partial\rho}+v_{s}^{pinch}\cdot n_{s}
                    :label: particle_density_gamma

            Heat transport equations

            Note:

                ion

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{i}T_{i}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{i}+T_{i}\gamma_{i}\right)=\
                            V^{\prime\frac{5}{3}}\left[Q_{i,exp}-Q_{i,imp}\cdot T_{i}+Q_{ei}+Q_{zi}+Q_{\gamma i}\right]
                    :label: transport_ion_temperature

                electron

                .. math:: \frac{3}{2}\left(\frac{\partial}{\partial t}-\frac{\dot{B}_{0}}{2B_{0}}\frac{\partial}{\partial\rho}\rho\right)\
                            \left(n_{e}T_{e}V^{\prime\frac{5}{3}}\right)+V^{\prime\frac{2}{3}}\frac{\partial}{\partial\rho}\left(q_{e}+T_{e}\gamma_{e}\right)=
                            V^{\prime\frac{5}{3}}\left[Q_{e,exp}-Q_{e,imp}\cdot T_{e}+Q_{ei}-Q_{\gamma i}\right]
                    :label: transport_electron_temperature
        """

    code: Code = {"name": "fy_trans", "copyright": "FyTok"}

    solver: str = "fy_trans_bvp_solver"

    primary_coordinate: str = "rho_tor_norm"

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        enable_momentum = self.code.parameters.enable_momentum or False
        enable_impurity = self.code.parameters.enable_impurity or False

        profiles_1d = self.profiles_1d

        ######################################################################################
        # 确定待求未知量

        unknowns = self.code.parameters.unknowns or list()

        # 极向磁通
        unknowns.append("psi_norm")

        # 电子
        # - 电子密度由准中性条件给出
        # - 电子温度，求解
        # - 电子转动，跟随离子，不求解
        unknowns.append("electrons/temperature")

        for s in self.ion_thermal:
            # 热化离子组份
            # - 密度 density,可求解，
            # - 温度 temperature,可求解，
            # - 环向转动 velocity/totoridal，可求解，

            unknowns.append(f"ion/{s}/density")
            unknowns.append(f"ion/{s}/temperature")
            if enable_momentum:
                unknowns.append(f"ion/{s}/velocity/toroidal")

        for s in self.ion_non_thermal:
            # 非热化离子组份，
            # - 密度 density ，可求解
            # - 温度 temperature, 无统一定义不求解，
            #    - He ash 温度与离子温度一致，alpha粒子满足慢化分布
            # - 环向转动 velocity/totoridal，无统一定义不求解
            #
            unknowns.append(f"ion/{s}/density")

        # if enable_impurity:
        #     # enable 杂质输运
        #     for s in self.impurities:
        #         unknowns.append(f"ion/{s}/density")
        #         unknowns.append(f"ion/{s}/temperature")
        #         if enable_momentum:
        #             unknowns.append(f"ion/{s}/velocity/toroidal")

        ######################################################################################
        # 声明主磁面坐标 primary_coordinate
        # 默认为 x=\bar{\rho}_{tor}=\sqrt{\frac{\Phi}{\Phi_{bdry}}}
        # \rho_{tor}= \sqrt{\frac{\Phi}{B_0 \pi}}

        x = Variable((i := 0), self.primary_coordinate)

        if self.primary_coordinate == "rho_tor_norm":
            x._metadata["label"] = r"\bar{\rho}_{tor}"

        ######################################################################################
        # 声明  variables 和 equations
        # profiles_1d = {self.primary_coordinate: x}
        # equations: typing.List[typing.Dict[str, typing.Any]] = []

        # 在 x=0 处边界条件唯一， flux=0 (n,T,u) or \farc{d \psi}{dx}=0 ( for psi )
        # 在 \rho_{bdry} 处边界条件类型可由参数指定
        bc_type = self.get_cache("boundary_condition_type", {})

        # 归一化/无量纲化单位
        # 在放入标准求解器前，系数矩阵需要无量纲、归一化
        units = self.code.parameters.units
        if units is _not_found_:
            units = {}

        else:
            units = units.__value__

        profiles_1d = self.profiles_1d

        profiles_1d[self.primary_coordinate] = x

        for s in unknowns:
            pth = Path(s)

            if pth[0] == "psi":
                label_p = r"\psi"
                label_f = r"\Psi"
                bc = bc_type.get(s, 1)

            if pth[0] == "psi_norm":
                label_p = r"\bar{\psi}"
                label_f = r"\bar{\Psi}"
                bc = bc_type.get(s, 1)

            if pth[-1] == "density":
                label_p = "n"
                label_f = r"\Gamma"
                bc = bc_type.get(s, None) or bc_type.get(f"*/density", 1)

            if pth[-1] == "temperature":
                label_p = "T"
                label_f = "H"
                bc = bc_type.get(s, None) or bc_type.get(f"*/temperature", 1)

            if pth[-1] == "toroidal":
                label_p = "u"
                label_f = r"\Phi"
                bc = bc_type.get(s, None) or bc_type.get(f"*/velocity/toroidal", 1)

            if pth[0] == "electrons":
                label_p += "_{e}"
                label_f += "_{e}"

            if pth[0] == "ion":
                label_p += f"_{{{pth[1]}}}"
                label_f += f"_{{{pth[1]}}}"

            profiles_1d[s] = Variable((i := i + 1), s, label=label_p)

            profiles_1d[f"{s}_flux"] = Variable((i := i + 1), f"{s}_flux", label=label_f)

            unit_profile = units.get(s, None) or units.get(f"*/{pth[-1]}", 1)

            unit_flux = units.get(f"{s}_flux", None) or units.get(f"*/{pth[-1]}_flux", 1)

            self.equations.append(
                {
                    "@name": s,
                    "identifier": s,
                    "units": (unit_profile, unit_flux),
                    "boundary_condition_type": bc,
                }
            )

        ###################################################################################################
        # 赋值属性
        # self.profiles_1d.update(profiles_1d)
        # self.equations = equations
        ##################################################################################################
        # 定义内部控制参数

        self._hyper_diff = self.code.parameters.hyper_diff or 0.001
        self._dc_pos = self.code.parameters.dc_pos or None

        # logger.debug([equ.identifier for equ in self.equations])

    def preprocess(self, *args, boundary_value=None, **kwargs) -> TransportSolverNumericsTimeSlice:
        """准备迭代求解
        - 方程 from self.equations
        - 初值 from initial_value
        - 边界值 from boundary_value
        """
        current: TransportSolverNumericsTimeSlice = super().preprocess(*args, **kwargs)

        profiles = self.profiles_1d

        rho_tor_norm = profiles.rho_tor_norm

        psi_norm = profiles.psi_norm

        eq0: Equilibrium.TimeSlice = self.inports["equilibrium/time_slice/0"].fetch()

        eq1: Equilibrium.TimeSlice = self.inports["equilibrium/time_slice/-1"].fetch()

        core0_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

        core1_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/-1/profiles_1d"].fetch()

        ni = sum([ion.z * ion.get("density", zero) for ion in profiles.ion], zero)

        ni_flux = sum([ion.z * ion.get("density_flux", zero) for ion in profiles.ion], zero)

        # 杂质密度
        # for label in self.impurities:
        #     logger.debug(label)
        #     imp = core0_1d.ion[label]
        #     ni += imp.z * imp.density(rho_tor_norm)
        # ni_flux += core0_1d.ion[label].density_flux(rho_tor_norm)

        impurity_fraction = kwargs.get("impurity_fraction", 0.0)

        profiles.electrons["density"] = ni / (1.0 - impurity_fraction)

        profiles.electrons["density_flux"] = ni_flux / (1.0 - impurity_fraction)

        tranport: AoS[CoreTransport.Model.TimeSlice] = self.inports["core_transport/model/*"].fetch(profiles)

        sources: AoS[CoreSources.Source.TimeSlice] = self.inports["core_sources/source/*"].fetch(profiles)

        eq0_1d = eq0.profiles_1d

        # if psi_norm is _not_found_:
        #     # psi_norm = profiles.psi / (eq0_1d.grid.psi_boundary - eq0_1d.grid.psi_axis)
        #     psi_norm = Function(
        #         eq0_1d.grid.rho_tor_norm,
        #         eq0_1d.grid.psi_norm,
        #         name="psi_norm",
        #         label=r"\bar{\psi}",
        #     )(rho_tor_norm)

        # 设定全局参数
        # $R_0$ characteristic major radius of the device   [m]
        R0 = eq0.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = eq0.vacuum_toroidal_field.b0

        rho_tor_boundary = Scalar(current.grid.rho_tor_boundary)

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr = eq0_1d.dvolume_drho_tor(psi_norm)

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq0_1d.f(psi_norm)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq0_1d.q(psi_norm)

        gm1 = eq0_1d.gm1(psi_norm)  # <1/R^2>
        gm2 = eq0_1d.gm2(psi_norm)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq0_1d.gm3(psi_norm)  # <|grad_rho_tor|^2>
        gm8 = eq0_1d.gm8(psi_norm)  # <R>

        if eq1 is _not_found_ or eq1 is None:
            one_over_dt = 0
            B1 = B0
            rho_tor_boundary_m = rho_tor_boundary
            vpr_m = vpr
            gm8_m = gm8
        else:
            dt = eq0.time - eq1.time

            if dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            elif np.isclose(dt, 0.0):
                one_over_dt = 0.0
            else:
                one_over_dt = one / dt

            B1 = eq1.vacuum_toroidal_field.b0
            rho_tor_boundary_m = eq1.profiles_1d.grid.rho_tor_boundary
            vpr_m = eq1.profiles_1d.dvolume_drho_tor(psi_norm)
            gm8_m = eq1.profiles_1d.gm8(psi_norm)

        k_B = (B0 - B1) / (B0 + B1) * one_over_dt

        k_rho_bdry = (rho_tor_boundary - rho_tor_boundary_m) / (rho_tor_boundary + rho_tor_boundary_m) * one_over_dt

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * rho_tor_norm

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        self._units = np.array(sum([equ.units for equ in self.equations], tuple()))

        X = current.grid.rho_tor_norm
        Y = np.zeros([len(self.equations) * 2, X.size])

        if boundary_value is None:
            boundary_value = {}

        if (initial_value := kwargs.get("initial_value", _not_found_)) is not _not_found_:
            for idx, equ in enumerate(self.equations):
                value = initial_value.get(equ.identifier, 0)
                Y[idx * 2] = value(X) if isinstance(value, Expression) else np.full_like(X, value)

        hyper_diff = self._hyper_diff

        for idx, equ in enumerate(self.equations):
            identifier = as_path(equ.identifier)
            path = identifier.parent
            quantity = identifier[-1]

            if quantity == "toroidal":
                quantity = "velocity/toroidal"
                path = path.parent

            bc_value = boundary_value.get(equ.identifier, Y[idx * 2][-1])

            match quantity:
                case "psi":
                    psi = identifier.get(profiles, zero)

                    psi_m = identifier.get(core1_1d, zero)(rho_tor_norm)

                    conductivity_parallel: Expression = zero

                    j_parallel: Expression = zero

                    for source in sources:
                        source_1d = source.profiles_1d
                        conductivity_parallel += as_path("conductivity_parallel").get(source_1d, zero)
                        j_parallel += as_path("j_parallel").get(source_1d, zero)

                    c = fpol2 / (scipy.constants.mu_0 * B0 * rho_tor * (rho_tor_boundary))

                    d_dt = one_over_dt * conductivity_parallel * (psi - psi_m) / c

                    D = vpr * gm2 / (fpol * rho_tor_boundary * 2.0 * scipy.constants.pi)

                    V = -k_phi * rho_tor_norm * conductivity_parallel

                    S = (
                        -vpr * (j_parallel) / (2.0 * scipy.constants.pi * rho_tor)
                        - k_phi
                        * conductivity_parallel
                        * (2 - 2 * rho_tor_norm * fpol.dln + rho_tor_norm * conductivity_parallel.dln)
                        * psi
                    ) / c

                    if bc_value is None:
                        bc_value = current.grid.psi_boundary

                    # at axis x=0 , dpsi_dx=0
                    bc = [[0, 1, 0]]

                    if bc_value is None:
                        assert equ.boundary_condition_type == 1
                        bc_value = current.grid.psi_boundary

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        # poloidal flux;
                        case 1:
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        # ip, total current inside x=1
                        case 2:
                            Ip = bc_value
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / fpol

                        # loop voltage;
                        case 3:
                            Uloop_bdry = bc_value
                            u = 0
                            v = 1
                            w = (dt * Uloop_bdry + psi_m) * (D - hyper_diff)

                        #  generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case "psi_norm":
                    dpsi = current.grid.psi_boundary - current.grid.psi_axis

                    psi_norm = profiles.psi_norm

                    psi_norm_m = identifier.get(core1_1d, zero)(rho_tor_norm)

                    conductivity_parallel: Expression = zero

                    j_parallel: Expression = zero

                    for source in sources:
                        source_1d = source.profiles_1d
                        conductivity_parallel += as_path("conductivity_parallel").get(source_1d, zero)
                        j_parallel += as_path("j_parallel").get(source_1d, zero)

                    c = fpol2 / (scipy.constants.mu_0 * B0 * rho_tor * (rho_tor_boundary))

                    d_dt = one_over_dt * conductivity_parallel * (psi_norm - psi_norm_m) / c

                    D = vpr * gm2 / (fpol * rho_tor_boundary * 2.0 * scipy.constants.pi)

                    V = -k_phi * rho_tor_norm * conductivity_parallel

                    S = (
                        (
                            -vpr * (j_parallel) / (2.0 * scipy.constants.pi * rho_tor)
                            - k_phi
                            * conductivity_parallel
                            * (2 - 2 * rho_tor_norm * fpol.dln + rho_tor_norm * conductivity_parallel.dln)
                            * psi_norm
                        )
                        / c
                        / dpsi
                    )

                    if bc_value is None:
                        bc_value = current.grid.psi_norm[-1]

                    # at axis x=0 , dpsi_dx=0
                    bc = [[0, 1, 0]]

                    if bc_value is None:
                        assert equ.boundary_condition_type == 1
                        bc_value = current.grid.psi_boundary

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        # poloidal flux;
                        case 1:
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        # ip, total current inside x=1
                        case 2:
                            Ip = bc_value
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / fpol

                        # loop voltage;
                        case 3:
                            Uloop_bdry = bc_value
                            u = 0
                            v = 1
                            w = (dt * Uloop_bdry + psi_m) * (D - hyper_diff)

                        #  generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case "density":
                    ns = (path / "density").get(profiles, zero)
                    ns_m = (path / "density").get(core1_1d, zero)(rho_tor_norm)

                    transp_D = zero
                    transp_V = zero

                    for transp in tranport:
                        transp_1d = transp.profiles_1d

                        transp_D += (path / "particles/d").get(transp_1d, zero)
                        transp_V += (path / "particles/v").get(transp_1d, zero)
                        # transp_F += (pth / "particles/flux").get(core_transp_1d, zero)

                    S = zero

                    for source in sources:
                        source_1d = source.profiles_1d
                        S += (path / "particles").get(source_1d, zero)

                    d_dt = one_over_dt * (vpr * ns - vpr_m * ns_m) * rho_tor_boundary

                    D = vpr * gm3 * transp_D / rho_tor_boundary  #

                    V = vpr * gm3 * (transp_V - rho_tor * k_phi)

                    S = vpr * (S - k_phi * ns) * rho_tor_boundary

                    # at axis x=0 , flux=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value
                        # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case "temperature":
                    ns = (path / "density").get(profiles, zero)
                    Gs = (path / "density_flux").get(profiles, zero)
                    Ts = (path / "temperature").get(profiles, zero)

                    ns_m = (path / "density").get(core1_1d, zero)(profiles)
                    Ts_m = (path / "temperature").get(core1_1d, zero)(profiles)

                    energy_D = zero
                    energy_V = zero
                    energy_F = zero

                    flux_multiplier = zero

                    for transp in tranport:
                        transp_1d = transp.profiles_1d
                        flux_multiplier += transp_1d._parent.flux_multiplier

                        energy_D += (path / "energy/d").get(transp_1d, zero)
                        energy_V += (path / "energy/v").get(transp_1d, zero)
                        energy_F += (path / "energy/flux").get(transp_1d, zero)

                    if flux_multiplier is zero:
                        flux_multiplier = one

                    Q = zero

                    for source in sources:
                        source_1d = source.profiles_1d
                        Q += (path / "energy").get(source_1d, zero)

                    d_dt = (
                        one_over_dt
                        * (3 / 2)
                        * (vpr * ns * Ts - (vpr_m ** (5 / 3)) * ns_m * Ts_m * inv_vpr23)
                        * rho_tor_boundary
                    )

                    D = vpr * gm3 * ns * energy_D / rho_tor_boundary

                    V = vpr * gm3 * ns * energy_V + Gs * flux_multiplier - (3 / 2) * k_phi * vpr * rho_tor * ns

                    S = vpr * (Q - k_vppr * ns * Ts) * rho_tor_boundary

                    # at axis x=0, dH_dx=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1] / equ.units[0]
                            v = 0
                            w = bc_value * equ.units[1] / equ.units[0]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value

                        case _:  # 5: generic boundary condition y expressed as a1y'+a2y=a3.
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case "velocity/toroidal":
                    us = (path / "velocity/toroidal").get(profiles, zero)
                    ns = (path / "density").get(profiles, zero)
                    Gs = (path / "density_flux").get(profiles, zero)

                    us_m = (path / "velocity/toroidal").get(core1_1d, zero)(rho_tor_norm)
                    ns_m = (path / "density").get(core1_1d, zero)(rho_tor_norm)

                    chi_u = zero
                    V_u_pinch = zero

                    for transp in tranport:
                        transp_1d = transp.profiles_1d

                        chi_u += (path / "momentum/toroidal/d").get(transp_1d, zero)
                        V_u_pinch += (path / "momentum/toroidal/v").get(transp_1d, zero)

                    U = zero

                    for source in sources:
                        source_1d = source.profiles_1d
                        U += (identifier / "../../momentum/toroidal").get(source_1d, zero)

                    U *= gm8

                    ms = identifier.get(atoms).a

                    d_dt = one_over_dt * ms * (vpr * gm8 * ns * us - vpr_m * gm8_m * ns_m * us_m) * rho_tor_boundary

                    D = vpr * gm3 * gm8 * ms * ns * chi_u / rho_tor_boundary

                    V = (vpr * gm3 * ns * V_u_pinch + Gs - k_phi * vpr * rho_tor * ns) * gm8 * ms

                    S = vpr * (U - k_rho_bdry * ms * ns * us) * rho_tor_boundary

                    # at axis x=0, du_dx=0
                    bc = [[0, 1, 0]]

                    # at boundary x=1
                    match equ.boundary_condition_type:
                        case 1:  # 1: value of the field y;
                            u = equ.units[1]
                            v = 0
                            w = bc_value * equ.units[1]

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V - (D - hyper_diff) / L
                            v = 1.0
                            w = 0
                        case 4:  # 4: flux;
                            u = 0
                            v = 1
                            w = bc_value

                        # 5: generic boundary condition y expressed as u y + v y'=w.
                        case _:
                            if not isinstance(bc_value, (tuple, list)) or len(bc_value) != 3:
                                raise NotImplementedError(f"5: generic boundary condition y expressed as a1y'+a2y=a3.")
                            u, v, w = bc_value

                    bc += [[u, v, w]]

                case _:
                    raise RuntimeError(f"Unknown equation of {equ.identifier}!")

            equ["coefficient"] = [d_dt, D, V, S]

            equ["boundary_condition_value"] = bc

        if (initial_value := kwargs.get("initial_value", _not_found_)) is not _not_found_:
            for idx, equ in enumerate(self.equations):
                d_dt, D, V, S = equ.coefficient
                y = Y[idx * 2]
                yp = derivative(y, X)
                Y[idx * 2 + 1] = -D(X, *Y) * yp + V(X, *Y) * y

        current.X = X
        current.Y = Y / self._units.reshape(-1, 1)

        return current

    def func(self, X: array_type, _Y: array_type, *args) -> array_type:
        dY = np.zeros([len(self.equations) * 2, X.size])

        hyper_diff = self._hyper_diff

        # 添加量纲和归一化系数，复原为物理量
        Y = _Y * self._units.reshape(-1, 1)

        for idx, equ in enumerate(self.equations):
            y = Y[idx * 2]
            flux = Y[idx * 2 + 1]

            _d_dt, _D, _V, _S = equ.coefficient

            try:
                d_dt = _d_dt(X, *Y, *args) if isinstance(_d_dt, Expression) else _d_dt
                D = _D(X, *Y, *args) if isinstance(_D, Expression) else _D
                V = _V(X, *Y, *args) if isinstance(_V, Expression) else _V
                S = _S(X, *Y, *args) if isinstance(_S, Expression) else _S
            except RuntimeError as error:
                raise RuntimeError(f"Error when calcuate {equ.identifier} {_S}") from error

            # yp = np.zeros_like(X)
            # yp[:-1] += 0.5 * ((y[1:] - y[:-1]) / (X[1:] - X[:-1]))  # derivative(flux, X)
            # yp[1:] += yp[:-1]
            # yp[0] = 0
            # yp[-1] *= 2
            yp = derivative(y, X)
            d_dr = (-flux + V * y + hyper_diff * yp) / (D + hyper_diff)

            # fluxp = np.zeros_like(X)
            # fluxp[:-1] = 0.5 * (flux[1:] - flux[:-1]) / (X[1:] - X[:-1])
            # fluxp[1:] += fluxp[:-1]
            # fluxp[0] = 0
            # flux[-1] *= 2

            fluxp = derivative(flux, X)
            dflux_dr = (S - d_dt + hyper_diff * fluxp) / (1.0 + hyper_diff)

            # if equ.identifier == "ion/alpha/density":
            #     dflux_dr[-1] = dflux_dr[-2]
            # if np.any(np.isnan(dflux_dr)):
            #     logger.exception(f"Error: {equ.identifier} nan in dflux_dr {_R._render_latex_()} {dflux_dr}")

            # 无量纲，归一化
            dY[idx * 2] = d_dr
            dY[idx * 2 + 1] = dflux_dr
            if equ.identifier in ["ion/alpha/density", "ion/He/density"]:
                #     dY[idx * 2, 0] = 0
                dY[idx * 2 + 1, -1] = 0

        dY /= self._units.reshape(-1, 1)

        return dY

    def bc(self, ya: array_type, yb: array_type, *args) -> array_type:
        x0, x1 = self.bc_pos

        bc = []

        ya = ya * self._units
        yb = yb * self._units
        for idx, equ in enumerate(self.equations):
            [u0, v0, w0], [u1, v1, w1] = equ.boundary_condition_value

            try:
                u0 = u0(x0, *ya, *args) if isinstance(u0, Expression) else u0
                v0 = v0(x0, *ya, *args) if isinstance(v0, Expression) else v0
                w0 = w0(x0, *ya, *args) if isinstance(w0, Expression) else w0
                u1 = u1(x1, *yb, *args) if isinstance(u1, Expression) else u1
                v1 = v1(x1, *yb, *args) if isinstance(v1, Expression) else v1
                w1 = w1(x1, *yb, *args) if isinstance(w1, Expression) else w1
            except Exception as error:
                logger.error(((u0, v0, w0), (u1, v1, w1)), exc_info=error)
                # raise RuntimeError(f"Boundary error of equation {equ.identifier}  ") from error

            y0 = ya[2 * idx]
            flux0 = ya[2 * idx + 1]

            y1 = yb[2 * idx]
            flux1 = yb[2 * idx + 1]

            # NOTE: 边界值量纲为 flux 通量，以 equ.units[1] 归一化
            bc.extend([(u0 * y0 + v0 * flux0 - w0) / equ.units[1], (u1 * y1 + v1 * flux1 - w1) / equ.units[1]])

        bc = np.array(bc)
        return bc

    def execute(
        self, current: TransportSolverNumericsTimeSlice, *previous: TransportSolverNumericsTimeSlice
    ) -> TransportSolverNumericsTimeSlice:
        current = super().execute(current, *previous)

        X = current.X
        Y = current.Y

        self.bc_pos = (X[0], X[-1])
        # 设定初值
        if Y is None:
            Y = np.zeros([len(self.equations) * 2, len(X)])

            for idx, equ in enumerate(self.equations):
                Y[2 * idx + 0] = (
                    equ.profile(X)
                    if isinstance(equ.profile, Expression)
                    else np.full_like(X, equ.profile if equ.profile is not _not_found_ else 0)
                )
                Y[2 * idx + 1] = (
                    equ.flux(X)
                    if isinstance(equ.flux, Expression)
                    else np.full_like(X, equ.flux if equ.flux is not _not_found_ else 0)
                )

        sol = solve_bvp(
            self.func,
            self.bc,
            X,
            Y,
            discontinuity=self.code.parameters.discontinuity or [],
            tol=self.code.parameters.tolerance or 1.0e-3,
            bc_tol=self.code.parameters.bc_tol or 1e6,
            max_nodes=self.code.parameters.max_nodes or 1000,
            verbose=self.code.parameters.verbose or 0,
        )

        current.X = sol.x
        current.Y = sol.y
        current.Yp = sol.yp
        current.rms_residuals = sol.rms_residuals

        logger.info(
            f"Solving the transport equation [{ 'success' if sol.success else 'failed'}]: {sol.message} , {sol.niter} iterations"
        )

        return current

    def postprocess(self, current: TransportSolverNumericsTimeSlice) -> TransportSolverNumericsTimeSlice:
        current = super().postprocess(current)

        X = current.X
        Y = current.Y * self._units.reshape(-1, 1)
        Yp = current.Yp * self._units.reshape(-1, 1)

        current["grid"] = CoreRadialGrid(
            {
                "rho_tor_norm": X,
                "psi_norm": Y[0],
                "psi_axis": current.grid.psi_axis,
                "psi_boundary": current.grid.psi_boundary,
                "rho_tor_boundary": current.grid.rho_tor_boundary,
            }
        )

        current["equations"] = []

        for idx, equ in enumerate(self.equations):
            d_dt, D, V, R = equ.coefficient

            current.equations.append(
                {
                    "@name": equ.identifier,
                    "boundary_condition_type": equ.boundary_condition_type,
                    "boundary_condition_value": equ.boundary_condition_value,
                    "profile": Y[2 * idx],
                    "flux": Y[2 * idx + 1],
                    "coefficient": [d_dt(X, *Y), D(X, *Y), V(X, *Y), R(X, *Y)],
                    "d_dr": Yp[2 * idx],
                    "dflux_dr": Yp[2 * idx + 1],
                }
            )

        return current


TransportSolverNumerics.register(["fy_trans"], FyTrans)
