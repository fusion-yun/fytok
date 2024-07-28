import typing
import numpy as np
import scipy.constants

from spdm.utils.type_hint import ArrayType
from spdm.utils.tags import _not_found_
from spdm.core.expression import Variable, Expression, one, zero
from spdm.core.path import as_path
from spdm.numlib.calculus import derivative
from fytok.utils.logger import logger
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.core_sources import CoreSources
from fytok.modules.transport_solver import TransportSolver
from fytok.modules.utilities import CoreRadialGrid


from .bvp import solve_bvp

EPSILON = 1.0e-32


def derivative_(y: ArrayType, x: ArrayType, dc_index=None):
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


class FyTrans(TransportSolver, code={"name": "fy_trans"}):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyper_diff = 0.001

    def initialize(self, *args, **kwargs):

        enable_momentum = self.code.parameters.enable_momentum or False
        enable_impurity = self.code.parameters.enable_impurity or False

        profiles_1d = self.profiles_1d

        ######################################################################################
        # 确定待求未知量

        unknowns = []

        if unknowns is _not_found_:
            unknowns = []

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
        bc_type = self.get("boundary_condition_type", {})

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
            pth = as_path(s)

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

        self._hyper_diff: float = self.code.parameters.hyper_diff or 0.001
        self._dc_pos = self.code.parameters.dc_pos or None

        # logger.debug([equ.identifier for equ in self.equations])

    def func(self, X: ArrayType, _Y: ArrayType, *args) -> ArrayType:
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

    def bc(self, ya: ArrayType, yb: ArrayType, *args) -> ArrayType:
        x0, x1 = self.bc_pos

        bc = []

        ya = ya * self._units
        yb = yb * self._units
        for idx, equ in enumerate(self.equations):
            [u0, v0, w0], [u1, v1, w1] = equ.boundary_condition_value

            try:
                u0 = u0(x0, *ya, *args)
                v0 = v0(x0, *ya, *args)
                w0 = w0(x0, *ya, *args)
                u1 = u1(x1, *yb, *args)
                v1 = v1(x1, *yb, *args)
                w1 = w1(x1, *yb, *args)
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
        self,
        *args,
        core_profiles: CoreProfiles,
        equilibrium: Equilibrium,
        core_transport: CoreTransport,
        core_sources: CoreSources,
        unknowns=None,
        impurity_fraction=0.0,
        boundary_value=None,
        **kwargs,
    ):
        """准备迭代求解
        - 方程 from self.equations
        - 初值 from initial_value
        - 边界值 from boundary_value
        """

        if boundary_value is None:
            boundary_value = {}

        core_profiles_in: CoreProfiles = core_profiles

        core_profiles_out = super().execute(
            *args,
            core_profiles=core_profiles,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
            **kwargs,
        )

        profiles_1d_in: CoreProfiles.Profiles1D = core_profiles_in.profiles_1d

        profiles_1d_out: CoreProfiles.Profiles1D = core_profiles_out.profiles_1d

        grid: CoreRadialGrid = equilibrium.profiles_1d.grid

        rho_tor_norm = grid.rho_tor_norm

        psi_norm = grid.psi_norm

        eq0_1d: Equilibrium.Profiles1D = equilibrium.profiles_1d

        eq_prev: Equilibrium = equilibrium.previous

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
        R0 = equilibrium.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium.vacuum_toroidal_field.b0

        rho_tor_boundary = grid.rho_tor_boundary

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

        if eq_prev is _not_found_ or eq_prev is None:
            one_over_dt = 0
            B0_prev = B0
            rho_tor_boundary_prev = rho_tor_boundary
            vpr_prev = vpr
            gm8_prev = gm8
            dt = 0
        else:
            dt = equilibrium.time - eq_prev.time

            if dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            elif np.isclose(dt, 0.0):
                one_over_dt = 0.0
            else:
                one_over_dt = one / dt

            B0_prev = eq_prev.vacuum_toroidal_field.b0
            rho_tor_boundary_prev = eq_prev.profiles_1d.grid.rho_tor_boundary
            vpr_prev = eq_prev.profiles_1d.dvolume_drho_tor(psi_norm)
            gm8_prev = eq_prev.profiles_1d.gm8(psi_norm)

        k_B = (B0 - B0_prev) / (B0 + B0_prev) * one_over_dt

        k_rho_bdry = (
            (rho_tor_boundary - rho_tor_boundary_prev) / (rho_tor_boundary + rho_tor_boundary_prev) * one_over_dt
        )

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * rho_tor_norm

        inv_vpr23 = vpr ** (-2 / 3)

        k_vppr = 0  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi).dln()

        self._units = np.array(sum([equ.units for equ in self.equations], tuple()))

        X, *_ = grid.coordinates

        Y = np.zeros([len(self.equations) * 2, X.size])

        # 准中性条件

        ni = sum(ion.z * ion.density for ion in profiles_1d_out.ion)

        ni_flux = sum(ion.z * ion.density_flux for ion in profiles_1d_out.ion)

        profiles_1d_out.electrons.density = ni / (1.0 - impurity_fraction)

        profiles_1d_out.electrons.density_flux = ni_flux / (1.0 - impurity_fraction)

        hyper_diff = 0.001  # self._hyper_diff

        primary_coordinate = self.primary_coordinate

        if unknowns is None:
            unknowns = (
                [core_profiles_in.profiles_1d.grid.primary_coordinate, "psi", "electrons/temperature"]
                + sum(
                    (
                        [f"{ion.label}/density", f"{ion.label}/tempetature"]
                        for ion in profiles_1d_in.ion
                        if ion.label != "alpha"
                    ),
                    [],
                )
                + ["ion/alpha/density"]
            )

        var_list: typing.List[Variable] = [Variable(idx, name) for idx, name in unknowns]

        core_profiles_var = CoreProfiles(
            {
                "profiles_1d": {
                    "grid": eq0_1d.grid.remesh(**{primary_coordinate: var_list[0]}),
                    "ion": [{"label": ion.label} for ion in profiles_1d_in.ion],
                }
            }
        )

        for var in var_list:
            core_profiles_var.profiles_1d.put(var.name, var)

        equations = []
        boundary_conditions = []
        boundary_condition = {}

        if True:  # "psi":
            psi = profiles_1d_out.get("psi", zero)

            psi_m = profiles_1d_in.get("psi", zero)(rho_tor_norm)

            conductivity_parallel = sum(
                (source.profiles_1d.get("conductivity_parallel", zero) for source in core_sources.source),
                zero,
            )

            j_parallel = sum((source.profiles_1d.get("j_parallel", zero) for source in core_sources.source), zero)

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
                bc_value = grid.psi_boundary

            # at axis x=0 , dpsi_dx=0
            bc = [[0, 1, 0]]

            if bc_value is None:
                assert equ.boundary_condition_type == 1
                bc_value = grid.psi_boundary

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
                        raise NotImplementedError("5: generic boundary condition y expressed as a1y'+a2y=a3.")
                    u, v, w = bc_value

            bc += [[u, v, w]]

        if True:  # "psi_norm":
            dpsi = grid.psi_boundary - grid.psi_axis

            psi_norm = profiles_1d_in.psi_norm(rho_tor_norm)

            if profiles_1d_in is not None:
                psi_norm_m = profiles_1d_in.get("psi_norm", zero)(rho_tor_norm)
            else:
                psi_norm_m = zero

            conductivity_parallel = sum(
                (source.profiles_1d.get("conductivity_parallel", zero) for source in core_sources.source),
                zero,
            )

            j_parallel = sum((source.profiles_1d.get("j_parallel", zero) for source in core_sources.source), zero)

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
                bc_value = grid.psi_norm[-1]

            # at axis x=0 , dpsi_dx=0
            bc = [[0, 1, 0]]

            if bc_value is None:
                assert equ.boundary_condition_type == 1
                bc_value = grid.psi_boundary

            # at boundary x=1
            match boundary_condition.get(equ, 1):
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
                        raise NotImplementedError("5: generic boundary condition y expressed as a1y'+a2y=a3.")
                    u, v, w = bc_value

            bc += [[u, v, w]]

        for ion in profiles_1d_in.ion:
            ion_label = f"ion/{ion.label}"
            ms = ion.a

            if True:  # density

                ns = profiles_1d_out.get(f"{ion_label}/density", zero)

                ns_m = profiles_1d_in.get(f"{ion_label}/density", zero)

                transp_D = sum(
                    (model.profiles_1d.get(f"{ion_label}/particles/d", zero) for model in core_transport.model), zero
                )

                transp_V = sum(
                    (model.profiles_1d.get(f"{ion_label}/particles/v", zero) for model in core_transport.model),
                    zero,
                )

                S = sum(
                    (source.profiles_1d.get(f"{ion_label}/particles", zero) for source in core_sources.source),
                    zero,
                )

                d_dt = one_over_dt * (vpr * ns - vpr_prev * ns_m) * rho_tor_boundary

                D = vpr * gm3 * transp_D / rho_tor_boundary

                V = vpr * gm3 * (transp_V - rho_tor * k_phi)

                S = vpr * (S - k_phi * ns) * rho_tor_boundary

                # at axis x=0 , flux=0
                bc = [[0, 1, 0]]

                # at boundary x=1
                match boundary_condition.get(f"{ion_label}/density", 1):
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
                            raise NotImplementedError("5: generic boundary condition y expressed as a1y'+a2y=a3.")
                        u, v, w = bc_value

                bc += [[u, v, w]]

            if True:  # "temperature":
                ns = profiles_1d_in.get(f"{ion_label}/density", zero)
                Gs = profiles_1d_in.get(f"{ion_label}/density_flux", zero)
                Ts = profiles_1d_in.get(f"{ion_label}/temperature", zero)

                if profiles_1d_in is not None:
                    ns_m = profiles_1d_in.get(f"{ion_label}/density", zero)
                    Ts_m = profiles_1d_in.get(f"{ion_label}/temperature", zero)
                else:
                    ns_m = zero
                    Ts_m = zero

                flux_multiplier = sum(
                    (model.get("flux_multiplier", 0) for model in core_transport.model),
                    0,
                )
                flux_multiplier = one

                energy_D = sum(
                    (model.profiles_1d.get(f"{ion_label}/energy/d", zero) for model in core_transport.model),
                    zero,
                )
                energy_V = sum(
                    (model.profiles_1d.get(f"{ion_label}/energy/v", zero) for model in core_transport.model),
                    zero,
                )

                Q = sum((source.profiles_1d.get(f"{ion_label}/energy", zero) for source in core_sources.source), zero)

                d_dt = (
                    one_over_dt
                    * (3 / 2)
                    * (vpr * ns * Ts - (vpr_prev ** (5 / 3)) * ns_m * Ts_m * inv_vpr23)
                    * rho_tor_boundary
                )

                D = vpr * gm3 * ns * energy_D / rho_tor_boundary

                V = vpr * gm3 * ns * energy_V + Gs * flux_multiplier - (3 / 2) * k_phi * vpr * rho_tor * ns

                S = vpr * (Q - k_vppr * ns * Ts) * rho_tor_boundary

                # at axis x=0, dH_dx=0
                bc = [[0, 1, 0]]

                # at boundary x=1
                match boundary_condition.get(equ, 1):
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
                            raise NotImplementedError("5: generic boundary condition y expressed as a1y'+a2y=a3.")
                        u, v, w = bc_value

                bc += [[u, v, w]]

            if True:  # "velocity/toroidal":
                us = profiles_1d_in.get(f"{ion_label}/velocity/toroidal", zero)
                ns = profiles_1d_in.get(f"{ion_label}/density", zero)
                Gs = profiles_1d_in.get(f"{ion_label}/density_flux", zero)

                if profiles_1d_in is not None:
                    us_m = profiles_1d_in.get(f"{ion_label}/velocity/toroidal", zero)
                    ns_m = profiles_1d_in.get(f"{ion_label}/density", zero)
                    Gs_m = profiles_1d_in.get(f"{ion_label}/density_flux", zero)
                else:
                    us_m = zero
                    ns_m = zero
                    Gs_m = zero

                chi_u = sum(
                    (
                        model.profiles_1d.get(f"{ion_label}/momentum/toroidal/d", zero)
                        for model in core_transport.model
                    ),
                    zero,
                )

                V_u_pinch = sum(
                    (
                        model.profiles_1d.get(f"{ion_label}/momentum/toroidal/v", zero)
                        for model in core_transport.model
                    ),
                    zero,
                )

                U = gm8 * sum(
                    (
                        source.profiles_1d.get(f"{ion_label}/momentum/toroidal", zero)
                        for source in core_sources.source
                    ),
                    zero,
                )

                d_dt = one_over_dt * ms * (vpr * gm8 * ns * us - vpr_prev * gm8_prev * ns_m * us_m) * rho_tor_boundary

                D = vpr * gm3 * gm8 * ms * ns * chi_u / rho_tor_boundary

                V = (vpr * gm3 * ns * V_u_pinch + Gs - k_phi * vpr * rho_tor * ns) * gm8 * ms

                S = vpr * (U - k_rho_bdry * ms * ns * us) * rho_tor_boundary

                # at axis x=0, du_dx=0
                bc = [[0, 1, 0]]

                # at boundary x=1
                match boundary_condition.get(f"{ion_label}/moment", 1):
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
                            raise NotImplementedError("5: generic boundary condition y expressed as a1y'+a2y=a3.")
                        u, v, w = bc_value

                bc += [[u, v, w]]

        for idx, equ in enumerate(self.equations):
            d_dt, D, V, S = equ.coefficient
            y = Y[idx * 2]
            yp = derivative(y, X)
            Y[idx * 2 + 1] = -D(X, *Y) * yp + V(X, *Y) * y

        Y = Y / self._units.reshape(-1, 1)

        # 设定边界值
        xa = X[0]
        xb = X[-1]

        # 设定初值
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
            lambda X, Y, *_args: np.stack([equ(X, *Y, *_args) for equ in equations]),
            lambda ya, yb, *_args: np.array(sum([bc(xa, ya, xb, yb, *_args) for bc in boundary_conditions], [])),
            X,
            Y,
            discontinuity=[],  # self.code.parameters.discontinuity or
            tol=self.code.parameters.tolerance if isinstance(self.code.parameters.tolerance, float) else 1.0e-3,
            bc_tol=self.code.parameters.bc_tol if isinstance(self.code.parameters.bc_tol, float) else 1e6,
            max_nodes=1000,
            verbose=0,
        )

        X = sol.X

        profiles_1d_out[grid.primary_coordinate] = X

        profiles_1d_out["grid"] = grid.remesh(**{self.primary_coordinate: X})

        Y = sol.Y * self._units.reshape(-1, 1)
        Yp = sol.Yp * self._units.reshape(-1, 1)

        for idx, equ in enumerate(self.equations):
            profiles_1d_out[f"{equ.identifier}"] = Y[2 * idx]
            profiles_1d_out[f"{equ.identifier}_flux"] = Y[2 * idx + 1]

        profiles_1d_out["/rms_residuals"] = sol.rms_residuals

        logger.info(
            f"Solving the transport equation [{ 'success' if sol.success else 'failed'}]: {sol.message} , {sol.niter} iterations"
        )

        for idx, equ in enumerate(self.equations):
            d_dt, D, V, R = equ.coefficient

            self.equations.append(
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

        return output
