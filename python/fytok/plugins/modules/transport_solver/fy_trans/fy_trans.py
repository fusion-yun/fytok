import typing
import numpy as np
import scipy.constants

from spdm.utils.tags import _not_found_
from spdm.core.expression import Variable, Expression, one, zero
from spdm.core.path import as_path
from spdm.numlib.calculus import derivative

from fytok.utils.envs import FY_VERSION, FY_COPYRIGHT
from fytok.utils.logger import logger
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.core_sources import CoreSources
from fytok.modules.transport_solver import TransportSolver
from fytok.modules.utilities import CoreRadialGrid


from .bvp import solve_bvp

EPSILON = 1.0e-32


class FyTrans(TransportSolver, code={"name": "fy_trans", "version": FY_VERSION, "copyright": FY_COPYRIGHT}):
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

    def execute(
        self,
        *args,
        core_profiles_prev: CoreProfiles,
        equilibrium_prev: Equilibrium,
        equilibrium_next: Equilibrium,
        core_transport: CoreTransport,
        core_sources: CoreSources,
        **kwargs,
    ):
        """准备迭代求解
        - 方程 from self.equations
        - 初值 from initial_value
        - 边界值 from boundary_value
        """
        enable_momentum: bool = self.code.parameters.get("enable_momentum", False)
        impurity_fraction: float = self.code.parameters.get("impurity_fraction", 0.0)
        normalize_factor: dict = self.code.parameters.get("normalize_factor", {})
        boundary_condition_cfg: dict = self.code.parameters.get("boundary_condition", {})
        hyper_diff: float = self.code.parameters.get("hyper_diff", 0.001)

        # 声明未知数变量
        var_list: typing.List[Variable] = []

        # 主坐标
        var_list.append(rho_tor_norm_v := Variable(len(var_list), "rho_tor_norm", label=r"\bar{\rho}_{tor}"))

        # 极向磁通

        var_list.append(psi_norm_v := Variable(len(var_list), "psi_norm", label=r"\bar{\psi}"))

        # 离子
        ni = zero
        ni_flux = zero

        Ui = zero
        Ui_flux = zero

        for ion in core_profiles_prev.profiles_1d.ion:

            var_list.append(
                ns_next := Variable(
                    len(var_list),
                    f"ion/{ion.label}/density",
                    label=rf"n_{{{ion.label}}}",
                )
            )
            var_list.append(
                ns_flux := Variable(
                    len(var_list),
                    f"ion/{ion.label}/density_flux",
                    label=rf"\Gamma_{{{ion.label}}}",
                )
            )
            ni = ni + ion.z * ns_next
            ni_flux = ni_flux + ion.z * ns_flux

            if ion.is_thermal:
                var_list.append(
                    Variable(
                        len(var_list),
                        f"ion/{ion.label}/temperature",
                        label=rf"T_{{{ion.label}}}",
                    )
                )
                var_list.append(
                    Variable(
                        len(var_list),
                        f"ion/{ion.label}/temperature_flux",
                        label=rf"G_{{{ion.label}}}",
                    )
                )

                if enable_momentum:
                    var_list.append(
                        Variable(
                            len(var_list),
                            f"ion/{ion.label}/velocity/toroidal",
                            label=rf"u_{{{ion.label}}}",
                        )
                    )
                    var_list.append(
                        Variable(
                            len(var_list),
                            f"ion/{ion.label}/velocity/toroidal_flux",
                            label=rf"U_{{{ion.label}}}",
                        )
                    )
                    Ui = Ui + ion.z * Us
                    Ui_flux = Ui_flux + ion.z * Us

        # 电子
        var_list.append(Variable(len(var_list), "electrons/temperature", label=r"T_e"))

        var_list.append(Variable(len(var_list), "electrons/temperature_flux", label=r"G_e"))

        profiles_v = {
            "electrons": {
                "density": ni / (1.0 - impurity_fraction),  # 准中性条件
                "density_flux": ni_flux / (1.0 - impurity_fraction),  # 准中性条件
            },
            "ion": [ion.label for ion in core_profiles_prev.profiles_1d.ion],
        }

        for var in var_list:
            as_path(var.name).update(profiles_v, var)

        core_profiles_iter = CoreProfiles({"profiles_1d": profiles_v})

        profiles_1d_iter: CoreProfiles.Profiles1D = core_profiles_iter.profiles_1d

        # 设定全局参数
        profiles_1d_prev: CoreProfiles.Profiles1D = core_profiles_prev.profiles_1d

        eq1d_next: Equilibrium.Profiles1D = equilibrium_next.profiles_1d

        # $R_0$ characteristic major radius of the device   [m]
        R0 = equilibrium_next.vacuum_toroidal_field.r0

        # $B_0$ magnetic field measured at $R_0$            [T]
        B0 = equilibrium_next.vacuum_toroidal_field.b0

        psi_axis = eq1d_next.grid.psi_axis

        psi_boundary = eq1d_next.grid.psi_boundary

        rho_tor_boundary = eq1d_next.grid.rho_tor_boundary

        rho_tor_norm_boundary = core_profiles_prev.profiles_1d.grid.rho_tor_norm[-1]

        psi_norm_boundary = core_profiles_prev.profiles_1d.grid.psi_norm[-1]

        # $\frac{\partial V}{\partial\rho}$ V',             [m^2]
        vpr_next = eq1d_next.dvolume_drho_tor(psi_norm_v)

        # diamagnetic function,$F=R B_\phi$                 [T*m]
        fpol = eq1d_next.f(psi_norm_v)

        fpol2 = fpol**2

        # $q$ safety factor                                 [-]
        qsf = eq1d_next.q(psi_norm_v)

        gm1 = eq1d_next.gm1(psi_norm_v)  # <1/R^2>
        gm2 = eq1d_next.gm2(psi_norm_v)  # <|grad_rho_tor|^2/R^2>
        gm3 = eq1d_next.gm3(psi_norm_v)  # <|grad_rho_tor|^2>
        gm8 = eq1d_next.gm8(psi_norm_v)  # <R>

        if equilibrium_prev is equilibrium_next:
            one_over_dt = 0
            B0_prev = B0
            rho_tor_boundary_prev = rho_tor_boundary
            vpr_prev = vpr_next
            gm8_prev = gm8
            dt = 0
        else:
            dt = equilibrium_next.time - equilibrium_prev.time

            if dt < 0:
                raise RuntimeError(f"dt={dt}<=0")
            elif np.isclose(dt, 0.0):
                one_over_dt = 0.0
            else:
                one_over_dt = one / dt

            B0_prev = equilibrium_prev.vacuum_toroidal_field.b0
            rho_tor_boundary_prev = equilibrium_prev.profiles_1d.grid.rho_tor_boundary
            vpr_prev = equilibrium_prev.profiles_1d.dvolume_drho_tor(psi_norm_v)
            gm8_prev = equilibrium_prev.profiles_1d.gm8(psi_norm_v)

        k_B = one_over_dt * ((B0 - B0_prev) / (B0 + B0_prev))

        k_rho_bdry = one_over_dt * (
            (rho_tor_boundary - rho_tor_boundary_prev) / (rho_tor_boundary + rho_tor_boundary_prev)
        )

        k_phi = k_B + k_rho_bdry

        rho_tor = rho_tor_boundary * rho_tor_norm_v

        inv_vpr23 = vpr_next ** (-2 / 3)

        k_vppr = zero  # (3 / 2) * k_rho_bdry - k_phi *　x * vpr(psi_norm_v).dln()

        equations: typing.List[Expression] = []

        boundary_conditions: typing.List[typing.Tuple[float, float, float]] = []

        x = var_list[0]

        for idx, var in enumerate(var_list[1::2]):
            bc = boundary_condition_cfg.get(var.name, None)

            var_path = var.name.split("/")

            prefix = "/".join(var_path[:-1])

            y = var

            flux = var_list[2 * idx + 2]

            match (var_path[-1]):

                case "psi":
                    psi = profiles_1d_iter.get("psi", zero)

                    psi_m = profiles_1d_prev.get("psi", zero)(rho_tor_norm_v)

                    conductivity_parallel = sum(
                        (source.profiles_1d.get("conductivity_parallel", zero) for source in core_sources.source),
                        zero,
                    )

                    j_parallel = sum(
                        (source.profiles_1d.get("j_parallel", zero) for source in core_sources.source), zero
                    )

                    c = fpol2 / (scipy.constants.mu_0 * B0 * rho_tor * (rho_tor_boundary))

                    dy_dt = one_over_dt * conductivity_parallel * (psi - psi_m) / c

                    D = vpr_next * gm2 / (fpol * rho_tor_boundary * 2.0 * scipy.constants.pi)

                    V = -k_phi * rho_tor_norm_v * conductivity_parallel

                    S = (
                        -vpr_next * (j_parallel) / (2.0 * scipy.constants.pi * rho_tor)
                        - k_phi
                        * conductivity_parallel
                        * (2 - 2 * rho_tor_norm_v * fpol.dln + rho_tor_norm_v * conductivity_parallel.dln)
                        * psi
                    ) / c

                    if bc_value is None:
                        bc_value = psi_boundary

                    # at axis x=0 , dpsi_dx=0
                    bc = [[0, 1, 0]]

                    if bc_value is None:
                        assert equ.boundary_condition_type == 1
                        bc_value = psi_boundary

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

                case "psi_norm":
                    dpsi = psi_boundary - psi_axis

                    psi_norm_m = profiles_1d_prev.psi_norm(rho_tor_norm_v)

                    conductivity_parallel = sum(
                        (source.profiles_1d.get("conductivity_parallel", zero) for source in core_sources.source),
                        zero,
                    )(rho_tor_norm_v)

                    j_parallel = sum(
                        (source.profiles_1d.get("j_parallel", zero) for source in core_sources.source), zero
                    )(rho_tor_norm_v)

                    c = fpol2 / (scipy.constants.mu_0 * B0 * rho_tor * (rho_tor_boundary))

                    dy_dt = one_over_dt * conductivity_parallel * (psi_norm_v - psi_norm_m) / c

                    D = vpr_next * gm2 / (fpol * rho_tor_boundary * 2.0 * scipy.constants.pi)

                    V = -k_phi * rho_tor_norm_v * conductivity_parallel

                    S = (
                        (
                            -vpr_next * (j_parallel) / (2.0 * scipy.constants.pi * rho_tor)
                            - k_phi
                            * conductivity_parallel
                            * (2 - 2 * rho_tor_norm_v * fpol.dln + rho_tor_norm_v * conductivity_parallel.dln)
                            * psi_norm_v
                        )
                        / c
                        / dpsi
                    )

                    # at axis x=0 , dpsi_dx=0
                    bc0 = (0, 1, 0)

                    # at boundary x=1
                    bc_type, bc_value = boundary_condition_cfg.get(var.name, (1, psi_boundary))
                    match bc_type:
                        # poloidal flux;
                        case 1:
                            u = 1.0
                            v = 0
                            w = bc_value

                        # ip, total current inside x=1
                        case 2:
                            Ip = bc_value
                            u = 0
                            v = 1
                            w = scipy.constants.mu_0 * Ip / equilibrium_next.profiles_1d.fpol(psi_norm_boundary[-1])

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

                    bc1 = (u, v, w)

                case "density":

                    ns_next = var

                    ns_prev = profiles_1d_prev.get(var.name, zero)(rho_tor_norm_v)

                    transp_D = sum(
                        (model.profiles_1d.get(f"{prefix}/particles/d", zero) for model in core_transport.model),
                        zero,
                    )(rho_tor_norm_v)

                    transp_V = sum(
                        (model.profiles_1d.get(f"{prefix}/particles/v", zero) for model in core_transport.model),
                        zero,
                    )(rho_tor_norm_v)

                    S = sum(
                        (source.profiles_1d.get(f"{prefix}/particles", zero) for source in core_sources.source),
                        zero,
                    )(rho_tor_norm_v)

                    dy_dt = one_over_dt * (vpr_next * ns_next - vpr_prev * ns_prev) * rho_tor_boundary

                    D = vpr_next * gm3 * transp_D / rho_tor_boundary

                    V = vpr_next * gm3 * (transp_V - rho_tor * k_phi)

                    S = vpr_next * (S - k_phi * ns_next) * rho_tor_boundary

                    # at axis x=0 , flux=0
                    bc0 = (0, 1, 0)

                    # at boundary x=1
                    bc_type, bc_value = boundary_condition_cfg.get(var.name, (1, psi_boundary))
                    match bc_type:
                        case 1:  # 1: value of the field y;
                            u = 1
                            v = 0
                            w = bc_value

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V
                            v = -1.0
                            w = bc_value * (D(rho_tor_norm_boundary) - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V(rho_tor_norm_boundary) - (D(rho_tor_norm_boundary) - hyper_diff) / L
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

                    bc1 = (u, v, w)

                case "temperature":
                    Ts_next = var

                    ns_next = profiles_1d_iter.get(f"{prefix}/density", zero)
                    Fs_next = profiles_1d_iter.get(f"{prefix}/density_flux", zero)

                    ns_prev = profiles_1d_prev.get(f"{prefix}/density", zero)(rho_tor_norm_v)
                    Ts_prev = profiles_1d_prev.get(f"{prefix}/temperature", zero)(rho_tor_norm_v)

                    flux_multiplier = sum(
                        (model.get("flux_multiplier", 0) for model in core_transport.model),
                        0,
                    )
                    flux_multiplier = one

                    energy_D = sum(
                        (model.profiles_1d.get(f"{prefix}/energy/d", zero) for model in core_transport.model),
                        zero,
                    )(rho_tor_norm_v)
                    energy_V = sum(
                        (model.profiles_1d.get(f"{prefix}/energy/v", zero) for model in core_transport.model),
                        zero,
                    )(rho_tor_norm_v)

                    Q = sum(
                        (source.profiles_1d.get(f"{prefix}/energy", zero) for source in core_sources.source),
                        zero,
                    )(rho_tor_norm_v)

                    dy_dt = one_over_dt * (
                        (3 / 2)
                        * (vpr_next * ns_next * Ts_next - (vpr_prev ** (5 / 3)) * ns_prev * Ts_prev * inv_vpr23)
                        * rho_tor_boundary
                    )

                    D = vpr_next * ns_next * gm3 * energy_D / rho_tor_boundary

                    V = (
                        vpr_next * ns_next * gm3 * energy_V
                        + Fs_next * flux_multiplier
                        - (3 / 2) * k_phi * vpr_next * ns_next * rho_tor
                    )

                    S = vpr_next * (Q - k_vppr * ns_next * Ts_next) * rho_tor_boundary

                    # at axis x=0, dH_dx=0
                    bc0 = (0, 1, 0)

                    # at boundary x=1
                    bc_type, bc_value = boundary_condition_cfg.get(var.name, (1, psi_boundary))

                    match bc_type:
                        case 1:  # 1: value of the field y;
                            u = 1.0
                            v = 0
                            w = bc_value

                        case 2:  # 2: radial derivative of the field (-dy/drho_tor);
                            u = V(rho_tor_norm_boundary)
                            v = -1.0
                            w = bc_value * (D(rho_tor_norm_boundary) - hyper_diff)

                        case 3:  # 3: scale length of the field y/(-dy/drho_tor);
                            L = bc_value
                            u = V(rho_tor_norm_boundary) - (D(rho_tor_norm_boundary) - hyper_diff) / L
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

                    bc1 = (u, v, w)

                case "toroidal":
                    ms = profiles_1d_prev[prefix].mass

                    us = var
                    ns_next = profiles_1d_prev.get(f"{prefix}/density", zero)
                    Fs_next = profiles_1d_prev.get(f"{prefix}/density_flux", zero)

                    us_m = profiles_1d_prev.get(f"{prefix}/velocity/toroidal", zero)(rho_tor_norm_v)
                    ns_prev = profiles_1d_prev.get(f"{prefix}/density", zero)(rho_tor_norm_v)

                    chi_u = sum(
                        (
                            model.profiles_1d.get(f"{prefix}/momentum/toroidal/d", zero)
                            for model in core_transport.model
                        ),
                        zero,
                    )(rho_tor_norm_v)

                    V_u_pinch = sum(
                        (
                            model.profiles_1d.get(f"{prefix}/momentum/toroidal/v", zero)
                            for model in core_transport.model
                        ),
                        zero,
                    )(rho_tor_norm_v)

                    U = gm8 * sum(
                        (
                            source.profiles_1d.get(f"{prefix}/momentum/toroidal", zero)
                            for source in core_sources.source
                        ),
                        zero,
                    )(rho_tor_norm_v)

                    dy_dt = (
                        one_over_dt
                        * ms
                        * (vpr_next * gm8 * ns_next * us - vpr_prev * gm8_prev * ns_prev * us_m)
                        * rho_tor_boundary
                    )

                    D = vpr_next * gm3 * gm8 * ms * ns_next * chi_u / rho_tor_boundary

                    V = (
                        (vpr_next * gm3 * ns_next * V_u_pinch + Fs_next - k_phi * vpr_next * rho_tor * ns_next)
                        * gm8
                        * ms
                    )

                    S = vpr_next * (U - k_rho_bdry * ms * ns_next * us) * rho_tor_boundary

                    # at axis x=0, du_dx=0
                    bc0 = (0, 1, 0)

                    # at boundary x=1
                    match boundary_condition_cfg.get(f"{prefix}/moment", 1):
                        case 1:  # 1: value of the field y;
                            u = 1.0
                            v = 0
                            w = bc_value

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

                    bc1 = (u, v, w)

            dy_dr = (-flux + V * y + hyper_diff * derivative(y, x)) / (D + hyper_diff)

            dflux_dr = (S - dy_dt + hyper_diff * derivative(flux, x)) / (1.0 + hyper_diff)

            equations.extend([dy_dr, dflux_dr])
            boundary_conditions.extend([bc0, bc1])

        # 设定初值

        X = core_profiles_prev.profiles_1d.grid.rho_tor_norm
        Y = np.zeros([len(var_list) - 1, X.size])
        for idx, var in enumerate(var_list[1:]):
            Y[idx] = as_path(var.name).get(profiles_1d_prev, zero)(X)

        sol = solve_bvp(
            lambda X, Y: np.stack([equ(X, *Y) for equ in equations]),
            lambda y0, y1: np.array(
                sum(
                    [
                        [u0 * y0[i * 2] + v0 * y0[i * 2 + 1] + w0, u1 * y1[i * 2] + v1 * y1[i * 2 + 1] + w1]
                        for i, (u0, v0, w0), (u1, v1, w1) in enumerate(boundary_conditions)
                    ],
                    [],
                )
            ),
            X,
            Y,
            discontinuity=self.code.parameters.get("discontinuity", []),
            tol=self.code.parameters.get("tolerance", 1.0e-3),
            bc_tol=self.code.parameters.get("bc_tol", 1e6),
            max_nodes=self.code.parameters.get("max_nodes", 1000),
            verbose=self.code.parameters.get("verbose", 1e6),
        )

        logger.info(
            f"Solving the transport equation [{ 'success' if sol.success else 'failed'}]: {sol.message} , {sol.niter} iterations"
        )

        profiles_1d_next = {}

        profiles_1d_next["rms_residuals"] = sol.rms_residuals

        profiles_1d_next[self.primary_coordinate] = sol.X

        profiles_1d_next["grid"] = equilibrium_next.profiles_1d.grid.remesh(**{self.primary_coordinate: sol.X})

        for idx, var in enumerate(var_list[1:]):
            as_path(var.name).update(profiles_1d_next, Y[idx])

        # for idx, equ in enumerate(self.equations):
        #     d_dt, D, V, R = equ.coefficient
        #     self.equations.append(
        #         {
        #             "@name": equ.identifier,
        #             "boundary_condition_type": equ.boundary_condition_type,
        #             "boundary_condition_value": equ.boundary_condition_value,
        #             "profile": Y[2 * idx],
        #             "flux": Y[2 * idx + 1],
        #             "coefficient": [d_dt(X, *Y), D(X, *Y), V(X, *Y), R(X, *Y)],
        #             "d_dr": Yp[2 * idx],
        #             "dflux_dr": Yp[2 * idx + 1],
        #         }
        #     )

        return {
            "time": equilibrium_next.time,
            "vacuum_toroidal_field": {"r0": R0, "b0": B0},
            "profiles_1d": profiles_1d_next,
        }
