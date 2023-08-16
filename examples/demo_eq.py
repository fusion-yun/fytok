
import pathlib

import numpy as np
import pandas as pd
import scipy.constants
from fytok.modules.Wall import Wall
from fytok.modules.PFActive import PFActive
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreProfiles import CoreProfiles

from fytok.Tokamak import Tokamak
from fytok.utils.load_scenario import load_scenario
from spdm.data.File import File
from spdm.data.Function import function_like
from spdm.data.open_entry import open_entry
from spdm.utils.logger import logger
from spdm.views.View import display

if __name__ == "__main__":

    output_path = pathlib.Path('/home/salmon/workspace/output')

    data_path = pathlib.Path("/home/salmon/workspace/data/15MA inductive - burn")

    eqdsk_file = File(
        data_path/"Increased domain R-Z/Medium resolution - 129x257/g900003.00230_ITER_15MA_eqdsk16VVMR.txt", format="GEQdsk").read()
    ###################################################################################################
    # Baseline

    # eqdsk_file = File(
    #     data_path/"Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    R0 = eqdsk_file.get("vacuum_toroidal_field/r0")
    B0 = eqdsk_file.get("vacuum_toroidal_field/b0")
    psi_axis = eqdsk_file.get("time_slice/0/global_quantities/psi_axis")
    psi_boundary = eqdsk_file.get("time_slice/0/global_quantities/psi_boundary")
    bs_eq_psi = eqdsk_file.get("time_slice/0/profiles_1d/psi")
    bs_eq_psi_norm = (bs_eq_psi-psi_axis)/(psi_boundary-psi_axis)
    bs_eq_fpol = function_like(eqdsk_file.get("time_slice/0/profiles_1d/f"), bs_eq_psi)

    profiles = pd.read_excel(data_path/'15MA Inductive at burn-ASTRA.xls',
                             sheet_name='15MA plasma', header=10, usecols="B:BN")

    bs_r_norm = profiles["x"].values

    b_Te = profiles["TE"].values*1000
    b_Ti = profiles["TI"].values*1000
    b_ne = profiles["NE"].values*1.0e19

    b_nHe = profiles["Nalf"].values * 1.0e19
    b_ni = profiles["Nd+t"].values * 1.0e19*0.5
    b_nImp = profiles["Nz"].values * 1.0e19
    b_zeff = profiles["Zeff"].values
    bs_psi_norm = profiles["Fp"].values
    bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    bs_xq = profiles["xq"].values
    bs_q = profiles["q"].values

    bs_r_norm = profiles["x"].values
    bs_line_style = {"marker": '.', "linestyle": ''}
    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(bs_r_norm-r_ped))

    scenario = load_scenario(data_path)

    ###################################################################################################

    entry = open_entry(None, source_schema="ITER", target_schema=f"imas/3")

    wall = Wall(entry=entry.child("wall"))

    pf_active = PFActive(entry=entry.child("pf_active"))

    equilibrium = Equilibrium(
        {**scenario["equilibrium"],
            "code": {
                "name": "freegs",  # 指定 equlibrium 模块 "eq_analyze",
                "parameters": {"boundary": "fixed", }},
            "$default_value": {
                "time_slice": {   # 默认参数， time_slice 是 TimeSeriesAoS
                    "boundary": {"psi_norm": 0.99},
                    "profiles_2d": {"grid": {"dim1": 256, "dim2": 128}},
                    "coordinate_system": {"grid": {"dim1": 128, "dim2": 128}}
                }}},

        wall=wall,
        pf_active=pf_active
    )

    core_profiles = CoreProfiles({
        **scenario["core_profiles"],
        "$default_value": {
            "profiles_1d": {  # 默认参数，profiles_1d  是 TimeSeriesAoS
                "grid": {     # 设定 grid
                    "rho_tor_norm": np.linspace(0, 1.0, 100),
                    "psi_norm": np.linspace(0, 1.0, 100),
                }}}
    })

    # tok = Tokamak(
    #     "ITER",  # 导入 ITER 装置描述（mapping）
    #     time=0.0,
    #     name=scenario["name"],
    #     description=scenario["description"],
    #     core_profiles={
    #         **scenario["core_profiles"],
    #         "$default_value": {
    #             "profiles_1d": {  # 默认参数，profiles_1d  是 TimeSeriesAoS
    #                 "grid": {     # 设定 grid
    #                               "rho_tor_norm": np.linspace(0, 1.0, 100),
    #                               "psi_norm": np.linspace(0, 1.0, 100),
    #                 }}}
    #     },
    #     equilibrium={
    #         **scenario["equilibrium"],
    #         "code": {
    #             "name": "freegs",  # 指定 equlibrium 模块 "eq_analyze",
    #             "parameters": {"boundary": "fixed", }},
    #         "$default_value": {
    #             "time_slice": {   # 默认参数， time_slice 是 TimeSeriesAoS
    #                 "boundary": {"psi_norm": 0.99},
    #                 "profiles_2d": {"grid": {"dim1": 256, "dim2": 128}},
    #                 "coordinate_system": {"grid": {"dim1": 128, "dim2": 128}}
    #             }}},

    # )

    eq_time_slice = equilibrium.time_slice.current

    display(  # plot equilibrium
        equilibrium,
        title=f"equilibrium time={equilibrium.time}s",
        output=output_path/"tokamak_prev.svg")

    if False:
        eq_profiles_1d = eq_time_slice.profiles_1d

        display(  # plot tokamak geometric profile
            [
                ((eq_profiles_1d.dvolume_dpsi, {"label": r"$\frac{dV}{d\psi}$"}), {"y_label": r"$[Wb]$"}),

                (eq_profiles_1d.dpsi_drho_tor, {"label": r"$\frac{d\psi}{d\rho_{tor}}$"}),
                ([
                    (bs_eq_fpol,  {"label": "astra", **bs_line_style}),
                    (eq_profiles_1d.f,  {"label": r"fytok"}),
                ], {"y_label": r"$F_{pol} [Wb\cdot m]$"}),
                ([
                    (function_like(profiles["q"].values, bs_psi), {"label": r"astra", **bs_line_style}),
                    # (function_like(eqdsk.get('profiles_1d.psi_norm'), eqdsk.get('profiles_1d.q')), "eqdsk"),
                    (eq_profiles_1d.q,  {"label": r"$fytok$"}),
                    # (magnetic_surface.dphi_dpsi,  r"$\frac{d\phi}{d\psi}$", r"$[Wb]$"),
                ], {"y_label": r"$q [-]$"}),
                ([
                    (function_like(profiles["rho"].values, bs_psi), {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.rho_tor,   {"label": r"$\rho$"}),
                ], {"y_label":  r"$\rho_{tor}[m]$", }),

                ([
                    (function_like(profiles["x"].values, bs_psi), {
                        "label": r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$ astra", **bs_line_style}),
                    (eq_profiles_1d.rho_tor_norm,   {"label": r"$\bar{\rho}$"}),
                ], {"y_label":  r"[-]", }),



                ([
                    (function_like(4*(scipy.constants.pi**2) * R0 * profiles["rho"].values, bs_psi),
                     {"label": r"$4\pi^2 R_0 \rho$", **bs_line_style}),
                    (eq_profiles_1d.dvolume_drho_tor,  {"label": r"$dV/d\rho_{tor}$", }),
                ], {"y_label":  r"$4\pi ^ 2 R_0 \rho[m ^ 2]$"}),

                (eq_profiles_1d.gm1,   {"label": r"$gm1=\left<\frac{1}{R^2}\right>$"}),
                (eq_profiles_1d.gm2,   {"label": r"$gm2=\left<\frac{\left|\nabla \rho\right|^2}{R^2}\right>$"}),
                (eq_profiles_1d.gm3,   {"label": r"$gm3=\left<\left|\nabla \rho\right|^2\right>$"}),
                (eq_profiles_1d.gm4,   {"label": r"$gm4=\left<1/B^2\right>$"}),
                (eq_profiles_1d.gm5,   {"label": r"$gm5=\left<B^2\right>$"}),
                (eq_profiles_1d.gm6,   {"label": r"$gm6=\left<\nabla \rho_{tor}^2/ B^2 \right>$"}),
                (eq_profiles_1d.gm7,   {"label": r"$gm7=\left<\left|\nabla \rho\right|\right>$"}),
                (eq_profiles_1d.gm8,   {"label": r"$gm8=\left<R\right>$"}),

            ],
            x_axis=(eq_profiles_1d.psi[1:], {"label":  r"$\psi$"}),
            title="Equilibrium",
            output=output_path/"Equilibrium_coord.svg",
            grid=True, fontsize=16)

    equilibrium.refresh(Ip=1.0e6, beta_p=1.0, xpoints=[(p.r, p.z) for p in eq_time_slice.boundary.x_point])
 
    display(  # plot equilibrium
        equilibrium,
        title=f" time={equilibrium.time}s",
        output=output_path/"tokamak_post.svg")
    if False:
        display(  # plot tokamak geometric profile
            [

                ([
                    (function_like(profiles["q"].values, bs_psi),   {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.q,                              {"label": r"fytok", }),
                    (eq_profiles_1d.dphi_dpsi*np.sign(B0)/scipy.constants.pi/2.0,
                     {"label": r"$\frac{\sigma_{B_{p}}}{\left(2\pi\right)^{1-e_{B_{p}}}}\frac{d\Phi_{tor}}{d\psi_{ref}}$"}),
                ], {"y_label": r"$q [-]$"}),
                ([
                    (function_like(profiles["rho"].values, bs_psi), {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.rho_tor,                        {"label": r"fytok"}),
                ], {"y_label": r"$\rho_{tor}[m]$", }),
                ([
                    (function_like(profiles["x"].values, bs_psi),   {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.rho_tor_norm,                   {"label": r"fytok"}),
                ], {"y_label": r"$\frac{\rho_{tor}}{\rho_{tor,bdry}}$"}),

                ([
                    (function_like(profiles["shif"].values, bs_psi),    {"label": r"astra", ** bs_line_style}),
                    (eq_profiles_1d.geometric_axis.r - R0,              {"label": r"fytok", }),
                ], {"y_label": "$\Delta$ shafranov \n shift $[m]$ "}),

                ([
                    (function_like(profiles["k"].values, bs_psi),       {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.elongation,                         {"label": r"fytok", }),
                ], {"y_label": r"$elongation[-]$", }),

                ([
                    (function_like(profiles["del"].values, bs_psi),     {"label": r"astra", **bs_line_style}),
                    (eq_profiles_1d.triangularity,                      {"label": r"$\Delta[-]$          fytok"}),
                    (eq_profiles_1d.triangularity_upper,                {"label": r"$\Delta_{upper}[-]$  fytok"}),
                    (eq_profiles_1d.triangularity_lower,                {"label": r"$\Delta_{lower}[-]$  fytok"}),
                ], {"y_label": r"$triangularity[-]$", }),
                ([
                    (4*(scipy.constants.pi**2) * R0*eq_profiles_1d.rho_tor,   {"label": r"$4\pi^2 R_0 \rho$", }),
                    (eq_profiles_1d.dvolume_drho_tor,                   {"label": r"$V^{\prime}$", }),
                ], {"y_label": r"$4\pi^2 R_0 \rho , dV/d\rho$"}),

                [
                    (eq_profiles_1d.geometric_axis.r,                   {"label":   r"$geometric_{axis.r}$"}),
                    (eq_profiles_1d.r_inboard,                          {"label":          r"$r_{inboard}$"}),
                    (eq_profiles_1d.r_outboard,                         {"label":         r"$r_{outboard}$"}),
                ],

                ([
                    (eq_profiles_1d.volume,                          {"label":   r"$V$",  **bs_line_style}),
                    (eq_profiles_1d.dvolume_dpsi.antiderivative(),   {
                        "label":   r"$\int \frac{dV}{d\psi}  d\psi$"}),
                ], {"y_label": r"volume", }),

                # ([
                #     (function_like(profiles["Jtot"].values*1e6),   {"label": r"astra", ** bs_line_style}),
                #     (eq_profiles_1d.j_parallel,                       {"label": r"fytok", }),
                # ], {"y_label": r"$j_{\parallel} [A\cdot m^{-2}]$", })

                # [
                #     (eq.coordinate_system.surface_integrate2(lambda r, z:1.0/r**2),
                #      r"$\left<\frac{1}{R^2}\right>$"),
                #     (eq.coordinate_system.surface_integrate(1/eq.coordinate_system.r**2),
                #      r"$\left<\frac{1}{R^2}\right>$"),
                # ]
            ],
            x_axis=(eq_profiles_1d.psi,  {"label": r"$\psi/\psi_{bdry}$"}),
            title="Equilibrium Geometric Shape",
            grid=True, fontsize=16, transparent=True,
            output=output_path/"equilibrium_profiles.svg")

    logger.info("Solve Equilibrium ")
