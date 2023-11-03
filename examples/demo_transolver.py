import pathlib
from fytok.Tokamak import Tokamak

if __name__ == "__main__":
    WORKSPACE = "/home/salmon/workspace"

    # f"{WORKSPACE}/gacode/neo/tools/input/profile_data"
    input_path = pathlib.Path("/home/salmon/workspace/fytok_data/data/15MA inductive - burn")
    output_path = f"{WORKSPACE}/output"

    tokamak = Tokamak(f"file+iterprofiles://{next(input_path.glob('*.xls')).as_posix()}",
                      f"file+geqdsk://{next(input_path.glob('**/*.txt')).as_posix()}",
                      device="iter",
                      transport_solver={"code":  {"name": "fytrans", }})

    core_profiles_1d = tokamak.core_profiles.time_slice.current.profiles_1d

    tokamak.transport_solver.refresh(
        equilibrium=tokamak.equilibrium,
        core_profiles=tokamak.core_profiles,
        core_transport=tokamak.core_transport,
        core_sources=tokamak.core_sources,
    )

    # eq_profiles_1d = equilibrium.time_slice.current.profiles_1d

    # core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D = core_profiles.time_slice.current.profiles_1d

    # logger.debug(core_profiles_1d.electrons.density(core_profiles_1d.grid.rho_tor_norm))

    # for ion in core_profiles_1d.ion:
    #     logger.debug(ion.label)
    #     logger.debug(ion.density(core_profiles_1d.grid.rho_tor_norm))
    #     # logger.debug(ion.temperature(core_profiles_1d.grid.rho_tor_norm))

    # sp_view.plot([
    #     ([
    #         (core_profiles_1d.electrons.density, {"label": r"$n_e$"}),
    #         *[(ion.density, {"label": rf"$n_{ion.label}$"}) for ion in core_profiles_1d.ion]
    #     ], {"y_label": r"$[m^{-3}]$"}),
    #     ([
    #         (core_profiles_1d.electrons.temperature, {"label": r"$n_e$"}),
    #         *[(ion.temperature, {"label": rf"$T_{ion.label}$"}) for ion in core_profiles_1d.ion]
    #     ], {"y_label": r"$[eV]$"}),

    # ],
    #     x_value=core_profiles_1d.grid.rho_tor_norm,
    #     x_label=r"$\bar{\rho}_{tor}$",
    #     stop_if_fail=False,
    #     output=f"{output_path}/core_profiles.svg")
    # logger.debug(tgyro.model[0].current.profiles_1d.grid_d.rho_tor_norm)