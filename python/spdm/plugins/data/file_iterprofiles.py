import pathlib
import re
import scipy.constants
import numpy as np
import pandas as pd
from spdm.utils.tags import _not_found_
from spdm.core.expression import Variable
from spdm.core.file import File
from spdm.core.entry import Entry
from spdm.numlib.smooth import smooth_1d

PI = scipy.constants.pi
TWOPI = scipy.constants.pi * 2.0


def step_function(x, scale=1.0e-2):
    return 1 / (1 + np.exp(-x / scale))


def read_iter_profiles(path, entry: Entry):
    path = pathlib.Path(path)

    excel_file = pd.read_excel(path, sheet_name=1)

    profiles_0D = {}

    for s in excel_file.iloc[0, 3:7]:
        res = re.match(r"(\w+)=(\d+\.?\d*)(\D+)", s)
        profiles_0D[res.group(1)] = (float(res.group(2)), str(res.group(3)))

    R0 = profiles_0D["R"][0]
    B0 = profiles_0D["B"][0]

    vacuum_toroidal_field = {"r0": R0, "b0": B0}
    profiles_1D = pd.read_excel(path, sheet_name=1, header=10, usecols="B:BN")
    # profiles_1D = pd.read_excel(path, header=1, usecols="A:BM")
    time = 0.0

    rho_tor_norm = profiles_1D["x"].values
    rho_tor = profiles_1D["rho"].values
    psi_norm = profiles_1D["Fp"].values

    grid = {
        "rho_tor_norm": rho_tor_norm,
        "rho_tor_boundary": rho_tor[-1],
        "psi_norm": psi_norm,
        "psi_boundary": _not_found_,
        "psi_axis": _not_found_,
    }

    entry["core_profiles"] = {"time": time, "vacuum_toroidal_field": vacuum_toroidal_field}

    # Core profile
    r_ped = 0.96  # np.sqrt(0.88)
    i_ped = np.argmin(np.abs(rho_tor_norm - r_ped))
    # fmt:off
    psi_norm = profiles_1D["Fp"].values
    # bs_psi = bs_psi_norm*(psi_boundary-psi_axis)+psi_axis

    b_Te =    smooth_1d(rho_tor_norm, profiles_1D["TE"].values,      i_end=i_ped-10, window_len=21)*1000
    b_Ti =    smooth_1d(rho_tor_norm, profiles_1D["TI"].values,      i_end=i_ped-10, window_len=21)*1000
    b_ne =    smooth_1d(rho_tor_norm, profiles_1D["NE"].values,      i_end=i_ped-10, window_len=21)*1.0e19
    b_nDT =   smooth_1d(rho_tor_norm, profiles_1D["Nd+t"].values,    i_end=i_ped-10, window_len=21)*1.0e19*0.5
    b_nath =   smooth_1d(rho_tor_norm, profiles_1D["Nath"].values,    i_end=i_ped-10, window_len=21)*1.0e19
    b_nalpha =   smooth_1d(rho_tor_norm, profiles_1D["Nalf"].values,    i_end=i_ped-10, window_len=21)*1.0e19
    b_nImp =  smooth_1d(rho_tor_norm, profiles_1D["Nz"].values,      i_end=i_ped-10, window_len=21)*1.0e19
    b_zeff = profiles_1D["Zeff"].values
    # fmt:on

    z_eff_star = b_zeff - (b_nDT * 2.0 + 4 * b_nath) / b_ne
    z_imp = 1 - (b_nDT * 2.0 + 2 * b_nath) / b_ne
    b = -2 * z_imp / (0.02 + 0.0012)
    c = (z_imp**2 - 0.02 * z_eff_star) / 0.0012 / (0.02 + 0.0012)

    z_Ar = np.asarray((-b + np.sqrt(b**2 - 4 * c)) / 2)
    z_Be = np.asarray((z_imp - 0.0012 * z_Ar) / 0.02)
    # b_nDT = b_ne * (1.0 - 0.02*4 - 0.0012*18) - b_nHe*2.0

    # Zeff = Function(bs_r_norm, baseline["Zeff"].values)
    # e_parallel = baseline["U"].values / (TWOPI * R0)

    entry["core_profiles/profiles_1d"] = {
        "grid": grid,
        # fmt:off
        "electrons": {"label": "e",                         "density": b_ne,            "temperature": b_Te },
        "ion": [
            {"label": "D",                                  "density": b_nDT,           "temperature": b_Ti },
            {"label": "T",                                  "density": b_nDT,           "temperature": b_Ti },
            {"label": "He",                                 "density": b_nath,          "temperature": b_Ti },
            {"label": "alpha",      "is_thermal": False,    "density": b_nalpha - b_nath                    },
            {"label": "Be",         "is_impurity": True,    "density": b_ne * 0.02,     "z_ion_1d": z_Be    },
            {"label": "Ar",         "is_impurity": True,    "density": b_ne * 0.0012,   "z_ion_1d": z_Ar    },
           
        ],
        # fmt:on
        # "e_field": {"parallel":  Function(e_parallel,bs_r_norm)},
        # "conductivity_parallel": Function(baseline["Joh"].values*1.0e6 / baseline["U"].values * (TWOPI * grid.r0),bs_r_norm),
        "rho_tor": profiles_1D["rho"].values,
        "q": profiles_1D["q"].values,
        "zeff": profiles_1D["Zeff"].values,
        "vloop": profiles_1D["U"].values,
        "j_ohmic": profiles_1D["Joh"].values * 1.0e6,
        "j_non_inductive": profiles_1D["Jnoh"].values * 1.0e6,
        "j_bootstrap": profiles_1D["Jbs"].values * 1.0e6,
        "j_total": profiles_1D["Jtot"].values * 1.0e6,
        "XiNC": profiles_1D["XiNC"].values,
        "ffprime": profiles_1D["EQFF"].values,
        "pprime": profiles_1D["EQPF"].values,
    }
    _x = Variable(0, name="rho_tor_norm", label=r"\bar{\rho}_{tor}")

    # entry["core_transport"] = {
    #     "model": [
    #         {
    #             "code": {"name": "dummy"},
    #             "time_slice": [
    #                 {
    #                     "time": time,
    #                     "vacuum_toroidal_field": vacuum_toroidal_field,
    #                 }
    #             ],
    #         }
    #     ]
    # }
    # # Core profiles
    # r_ped = 0.96  # np.sqrt(0.88)
    # i_ped = np.argmin(np.abs(rho_tor_norm - r_ped))
    # # Core Transport
    # Cped = 0.17
    # Ccore = 0.4
    # # Function( profiles["Xi"].values,bs_r_norm)  Cped = 0.2
    # # chi = Piecewise(
    # #     [
    # #         (Ccore * (1.0 + 3 * (_x**2)), (_x < r_ped)),
    # #         (Cped, (_x >= r_ped)),
    # #     ],
    # #     label=r"\chi",
    # # )
    # # chi_e = Piecewise(
    # #     [
    # #         (0.5 * Ccore * (1.0 + 3 * (_x**2)), (_x < r_ped)),
    # #         (Cped, (_x >= r_ped)),
    # #     ],
    # #     label=r"\chi_e",
    # # )
    # delta = step_function_approx(_x - r_ped, scale=0.005)
    # chi = (Ccore * (1.0 + 3 * (_x**2))) * (1 - delta) + Cped * delta
    # chi_e = (0.5 * Ccore * (1.0 + 3 * (_x**2))) * (1 - delta) + Cped * delta
    # D = 0.1 * (chi + chi_e)
    # v_pinch_ne = 0.6 * D * _x / R0
    # v_pinch_Te = 2.5 * chi_e * _x / R0
    # v_pinch_ni = D * _x / R0
    # v_pinch_Ti = chi * _x / R0
    # entry["core_transport/model/0/time_slice/0/flux_multiplier"] = 3 / 2
    # entry["core_transport/model/0/time_slice/0/profiles_1d"] = {
    #     "grid_d": grid,
    #     "electrons": {"@name": "e", "particles": {"d": D, "v": v_pinch_ne}, "energy": {"d": chi_e, "v": v_pinch_Te}},
    #     "ion": [
    #         {"@name": "D", "particles": {"d": D, "v": v_pinch_ni}, "energy": {"d": chi, "v": v_pinch_Ti}},
    #         {"@name": "T", "particles": {"d": D, "v": v_pinch_ni}, "energy": {"d": chi, "v": v_pinch_Ti}},
    #         {"@name": "He", "particles": {"d": D, "v": v_pinch_ni}, "energy": {"d": chi, "v": v_pinch_Ti}},
    #         {"@name": "alpha", "particles": {"d": 0.001 * D, "v": 0}},
    #     ],
    # }

    entry["core_sources"] = {
        "source": [
            {
                "code": {"name": "dummy"},
                "time": time,
                "vacuum_toroidal_field": vacuum_toroidal_field,
            }
        ]
    }

    S = 9e20 * np.exp(15.0 * (_x**2 - 1.0))

    Q_e = (
        (
            # profiles_1D["Poh"].values
            +profiles_1D["Paux"].values
            - profiles_1D["Prad"].values
            - profiles_1D["Pneu"].values
            # - profiles_1D["Peic"].values
            # + profiles_1D["Pdte"].values
        )
        * 1e6
        / scipy.constants.electron_volt
    )

    Q_DT = (
        (
            profiles_1D["Pibm"].values
            # + profiles_1D["Peic"].values
            # + profiles_1D["Pdti"].values
        )
        * 1e6
        / scipy.constants.electron_volt
    )

    # Q_He = (profiles_1D["Pdti"].values + profiles_1D["Pdte"].values) * 1e6 / scipy.constants.electron_volt

    # Core Source
    entry["core_sources/source/0/profiles_1d"] = {
        "grid": grid,
        "conductivity_parallel": profiles_1D["Joh"].values * 1.0e6 / profiles_1D["U"].values * (TWOPI * R0),
        "j_parallel": -profiles_1D["Jtot"].values * 1e6,  # A/m^2
        "electrons": {"@name": "e", "particles": S, "energy": Q_e},
        "ion": [
            {"@name": "D", "particles": S * 0.5, "energy": Q_DT * 0.5},
            {"@name": "T", "particles": S * 0.5, "energy": Q_DT * 0.5},
            # {"@name": "He", "particles": S * 0.00, "energy": Q_DT * 0.00},  #
            # {"@name": "alpha", "particles": S * 0.2, "energy": Q_DT * 0.01},
        ],
    }

    return entry


class ITERProfiles(File, plugin_name="iterprofiles"):
    """Read iter_profiles.xslx file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cache = {
            "dataset_fair": {
                "identifier": "15MA Inductive at burn-ASTRA",
                "provenance": {"node": [{"path": "core_profiles", "sources": self.uri}]},
            }
        }

    def open(self) -> Entry:
        entry = Entry(self._cache)
        return read_iter_profiles(pathlib.Path(self.uri.path), entry)

    def flush(self):
        raise NotImplementedError()
