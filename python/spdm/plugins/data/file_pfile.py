import pathlib
import itertools
import re
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from spdm.utils.logger import logger
from spdm.core.file import File
from spdm.core.entry import Entry
from spdm.core.expression import Expression


def sp_read_pfile_txt(path: str | pathlib.Path):
    """
    :param file: input file / file path
    :return: profile object
    """
    s = "ne(10^20/m^3)"
    match = re.match(r"(\w+)\((.*)\)", s)

    res = {}
    with open("/home/salmon/workspace/fytok_ext/examples/data/Pfile_H_6_2", mode="r") as fid:
        while True:
            head = fid.readline()
            if head == "":
                break
            line_num, psi_norm, profile_name, dprofile_dpsi = head.split()
            line_num = int(line_num)
            match = re.match(r"(\w+)\((.*)\)", profile_name)
            if match:
                profile_name = match.group(1)
                unit = match.group(2)
            else:
                unit = "-"

            data = np.asarray([[float(v) for v in x.split()] for x in itertools.islice(fid, line_num)])
            ppoly = CubicHermiteSpline(data[:, 0], data[:, 1], data[:, 2])
            res[profile_name] = Expression(ppoly, domain=data[:, 0], name=profile_name, unit=unit)
    return res


def sp_to_imas(data: dict):
    entry = Entry({})

    core_profiles_1d = entry.child("core_profiles/profiles_1d/")

    core_profiles_1d["electrons/temperature"] = data["te"]
    core_profiles_1d["electrons/density"] = data["ne"]
    core_profiles_1d["t_i_average"] = data["ti"]
    core_profiles_1d["n_i_thermal_total"] = data["ni"]
    core_profiles_1d["pressure_parallel"] = data["pb"]
    core_profiles_1d["pressure_ion_total"] = data["ptot"]
    # fmt:on

    return entry


class PFile(File, plugin_name="pfile"):
    """Read pfile file  (from gacode)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self) -> Entry:
        if self.uri.netloc:
            raise NotImplementedError(f"{self.uri}")

        path = pathlib.Path(self.uri.path)

        if path.suffix.lower() in [".nc", ".h5"]:
            data = File(path, mode="r").read().dump()
        else:
            data = sp_read_pfile_txt(path)

        return sp_to_imas(data)

    def write(self, d, *args, **kwargs):
        raise NotImplementedError(f"TODO: write ITERDB {self.uri}")
