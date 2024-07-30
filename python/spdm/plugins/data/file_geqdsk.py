""" 访问 GEqdsk 文件 """

import typing
import pathlib
import numpy as np
from spdm.core.entry import Entry, as_entry
from spdm.core.file import File
from spdm.core.function import Function
from spdm.utils.logger import logger


def sp_read_geqdsk(file):
    """
    :param file: input file / file path
    :return: profile object
    """

    description = file.read(48)
    idum = int(file.read(4))
    nw = int(file.read(4))
    nh = int(file.read(4))
    file.readline()

    rdim = float(file.read(16))
    zdim = float(file.read(16))
    rcentr = float(file.read(16))
    rleft = float(file.read(16))
    zmid = float(file.read(16))
    file.readline()
    rmaxis = float(file.read(16))
    zmaxis = float(file.read(16))
    simag = float(file.read(16))
    sibry = float(file.read(16))
    bcentr = float(file.read(16))
    file.readline()
    current = float(file.read(16))
    simag = float(file.read(16))
    xdum = float(file.read(16))
    rmaxis = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    zmaxis = float(file.read(16))
    xdum = float(file.read(16))
    sibry = float(file.read(16))
    xdum = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    def _read_data(count, width=16):
        data = []
        if count == 0:
            return data

        for n in range(count):
            d = file.read(width)

            try:
                v = float(d)
            except Exception as error:
                raise RuntimeError(f"Error reading data {n} {count} {data[-4:]} '{d}'") from error
            data.append(v)
            if n >= count - 1 or ((n + 1) % 5 == 0):
                file.readline()
        data = np.asarray(data)
        return data

    #
    fpol = _read_data(nw)
    pres = _read_data(nw)
    ffprim = _read_data(nw)
    pprim = _read_data(nw)

    psirz = _read_data(nw * nh).reshape([nh, nw])

    qpsi = _read_data(nw)

    # try:
    nbbs = int(file.read(5))
    nlimitr = int(file.read(5))
    file.readline()

    bbsrz = _read_data(nbbs * 2).reshape([nbbs, 2]) if nbbs > 0 else None

    limrz = _read_data(nlimitr * 2).reshape([nlimitr, 2]) if nlimitr > 0 else None
    # except Exception as error:

    #     nbbs = 0
    #     limitr = 0
    #     bbsrz = None
    #     limrz = None

    data = {
        "description": description,
        # "idum": idum,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        # "simag": simag,
        # "rmaxis": rmaxis,
        # "zmaxis": zmaxis,
        # "sibry": sibry,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprim": pprim,
        "psirz": psirz,
        "qpsi": qpsi,
        "bbsrz": bbsrz,
        "limrz": limrz,
    }
    return data


def sp_write_geqdsk(p, file):
    """
    :param profile: object

    :param file: file path / file
    :return:
    """
    # logger.debug(p)
    nw = p["nw"]
    nh = p["nh"]

    file.write("%48s%4i%4i%4i\n" % (p.get("description", "NO DESCRIPTION"), 3, nw, nh))
    file.write("%16.8e%16.8e%16.8e%16.8e%16.8e\n" % (p["rdim"], p["zdim"], p["rcentr"], p["rleft"], p["zmid"]))
    file.write("%16.8e%16.8e%16.8e%16.8e%16.8e\n" % (p["rmaxis"], p["zmaxis"], p["simag"], p["sibry"], p["bcentr"]))
    file.write("%16.8e%16.8e%16.8e%16.8e%16.8e\n" % (p["current"], p["simag"], 0, p["rmaxis"], 0))
    file.write("%16.8e%16.8e%16.8e%16.8e%16.8e\n" % (p["zmaxis"], 0, p["sibry"], 0, 0))

    def _write_data(d):
        if not isinstance(d, np.ndarray):
            logger.debug(d)
        count = len(d)
        for n in range(count):
            file.write("%16.8e" % d[n])
            if (n == count - 1) or ((n + 1) % 5 == 0):
                file.write("\n")
            # else:
            #     file.write(" ")

    _write_data(p["fpol"])
    _write_data(p["pres"])
    _write_data(p["ffprim"])
    _write_data(p["pprim"])
    _write_data(p["psirz"].reshape([nw * nh]))
    _write_data(p["qpsi"])

    bbsrz = p.get("bbsrz", np.zeros([0, 2]))
    limrz = p.get("limrz", np.zeros([0, 2]))

    file.write("%5i%5i\n" % (bbsrz.shape[0], limrz.shape[0]))

    _write_data(bbsrz.reshape([bbsrz.size]))
    _write_data(limrz.reshape([limrz.size]))

    return


def sp_to_geqdsk(d, description: str | None = None, time_slice=0, **kwargs) -> dict:
    entry: Entry = as_entry(d)

    geqdsk: dict = {"description": description or entry.get("description", "NOTHING TO SAY")}

    limiter_r = entry.get("wall/description_2d/0/limiter/unit/0/outline/r", None)
    limiter_z = entry.get("wall/description_2d/0/limiter/unit/0/outline/z", None)

    if isinstance(limiter_r, np.ndarray):
        geqdsk["limrz"] = np.append(
            limiter_r.reshape([1, limiter_r.size]),
            limiter_z.reshape([1, limiter_z.size]),
            axis=0,
        ).transpose()

    eq = entry.child(f"equilibrium/time_slice/{time_slice}")

    # rdim = 0.0
    # zdim = 0.0
    geqdsk["rcentr"] = eq.get("boundary/geometric_axis/r")
    # rleft = 0.0
    geqdsk["zmid"] = zmid = eq.get("boundary/geometric_axis/z")

    geqdsk["rmaxis"] = eq.get("global_quantities/magnetic_axis/r")
    geqdsk["zmaxis"] = eq.get("global_quantities/magnetic_axis/z")
    geqdsk["simag"] = eq.get("global_quantities/psi_axis")
    geqdsk["sibry"] = eq.get("global_quantities/psi_boundary")
    geqdsk["bcentr"] = eq.get("vacuum_toroidal_field/b0")
    geqdsk["current"] = eq.get("global_quantities/ip")

    # boundary

    rbbs = eq.get("boundary/outline/r", np.zeros([0]))
    zbbs = eq.get("boundary/outline/z", np.zeros([0]))

    geqdsk["bbsrz"] = np.append(rbbs.reshape([1, rbbs.size]), zbbs.reshape([1, rbbs.size]), axis=0).transpose()

    # psi

    psirz = eq.get("profiles_2d/psi")

    if eq.get("profiles_2d/grid_type/index", None) != 1:
        raise NotImplementedError(f"TODO: {eq.get('profiles_2d/grid_type/index', None)}")

    dim1 = eq.get("profiles_2d/grid/dim1")
    dim2 = eq.get("profiles_2d/grid/dim2")

    nw = dim1.size
    nh = dim2.size

    geqdsk["rleft"] = dim1.min()
    geqdsk["rmid"] = rmid = 0.5 * (dim1.max() + dim1.min())
    geqdsk["rdim"] = rdim = dim1.max() - dim1.min()
    geqdsk["zdim"] = zdim = dim2.max() - dim2.min()

    # coord_r = np.append(coord_r[:, :], coord_r[:, 0].reshape(coord_r.shape[0], 1), axis=1)
    # coord_z = np.append(coord_z[:, :], coord_z[:, 0].reshape(coord_z.shape[0], 1), axis=1)

    # points = np.append(coord_r.reshape([coord_r.size, 1]), coord_z.reshape([coord_z.size, 1]), axis=1)

    if not isinstance(psirz, np.ndarray):
        psirz = psirz.__array__()

    geqdsk["psirz"] = psirz.T

    geqdsk["nw"] = nw

    geqdsk["nh"] = nh

    # psi = np.append(psi[:, :], psi[:, 0].reshape(psi.shape[0], 1), axis=1)
    # values = psi[:coord_r.shape[0], :coord_r.shape[1]].reshape(points.shape[0])
    # psirz = interpolate.griddata(points, values, (grid_r, grid_z), method='cubic').transpose()

    # profile

    psi = eq.get("profiles_1d/psi", None)

    psi_axis = eq.get("global_quantities/psi_axis")
    psi_boundary = eq.get("global_quantities/psi_boundary")

    psi_ = np.linspace(psi_axis, psi_boundary, nw)
    nan_array = np.full(nw, np.nan)

    def _profile(f, psi_s):
        if f is None or psi_s is None:
            return nan_array
        else:
            if psi is None:
                length = len(f)
                x = np.linspace(psi_axis, psi_boundary, length)
            else:
                x = psi
            try:
                res = Function(x, f)(psi_s)
            except Exception:
                return nan_array
            else:
                return res

    geqdsk["fpol"] = _profile(eq.get("profiles_1d/f", None), psi_)
    geqdsk["pres"] = _profile(eq.get("profiles_1d/pressure", None), psi_)
    geqdsk["ffprim"] = _profile(eq.get("profiles_1d/f_df_dpsi", None), psi_)
    geqdsk["pprim"] = _profile(eq.get("profiles_1d/dpressure_dpsi", None), psi_)
    geqdsk["qpsi"] = _profile(eq.get("profiles_1d/q", None), psi_)

    return geqdsk

    # return Entry({
    #     "nw": nw,
    #     "nh": nh,
    #     "rdim": rdim,
    #     "zdim": zdim,
    #     "rcentr": rcentr,
    #     "rleft": rleft,
    #     "zmid": zmid,
    #     "rmaxis": rmaxis,
    #     "zmaxis": zmaxis,
    #     "simag": simag,
    #     "sibry": sibry,
    #     "bcentr": bcentr,
    #     "current": current,
    #     "bbsrz": bbsrz,
    #     "psirz": psirz,
    #     "fpol": fpol,
    #     "pres": pres,
    #     "ffprim": ffprim,
    #     "pprim": pprim,
    #     "qpsi": qpsi,
    #     "limrz": limrz

    # })


def sp_from_geqdsk(geqdsk: dict, eq: typing.Optional[Entry] = None) -> Entry:
    """Converts a GEQDSK file to an IMAS equilibrium entry.
    @TODO:
        - convert to COCOS 11 !!!
    """

    if eq is None:
        eq = Entry({})

    r0 = geqdsk["rcentr"]
    b0 = geqdsk["bcentr"]
    Ip = geqdsk["current"]
    psi_axis = geqdsk["simag"]
    psi_boundary = geqdsk["sibry"]
    q = geqdsk["qpsi"]

    s_Bp = np.sign(b0)
    s_Ip = np.sign(Ip)
    s_rtp = np.mean(np.sign(q)) / (s_Bp * s_Ip)
    assert np.sign(psi_boundary - psi_axis) == s_Ip

    e_Bp_TWOPI = 1.0
    limrz = geqdsk.get("limrz", None)
    if isinstance(limrz, np.ndarray):
        eq["wall"] = {"description_2d": [{"limiter": {"unit": [{"outline": {"r": limrz[:, 0], "z": limrz[:, 1]}}]}}]}

    # eq["equilibrium/time"] = [0.0]
    # eq["equilibrium/vacuum_toroidal_field/r0"] = r0
    # eq["equilibrium/vacuum_toroidal_field/b0"] = [b0]

    # rleft = 0.0

    # eq["global_quantities.magnetic_axis.b_field_tor"] = geqdsk["bcentr"]
    nw = geqdsk["nw"]
    nh = geqdsk["nh"]
    rmin = geqdsk["rleft"]
    rmax = geqdsk["rleft"] + geqdsk["rdim"]
    zmin = geqdsk["zmid"] - geqdsk["zdim"] / 2
    zmax = geqdsk["zmid"] + geqdsk["zdim"] / 2

    psirz = geqdsk["psirz"]

    if psirz.shape == (nh, nw):
        psirz = psirz.T
        # logger.warning(f"Transposing psirz from {(nh, nw)} to {(nw,nh)}")

    if psirz.shape != (nw, nh):
        raise ValueError(f"Invalid shape for psirz: {psirz.shape}!={(nw, nh)}")

    eq["equilibrium"] = {
        "time": 0.0,
        "vacuum_toroidal_field": {"r0": r0, "b0": b0},
        "global_quantities": {
            "magnetic_axis": {
                "r": geqdsk["rmaxis"],
                "z": geqdsk["zmaxis"],
            },
            "psi_axis": psi_axis,
            "psi_boundary": psi_boundary,
            "ip": Ip,
        },
        "boundary": {  # Boundary
            "outline": {
                "r": geqdsk["bbsrz"][:, 0],
                "z": geqdsk["bbsrz"][:, 1],
            },
            "geometric_axis": {
                "r": geqdsk["rcentr"],
                "z": geqdsk["zmid"],
            },
        },
        "profiles_1d": {  # Profiles 1D
            "grid": {
                "psi_axis": psi_axis,
                "psi_boundary": psi_boundary,
            },
            "psi_norm": np.linspace(0.0, 1.0, nw),
            "f": geqdsk["fpol"],
            "f_df_dpsi": geqdsk["ffprim"],
            "pressure": geqdsk["pres"],
            "dpressure_dpsi": geqdsk["pprim"],
            "q": geqdsk["qpsi"],
        },
        "profiles_2d": {  # profiles 2D
            "type": "total",  # total field
            "grid_type": {"name": "rectangular"},
            "grid": {
                "dim1": np.linspace(rmin, rmax, nw),
                "dim2": np.linspace(zmin, zmax, nh),
                "type": "rectangular",
            },
            "psi": psirz,
        },
    }

    return eq


class GEQdskFile(File, plugin_name=["gfile", "geqdsk"]):
    """GEqdsk File"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root = None

        # else:
        #     logger.debug(f"Open File mode={self.mode}  {self.url} ")

    def close(self):
        self._root = None

    # def flush(self, *args, **kwargs):
    #     if self.mode & File.Mode.write:
    #         self.save(self.path)

    def open(self) -> Entry:
        if self._root is None:
            with open(
                pathlib.Path(self.uri.path).expanduser().resolve(),
                mode="r",
                encoding="utf-8",
            ) as fid:
                self._root = sp_from_geqdsk(sp_read_geqdsk(fid))

        return super().open()

    def read(self, *args, **kwargs) -> Entry:
        self.open()
        return self._root.child(*args[:1]).find(*args[1:], **kwargs)

    def write(self, *args, **kwargs):
        self.open()
        self._root.child(*args[:-1]).update(*args[-1:], **kwargs)

    def flush(self):
        if self._root is not None:
            with open(
                pathlib.Path(self.uri.path).expanduser().resolve(),
                mode="x",
                encoding="utf-8",
            ) as fid:
                geqdsk = sp_to_geqdsk(self._root.get())
                sp_write_geqdsk(geqdsk, fid)
                fid.flush()
