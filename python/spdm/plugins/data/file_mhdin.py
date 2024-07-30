import pathlib

import numpy as np
from spdm.core.entry import Entry
from spdm.core.file import File
from spdm.utils.logger import logger


def mhdin_outline(r, z, w, h, a0, a1):
    return r, z, r + w, z + h


def sp_to_imas(data: dict):

    entry = Entry({})

    # entry["wall.description_2d[0].limiter.unit[0].outline.r"] = np.array(data["rsi"])
    # entry["wall.description_2d[0].limiter.unit[0].outline.z"] = np.array(data["zsi"])

    n_pf = 0
    if "rvs" in data:
        rvs = data["rvs"]
        zvs = data["zvs"]
        wvs = data["wvs"]
        hvs = data["hvs"]
        avs = data["avs"]
        avs2 = data["avs2"]

        for i, unit_name in enumerate(data["vsid"]):
            a0 = -avs[i] * np.pi / 180.0
            a1 = -(avs2[i] * np.pi / 180.0 if avs2[i] != 0 else np.pi / 2)
            entry[f"wall.description_2d[0].vessel.unit[{i}].name"] = str(f"unit{int(unit_name)}")
            entry[f"wall.description_2d[0].vessel.unit[{i}].element[0].outline.r"] = np.array(
                [
                    rvs[i] - wvs[i] / 2.0 - hvs[i] / 2.0 * np.tan((np.pi / 2.0 + a1)),
                    rvs[i] - wvs[i] / 2.0 + hvs[i] / 2.0 * np.tan((np.pi / 2.0 + a1)),
                    rvs[i] + wvs[i] / 2.0 + hvs[i] / 2.0 * np.tan((np.pi / 2.0 + a1)),
                    rvs[i] + wvs[i] / 2.0 - hvs[i] / 2.0 * np.tan((np.pi / 2.0 + a1)),
                ]
            )

            entry[f"wall.description_2d[0].vessel.unit[{i}].element[0].outline.z"] = np.array(
                [
                    zvs[i] - hvs[i] / 2.0 - wvs[i] / 2.0 * np.tan(-a0),
                    zvs[i] + hvs[i] / 2.0 - wvs[i] / 2.0 * np.tan(-a0),
                    zvs[i] + hvs[i] / 2.0 + wvs[i] / 2.0 * np.tan(-a0),
                    zvs[i] - hvs[i] / 2.0 + wvs[i] / 2.0 * np.tan(-a0),
                ]
            )
            entry[f"wall.description_2d.0.vessel.unit.{i}.element.0.resistivity"] = data["rsisvs"][i]

    if "rf" in data:
        n_pf = len(data["rf"])
        for i in range(n_pf):
            # fmt:off
            pf_id=int(data["fcid"][i])-1
            entry[f"pf_active.coil[{i}].name"]                                  = f"PF{pf_id}"
            entry[f"pf_active.coil[{i}].identifier"]                            = f"PF{pf_id}"
            entry[f"pf_active.coil[{i}].element[0].geometry.geometry_type"]     = 2
            entry[f"pf_active.coil[{i}].element[0].geometry.rectangle.r"]       = float(data["rf"][i])
            entry[f"pf_active.coil[{i}].element[0].geometry.rectangle.z"]       = float(data["zf"][i])
            entry[f"pf_active.coil[{i}].element[0].geometry.rectangle.width"]   = float(data["wf"][i])
            entry[f"pf_active.coil[{i}].element[0].geometry.rectangle.height"]  = float(data["hf"][i])
            entry[f"pf_active.coil[{i}].element[0].turns_with_sign"]            = int(data['fcturn'][i])
            # fmt:on
    if "re" in data:
        re = data["re"]
        ze = data["ze"]
        we = data["we"]
        he = data["he"]
        ecturn = data["ecturn"]
        elements_id = (np.array(data["ecid"]) - 1).astype(int)
        for i in range(len(re)):
            c = elements_id[i]
            e = sum(elements_id[:i] == elements_id[i])
            entry[f"pf_active.coil[{i+n_pf}].name"] = f"OH{c}"
            entry[f"pf_active.coil[{i+n_pf}].identifier"] = f"OH{c}"
            entry[f"pf_active.coil[{i+n_pf}].element[0].name"] = f"OH{c}_{e}"
            entry[f"pf_active.coil[{i+n_pf}].element[0].identifier"] = f"OH{c}_{e}"
            entry[f"pf_active.coil[{i+n_pf}].element[0].turns_with_sign"] = ecturn[i]
            entry[f"pf_active.coil[{i+n_pf}].element[0].geometry.rectangle.r"] = re[i]
            entry[f"pf_active.coil[{i+n_pf}].element[0].geometry.rectangle.z"] = ze[i]
            entry[f"pf_active.coil[{i+n_pf}].element[0].geometry.rectangle.width"] = we[i]
            entry[f"pf_active.coil[{i+n_pf}].element[0].geometry.rectangle.height"] = he[i]
            entry[f"pf_active.coil[{i+n_pf}].element[0].geometry.geometry_type"] = 1

    if "lpname" in data:
        n_flux = len(data["lpname"])
        for i in range(n_flux):
            # fmt:off
            flux_name                                                           = data['lpname'][i]
            entry[f'magnetics.flux_loop.{i}.name']                              = flux_name
            entry[f'magnetics.flux_loop.{i}.identifier']                        = flux_name
            entry[f'magnetics.flux_loop.{i}.position[0].r']                     = data["rsi"][i]
            entry[f'magnetics.flux_loop.{i}.position[0].z']                     = data["zsi"][i]
            entry[f'magnetics.flux_loop.{i}.type.index']                        = 1
            # fmt:on

    if "mpnam2" in data:
        n_bp = len(data["mpnam2"])
        for i in range(n_bp):
            # fmt:off
            entry[f'magnetics.b_field_pol_probe.{i}.name']                      = data['mpnam2'][i]
            entry[f'magnetics.b_field_pol_probe.{i}.identifier']                = data['mpnam2'][i]
            entry[f'magnetics.b_field_pol_probe.{i}.position.r']                = data['xmp2'][i]
            entry[f'magnetics.b_field_pol_probe.{i}.position.z']                = data['ymp2'][i]
            entry[f'magnetics.b_field_pol_probe.{i}.length']                    = data['smp2'][i]
            entry[f'magnetics.b_field_pol_probe.{i}.poloidal_angle']            = -data['amp2'][i] / 180 * np.pi
            entry[f'magnetics.b_field_pol_probe.{i}.toroidal_angle']            = 0.0 / 180 * np.pi
            entry[f'magnetics.b_field_pol_probe.{i}.type.index']                = 1
            entry[f'magnetics.b_field_pol_probe.{i}.turns']                     = 1
            # fmt:on

    return entry


class MHDINFile(File, plugin_name="mhdin"):
    """READ mahchine description file (MHDIN)
    learn from omas
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self) -> Entry:
        if self.url.authority:
            raise NotImplementedError(f"{self.url}")

        data = File(self.url.path, mode="r", format="namelist").read().dump()

        return sp_to_imas(data["in3"])
