import os
import pathlib

import numpy as np
from spdm.data.File import File
from spdm.utils.logger import logger
from spdm.view.View import display

from fytok.Tokamak import Tokamak

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"


if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(f"EAST+MDSplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east&shot=70754")

    tok.refresh(time=5.0)

    display(tok, title=f"EAST RZ  View", output=output_path/"east_rz.svg")
    display(tok, title=f"EAST Top View", output=output_path/"east_top.svg", view="TOP")

    
