import pathlib
import os
import numpy as np
from spdm.data.File import File
from spdm.view.View import display
from spdm.utils.logger import logger
from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak


WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"


if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(f"EAST+MDSplus://{WORKSPACE}/fytok_data/mdsplus/~t/?shot=70754")

    display(tok, title=f"Tokamak", output=output_path/"tok.svg")

    logger.info("Done")