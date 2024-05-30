import pathlib
from fytok.tokamak import Tokamak

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"
output_path = pathlib.Path(f"{WORKSPACE}/output/")

tok = Tokamak(
    "file+geqdsk:///home/salmon/workspace/fytok_tutorial/tutorial/data/g900003.00230_ITER_15MA_eqdsk16HR.txt",
    device="east",
    equilibrium={"code": {"name": "freegs"}},
)

tok.refresh()