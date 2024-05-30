import os
import pathlib

from fytok.tokamak import Tokamak
from fytok.utils.logger import logger

from spdm.view.sp_view import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

if __name__ == "__main__":
    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(f"east+mdsplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east", device="EAST", shot=70754)

    # f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890", device="d3d"
    #     tok = Tokamak("file+geqdsk:///home/salmon/workspace/fytok_tutorial/tutorial/data/g900003.00230_ITER_15MA_eqdsk16HR.txt",
    #                   device='ITER', shot='900003', time=2.3)

    tok.refresh(time=5.0)
    
    prof2d = tok.equilibrium.time_slice.current.profiles_2d
    
    display(
        tok.equilibrium.time_slice.current.profiles_2d.psi,
        title=tok.title,
        styles={"interferometer": False},
        output=output_path / f"{tok.tag}_rz.svg",
    )

#     tok.advance()

#     display(tok,
#             title=tok.title,
#             styles={"interferometer": False},
#             output=output_path/f"{tok.tag}_rz.svg")
#     tok.advance()

#     display(tok,
#             title=tok.title,
#             styles={"interferometer": False},
#             output=output_path/f"{tok.tag}_rz.svg")
