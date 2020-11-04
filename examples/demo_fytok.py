import matplotlib.pyplot as plt
import pprint
import sys

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


##################################################################################################

if __name__ == "__main__":

    from fytok.Tokamak import Tokamak
    from fytok.PFActive import PFActive
    from fytok.Wall import Wall
    from spdm.data.Entry import open_entry
    from spdm.util.logger import logger

    device = open_entry("east+mdsplus:///home/salmon/public_data/~t/?default_tree_name=efit_east#shot=55555")

    # wall = Wall(
    #     limiter=device.wall.description_2d[0].limiter.unit[0].outline,
    #     vessel=device.wall.description_2d[0].vessel.annular
    # )

    # pf_active = PFActive(device.pf_active)

    tok = Tokamak()

    tok.load(device)

    # logger.debug(tok)
    # logger.debug(tok.entry.wall())
    lfcs_r = device.equilibrium.time_slice[10].boundary.outline.r.__value__()[:, 0]
    lfcs_z = device.equilibrium.time_slice[10].boundary.outline.z.__value__()[:, 0]
    psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]

    # tok.entry.equilibrium.solve(core_profiles=None, psivals=psivals)

    # tok.entry.vacuum_toroidal_field.b0 = 1.0
    # tok.entry.vacuum_toroidal_field.r0 = 1.0
    # tok.entry.core_profiles.profiles_1d.conductivity_parallel = 1.0

    tok.solve(0.1, max_iters=1, psivals=psivals)
    # fig = plt.figure()

    fig = tok.plot_full()

    # axs[0].axis("scaled")
    # axs[1].axis("scaled")
    fig.savefig("a.svg")
