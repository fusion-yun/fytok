import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import math


sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == "__main__":
    from spdm.util.logger import logger
    from spdm.data.Collection import Collection
    from fytok.Tokamak import Tokamak
    from spdm.util.logger import logger
    from spdm.util.Profiles import Profile
    from spdm.data.Entry import open_entry
    from fytok.Plot import plot_profiles
    from spdm.util.AttributeTree import _next_
    import freegs

    tok = Tokamak(open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east", shot=55555, time_slice=20))

    control_line = [

        ((1.99, 0),    (2.40, 0)),     # isoflux01
        ((1.93, 0.15),    (2.24, 0.52)),   # isoflux03
        ((1.72, 0.24),    (1.35, 0.45)),   # isoflux04
        ((1.66, 0),    (1.35, 0)),     # isoflux06
        ((1.72, -0.24),   (1.35, -0.45)),    # isoflux08
        ((1.93, -0.15),    (2.24, -0.52)),   # isoflux09
    ]

    r = np.linspace(0.1, 1.0, 10, endpoint=False)

    itime = 9

    prefix = "../output/magnetic/{itime}.png"

    s = r[itime]
    logger.debug(s)
    pts = [((1.0-s)*b[0] + s*e[0], (1.0-s)*b[1] + s*e[1]) for b, e in control_line]

    isoflux = [
        (*pts[0], *pts[3]),
        (*pts[1], *pts[4]),
        (*pts[2], *pts[5]),
    ]
    logger.debug(isoflux)

    # for r0, z0, r1, z1 in isoflux:
    #     plt.plot([r0, r1], [z0, z1])

    R = tok.equilibrium.profiles_2d.r
    Z = tok.equilibrium.profiles_2d.z

    tok.equilibrium.update(
        constraints={
            # "psivals": psivals,
            # "xpoints": xpoints,
            "isoflux": isoflux
        })

    plt.cla()

    tok.wall.plot()

    tok.pf_active.plot()

    for p0, p1 in control_line:
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]])

    logger.debug(tok.equilibrium.boundary.x_point)

    psi = tok.equilibrium.profiles_2d.psi

    # plt.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=20,  linewidths=0.2)

    tok.equilibrium.plot()

    plt.gca().set_aspect('equal')

    plt.savefig(prefix.format(itime=itime))