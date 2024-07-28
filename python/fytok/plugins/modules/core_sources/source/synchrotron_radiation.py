import typing
import scipy.constants
from spdm.core.expression import Variable, Expression, zero
from spdm.core.sp_tree import sp_tree


from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.utilities import *

from fytok.utils.atoms import atoms

from fytok.utils.logger import logger

PI = scipy.constants.pi


class SynchrotronRadiation(
    CoreSources.Source,
    identifier="synchrotron",
    code={
        "name": "synchrotron_radiation",
        "description": """   Source from Synchrotron radition  """,
    },
):
    """Source from Synchrotron radition"""

    def execute(self, *args, core_profiles: CoreProfiles, **kwargs):
        current = super().execute(*args, **kwargs)
        profiles_1d = core_profiles.profiles_1d
        source_1d = current.profiles_1d

        ne = profiles_1d.electrons.density
        Te = profiles_1d.electrons.temperature

        eq_1d: Equilibrium.TimeSlice.Profiles1D = self.inports["equilibrium/time_slice/current/profiles_1d"].fetch()

        B0 = np.abs(eq_1d._parent.vacuum_toroidal_field.b0)
        R0 = eq_1d._parent.vacuum_toroidal_field.r0

        x = profiles_1d.rho_tor_norm

        if True:
            #   Reference: (GACODE)
            #    Synchrotron synchrotron
            #        - Trubnikov, JETP Lett. 16 (1972) 25.

            psi_norm = Function(eq_1d.grid.rho_tor_norm, eq_1d.grid.psi_norm, label=r"\bar{\psi}")(x)

            r_min = eq_1d.major_radius(psi_norm)

            aspect_rat = R0 / r_min

            r_coeff = 0.8  # Reflection coefficient (Rosenbluth)

            me = scipy.constants.electron_mass
            e = scipy.constants.elementary_charge
            PI = scipy.constants.pi
            c = scipy.constants.speed_of_light

            wpe = np.sqrt(4 * PI * ne * e**2 / me)
            wce = e * B0 / (me * c)
            g = Te / scipy.constants.electron_volt / (me * c**2)
            phi = (
                60
                * g**1.5
                * np.sqrt((1.0 - r_coeff) * (1 + 1 / aspect_rat / np.sqrt(g)) / (r_min * wpe**2 / c / wce))
            )

            qsync = me / (3 * PI * c) * g * (wpe * wce) ** 2 * phi

        else:
            qsync = 6.2e-22 * B0**2.0 * ne * Te

        source_1d.electrons.energy -= qsync

        return current
