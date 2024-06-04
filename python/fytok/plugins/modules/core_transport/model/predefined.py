import collections

import numpy as np
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium
from spdm.utils.logger import logger
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.core.sp_property import sp_tree


@sp_tree
class PredefinedTransport(CoreTransport.Model):
    """ """

    identifier = "predefined"
    code = {"name": "predefined", "description": f"predefined", "copyright": "fytok"}

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args, **kwargs) -> CoreTransport.Model.TimeSlice:
        current: CoreTransport.Model.TimeSlice = super().fetch(*args, **kwargs)

        rho_tor_norm = profiles_1d.rho_tor_norm

        eq: Equilibrium.TimeSlice = self.inports["equilibrium/time_slice/current"].fetch()

        eq_1d = eq.profiles_1d

        B0 = np.abs(eq.vacuum_toroidal_field.b0)

        R0 = eq.vacuum_toroidal_field.r0

        # rho_tor_norm = Variable(0, name="rho_tor_norm", label=r"\bar{\rho}_{tor}")

        _x = rho_tor_norm

        # Core profiles
        r_ped = 0.96  # np.sqrt(0.88)

        # Core Transport
        Cped = 0.17
        Ccore = 0.4

        delta = np.heaviside(_x - r_ped, 0.5)

        chi = (Ccore * (1.0 + 3 * (_x**2))) * (1 - delta) + Cped * delta
        chi_e = Ccore * (1.0 + 3 * (_x**2)) * (1 - delta) * 0.5 + Cped * delta

        D = 0.1 * (chi + chi_e)

        v_pinch_ne = 0.6 * D * _x / R0
        v_pinch_Te = 2.5 * chi_e * _x / R0
        v_pinch_ni = D * _x / R0
        v_pinch_Ti = chi * _x / R0

        current.flux_multiplier = 3 / 2

        current.profiles_1d.grid_d = eq_1d.grid
        current.profiles_1d.electrons.particles.d = D
        current.profiles_1d.electrons.particles.v = -v_pinch_ne
        current.profiles_1d.electrons.energy.d = chi_e
        current.profiles_1d.electrons.energy.v = -v_pinch_Te

        current.profiles_1d.ion.extend(
            [
                {"@name": "D", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}},
                {"@name": "T", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}},
                {"@name": "He", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}},
            ]
        )

        return current


CoreTransport.Model.register(["predefined"], PredefinedTransport)
