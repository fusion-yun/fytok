import typing
import numpy as np
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransportModel
from fytok.modules.equilibrium import Equilibrium


class PredefinedTransport(CoreTransportModel, identifier="predefined", code={"name": "predefined"}):
    """Predefined transport model"""

    def execute(self, *args, core_profiles: CoreProfiles, equilibrium: Equilibrium, **kwargs) -> typing.Self:
        current = super().execute(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)

        rho_tor_norm = current.profiles_1d.grid.rho_tor_norm

        eq_1d = equilibrium.profiles_1d

        B0 = np.abs(equilibrium.vacuum_toroidal_field.b0)

        R0 = equilibrium.vacuum_toroidal_field.r0

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

        prof_1d: CoreTransport.Model.Profiles1D = current.profiles_1d

        prof_1d.grid_d = eq_1d.grid
        prof_1d.electrons.particles.d = D
        prof_1d.electrons.particles.v = -v_pinch_ne
        prof_1d.electrons.energy.d = chi_e
        prof_1d.electrons.energy.v = -v_pinch_Te

        prof_1d.ion.insert(
            {"name": "D", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}}
        )
        prof_1d.ion.insert(
            {"name": "T", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}}
        )
        prof_1d.ion.insert(
            {"name": "He", "particles": {"d": D, "v": -v_pinch_ni}, "energy": {"d": chi, "v": -v_pinch_Ti}}
        )

        return current
