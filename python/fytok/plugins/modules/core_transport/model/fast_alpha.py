import numpy as np

from spdm.core.expression import derivative
from spdm.core.sp_property import sp_tree

from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium

from .predefined import PredefinedTransport


@sp_tree
class FastAlpha(PredefinedTransport):
    """
    FastAlpha   Model

    Reference:

    [1] Angioni, C., Peeters, A. G., Pereverzev, G. V., Bottino, A., Candy, J., Dux, R., Fable, E., Hein, T., & Waltz, R. E. (2009).
        Gyrokinetic simulations of impurity, He ash and \\alpha particle transport and consequences on ITER transport modelling.
        Nuclear Fusion, 49(5), 055013. https://doi.org/10.1088/0029-5515/49/5/055013
    [2] Waltz, R. E., & Bass, E. M. (2014).
        Prediction of the fusion alpha density profile in ITER from local marginal stability to Alfvén eigenmodes.
        Nuclear Fusion, 54(10), 104006. https://doi.org/10.1088/0029-5515/54/10/104006

    """

    identifier = "alpha"
    code = {"name": "fast_alpha", "description": f" Fast alpha, Angioni's model"}

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args, **kwargs) -> CoreTransport.Model.TimeSlice:
        current: CoreTransport.Model.TimeSlice = super().fetch(profiles_1d, *args, **kwargs)

        eq: Equilibrium.TimeSlice = self.inports["equilibrium/time_slice/current"].fetch()

        R0 = eq.vacuum_toroidal_field.r0

        rho_tor_norm = profiles_1d.rho_tor_norm

        # _x = rho_tor_norm

        # chi = current.profiles_1d.ion["D"].energy.d

        # chi_e = current.profiles_1d.electrons.energy.d

        D = current.profiles_1d.ion["D"].particles.d  # 0.1 * (chi + chi_e)

        Te = profiles_1d.electrons.temperature

        inv_L_Te = -derivative(Te, rho_tor_norm) / Te

        Te_Ea = Te / 3.5e6  # Te/ 3.5MeV

        Ec_Ea = 33.05 * Te_Ea

        r_ped = 0.96  #

        delta = np.heaviside(rho_tor_norm - r_ped, 0.5)

        fast_factor_d = (0.02 + 4.5 * (Te_Ea) + 8.0 * (Te_Ea**2) + 350 * (Te_Ea**3)) * D

        Cs = 1.5 * (1.0 / np.log((Ec_Ea ** (-1.5) + 1) * (Ec_Ea**1.5 + 1)) - 1) * inv_L_Te

        fast_factor_v = fast_factor_d * rho_tor_norm / R0  # Cs * (1 - delta)

        current.profiles_1d.ion = [{"@name": "alpha", "particles": {"d": fast_factor_d, "v": -fast_factor_v}}]

        return current


CoreTransport.Model.register(["fast_alpha"], FastAlpha)
