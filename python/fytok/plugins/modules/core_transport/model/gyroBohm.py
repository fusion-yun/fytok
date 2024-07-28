import numpy as np
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium


class GyroBohm(CoreTransport.Model, identifier="turbulence", code={"name": "gyroBohm"}):
    """
    Heat conductivity Anomalous gyroBohm
    ===============================

    References:
    =============
    - Tokamaks, Third Edition, Chapter  4.16  ,p197,  J.A.Wesson 2003
    """

    def execute(self, *args, core_profiles: CoreProfiles, equilibrium: Equilibrium, **kwargs) -> dict:
        res = super().execute(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)

        prof = self.profiles_1d[-1]
        rho_tor_norm = core_profiles.profiles_1d.grid.rho_tor_norm
        psi_norm = core_profiles.profiles_1d.grid.psi_norm

        Te = np.asarray(core_profiles.profiles_1d.electrons.temperature) / 1.0e3
        ne = np.asarray(core_profiles.profiles_1d.electrons.density) / 1.0e19
        mu = 1.0 / np.asarray(equilibrium.profiles_1d.q(psi_norm))

        for ion in prof.ion:
            # ion.particles.d = 0
            # ion.particles.v = 0
            ion.energy.d = Chi_i
            ion.energy.v = 0

        # prof.electrons.particles.d = 0
        # prof.electrons.particles.v = 0
        # prof.electrons.energy.d = Chi_e
        # prof.electrons.energy.v = 0

        return res
