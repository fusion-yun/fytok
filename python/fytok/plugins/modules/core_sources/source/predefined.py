import numpy as np
import scipy.constants
from spdm.core.sp_property import sp_tree
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.utilities import *

PI = scipy.constants.pi


@sp_tree
class PredefinedSource(CoreSources.Source):
    identifier = "predefined"
    code = {"name": "predefined", "description": f"predefined"}

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D, *args, **kwargs) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(*args, **kwargs)

        rho_tor_norm = profiles_1d.rho_tor_norm

        _x = rho_tor_norm

        S = 9e20 * np.exp(15.0 * (_x**2 - 1.0))

        current.profiles_1d.grid = profiles_1d.grid
        current.profiles_1d.electrons.particles = S

        current.profiles_1d.ion.extend(
            [
                {"@name": "D", "particles": S * 0.5},
                {"@name": "T", "particles": S * 0.5},
                # {"@name": "He", "particles": S * 0.02},
            ]
        )

        return current


CoreSources.Source.register(["predefined"], PredefinedSource)
