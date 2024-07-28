import numpy as np
import scipy.constants
from spdm.core.sp_tree import sp_tree
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.utilities import *

PI = scipy.constants.pi


class PredefinedSource(CoreSources.Source, identifier="predefined", code={"name": "predefined"}):
    """Predefined source"""

    def execute(self, core_profiles: CoreProfiles, *args, **kwargs) -> CoreSources.Source:
        current: CoreSources.Source = super().fetch(*args, **kwargs)

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
