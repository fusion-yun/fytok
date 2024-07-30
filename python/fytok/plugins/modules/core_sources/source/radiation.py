import scipy.constants
from spdm.core.expression import zero
from spdm.core.sp_tree import sp_tree
from spdm.utils.tags import _not_found_
from fytok.modules.amns_data import amns
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.utilities import *

PI = scipy.constants.pi


class Radiation(
    CoreSources.Source,
    identifier="radiation",
    code={
        "name": "radiation",
        "description": """
    Source from   bremsstrahlung and impurity line radiation, and synchrotron radiation 
    Reference:
        Synchrotron radiation
            - Trubnikov, JETP Lett. 16 (1972) 25.
    """,
    },
):
    """Source from   bremsstrahlung and impurity line radiation, and synchrotron radiation"""

    def execute(self, profiles_1d: CoreProfiles) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(profiles_1d)

        source_1d = current.profiles_1d

        ne = profiles_1d.electrons.density
        Te = profiles_1d.electrons.temperature

        if ne is _not_found_ or Te is _not_found_:
            raise RuntimeError(f"{ne} {Te}")

        Qrad = sum(
            [
                ne * ion.density * amns[ion.label].radiation(Te)
                for ion in profiles_1d.ion
                if ion.density is not _not_found_
            ],
            zero,
        )

        source_1d.electrons.energy -= Qrad

        return current
