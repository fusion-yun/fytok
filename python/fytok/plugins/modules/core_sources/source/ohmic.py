import numpy as np
import scipy.constants
import typing
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium

from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_
from spdm.core.expression import Expression, Variable, zero
from spdm.core.sp_property import sp_tree

from fytok.modules.core_sources import CoreSources
from fytok.utils.atoms import atoms
from fytok.utils.logger import logger


@sp_tree
class Ohmic(CoreSources.Source):
    identifier = "collisional_equipartition"

    code = {"name": "ohmic", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(profiles_1d)

        #! +++ Radial electric field and ohmic heating:
        #      rho_loop13: DO irho=1,nrho
        #         if (sigma(irho).eq.0.0) then
        #           e_par(irho)=0.0
        #         else
        #           e_par(irho)              = ( curr_par(irho) - curr_ni_exp(irho)            &
        #                                   - curr_ni_imp(irho)*y(irho) ) / sigma(irho)
        #         end if
        #         qoh(irho)                = sigma(irho)*e_par(irho)**2 * control%ohmic_heating_multiplier
        #      END DO rho_loop13

        #    #!+++ Current diagnostics:
        #      fun7                       =  vpr * curr_par / 2.0e0_r8 / itm_pi * bt / fdia**2
        #      CALL integr2(nrho, rho, fun7, intfun7)
        #      curr_tot                   =  intfun7(nrho) * fdia(nrho)
        #      fun7                       =  vpr * (curr_ni_exp + curr_ni_imp * psi) / 2.0e0_r8 / itm_pi * bt / fdia**2
        #      CALL integr2(nrho, rho, fun7, intfun7)
        #      curr_ni                    =  intfun7(nrho) * fdia(nrho)

        #      fun7                       = qoh * vpr
        #      CALL integr2(nrho, rho, fun7, intfun7)
        #      qoh_tot                    = intfun7(nrho)
