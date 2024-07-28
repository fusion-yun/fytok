from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_transport import CoreTransport
from fytok.modules.equilibrium import Equilibrium


class ChangHiton(CoreTransport.Model, identifier="turbulence", code={"name": "chang_hiton"}):
    """
    Chang-Hiton formula for \Chi_{i}
    ===============================

    References:
    =============
    - Tokamaks, Third Edition, Chapter 14.11  ,p737,  J.A.Wesson 2003
    """

    def refresh(
        self,
        *args,
        core_profiles_1d: CoreProfiles,
        equilibrium: Equilibrium,
        **kwargs,
    ):
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = ChangHiton
