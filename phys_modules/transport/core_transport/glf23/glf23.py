
import collections

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import (CoreTransport,
                                                   CoreTransportProfiles1D)
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Function import Function
from spdm.data.Node import _next_
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger

from .glf23 import glf2d


class GLF23(CoreTransport.Model):
    r"""
        GLF23
        ===============================
            - 2D GLF equations with massless isothermal passing electrons from

        References:
        =============
            - Waltz et al, Phys. of Plasmas 6(1995)2408
    """

    def __init__(self, d, *args,  **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": "anomalous",
                "index": 6,
                "description": f"{self.__class__.__name__}"
            }}, d or {}), *args, **kwargs)

    def update(self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs):
        super().update(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
        prof = self.profiles_1d[-1]
        return 0.0