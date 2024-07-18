import typing
from spdm.core.domain import WithDomain
from spdm.core.spacetime import SpacetimeVolume
from spdm.core.sp_tree import SpTree
from spdm.core.mesh import Mesh
from spdm.model.entity import Entity

from fytok.utils.base import IDS, FyModule
from fytok.modules.utilities import VacuumToroidalField


class PlasmaGlobalQuantities(SpTree):
    pass


class PlasmaProfiles1D(WithDomain, SpTree, domain="psi_norm"):
    pass


class PlasmaProfiles2D(WithDomain, SpTree, domain="grid"):
    grid: Mesh


class PlasmaProfiles(IDS, FyModule, SpacetimeVolume, Entity, code={"name": "plasma_profiles"}):
    """
    Plasma profiles core+edge
    """

    vacuum_toroidal_field: VacuumToroidalField

    GlobalQuantities = PlasmaGlobalQuantities
    global_quantities: PlasmaGlobalQuantities

    Profiles1D = PlasmaProfiles1D
    profiles_1d: PlasmaProfiles1D

    Profiles2D = PlasmaProfiles2D
    profiles_2d: PlasmaProfiles2D
