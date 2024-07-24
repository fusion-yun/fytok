""" 等离子体分布 core+edge """

from spdm.core.time import WithTime
from spdm.core.domain import WithDomain
from spdm.core.sp_tree import SpTree
from spdm.core.mesh import Mesh

from fytok.utils.base import IDS, FyEntity
from fytok.modules.utilities import VacuumToroidalField


class PlasmaGlobalQuantities(SpTree):
    """零维量"""


class PlasmaProfiles1D(WithDomain, SpTree, domain="psi_norm"):
    """一维分布磁面"""


class PlasmaProfiles2D(WithDomain, SpTree, domain="grid"):
    """二维分布"""

    grid: Mesh


class PlasmaProfiles(FyEntity, WithTime, IDS, code={"name": "plasma_profiles"}):
    """
    等离子体分布 core+edge
    """

    vacuum_toroidal_field: VacuumToroidalField

    GlobalQuantities = PlasmaGlobalQuantities
    global_quantities: PlasmaGlobalQuantities

    Profiles1D = PlasmaProfiles1D
    profiles_1d: PlasmaProfiles1D

    Profiles2D = PlasmaProfiles2D
    profiles_2d: PlasmaProfiles2D
