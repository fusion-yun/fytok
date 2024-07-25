import typing
from scipy import constants

from spdm.utils.tags import _not_found_
from spdm.utils.type_hint import array_type

from spdm.core.htree import List, Dict, HTree
from spdm.core.expression import Expression
from spdm.core.sp_tree import annotation, sp_property, SpTree, AttributeTree

from spdm.model.process import Process
from spdm.model.port import Ports

from fytok.utils.logger import logger
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.wall import Wall
from fytok.modules.magnetics import Magnetics
from fytok.modules.pf_active import PFActive
from fytok.modules.tf import TF
from fytok.utils.base import IDS, FyEntity

from fytok.modules.utilities import CoreRadialGrid

from fytok.ontology import equilibrium


class EequilibriumConstraints(equilibrium.equilibrium_constraints):
    pass


class EquilibriumSolver(
    IDS,
    FyEntity,
    Process,
    plugin_prefix="equilibrium_solver/",
):
    r"""Solve  GS equaiton"""

    class InPorts(Ports):
        wall: Wall
        magnetics: Magnetics
        pf_active: PFActive
        tf: TF
        core_profiles: CoreProfiles
        equilibrium: Equilibrium

    class OutPorts(Ports):
        equilibrium: Equilibrium

    Constraints = EequilibriumConstraints
    constraints: EequilibriumConstraints
