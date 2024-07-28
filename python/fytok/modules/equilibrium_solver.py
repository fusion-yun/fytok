from spdm.utils.tags import _not_found_
from spdm.model.process import Process

from fytok.utils.logger import logger
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.wall import Wall
from fytok.modules.magnetics import Magnetics
from fytok.modules.pf_active import PFActive
from fytok.modules.tf import TF
from fytok.utils.base import IDS, FyEntity


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

    Constraints = EequilibriumConstraints

    class InPorts(Process.InPorts):
        time: float
        wall: Wall
        magnetics: Magnetics
        pf_active: PFActive
        tf: TF
        core_profiles: CoreProfiles
        equilibrium: Equilibrium
        constraints: EequilibriumConstraints

    OutPorts = Equilibrium

    def execute(self, *args, time: float, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs):
        res = {
            "time": time,
            "vacuum_toroidal_field": equilibrium.vacuum_toroidal_field,
        }

        return Equilibrium(res)
