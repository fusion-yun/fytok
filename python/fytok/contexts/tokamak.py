import typing

from spdm.utils.tags import _not_found_
from spdm.core.htree import List, Set
from spdm.core.history import WithHistory

from spdm.model.context import Context

# ---------------------------------
from fytok.utils.envs import *
from fytok.utils.logger import logger
from fytok.utils.base import IDS, FyModule

# ---------------------------------
from fytok.modules.dataset_fair import DatasetFAIR
from fytok.modules.summary import Summary

from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSourcesSource
from fytok.modules.core_transport import CoreTransportModel

from fytok.modules.ec_launchers import ECLaunchers
from fytok.modules.ic_antennas import ICAntennas
from fytok.modules.interferometer import Interferometer
from fytok.modules.lh_antennas import LHAntennas
from fytok.modules.magnetics import Magnetics
from fytok.modules.nbi import NBI
from fytok.modules.pellets import Pellets
from fytok.modules.pf_active import PFActive
from fytok.modules.tf import TF
from fytok.modules.wall import Wall

from fytok.modules.transport_solver import TransportSolver
from fytok.modules.equilibrium_solver import EquilibriumSolver

# from fytok.ontology import GLOBAL_ONTOLOGY

# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------


class Tokamak(IDS, FyModule, WithHistory, Context, code={"name": "fy_tok"}):
    # fmt:off
    dataset_fair            : DatasetFAIR
    summary                 : Summary

    # device
    wall                    : Wall

    # magnetics
    tf                      : TF
    pf_active               : PFActive
    magnetics               : Magnetics

    # aux
    ec_launchers            : ECLaunchers
    ic_antennas             : ICAntennas
    lh_antennas             : LHAntennas
    nbi                     : NBI
    pellets                 : Pellets

    # diag
    interferometer          : Interferometer

    # transport: state of device
    equilibrium             : Equilibrium

    core_profiles           : CoreProfiles
    
    core_transport          : Set[CoreTransportModel]
    core_sources            : Set[CoreSourcesSource]

    # edge_profiles         : EdgeProfiles
    # edge_transport        : EdgeTransport
    # edge_sources          : EdgeSources
    # edge_transport_solver : EdgeTransportSolver

    # solver
    equilibrium_solver      : EquilibriumSolver
    transport_solver        : TransportSolver

    # fmt:on

    def __str__(self) -> str:
        return f"""------------------------------------------------------------------------------------------------------------------------
{self.dataset_fair}
------------------------------------------------------------------------------------------------------------------------
{super().__str__()}
------------------------------------------------------------------------------------------------------------------------
"""

    @property
    def title(self) -> str:
        """标题，由初始化信息 dataset_fair.description"""
        return f"{self.dataset_fair.description}  time={self.time:.2f}s"

    @property
    def tag(self) -> str:
        """当前状态标签，由程序版本、用户名、时间戳等信息确定"""
        return f"{self.dataset_fair.tag}_{int(self.time*100):06d}"

    def __init__(
        self,
        *args,
        device: str = None,
        shot: int = None,
        run: int = None,
        **kwargs,
    ):
        """
        用于集成子模块，以实现工作流。

        现有子模块包括： wall, tf, pf_active, magnetics, equilibrium, core_profiles, core_transport, core_sources, transport_solver

        :param args:   初始化数据，可以为 dict，str 或者  Entry。 输入会通过数据集成合并为单一的HTree，其子节点会作为子模块的初始化数据。
        :param device: 指定装置名称，例如， east，ITER, d3d 等
        :param shot:   指定实验炮号
        :param run:    指定模拟计算的序号
        :param time:   指定当前时间
        :param kwargs: 指定子模块的初始化数据，，会与args中指定的数据源子节点合并。
        """

        dataset_fair = {"description": {"device": device, "shot": shot or 0, "run": run or 0}}

        if device is not None:
            args = (f"{device}://", *args)

        super().__init__(*args, **kwargs, dataset_fair=dataset_fair)

    def solve(self, *args, **kwargs) -> None:
        solver_1d = self.transport_solver.refresh(*args, time=self.time, **kwargs)
        profiles_1d = self.transport_solver.fetch()

        self.core_profiles["profiles_1d"] = profiles_1d

        return solver_1d
