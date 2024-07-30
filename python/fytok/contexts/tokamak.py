""" Tokamak
    用于集成子模块，以实现工作流。

"""

import typing
import numpy as np
from spdm.utils.tags import _not_found_
from spdm.core.sp_tree import annotation
from spdm.core.time import WithTime
from spdm.model.context import Context
from spdm.model.component import Component

# ---------------------------------
from fytok.utils.envs import FY_VERSION
from fytok.utils.base import IDS, FyEntity

# ---------------------------------
from fytok.modules.dataset_fair import DatasetFAIR
from fytok.modules.summary import Summary

from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_transport import CoreTransport

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


class Tokamak(WithTime, IDS, Context, FyEntity, code={"name": "fy_tok", "version": FY_VERSION}):
    """Tokamak 整合子模块"""

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
        :param kwargs: 指定子模块的初始化数据，会与args中指定的数据源子节点合并。
        """

        dataset_fair = {"description": {"device": device, "shot": shot or 0, "run": run or 0}}

        if device is not None:
            args = (f"{device}://", *args)

        super().__init__(*args, **kwargs, dataset_fair=dataset_fair)

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
    equilibrium             : Equilibrium = annotation(input=True, output=True)
    core_profiles           : CoreProfiles = annotation(input=True, output=True)

    core_transport          : CoreTransport
    core_sources            : CoreSources

    # edge_profiles         : EdgeProfiles
    # edge_transport        : EdgeTransport
    # edge_sources          : EdgeSources
    # edge_transport_solver : EdgeTransportSolver

    # solver
    equilibrium_solver      : EquilibriumSolver
    transport_solver        : TransportSolver

    # fmt:on
    def __str__(self) -> str:
        """返回 Context 的字符串表示形式。

        Returns:
            str: Context 的字符串表示形式。
        """
        processor_summary = "\n".join(f"{k:>19s} : {e} " for k, e in self.processes())
        component_summary = ",".join(k for k, e in self.entities(Component))

        return f"""------------------------------------------------------------------------------------------------------------------------
{self.dataset_fair}
------------------------------------------------------------------------------------------------------------------------
- Context           : {self.code}
- Actors/Processors :
{processor_summary}
- Components        : {component_summary}
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

    def initialize(self):
        self.equilibrium_solver.in_ports.connect(self)
        self.transport_solver.in_ports.connect(self)

    def execute(self, *args, time: float, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs):
        """刷新 Context 的状态，将执行结果更新到各个子模块。"""
        tolerance = self.code.parameters.get("tolerance", 1.0e-6)

        rms = 10000

        core_profiles_prev = core_profiles
        core_profiles_next = core_profiles

        equilibrium_prev = equilibrium
        if time > equilibrium_prev.time:
            equilibrium_next = self.equilibrium_solver.execute(
                time=time,
                core_profiles=core_profiles_next,
                equilibrium=equilibrium,
            )
        else:
            equilibrium_next = equilibrium

        while rms > tolerance:
            equilibrium_iter = equilibrium_next

            core_profiles_next = self.transport_solver.execute(
                core_profiles_prev=core_profiles_prev,
                equilibrium_prev=equilibrium_prev,
                equilibrium_next=equilibrium_iter,
                core_transport=self.core_transport,
                core_sources=self.core_sources,
            )

            equilibrium_next = self.equilibrium_solver.execute(
                time=time,
                core_profiles=core_profiles_next,
                equilibrium=equilibrium_iter,
            )

            rms = np.sqrt(np.mean((equilibrium_next.profiles_2d.psi - equilibrium_iter.profiles_2d.psi) ** 2))

        return {
            "time": time,
            "core_profiles": core_profiles_next,
            "equilibrium": equilibrium_next,
        }
