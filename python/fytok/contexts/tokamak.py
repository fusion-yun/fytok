import typing
from spdm.core.path import update_tree
from spdm.core.entry import open_entry
from spdm.core.htree import HTree, List
from spdm.core.context import Context
from spdm.core.geo_object import GeoObject
from spdm.utils.tags import _not_found_

# ---------------------------------
from fytok.utils.envs import *
from fytok.utils.logger import logger

# ---------------------------------
from fytok.modules.dataset_fair import DatasetFAIR
from fytok.modules.summary import Summary
from fytok.modules.core_profiles import CoreProfiles
from fytok.modules.core_sources import CoreSources
from fytok.modules.core_transport import CoreTransport
from fytok.modules.ec_launchers import ECLaunchers
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.ic_antennas import ICAntennas
from fytok.modules.interferometer import Interferometer
from fytok.modules.lh_antennas import LHAntennas
from fytok.modules.magnetics import Magnetics
from fytok.modules.nbi import NBI
from fytok.modules.pellets import Pellets
from fytok.modules.pf_active import PFActive
from fytok.modules.tf import TF
from fytok.modules.wall import Wall
from fytok.modules.transport_solver_numerics import TransportSolverNumerics
from fytok.modules.utilities import Code, FyModule, IDS

# from fytok.ontology import GLOBAL_ONTOLOGY

# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------


class Tokamak(IDS, Context):
    # fmt:off

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
    core_transport          : List[CoreTransport.Model]  
    core_sources            : List[CoreSources.Source]   

    # edge_profiles         : EdgeProfiles              
    # edge_transport        : EdgeTransport             
    # edge_sources          : EdgeSources               
    # edge_transport_solver : EdgeTransportSolver       

    # solver
    transport_solver        : TransportSolverNumerics   

    summary                 : Summary

    code                    : Code = {"name": "fy_tok"}
    
    dataset_fair            : DatasetFAIR
    # fmt:on

    @property
    def description(self) -> str:
        """综述模拟内容"""
        return f"""{FY_LOGO}
---------------------------------------------------------------------------------------------------
                                                Description
---------------------------------------------------------------------------------------------------
{self.dataset_fair}
---------------------------------------------------------------------------------------------------
Modules:
    transport_solver        : {self.transport_solver.code }
    equilibrium             : {self.equilibrium.code }

    core_profiles           : {self.core_profiles.code }             
    core_transport          : {', '.join([s.code.name for s in self.core_transport])}
    core_sources            : {', '.join([s.code.name  for s in self.core_sources])}
---------------------------------------------------------------------------------------------------
"""

    @property
    def title(self) -> str:
        """标题，由初始化信息 dataset_fair.description"""
        return f"{self.dataset_fair.description}  time={self.time:.2f}s"

    @property
    def tag(self) -> str:
        """当前状态标签，由程序版本、用户名、时间戳等信息确定"""
        return f"{self.dataset_fair.description.tag}_{int(self.time*100):06d}"

    @property
    def shot(self) -> int:
        return self._shot

    @property
    def run(self) -> int:
        return self._run

    @property
    def device(self) -> str:
        return self._device

    def __init__(
        self,
        *args,
        device: str = _not_found_,
        shot: int = _not_found_,
        run: int = _not_found_,
        _entry=None,
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
        super().__init__(
            *args,
            **kwargs,
            dataset_fair={
                "description": {"device": device, "shot": shot or 0, "run": run or 0}
            },
        )

        self._shot = shot
        self._run = run
        self._device = device

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.core_profiles.initialize(time=self.time)
        self.equilibrium.initialize(
            time=self.time,
            pf_active=self.pf_active,
            wall=self.wall,
            magnetics=self.magnetics,
        )
        self.core_sources.initialize(
            time=self.time,
            equilibrium=self.equilibrium,
            core_profiles=self.core_profiles,
        )
        self.core_transport.initialize(
            time=self.time,
            equilibrium=self.equilibrium,
            core_profiles=self.core_profiles,
        )
        self.transport_solver.initialize(
            time=self.time,
            equilibrium=self.equilibrium,
            core_profiles=self.core_profiles,
            core_sources=self.core_sources,
            core_transport=self.core_transport,
        )

    def refresh(self, *args, **kwargs) -> None:
        super().refresh(*args, **kwargs)

        self.core_profiles.refresh(time=self.time)
        self.equilibrium.refresh(time=self.time)
        self.core_sources.refresh(time=self.time)
        self.core_transport.refresh(time=self.time)

    def solve(self, *args, **kwargs) -> None:
        solver_1d = self.transport_solver.refresh(*args, time=self.time, **kwargs)
        profiles_1d = self.transport_solver.fetch()

        self.core_profiles.time_slice.current["profiles_1d"] = profiles_1d

        return solver_1d

    def flush(self):
        profiles_1d = self.transport_solver.fetch()

        self.core_profiles.time_slice.current["profiles_1d"] = profiles_1d

        self.core_profiles.flush()
        self.equilibrium.flush()
        self.core_transport.flush()
        self.core_sources.flush()
        self.transport_solver.flush()

        super().flush()

    def __view__(self, **kwargs) -> GeoObject | typing.Dict:
        geo = {}

        o_list = [
            "wall",
            "equilibrium",
            "pf_active",
            "magnetics",
            "interferometer",
            "tf",
            # "ec_launchers",
            # "ic_antennas",
            # "lh_antennas",
            # "nbi",
            # "pellets",
        ]

        for o_name in o_list:
            try:
                g = getattr(self, o_name, None)
                if g is None:
                    continue
                g = g.__view__(**kwargs)

            except Exception as error:
                logger.error(
                    "Failed to get %s.__view__ ! ", g.__class__.__name__, exc_info=error
                )
                # raise RuntimeError(f"Can not get {g.__class__.__name__}.__view__ !") from error
            else:
                geo[o_name] = g

        view_point = (kwargs.get("view_point", None) or "rz").lower()

        styles = {}

        if view_point == "rz":
            styles["xlabel"] = r"Major radius $R [m] $"
            styles["ylabel"] = r"Height $Z [m]$"

        styles["title"] = kwargs.pop("title", None) or self.title

        geo["$styles"] = styles

        return geo
