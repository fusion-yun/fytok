from __future__ import annotations
import typing
from spdm.core.path import update_tree
from spdm.core.entry import open_entry
from spdm.core.htree import HTree
from spdm.core.actor import Actor
from spdm.core.context import Context
from spdm.core.sp_property import sp_tree
from spdm.core.geo_object import GeoObject
from spdm.utils.tags import _not_found_

# ---------------------------------
from .utils.envs import *
from .utils.logger import logger

# ---------------------------------
from .modules.dataset_fair import DatasetFAIR
from .modules.summary import Summary
from .modules.core_profiles import CoreProfiles
from .modules.core_sources import CoreSources
from .modules.core_transport import CoreTransport
from .modules.ec_launchers import ECLaunchers
from .modules.equilibrium import Equilibrium
from .modules.ic_antennas import ICAntennas
from .modules.interferometer import Interferometer
from .modules.lh_antennas import LHAntennas
from .modules.magnetics import Magnetics
from .modules.nbi import NBI
from .modules.pellets import Pellets
from .modules.pf_active import PFActive
from .modules.tf import TF
from .modules.wall import Wall
from .modules.transport_solver_numerics import TransportSolverNumerics
from .modules.utilities import *

from .ontology import GLOBAL_ONTOLOGY
# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------


@sp_tree
class Tokamak(Context):
    # fmt:off
    dataset_fair            : DatasetFAIR               

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
    core_transport          : CoreTransport             
    core_sources            : CoreSources               

    # edge_profiles         : EdgeProfiles              
    # edge_transport        : EdgeTransport             
    # edge_sources          : EdgeSources               
    # edge_transport_solver : EdgeTransportSolver       

    # solver
    transport_solver        : TransportSolverNumerics   

    summary                 : Summary                   
    # fmt:on

    @property
    def brief_summary(self) -> str:
        """综述模拟内容"""
        return f"""{FY_LOGO}
---------------------------------------------------------------------------------------------------
                                                Brief Summary
---------------------------------------------------------------------------------------------------
Dataset Description:
{self.dataset_fair}
---------------------------------------------------------------------------------------------------
Modules:
    transport_solver        : {self.transport_solver.code }
    equilibrium             : {self.equilibrium.code }

    core_profiles           : {self.core_profiles.code }             
    core_transport          : {', '.join([str(s.code).split(".")[-1] for s in self.core_transport.model])}
    core_sources            : {', '.join([str(s.code).split(".")[-1]  for s in self.core_sources.source])}
---------------------------------------------------------------------------------------------------
"""

    code: Code = {"name": "fy_tok"}

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
        time: float = None,
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
        cache, entry, parent, kwargs = HTree._parser_args(*args, **kwargs)

        cache = update_tree(cache, kwargs)

        cache["dataset_fair"] = {"description": {"entry": entry, "device": device, "shot": shot or 0, "run": run or 0}}

        entry = open_entry(entry, shot=shot, run=run, local_schema=device, global_schema=GLOBAL_ONTOLOGY)

        super().__init__(cache, _entry=entry, _parent=parent)

        self._shot = shot
        self._run = run
        self._device = device
        self._metadata.setdefault("name", device)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.core_profiles.initialize(time=self.time)
        self.equilibrium.initialize(time=self.time, pf_active=self.pf_active, wall=self.wall, magnetics=self.magnetics)
        self.core_sources.initialize(time=self.time, equilibrium=self.equilibrium, core_profiles=self.core_profiles)
        self.core_transport.initialize(time=self.time, equilibrium=self.equilibrium, core_profiles=self.core_profiles)
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
                logger.warning(f"Failed to get {g.__class__.__name__}.__view__ ! {error}")  # , exc_info=error
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
