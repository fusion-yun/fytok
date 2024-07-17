import abc
from spdm.utils.type_hint import array_type
from spdm.utils.tags import _not_found_
from spdm.core.path import Path
from spdm.core.htree import List
from spdm.core.sp_tree import SpTree
from spdm.core.sp_object import SpObject
from spdm.core.sp_tree import AttributeTree
from spdm.core.spacetime import SpacetimeVolume
from spdm.core.pluggable import Pluggable

from spdm.model.entity import Entity
from spdm.model.actor import Actor
from spdm.model.context import Context
from spdm.model.component import Component

from fytok.utils.envs import FY_VERSION, FY_COPYRIGHT
from fytok.utils.logger import logger


class IDSProperties(SpTree):
    comment: str
    homogeneous_time: int
    provider: str
    creation_date: str
    version_put: AttributeTree
    provenance: AttributeTree


class Library(SpTree):
    name: str
    commit: str
    version: str = "0.0.0"
    repository: str = ""
    parameters: AttributeTree


class Code(SpTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cache = Path().update(self._cache, self._parent._metadata.get("code", _not_found_))

    name: str
    """代码名称，也是调用 plugin 的 identifier"""

    module_path: str
    """模块路径， 可用于 import 模块"""

    commit: str
    version: str = FY_VERSION
    copyright: str = FY_COPYRIGHT
    repository: str = ""
    output_flag: array_type
    library: List[Library]

    parameters: AttributeTree
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    def __str__(self) -> str:
        return "-".join([s for s in [self.name, self.version] if isinstance(s, str)])


class Identifier(SpTree):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            pass
        elif isinstance(args[0], str):
            args = ({"name": args[0]}, *args[1:])
        elif isinstance(args[0], int):
            args = ({"int": args[0]}, *args[1:])
        super().__init__(*args, **kwargs)

    name: str
    index: int
    description: str


class IDS(abc.ABC):
    ids_properties: IDSProperties


class FyModule(Pluggable, plugin_prefix="fytok/modules/"):

    _plugin_registry = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs) -> None:
        if plugin_name is None:
            plugin_name = Path("code/name").get(kwargs, None)
        return super().__init_subclass__(plugin_name=plugin_name, **kwargs)

    identifier: str

    code: Code
    """代码信息"""


class FySpacetimeVolume(SpacetimeVolume):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._entry is not None:
            time = self._cache.get("time", _not_found_)
            if time is _not_found_:
                self._entry = self._entry.child(["time_slice", -1])
            else:
                self._entry = self._entry.child(["time_slice", {"time": time}])
