import typing

from spdm.core.path import Path
from spdm.core.htree import List
from spdm.core.sp_tree import SpTree
from spdm.core.sp_object import SpObject
from spdm.core.sp_tree import AttributeTree
from spdm.core.actor import Actor
from spdm.core.component import Component
from spdm.core.processor import Processor
from spdm.core.context import Context

from spdm.utils.type_hint import array_type
from spdm.utils.tags import _not_found_

from fytok.utils.logger import logger
from fytok.utils.envs import FY_JOBID


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

        self._cache = Path().update(
            self._cache,
            {
                "name": self._parent.__class__.__name__,
                "module_path": self._parent.__module__ + "." + self._parent.__class__.__name__,
            },
        )

    name: str
    """代码名称，也是调用 plugin 的 identifier"""

    module_path: str
    """模块路径， 可用于 import 模块"""

    commit: str
    version: str = "0.0.0"
    copyright: str = "NO_COPYRIGHT_STATEMENT"
    repository: str = ""
    output_flag: array_type
    library: List[Library]

    parameters: AttributeTree = {}
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    def __str__(self) -> str:
        return "-".join([s for s in [self.name, self.version.replace(".", "_")] if isinstance(s, str)])

    def __repr__(self) -> str:
        desc = {
            "name": self.name,
            "version": self.version,
            "copyright": self.copyright,
        }

        return ", ".join(
            [
                f"{key}='{value}'"
                for key, value in desc.items()
                if value is not _not_found_ and value is not None and value != ""
            ]
        )


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


class IDS(SpTree):
    ids_properties: IDSProperties


class FyModule(SpObject):

    _plugin_prefix = "fytok.modules."
    _plugin_registry = {}

    identifier: str

    code: Code

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self.code.module_path}"


_TSlice = typing.TypeVar("_TSlice")


class FyActor(FyModule, Actor[_TSlice]):

    def refresh(self, *args, **kwargs) -> typing.Type[_TSlice]:
        """更新当前 Actor 的状态。
        更新当前状态树 （time_slice），并执行 self.iteration+=1

        """
        logger.verbose(f"Refresh module {self.code.module_path}")

        current = super().refresh(*args, **kwargs)

        return current


class FyComponent(FyModule, Component):
    pass


class FyProcessor(FyModule, Processor):
    pass


class FyContext(FyModule, Context):
    pass
