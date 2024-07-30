"""
This module contains the base classes and definitions for the fytok package.

Classes:
- IDSProperties: Represents the properties of an IDS (Integrated Development System).
- Library: Represents a library used by the code.
- Code: Represents a code module.
- Identifier: Represents an identifier.
- IDS: Abstract base class for IDS.
- FyEntity: Represents an entity in the fytok package.

"""

import abc
import typing
from spdm.utils.type_hint import array_type
from spdm.utils.tags import _not_found_
from spdm.core.path import Path
from spdm.core.htree import List
from spdm.core.sp_tree import SpTree, annotation, sp_property
from spdm.core.sp_tree import AttributeTree

from spdm.model.entity import Entity


from fytok.utils.envs import FY_VERSION, FY_COPYRIGHT


class IDSProperties(SpTree):
    """
    Represents the properties of an IDS (Integrated Development System).

    Attributes:
    - comment: A comment for the IDS.
    - homogeneous_time: The homogeneous time of the IDS.
    - provider: The provider of the IDS.
    - creation_date: The creation date of the IDS.
    - version_put: The version put of the IDS.
    - provenance: The provenance of the IDS.
    """

    comment: str
    homogeneous_time: int
    provider: str
    creation_date: str
    version_put: AttributeTree
    provenance: AttributeTree


class Code(SpTree):
    """
    Represents a code module.

    Attributes:
    - name: The name of the code.
    - version: The version of the code.
    - copyright: The copyright of the code.
    - parameters: The parameters of the code.
    """

    name: str = "dummy"
    version: str
    copyright: str

    parameters: AttributeTree = {}
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    def __str__(self) -> str:
        return "-".join([s for s in [self.name, self.version] if isinstance(s, str)])


class Identifier(SpTree):
    """
    Represents an identifier.

    Attributes:
    - name: The name of the identifier.
    - index: The index of the identifier.
    - description: The description of the identifier.
    """

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


class WithIdentifier(abc.ABC):
    identifier: Identifier = annotation(alias="_metadata/identifier", default_value="unamed")
    """标识符"""

    def __hash__(self) -> int:
        return hash(self.identifier.index)


class IDS(abc.ABC):
    """
    Abstract base class for IDS (Interface Data Structure).

    Attributes:
    - ids_properties: The properties of the IDS.
    """

    ids_properties: IDSProperties


class FyEntity(Entity, plugin_prefix="fytok/plugins/modules/"):
    """
    Represents an entity in the fytok package.

    Attributes:
    - identifier: The identifier of the entity.
    - code: The code module of the entity.
    """

    _plugin_registry = {}

    def __new__(cls, *args, plugin_name: str = None, **kwargs) -> typing.Self:
        if plugin_name is None and len(args) > 0 and isinstance(args[0], dict):
            plugin_name = Path("code/name").get(args[0], None)
        if plugin_name is None:
            plugin_name = Path("code/name").get(kwargs, None)
        return super().__new__(cls, *args, plugin_name=plugin_name, **kwargs)

    def __init_subclass__(cls, plugin_name: str = None, **kwargs) -> None:
        if plugin_name is None:
            plugin_name = Path("code/name").get(kwargs, None)
        super().__init_subclass__(plugin_name=plugin_name, **kwargs)

    identifier: str = annotation(alias="_metadata/identifier")  # type:ignore
    """模块标识符"""

    code: Code = annotation(alias="_metadata/code", default_value={})  # type:ignore
    """代码信息"""

    def __hash__(self) -> int:
        label = self.identifier
        if label is _not_found_:
            label = self.code.name
        return hash(label)

    def __str__(self) -> str:
        return str(self.code)
