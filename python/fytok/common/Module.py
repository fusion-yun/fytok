
import collections
import collections.abc

from spdm.common.tags import _undefined_
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from ..common.Identifier import Identifier


class Module(Dict[Node]):

    @classmethod
    def create(cls, desc=None, *args, metadata=None, **kwargs):

        prefix = getattr(cls, "_fy_module_prefix", None)

        cls_name = None
        if metadata is None:
            metadata = desc

        if cls is not Module and prefix is None:
            pass
        elif isinstance(metadata, collections.abc.Mapping):
            cls_name = metadata.get("code", {}).get("name", None)
        elif isinstance(metadata, Entry):
            cls_name = metadata.get("code.name", None)

        if isinstance(cls_name, str):
            cls_name = f"{prefix}{cls_name}"

        if isinstance(cls_name, str):
            return super().create(cls_name)
        else:
            return object.__new__(cls)

    def __init__(self, d=None,  /, time=None,  **kwargs):
        super().__init__(d,  **kwargs)
        self._job_id = 0  # Session.current().job_id(self.__class__.__name__)
        self._time = time if time is not None else 0.0

    def __del__(self):
        # logger.debug(f"Delete Module {guess_class_name(self.__class__)}")
        pass

    @property
    def time(self):
        return self._time

    def job_id(self):
        return self._job_id

    code: Dict = sp_property()

    identifier: Identifier = sp_property()

    comment: str = sp_property()

    def refresh(self,  *args, time=_undefined_, ** kwargs) -> float:
        """
            Function: update the current state of the Module without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        if time is not _undefined_:
            self._time = time
        return 0.0