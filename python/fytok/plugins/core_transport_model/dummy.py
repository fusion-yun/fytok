
import collections

from fytok.modules.CoreTransport import CoreTransport


class TransportModelDummy(CoreTransport.Model):
    """
       Dummy CoreTransport.Model
       ===============================

    """

    def __init__(self, d, *args,   **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {"name": "dummy", "index": 5,
                           "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "},
            "code": {"name": "dummy"}}, d or {}),
            *args, **kwargs)

    def update(self, *args, **kwargs) -> float:
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = TransportModelDummy