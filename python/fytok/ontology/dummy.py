import functools

from spdm.core.property_tree import PropertyTree
from spdm.core.sp_tree import SpTree
from spdm.utils.misc import camel_to_snake
from fytok.ontology.catalogy import catalogy
from fytok.modules.utilities import IDS


__version__ = "dummy"


class DummyModule(object):
    def __init__(self, name, cache=None):
        self._module = name
        self._cache = cache or {}

    def __getattr__(self, name: str):
        cls = self._cache.get(name, None)

        if cls is not None:
            return cls
        tp_bases = catalogy.get(camel_to_snake(name).lower(), None)

        if tp_bases is None:
            tp_bases = ()
        else:
            if not isinstance(tp_bases, tuple):
                tp_bases = (IDS, tp_bases)
            else:
                tp_bases = (IDS, *tp_bases)

        if not any(issubclass(tp, PropertyTree) for tp in tp_bases):
            tp_bases = tp_bases + (PropertyTree, SpTree)

        new_cls = type(
            name,
            tp_bases,
            {
                "__module__": f"{__package__}.{self._module}",
                "__package__": __package__,
            },
        )
        self._cache[name] = new_cls
        return new_cls


@functools.lru_cache
def _find_module(key):
    return DummyModule(key)


def __getattr__(key: str):
    return _find_module(key)
