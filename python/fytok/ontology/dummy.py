import functools

from spdm.core.htree import HTree

from spdm.core.sp_tree import WithAttribute, WithProperty

__implement__ = "dummy"


class DummyModule(object):
    def __init__(self, name, cache=None):
        self._module = name
        self._cache = cache or {}

    def __getattr__(self, name: str):
        cls = self._cache.get(name, None)

        if cls is not None:
            return cls

        # tp_bases = catalogy.get(camel_to_snake(name).lower(), None)
        # if tp_bases is None:
        #     tp_bases = ()
        # else:
        #     if not isinstance(tp_bases, tuple):
        #         tp_bases = (tp_bases,)
        # if not any(issubclass(tp, AttributeTree) for tp in tp_bases):
        #     tp_bases = tp_bases + (AttributeTree,)

        new_cls = type(
            name,
            (WithAttribute, WithProperty, HTree),
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
