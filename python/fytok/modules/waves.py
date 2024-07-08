import typing
from fytok.utils.base import IDS, FyActor
from fytok.ontology import waves

_TWavesSlice = typing.TypeVar("_TWavesSlice")


class Waves(
    IDS,
    FyActor[_TWavesSlice],
    waves.waves,
    plugin_default="fy_eq",
    plugin_prefix="waves/",
):
    """描述电磁波/等离子体波。。。
    =========================================
    """
