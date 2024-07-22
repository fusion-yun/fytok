import typing

from spdm.model.actor import Actor
from fytok.utils.base import IDS, FyModule
from fytok.ontology import waves


class Waves(
    IDS,
    FyModule,
    Actor,
    waves.waves,
    plugin_prefix="waves/",
):
    """描述电磁波/等离子体波。。。
    =========================================
    """
