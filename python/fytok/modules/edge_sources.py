"""Source terms for the edge transport equations"""

from spdm.core.time import WithTime
from spdm.model.context import Context

from fytok.utils.base import IDS, FyEntity


class EdgeSources(WithTime, IDS, Context, FyEntity, code={"name": "edge_sources"}):
    pass
