import typing
from spdm.model.entity import Entity
from spdm.model.context import Context

from fytok.utils.base import IDS
from fytok.ontology import summary


class Summary(IDS, Entity, summary.summary):
    pass
