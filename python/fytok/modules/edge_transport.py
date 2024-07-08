import typing
from spdm.core.htree import List
from spdm.core.actor import Actor

from fytok.utils.base import IDS, FyModule
from fytok.ontology import edge_transport

_TSlice = typing.TypeVar("_TSlice")


class EdgeTransportModel(FyModule, Actor[_TSlice], edge_transport.edge_transport_model):
    pass


class EdgeTransport(IDS, edge_transport.edge_transport):

    Model = edge_transport.edge_transport_model

    model = List[edge_transport.edge_transport_model]

    def update(self, *args, **kwargs) -> float:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]
        return sum([model.refresh(*args, **kwargs) for model in self.model])
