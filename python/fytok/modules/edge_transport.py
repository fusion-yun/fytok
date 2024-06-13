from functools import cached_property

from spdm.utils.tags import _undefined_
from  ..ontology import edge_transport import _T_edge_transport, _T_edge_transport_model


class EdgeTransport(_T_edge_transport):

    Model = _T_edge_transport_model

    @cached_property
    def model_combiner(self) -> Model:
        return self.model.combine({
            "identifier": {"name": "combined", "index": 1,
                           "description": """Combination of data from all available transport models"""},
            "code": {"name": _undefined_}
        })

    def update(self, *args, **kwargs) -> float:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]
        return sum([model.refresh(*args,   **kwargs) for model in self.model])
