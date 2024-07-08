__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from fytok.utils.logger import logger
from fytok.utils.envs import FY_ONTOLOGY

from spdm.core import mapper

FY_MAPPING_PATH = os.environ.get("FY_MAPPING_PATH", "")

mapper.path = ":".join(
    map(
        lambda v: v + "/" + FY_ONTOLOGY,
        filter(
            lambda v: v != "",
            [
                "fytok/mappings/{schema}",
                *FY_MAPPING_PATH.split(":"),
                *(mapper.path.split(":") if isinstance(mapper.path, str) else mapper.path),
            ],
        ),
    )
)


try:
    from . import imas_latest as latest
except ImportError as error:
    from . import dummy as latest

logger.verbose(f"Using ontology: {FY_ONTOLOGY} ({latest.__implement__}) at {mapper.path}")

__VERSION__ = FY_ONTOLOGY


def __getattr__(key: str):
    mod = getattr(latest, key, None)
    if mod is None:
        raise ModuleNotFoundError(f"Module {key} not found in ontology")
    return mod
