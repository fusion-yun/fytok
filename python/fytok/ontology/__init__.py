__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os

from fytok.utils.logger import logger

GLOBAL_ONTOLOGY = os.environ.get("FY_ONTOLOGY", f"imas/3")

try:
    from . import imas_latest as latest
except ImportError as error:
    from . import dummy as latest

logger.verbose(f"Using ontology: {latest.__version__}")


def __getattr__(key: str):
    mod = getattr(latest, key, None)
    if mod is None:
        raise ModuleNotFoundError(f"Module {key} not found in ontology")
    return mod
