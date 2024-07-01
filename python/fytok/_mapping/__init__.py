__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import spdm.core.entry as entry

from fytok.utils.logger import logger

entry.mapping_path.extend(__path__)

logger.verbose(f"Mapping path: {__path__}")

# from importlib import resources as impresources
# try:
#     import spdm.core.entry as entry_
#     entry_._mapping_path.extend([p.resolve() for p in impresources.files(_mapping)._paths])
# except Exception as error:
#     raise FileNotFoundError(f"Can not find mappings!") from error
