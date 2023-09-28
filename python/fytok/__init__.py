__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from .utils.logger import logger
from .__version__ import __version__

logger.info(
    rf"""
#######################################################################################################################
    ______      _____     _
   / ____/_  __|_   _|__ | | __
  / /_  / / / /  | |/ _ \| |/ /
 / __/ / /_/ /   | | (_) |   <
/_/    \__, /    |_|\___/|_|\_\
      /____/      
Copyright (c) 2021-present Zhi YU (Institute of Plasma Physics Chinese Academy of Sciences) 
URL         : https://github.com/fusion-yun/fytok
version = {__version__}
#######################################################################################################################
"""
)

try:
    from importlib import resources as impresources
    from . import _mapping
    from spdm.data.Entry import EntryProxy

    EntryProxy._mapping_path.extend(impresources.files(_mapping)._paths)

except Exception as error:
    raise FileNotFoundError(f"Can not find mappings!") from error
else:
    logger.info(f"Mapping path {EntryProxy._mapping_path}")
