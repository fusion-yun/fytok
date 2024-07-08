import os
from spdm.utils import envs as sp_envs
import spdm.core.mapper as mapper


FY_ONTOLOGY = "imas/3"
""" 本体版本 """


FY_COPYRIGHT = "(C) 2021-2024 Zhi YU @ Institute of Plasma Physics Chinese Academy of Sciences. All rights reserved."
""" 版权信息 """

try:
    from fytok.__version__ import version
except ModuleNotFoundError as error:
    FY_VERSION = "alpha"
else:
    FY_VERSION = version

try:
    from fytok.extension import tags as extension_tags
except ImportError:
    FY_EXT_VERSION = "n/a"
else:
    FY_EXT_VERSION = extension_tags

FY_LABEL = os.environ.get("FY_LABEL", "fytok")

FY_DEBUG = os.environ.get("FY_DEBUG", False)

FY_VERBOSE = os.environ.get("FY_VERBOSE", "info")


for k, v in os.environ.items():
    if k.startswith("FY_"):
        os.environ[f"SP_{k[3:]}"] = v


FY_LOGO = rf"""
########################################################################################################################
                                            ______      _____     _
                                           / ____/_  __|_   _|__ | | __
                                          / /_  / / / /  | |/ _ \| |/ /
                                         / __/ / /_/ /   | | (_) |   <
                                        /_/    \__, /    |_|\___/|_|\_\
                                              /____/

{FY_COPYRIGHT}
 version  = {FY_VERSION}  {FY_EXT_VERSION} spdm version = {sp_envs.SP_VERSION}  ontology = {FY_ONTOLOGY} 
########################################################################################################################
"""


__all__ = ["FY_LOGO", "FY_VERSION", "FY_EXT_VERSION", "FY_ONTOLOGY", "FY_COPYRIGHT", "FY_DEBUG", "FY_VERBOSE"]
