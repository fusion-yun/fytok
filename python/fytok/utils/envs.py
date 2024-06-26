import os
import getpass
import datetime

try:
    from ..__version__ import version
except Exception:
    FY_VERSION = "alpha"
else:
    FY_VERSION = version

try:
    from ..extension import tags as extension_tags
except ImportError:
    FY_EXT_VERSION = "n/a"
else:
    FY_EXT_VERSION = extension_tags


for k, v in os.environ.items():
    if k.startswith("FY_"):
        os.environ[f"SP_{k[3:]}"] = v


FY_LABEL = os.environ.get("FY_LABEL", "fytok")

FY_DEBUG = os.environ.get("FY_DEBUG", True)

FY_LOG = os.environ.get("FY_LOG", FY_DEBUG)

FY_JOBID = f"fytok_{getpass.getuser().lower()}_{os.uname().nodename.lower()}_{os.getpid()}"

from spdm.utils import envs as sp_envs

FY_NO_LOGO = os.environ.get("FY_NO_LOGO", False)
if FY_NO_LOGO:
    FY_LOGO = False
else:
    FY_LOGO = rf"""
###################################################################################################

    ______      _____     _
   / ____/_  __|_   _|__ | | __
  / /_  / / / /  | |/ _ \| |/ /
 / __/ / /_/ /   | | (_) |   <
/_/    \__, /    |_|\___/|_|\_\
      /____/

 Copyright (c) 2021-present Zhi YU @ Institute of Plasma Physics Chinese Academy of Sciences 
            
 version = {FY_VERSION}  (spdm={sp_envs.SP_VERSION} {FY_EXT_VERSION})

 Run by {getpass.getuser()} at {datetime.datetime.now().isoformat()}.
 Job ID: {FY_JOBID}

###################################################################################################
"""

__all__ = ["FY_DEBUG", "FY_JOBID", "FY_LOGO", "FY_LOG", "FY_VERSION", "FY_EXT_VERSION"]
