import getpass
import datetime
import os

from fytok.utils.envs import FY_LOGO

FY_JOBID = f"fytok_{getpass.getuser().lower()}_{os.uname().nodename.lower()}_{os.getpid()}"

FY_RUNTIME_INFO = f"""
{FY_LOGO}
 Run by {getpass.getuser()} at {datetime.datetime.now().isoformat()}.
 Job ID: {FY_JOBID}
"""

__all__ = ["FY_JOBID", "FY_RUNTIME_INFO"]
