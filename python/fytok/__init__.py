__path__ = __import__("pkgutil").extend_path(__path__, __name__)


import os
from fytok.__version__ import __version__

if os.environ.get("FY_VERBOSE", None) != "quiet":
    from spdm.utils.logger import logger
    from spdm.utils.envs import SP_MPI
    from .utils.envs import FY_LOGO

    if SP_MPI is None or SP_MPI.COMM_WORLD.Get_rank() == 0:  # 粗略猜测是否在交互环境下运行
        # from fytok.utils.runtime_info import FY_RUNTIME_INFO
        logger.info(FY_LOGO)
