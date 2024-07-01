__path__ = __import__("pkgutil").extend_path(__path__, __name__)


from .utils.envs import *
from . import _mapping

__version__ = FY_VERSION


if FY_VERBOSE != "quiet":
    from spdm.utils.logger import logger
    from spdm.utils.envs import SP_MPI

    if SP_MPI is None or SP_MPI.COMM_WORLD.Get_rank() == 0:  # 粗略猜测是否在交互环境下运行
        logger.info(FY_LOGO)
