__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import spdm.core.mapper as mapper
from fytok.utils.envs import FY_VERBOSE, FY_LOGO
from fytok.__version__ import __version__


if FY_VERBOSE != "quiet":
    from spdm.utils.logger import logger
    from spdm.utils.envs import SP_MPI

    if SP_MPI is None or SP_MPI.COMM_WORLD.Get_rank() == 0:  # 粗略猜测是否在交互环境下运行
        logger.info(FY_LOGO)

mapper.default_namespace = "fytok/mappings/{schema}/imas/3"
