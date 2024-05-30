from spdm.utils.logger import logger
from .envs import FY_LABEL,FY_LOG

logger.name = FY_LABEL
logger.setLevel(FY_LOG)

__all__ = ["logger"]
