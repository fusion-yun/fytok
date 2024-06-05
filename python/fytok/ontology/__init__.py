__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
import typing

from ..utils.logger import logger


GLOBAL_ONTOLOGY = os.environ.get("FY_ONTOLOGY", f"imas/3")


try:
    from .imas_lastest.__version__ import __version__ as imas_version

    from .imas_lastest import (
        dataset_fair,
        summary,
        pulse_schedule,
        equilibrium,
        core_profiles,
        core_sources,
        ic_antennas,
        interferometer,
        lh_antennas,
        magnetics,
        nbi,
        pellets,
        core_transport,
        pf_active,
        tf,
        transport_solver_numerics,
        utilities,
        ec_launchers,
        amns_data,
        wall,
        waves,
    )

    __all__ = [
        "dataset_fair",
        "summary",
        "pulse_schedule",
        "equilibrium",
        "core_profiles",
        "core_sources",
        "ec_launchers",
        "ic_antennas",
        "interferometer",
        "lh_antennas",
        "magnetics",
        "nbi",
        "pellets",
        "amns_data",
        "core_transport",
        "wall",
        "pf_active",
        "tf",
        "transport_solver_numerics",
        "utilities",
        "waves",
    ]


except ModuleNotFoundError as error:
    logger.verbose(f"Failed to import IMAS ontology:")

    logger.info(f"Failed to import IMAS wrapper. Using dummy ontology as fallback.")

    from spdm.core.sp_property import PropertyTree

    imas_version = "None"

    class DummyModule:
        def __init__(self, name):
            self._module = name

        def __str__(self) -> str:
            return f"<dummy_module '{__package__}.dummy.{self._module}'>"

        def __getattr__(self, __name: str) -> typing.Type[PropertyTree]:
            cls = type(__name, (PropertyTree,), {})
            cls.__module__ = f"{__package__}.dummy.{self._module}"
            return cls

    def __getattr__(key: str) -> DummyModule:
        return DummyModule(key)

else:
    if GLOBAL_ONTOLOGY != f"imas/{imas_version[1:].split('_')[0]}":
        raise RuntimeError(
            f"Global ontology {GLOBAL_ONTOLOGY} is not compatible with IMAS version {imas_version[1:].split('.')[0]}"
        )
