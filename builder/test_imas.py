from _imas import _T_equilibrium

from spdm.util.logger import logger


if __name__ == "__main__":

    eq = _T_equilibrium({"time_slice": [{"global_quantities": {"beta_pol": 1.23450}, }]})

    logger.debug(eq.time_slice[0].global_quantities.beta_pol)